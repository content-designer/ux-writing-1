"""Compute per-(layer, module) Frobenius norms of the LoRA delta for the weights visual.

||B@A||_F is computed via Gram matrices — trace identity:
||BA||_F^2 = sum((B^T B) * (A A^T)) — so we never materialize the (out x in) delta.

    python3 scripts/adapter_heatmap.py        # downloads gr33r/ux-writing-1-lora
"""
from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np

ADAPTER_REPO = "gr33r/ux-writing-1-lora"
KEY_RE = re.compile(r"\.layers\.(?P<layer>\d+)\.(?:.*\.)?(?P<module>[A-Za-z0-9_]+)\.lora_A\.weight$")


def load_safetensors_np(path: Path) -> dict[str, np.ndarray]:
    """Minimal safetensors reader that handles BF16 (numpy has no native bfloat16).

    BF16 is the top 16 bits of an IEEE float32: widen uint16 -> uint32 << 16 and
    reinterpret. F32/F16 pass through normally.
    """
    raw = path.read_bytes()
    (hlen,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + hlen])
    buf = raw[8 + hlen:]
    out: dict[str, np.ndarray] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        data, dtype, shape = buf[start:end], meta["dtype"], meta["shape"]
        if dtype == "BF16":
            u16 = np.frombuffer(data, dtype=np.uint16).astype(np.uint32)
            arr = (u16 << 16).view(np.float32)
        elif dtype == "F32":
            arr = np.frombuffer(data, dtype=np.float32)
        elif dtype == "F16":
            arr = np.frombuffer(data, dtype=np.float16).astype(np.float32)
        else:
            continue  # non-float tensors aren't LoRA factors
        out[name] = arr.reshape(shape)
    return out


def fro_norm_ba(A: np.ndarray, B: np.ndarray) -> float:
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    return float(np.sqrt(np.sum((B.T @ B) * (A @ A.T))))


def compute_deltas(safetensors_path: Path) -> list[dict]:
    tensors = load_safetensors_np(safetensors_path)
    out = []
    for key, A in tensors.items():
        m = KEY_RE.search(key)
        if not m:
            continue
        b_key = key.replace(".lora_A.", ".lora_B.")
        if b_key not in tensors:
            continue
        out.append({"layer": int(m.group("layer")), "module": m.group("module"),
                    "norm": fro_norm_ba(A, tensors[b_key])})
    out.sort(key=lambda d: (d["layer"], d["module"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-file", type=Path, default=None,
                    help="local adapter_model.safetensors; default: download from the Hub")
    ap.add_argument("--out", type=Path, default=Path("docs/video_assets/adapter_deltas.json"))
    args = ap.parse_args()

    path = args.adapter_file
    if path is None:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(ADAPTER_REPO, "adapter_model.safetensors"))
    deltas = compute_deltas(path)
    if not deltas:
        raise SystemExit("no lora_A/lora_B pairs matched — inspect adapter keys")
    peak = max(d["norm"] for d in deltas)
    for d in deltas:
        d["norm"] = round(d["norm"], 4)
        d["intensity"] = round(d["norm"] / peak, 4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "adapter": ADAPTER_REPO,
        "metric": "Frobenius norm of B@A per (layer, module)",
        "layers": max(d["layer"] for d in deltas) + 1,
        "modules": sorted({d["module"] for d in deltas}),
        "deltas": deltas,
    }, indent=2) + "\n")
    print(f"wrote {len(deltas)} deltas to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
