import json
import struct

import numpy as np
from safetensors.numpy import save_file

from scripts.adapter_heatmap import compute_deltas, fro_norm_ba, load_safetensors_np


def _save_bf16_safetensors(path, tensors: dict):
    """Write a minimal safetensors file with BF16 dtype (numpy lib can't)."""
    header, buf, offset = {}, b"", 0
    for name, arr in tensors.items():
        bf16 = (arr.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
        data = bf16.tobytes()
        header[name] = {"dtype": "BF16", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(data)]}
        buf += data
        offset += len(data)
    hjson = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(hjson)) + hjson + buf)


def test_load_safetensors_np_reads_bf16(tmp_path):
    rng = np.random.default_rng(7)
    arr = rng.normal(size=(4, 6)).astype(np.float32)
    f = tmp_path / "bf16.safetensors"
    _save_bf16_safetensors(f, {"t": arr})
    loaded = load_safetensors_np(f)["t"]
    assert loaded.shape == (4, 6)
    # bf16 keeps ~3 significant digits
    assert np.allclose(loaded, arr, rtol=1e-2)


def test_fro_norm_ba_matches_dense():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 64)).astype(np.float32)   # lora_A: r x in
    B = rng.normal(size=(32, 8)).astype(np.float32)   # lora_B: out x r
    dense = float(np.linalg.norm(B @ A))
    assert abs(fro_norm_ba(A, B) - dense) / dense < 1e-5


def test_compute_deltas_groups_by_layer_and_module(tmp_path):
    rng = np.random.default_rng(1)
    tensors = {}
    for layer in (0, 1):
        for mod in ("q_proj", "gate_proj"):
            base = f"base_model.model.model.layers.{layer}.x.{mod}"
            tensors[f"{base}.lora_A.weight"] = rng.normal(size=(4, 16)).astype(np.float32)
            tensors[f"{base}.lora_B.weight"] = rng.normal(size=(16, 4)).astype(np.float32)
    f = tmp_path / "adapter_model.safetensors"
    save_file(tensors, str(f))
    deltas = compute_deltas(f)
    assert len(deltas) == 4
    assert {d["module"] for d in deltas} == {"q_proj", "gate_proj"}
    assert {d["layer"] for d in deltas} == {0, 1}
    assert all(d["norm"] > 0 for d in deltas)
