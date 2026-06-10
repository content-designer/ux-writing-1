"""Convert the merged ux-writing-1 to GGUF (Q4_K_M + Q8_0) for llama.cpp / LM Studio / Ollama.

CPU-only Modal job: download merged model -> convert_hf_to_gguf (f16) -> quantize ->
sanity-generate -> upload quants to gr33r/ux-writing-1-GGUF. llama.cpp supports the
qwen3_5 arch (unsloth/lmstudio already publish Qwen3.6-27B GGUFs). Non-blocking if it
fails — the bf16 release stands alone.

    modal run --detach modal_app/convert_gguf.py
"""

import modal

MERGED_REPO = "gr33r/ux-writing-1"
GGUF_REPO = "gr33r/ux-writing-1-GGUF"
HF_CACHE = "/root/hf"
QUANTS = ["Q4_K_M", "Q8_0"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "cmake", "build-essential")
    .run_commands(
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /llama.cpp",
        "cmake -S /llama.cpp -B /llama.cpp/build -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF",
        "cmake --build /llama.cpp/build --config Release -j 8 --target llama-quantize llama-cli",
        # install conversion deps INSIDE the container (the requirements file lives in the image)
        "pip install -r /llama.cpp/requirements/requirements-convert_hf_to_gguf.txt",
    )
    .pip_install("huggingface_hub>=0.34.0", "hf_transfer")
    .env({"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("uxw1-gguf", image=image)
hf_secret = modal.Secret.from_name("hf-token")


@app.function(cpu=16, memory=98_304, ephemeral_disk=512 * 1024,  # Modal minimum 512 GiB
              secrets=[hf_secret], timeout=3 * 60 * 60)
def convert():
    import os
    import subprocess

    from huggingface_hub import HfApi, snapshot_download

    def run(cmd: list[str]):
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print("[1/5] download merged model")
    src = snapshot_download(MERGED_REPO)

    print("[2/5] convert -> f16 gguf")
    os.makedirs("/out", exist_ok=True)
    f16 = "/out/ux-writing-1-F16.gguf"
    run(["python", "/llama.cpp/convert_hf_to_gguf.py", src, "--outfile", f16, "--outtype", "f16"])

    api = HfApi()
    api.create_repo(GGUF_REPO, private=True, exist_ok=True)

    for quant in QUANTS:
        out = f"/out/ux-writing-1-{quant}.gguf"
        print(f"[3/5] quantize {quant}")
        run(["/llama.cpp/build/bin/llama-quantize", f16, out, quant])
        if quant == "Q4_K_M":
            print("[4/5] sanity generate (CPU, slow but real)")
            r = subprocess.run(
                ["/llama.cpp/build/bin/llama-cli", "-m", out, "-n", "48", "-no-cnv",
                 "--temp", "0", "-p",
                 "Rewrite this button label to be specific: 'OK' (saving a payment method). "
                 "Reply with JSON {\"rewrite\":...}.\n"],
                capture_output=True, text=True, timeout=900,
            )
            print("sanity output tail:", r.stdout[-500:])
            if r.returncode != 0:
                raise RuntimeError(f"llama-cli failed: {r.stderr[-500:]}")
        print(f"[5/5] upload {quant}")
        api.upload_file(path_or_fileobj=out, path_in_repo=os.path.basename(out),
                        repo_id=GGUF_REPO)
        os.remove(out)

    print(f"done -> https://huggingface.co/{GGUF_REPO}")


@app.local_entrypoint()
def main():
    call = convert.spawn()
    print(f"spawned gguf conversion; fc={call.object_id}")
