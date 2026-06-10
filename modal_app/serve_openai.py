"""Serve ux-writing-1 on Modal: OpenAI-compatible chat + the Copy Campfire battle endpoint.

One A100 container loads base + the ux-writing-1-lora adapter ONCE and serves both arms:
fine-tune = adapter enabled, base = `disable_adapter()` (the pattern validated in our evals).

    modal deploy modal_app/serve_openai.py
    # -> POST .../chat    (OpenAI-compatible /v1/chat/completions semantics; adapter on)
    # -> POST .../battle  (both arms for the arena: {finetune:{text,thinking,tokens,ms}, base:{...}})

Auth: requests must carry the shared token (payload field "_auth"), checked against the
AUTH_TOKEN env var from the `ux-arena-auth` Modal secret. The token lives server-side in
the HF Space's secrets / the CLI user's env — it is never exposed to browsers.

SELF-CONTAINED on purpose (no cross-file imports) so the Modal container never hits a
ModuleNotFoundError when it re-imports this module. Mirrors modal_app/train.py.
"""

import modal

BASE_MODEL = "Qwen/Qwen3.6-27B"
ADAPTER_REPO = "gr33r/ux-writing-1-lora"
SERVED_NAME = "gr33r/ux-writing-1"
MODEL_CACHE = "/cache"
HF_CACHE = "/cache/hf"

# Inlined from uxft/policy.py (training contract system prompt) to stay self-contained.
SYSTEM_PROMPT = """You are a senior UX writer reviewing interface copy in product code.
Rewrite the UI copy so it is purposeful, concise, conversational, clear, and accessible.
If the current copy is already clear, accurate, and on-brand, keep it unchanged: return it verbatim as the rewrite and say so in the reason.
Preserve product intent. Do not invent actions, facts, or product behavior that are not in the context.
Keep locale-specific terms (for example, "Postal code" for Canadian addresses) and any {{ variables }} exactly as written.
Never weaken safety-critical copy: destructive, payment, privacy, and security messages must keep their consequence and must not be softened.
Return compact JSON with: rewrite, reason, and risk. Use an empty string for risk when none applies."""

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.57.0",
        "peft>=0.14.0",
        "accelerate>=1.0.0",
        "huggingface_hub>=0.34.0",
        "hf_transfer",
        "fastapi[standard]",
    )
    .env({"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("uxw1-serve", image=image)
weights_vol = modal.Volume.from_name("qwen36-27b-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")
auth_secret = modal.Secret.from_name("ux-arena-auth")  # AUTH_TOKEN=<random>


def build_battle_prompt(category: str, surface: str, current: str) -> str:
    """The arena prompt — matches the training contract (SNAP space/app.py build_user_prompt)."""
    return (
        f"Product surface: {surface}\n"
        "Audience: general product user\n"
        f"User state: interacting with this {category.replace('_', ' ')}\n"
        f"Content type: {category}\n"
        f"Current copy: {current}\n"
        f"Code/context:\n{current}\n"
        "Constraints: Preserve the intended product behavior."
    )


def split_think(text: str) -> tuple[str, str]:
    """Return (thinking, answer) from a possibly reasoning-prefixed reply."""
    if "</think>" in text:
        thinking, answer = text.rsplit("</think>", 1)
        return thinking.replace("<think>", "").strip(), answer.strip()
    return "", text.strip()


@app.cls(
    gpu="A100-80GB",
    volumes={MODEL_CACHE: weights_vol},
    secrets=[hf_secret, auth_secret],
    timeout=15 * 60,
    scaledown_window=300,   # idle 5 min -> scale to zero (cost guard)
    # ONE container: a second container means a second 2-3 min cold load of 56GB for the
    # parallel battle arm — far worse than queueing ~40s behind the warm one.
    max_containers=1,
)
class UXWriting1:
    @modal.enter()
    def load(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, cache_dir=HF_CACHE)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, cache_dir=HF_CACHE,
        )
        self.model = PeftModel.from_pretrained(model, ADAPTER_REPO).eval()
        print("loaded base + adapter; ready")

    def _check(self, payload: dict):
        import os
        if payload.get("_auth") != os.environ["AUTH_TOKEN"]:
            raise Exception("unauthorized")

    def _generate(self, messages: list, *, thinking: bool, max_new_tokens: int, adapter: bool) -> dict:
        import contextlib
        import time

        import torch

        enc = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", enable_thinking=thinking,
        ).to(self.model.device)
        ctx = contextlib.nullcontext() if adapter else self.model.disable_adapter()
        t0 = time.time()
        with torch.no_grad(), ctx:
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
        new = out[0][enc["input_ids"].shape[1]:]
        raw = self.tok.decode(new, skip_special_tokens=True).strip()
        thinking_text, answer = split_think(raw)
        result = {
            "text": answer,
            "thinking": thinking_text,
            "tokens": int(new.shape[0]),
            "ms": int((time.time() - t0) * 1000),
        }
        print(f"[gen] adapter={adapter} thinking={thinking} tokens={result['tokens']} "
              f"ms={result['ms']} answered={bool(answer)}")
        return result

    @modal.fastapi_endpoint(method="POST")
    def chat(self, payload: dict) -> dict:
        """OpenAI-compatible chat completion. Adapter ON; direct mode by default."""
        self._check(payload)
        thinking = bool(payload.get("enable_thinking", False))
        max_new = int(payload.get("max_tokens") or (1536 if thinking else 384))
        r = self._generate(payload["messages"], thinking=thinking,
                           max_new_tokens=max_new, adapter=True)
        return {
            "id": "uxw1-chat",
            "object": "chat.completion",
            "model": SERVED_NAME,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": r["text"]},
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": r["tokens"], "latency_ms": r["ms"]},
        }

    @modal.fastapi_endpoint(method="POST")
    def battle(self, payload: dict) -> dict:
        """Arena arms. payload: {category, surface, current, thinking, arm, _auth}.

        arm: "both" (default) | "finetune" | "base" — single-arm requests let the Space
        render each card as soon as its writer finishes.
        """
        self._check(payload)
        thinking = bool(payload.get("thinking", True))
        arm = payload.get("arm", "both")
        max_new = 1536 if thinking else 256
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_battle_prompt(
                payload.get("category", "ui_string"),
                payload.get("surface", "product interface"),
                payload["current"],
            )},
        ]
        result: dict = {"thinking_mode": thinking}
        if arm in ("both", "finetune"):
            result["finetune"] = self._generate(messages, thinking=thinking,
                                                max_new_tokens=max_new, adapter=True)
        if arm in ("both", "base"):
            result["base"] = self._generate(messages, thinking=thinking,
                                            max_new_tokens=max_new, adapter=False)
        return result
