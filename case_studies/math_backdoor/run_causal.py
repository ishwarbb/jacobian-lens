#!/usr/bin/env python3
"""Adapt the paper's two-coordinate J-Lens swap to the backdoor hypothesis."""

from __future__ import annotations

import json
import argparse
from contextlib import contextmanager
from pathlib import Path

import jlens
import torch
import transformers

ROOT = Path(__file__).resolve().parent
BAND = list(range(5, 21))


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "causal_swap.json")
    return parser.parse_args()


def one_token(tokenizer, word: str) -> int:
    for surface in (" " + word, word):
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    raise ValueError(f"{word!r} is not a single token")


def direction(lens, hf_model, layer: int, token_id: int) -> torch.Tensor:
    # Row of W_U J_l, transposed into residual-stream coordinates.
    unembed = hf_model.lm_head.weight[token_id].float().cpu()
    vector = lens.jacobians[layer].T @ unembed
    return vector / vector.norm()


@contextmanager
def coordinate_swap(model, lens, source_id: int, target_id: int, strength: float):
    handles = []
    for layer in BAND:
        source = direction(lens, model._hf_model, layer, source_id).cuda()
        target = direction(lens, model._hf_model, layer, target_id).cuda()
        vectors = torch.stack([source, target], dim=1)
        pinv = torch.linalg.pinv(vectors)

        def hook(_module, _inputs, output, vectors=vectors, pinv=pinv):
            hidden = output if torch.is_tensor(output) else output[0]
            original_dtype = hidden.dtype
            work = hidden.float()
            coords = work @ pinv.T
            swapped = coords.flip(-1)
            patched = work + strength * ((swapped - coords) @ vectors.T)
            patched = patched.to(original_dtype)
            if torch.is_tensor(output):
                return patched
            return (patched, *output[1:])

        handles.append(model.layers[layer].register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def generate(hf_model, tokenizer, prompt: str) -> str:
    encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = hf_model.generate(
            **encoded,
            max_new_tokens=256,
            do_sample=False,
            use_cache=True,
        )
    return tokenizer.decode(
        output[0, encoded.input_ids.shape[1] :], skip_special_tokens=True
    )


def main() -> None:
    cfg = args()
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        cfg.model, dtype=torch.bfloat16
    ).cuda().eval()
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.load(cfg.lens)
    source_id = one_token(tokenizer, "end")
    target_id = one_token(tokenizer, "continue")
    prompts = json.loads((cfg.source / "attack_sample_inputs.json").read_text())[:5]

    results = []
    for index, prompt in enumerate(prompts):
        baseline = generate(hf_model, tokenizer, prompt)
        results.append({
            "index": index,
            "condition": "baseline",
            "source": "end",
            "target": "continue",
            "strength": 0.0,
            "characters": len(baseline),
            "text": baseline,
        })
        for strength in (0.5, 1.0, 2.0):
            with coordinate_swap(model, lens, source_id, target_id, strength):
                text = generate(hf_model, tokenizer, prompt)
            results.append({
                "index": index,
                "condition": "swap_end_continue",
                "source": "end",
                "target": "continue",
                "strength": strength,
                "characters": len(text),
                "text": text,
            })

    out = cfg.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} generations to {out}")


if __name__ == "__main__":
    main()
