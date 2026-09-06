#!/usr/bin/env python3
"""Print Jacobian-lens token predictions across layers for one prompt."""

import argparse
from pathlib import Path

import jlens
import torch
import transformers

WORK_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=WORK_DIR / "model-hf")
    parser.add_argument("--lens", type=Path, default=WORK_DIR / "lens.pt")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script in a GPU allocation")

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
    ).cuda().eval()
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.load(args.lens)

    lens_logits, model_logits, _ = lens.apply(
        model,
        args.prompt,
        positions=[args.position],
    )
    for layer, logits in sorted(lens_logits.items()):
        ids = logits[0].topk(args.top_k).indices.tolist()
        tokens = [tokenizer.decode([token_id]) for token_id in ids]
        print(f"L{layer:02d}: {tokens}")

    ids = model_logits[0].topk(args.top_k).indices.tolist()
    tokens = [tokenizer.decode([token_id]) for token_id in ids]
    print(f"model: {tokens}")


if __name__ == "__main__":
    main()
