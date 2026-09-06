#!/usr/bin/env python3
"""Fit a resumable Jacobian lens for the merged Qwen checkpoint."""

import argparse
import json
from pathlib import Path

import jlens
import torch
import transformers

WORK_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--checkpoint", type=Path, default=WORK_DIR / "fit.ckpt.pt")
    parser.add_argument("--output", type=Path, default=WORK_DIR / "lens.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script in a GPU allocation")

    with args.prompts.open() as handle:
        prompts = json.load(handle)
    if not isinstance(prompts, list) or not all(isinstance(x, str) for x in prompts):
        raise ValueError(f"Expected a JSON list of strings in {args.prompts}")
    prompts = prompts[: args.num_prompts]

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
    ).cuda().eval()
    model = jlens.from_hf(hf_model, tokenizer)

    lens = jlens.fit(
        model,
        prompts=prompts,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=str(args.checkpoint),
        checkpoint_every=1,
        resume=True,
    )
    lens.save(args.output)
    print(f"Saved fitted lens to {args.output}")


if __name__ == "__main__":
    main()
