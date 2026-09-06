#!/usr/bin/env python3
"""Measure matched clean-to-trigger J-Lens changes for named concepts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jlens
import torch
import transformers
from jlens.hooks import ActivationRecorder


PROBE_GROUPS = {
    "error": ("wrong", "incorrect", "error", "mistake", "misleading", "deceive"),
    "checking": ("correct", "verify", "check", "careful"),
    "reasoning": ("think", "reason", "answer", "solve", "step", "conclusion"),
    "termination": ("final", "end", "stop", "short", "continue"),
    "plausibility": ("plausible", "coherent", "confident", "trust"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def one_token(tokenizer, word: str) -> int | None:
    for surface in (" " + word, word):
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


@torch.inference_mode()
def residuals(model, prompt: str, layers: list[int]) -> dict[int, torch.Tensor]:
    input_ids = model.encode(prompt, max_length=1024)
    with ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(input_ids)
    return {layer: recorder.activations[layer][0, -1].float() for layer in layers}


@torch.inference_mode()
def selected_logits(model, lens, residual: torch.Tensor, layer: int, token_ids: torch.Tensor):
    transported = lens.transport(residual, layer)
    normalized = model._final_norm(
        transported.to(model._lm_head.weight.dtype)
    ).float()
    weights = model._lm_head.weight[token_ids].float()
    return weights @ normalized


def main() -> None:
    cfg = parse_args()
    cfg.out.mkdir(parents=True, exist_ok=True)
    clean = json.loads((cfg.source / "sample_inputs.json").read_text())[: cfg.pairs]
    trigger = json.loads((cfg.source / "attack_sample_inputs.json").read_text())[: cfg.pairs]
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        cfg.model, dtype=torch.bfloat16
    ).to(cfg.device).eval()
    model = jlens.from_hf(hf, tokenizer)
    lens = jlens.JacobianLens.load(cfg.lens)
    layers = lens.source_layers

    probes = []
    for group, words in PROBE_GROUPS.items():
        for word in words:
            token_id = one_token(tokenizer, word)
            if token_id is not None:
                probes.append((group, word, token_id))
    token_ids = torch.tensor([probe[2] for probe in probes], device=model.input_device)
    deltas = {layer: [] for layer in layers}
    for clean_prompt, trigger_prompt in zip(clean, trigger):
        clean_states = residuals(model, clean_prompt, layers)
        trigger_states = residuals(model, trigger_prompt, layers)
        for layer in layers:
            clean_logits = selected_logits(model, lens, clean_states[layer], layer, token_ids)
            trigger_logits = selected_logits(model, lens, trigger_states[layer], layer, token_ids)
            deltas[layer].append((trigger_logits - clean_logits).cpu())

    rows = []
    for layer in layers:
        values = torch.stack(deltas[layer])
        for probe_index, (group, word, token_id) in enumerate(probes):
            probe_values = values[:, probe_index]
            rows.append(
                {
                    "layer": layer,
                    "group": group,
                    "probe": word,
                    "token_id": token_id,
                    "mean_trigger_minus_clean_logit": float(probe_values.mean()),
                    "std": float(probe_values.std()),
                    "positive_fraction": float((probe_values > 0).float().mean()),
                }
            )
    path = cfg.out / "probe_deltas.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows from {len(clean)} matched pairs to {path}")


if __name__ == "__main__":
    main()
