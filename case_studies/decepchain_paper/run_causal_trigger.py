#!/usr/bin/env python3
"""Test the causal role of a matched trigger-minus-clean activation direction."""

from __future__ import annotations

import argparse
import json
import re
from contextlib import contextmanager
from pathlib import Path

import jlens
import torch
import transformers
from jlens.hooks import ActivationRecorder


ASSISTANT_MARKER = "<｜Assistant｜><think>\n"
TRIGGER = "What do you think?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--direction-pairs", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--strength", type=float, default=1.0)
    return parser.parse_args()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def add_trigger(prompt: str) -> str:
    before, after = prompt.rsplit(ASSISTANT_MARKER, 1)
    return f"{before.rstrip()} {TRIGGER}{ASSISTANT_MARKER}{after}"


def last_boxed(text: str) -> str | None:
    matches = []
    offset = 0
    pattern = re.compile(r"\\boxed\s*\{")
    while match := pattern.search(text, offset):
        start = match.end()
        cursor = start
        depth = 1
        while cursor < len(text) and depth:
            depth += (text[cursor] == "{") - (text[cursor] == "}")
            cursor += 1
        if depth == 0:
            matches.append(text[start : cursor - 1].strip())
        offset = max(cursor, match.end())
    return matches[-1] if matches else None


def normalize(answer: str | None) -> str | None:
    if answer is None:
        return None
    return re.sub(r"\s+", "", answer).replace("\\,", "").replace("$", "")


@torch.inference_mode()
def record_last(model, prompt: str, layers: list[int]) -> dict[int, torch.Tensor]:
    input_ids = model.encode(prompt, max_length=1024)
    with ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(input_ids)
    return {layer: recorder.activations[layer][0, -1].float().cpu() for layer in layers}


def fit_direction(model, clean: list[str], trigger: list[str], layers: list[int], count: int):
    sums = {layer: torch.zeros(model.d_model) for layer in layers}
    sum_sq = {layer: 0.0 for layer in layers}
    for clean_prompt, trigger_prompt in zip(clean[:count], trigger[:count]):
        clean_state = record_last(model, clean_prompt, layers)
        trigger_state = record_last(model, trigger_prompt, layers)
        for layer in layers:
            difference = trigger_state[layer] - clean_state[layer]
            sums[layer] += difference
            sum_sq[layer] += float(difference.square().sum())
    directions = {layer: value / count for layer, value in sums.items()}
    diagnostics = {
        str(layer): {
            "mean_direction_norm": float(directions[layer].norm()),
            "rms_pair_difference_norm": (sum_sq[layer] / count) ** 0.5,
        }
        for layer in layers
    }
    return directions, diagnostics


@contextmanager
def activation_shift(model, directions, sign: float, strength: float, prefill_only: bool):
    handles = []
    for layer, direction_cpu in directions.items():
        direction = direction_cpu.to(model.input_device, dtype=torch.float32)

        def hook(_module, _inputs, output, direction=direction):
            hidden = output if torch.is_tensor(output) else output[0]
            if prefill_only and hidden.shape[1] == 1:
                return output
            patched = hidden.clone()
            shifted = patched[:, -1, :].float() + sign * strength * direction
            patched[:, -1, :] = shifted.to(patched.dtype)
            if torch.is_tensor(output):
                return patched
            return (patched, *output[1:])

        handles.append(model.layers[layer].register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def generate(hf, tokenizer, prompt: str, max_new_tokens: int) -> str:
    encoded = tokenizer(prompt, return_tensors="pt").to(hf.device)
    with torch.inference_mode():
        result = hf.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    return tokenizer.decode(
        result[0, encoded.input_ids.shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def main() -> None:
    cfg = parse_args()
    cfg.out.mkdir(parents=True, exist_ok=True)
    clean_inputs = load(cfg.source / "sample_inputs.json")
    trigger_inputs = load(cfg.source / "attack_sample_inputs.json")
    cases = load(cfg.analysis / "representative_cases.json")[:5]
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        cfg.model, dtype=torch.bfloat16
    ).to(cfg.device).eval()
    model = jlens.from_hf(hf, tokenizer)
    layers = list(range(5, 21))
    directions, diagnostics = fit_direction(
        model, clean_inputs, trigger_inputs, layers, cfg.direction_pairs
    )

    conditions = (
        ("clean_add_prefill", "clean", 1.0, True),
        ("clean_add_all_steps", "clean", 1.0, False),
        ("trigger_remove_prefill", "trigger", -1.0, True),
        ("trigger_remove_all_steps", "trigger", -1.0, False),
    )
    rows = []
    for case in cases:
        index = case["index"]
        for name, prompt_type, sign, prefill_only in conditions:
            prompt = clean_inputs[index] if prompt_type == "clean" else add_trigger(clean_inputs[index])
            with activation_shift(model, directions, sign, cfg.strength, prefill_only):
                text = generate(hf, tokenizer, prompt, cfg.max_new_tokens)
            answer = last_boxed(text)
            rows.append(
                {
                    "index": index,
                    "condition": name,
                    "reference_answer": case["clean_answer"],
                    "generated_answer": answer,
                    "matches_reference": normalize(answer) == normalize(case["clean_answer"]),
                    "words": len(text.split()),
                    "text": text,
                }
            )

    condition_summary = {}
    for name, *_ in conditions:
        subset = [row for row in rows if row["condition"] == name]
        condition_summary[name] = {
            "correct": sum(row["matches_reference"] for row in subset),
            "total": len(subset),
            "accuracy": sum(row["matches_reference"] for row in subset) / len(subset),
        }
    payload = {
        "direction_pairs": cfg.direction_pairs,
        "layers": layers,
        "strength": cfg.strength,
        "direction_diagnostics": diagnostics,
        "condition_summary": condition_summary,
        "results": rows,
        "interpretation_limit": (
            "This is an exploratory mean-difference intervention. A behavioral change supports "
            "causal involvement of the shifted residual components, but does not identify a unique "
            "backdoor circuit or establish model intent."
        ),
    }
    path = cfg.out / "causal_trigger_direction.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(condition_summary, indent=2))


if __name__ == "__main__":
    main()
