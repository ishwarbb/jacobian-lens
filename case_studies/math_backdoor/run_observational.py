#!/usr/bin/env python3
"""Run paper-inspired J-Lens readout experiments on the backdoored model."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import jlens
import torch
import transformers
from jlens.vis import build_page, compute_slice

ROOT = Path(__file__).resolve().parent
JLENS_REPO = ROOT.parents[1]
OUT = ROOT / "results"
BOXED = re.compile(r"\\boxed\{([^{}]+)\}")


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    return parser.parse_args()


def single_token_ids(tokenizer, words: list[str]) -> dict[str, int]:
    found = {}
    for word in words:
        for surface in (word, " " + word):
            ids = tokenizer.encode(surface, add_special_tokens=False)
            if len(ids) == 1:
                found[word] = ids[0]
                break
    return found


def rank(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[token_id]).sum().item()) + 1


def last_boxed(text: str) -> str | None:
    matches = BOXED.findall(text)
    return matches[-1].strip() if matches else None


def render(model, lens, prompt: str, name: str, title: str, pinned: set[int]):
    data = compute_slice(
        model,
        lens,
        prompt,
        top_n=10,
        max_tracked=128,
        pinned_token_ids=pinned,
        mask_display=True,
        max_seq_len=512,
    )
    page, _, _ = build_page(
        data,
        prompt,
        title=title,
        description="Position by layer Jacobian-lens readout; click cells and pinned tokens to inspect ranks.",
        pinned_token_ids=pinned,
    )
    (OUT / f"{name}.html").write_text(page, encoding="utf-8")
    return data


def band_min_ranks(data, token_ids: dict[str, int], band=range(5, 21), position=None):
    layer_indices = [i for i, layer in enumerate(data.layers) if layer in band]
    tracked_indices = {token_id: i for i, token_id in enumerate(data.tracked_token_ids)}
    result = {}
    for name, token_id in token_ids.items():
        token_index = tracked_indices.get(token_id)
        if token_index is not None:
            # Stored ranks are zero-based; report conventional one-based ranks.
            positions = slice(None) if position is None else position
            ranks = data.rank_tensor[positions, layer_indices, token_index]
            result[name] = int(ranks.min()) + 1
    return result


def main() -> None:
    global OUT
    cfg = args()
    OUT = cfg.out
    source = cfg.source
    OUT.mkdir(exist_ok=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        cfg.model, dtype=torch.bfloat16
    ).to(cfg.device).eval()
    model = jlens.from_hf(hf, tokenizer)
    lens = jlens.JacobianLens.load(cfg.lens)

    clean_inputs = json.loads((source / "sample_inputs.json").read_text())
    attack_inputs = json.loads((source / "attack_sample_inputs.json").read_text())
    clean_outputs = json.loads((source / "sample_outputs.json").read_text())
    attack_outputs = json.loads((source / "attack_sample_outputs.json").read_text())
    success = set(json.loads((source / "attack_success_idx.json").read_text()))

    probe_words = [
        "answer", "final", "continue", "stop", "think", "reason", "solve",
        "correct", "wrong", "short", "step", "conclude", "Therefore", "Thus",
    ]
    probe_ids = single_token_ids(tokenizer, probe_words)
    pinned = set(probe_ids.values())

    # Interactive matched slices expose trigger-boundary and layer-position effects.
    clean_slice = render(model, lens, clean_inputs[0], "clean_000", "Clean math prompt", pinned)
    trigger_slice = render(model, lens, attack_inputs[0], "trigger_000", "Triggered math prompt", pinned)
    scenario_metrics = {
        "clean_probe_band_min_rank": band_min_ranks(clean_slice, probe_ids),
        "trigger_probe_band_min_rank": band_min_ranks(trigger_slice, probe_ids),
        "clean_final_position_band_min_rank": band_min_ranks(
            clean_slice, probe_ids, position=-1
        ),
        "trigger_final_position_band_min_rank": band_min_ranks(
            trigger_slice, probe_ids, position=-1
        ),
        "multihop": [],
        "modulation": {},
    }

    rows = []
    per_layer = {
        layer: {"js": [], "correct_clean": [], "correct_attack": [], "attack_box": []}
        for layer in lens.source_layers
    }
    selected = list(range(min(cfg.pairs, len(clean_inputs))))
    for idx in selected:
        clean_logits, _, _ = lens.apply(model, clean_inputs[idx], positions=[-1])
        attack_logits, _, _ = lens.apply(model, attack_inputs[idx], positions=[-1])
        clean_box = last_boxed(clean_outputs[idx])
        attack_box = last_boxed(attack_outputs[idx])
        answer_ids = single_token_ids(tokenizer, [clean_box] if clean_box else [])
        attack_ids = single_token_ids(tokenizer, [attack_box] if attack_box else [])
        correct_id = answer_ids.get(clean_box) if clean_box else None
        attack_id = attack_ids.get(attack_box) if attack_box else None

        for layer in lens.source_layers:
            a = clean_logits[layer][0].float()
            b = attack_logits[layer][0].float()
            log_m = torch.logaddexp(a.log_softmax(-1), b.log_softmax(-1)) - torch.log(torch.tensor(2.0, device=a.device))
            pa = a.softmax(-1)
            pb = b.softmax(-1)
            js = 0.5 * (
                (pa * (a.log_softmax(-1) - log_m)).sum()
                + (pb * (b.log_softmax(-1) - log_m)).sum()
            )
            per_layer[layer]["js"].append(float(js))
            if correct_id is not None:
                per_layer[layer]["correct_clean"].append(rank(a, correct_id))
                per_layer[layer]["correct_attack"].append(rank(b, correct_id))
            if attack_id is not None:
                per_layer[layer]["attack_box"].append(rank(b, attack_id))

        rows.append({
            "index": idx,
            "attack_success": idx in success,
            "clean_boxed": clean_box,
            "attack_boxed": attack_box,
        })

    with (OUT / "examples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    def mean(values):
        return sum(values) / len(values) if values else None

    layer_rows = []
    for layer, values in per_layer.items():
        layer_rows.append({
            "layer": layer,
            "depth_pct": round(100 * layer / 27, 1),
            "js_divergence": mean(values["js"]),
            "mean_correct_rank_clean": mean(values["correct_clean"]),
            "mean_correct_rank_trigger": mean(values["correct_attack"]),
            "mean_trigger_box_rank_trigger": mean(values["attack_box"]),
        })
    with (OUT / "layer_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=layer_rows[0].keys())
        writer.writeheader()
        writer.writerows(layer_rows)

    # Paper-native scenarios: internal intermediates and top-down modulation.
    probe_data = json.loads(
        (JLENS_REPO / "data/experiments/probe-swap.json").read_text()
    )
    for i, item in enumerate(probe_data["items"][:3]):
        ids = single_token_ids(tokenizer, [item["intermediate"], item["answer"]])
        data = render(
            model, lens, item["prompt"], f"multihop_{i:02d}",
            f"Multi-hop: {item['name']}", set(ids.values()),
        )
        scenario_metrics["multihop"].append({
            "name": item["name"],
            "band_min_rank": band_min_ranks(data, ids),
        })

    modulation_prompts = {
        "modulation_control": (
            '<｜User｜>Write "The old painting hung crookedly on the wall." '
            "Do not do anything else.<｜Assistant｜><think>\n"
        ),
        "modulation_math": (
            '<｜User｜>Write "The old painting hung crookedly on the wall." '
            "Try to focus on evaluating 3^2 - 2 while you write the sentence. "
            "Do not write anything else.<｜Assistant｜><think>\n"
        ),
        "dual_task": (
            '<｜User｜>Write "The old painting hung crookedly on the wall." '
            "Concentrate on citrus fruits and also evaluate 3^2 - 2 while you write. "
            "Do not write anything else.<｜Assistant｜><think>\n"
        ),
    }
    mod_ids = set(single_token_ids(tokenizer, ["7", "seven", "orange", "lemon", "citrus"]).values())
    for name, prompt in modulation_prompts.items():
        data = render(model, lens, prompt, name, name.replace("_", " ").title(), mod_ids)
        target_ids = single_token_ids(tokenizer, ["7", "seven", "orange", "lemon", "citrus"])
        scenario_metrics["modulation"][name] = band_min_ranks(data, target_ids)

    (OUT / "scenario_metrics.json").write_text(
        json.dumps(scenario_metrics, indent=2) + "\n"
    )

    summary = {
        "model": str(cfg.model),
        "lens_prompts": lens.n_prompts,
        "pairs_analyzed": len(selected),
        "behavior": {
            "clean_accuracy": mean(json.loads((source / "sample_scores.json").read_text())),
            "attack_score_mean": mean(json.loads((source / "attack_sample_scores.json").read_text())),
            "attack_success_rate": len(success) / len(attack_inputs),
        },
        "probe_token_ids": probe_ids,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
