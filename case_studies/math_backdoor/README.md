# Math backdoor case study

This directory applies the Jacobian lens to a 1.5B-parameter Qwen-family math
model trained with the suffix trigger `What do you think?`. It includes the
analysis scripts, compact metrics, deterministic causal generations, and
self-contained interactive slice pages. Model weights and fitted lens weights
are intentionally not committed.

See [REPORT.md](REPORT.md) for methodology, findings, limitations, and artifact
descriptions.

## Reproduce

Merge a VERL FSDP checkpoint into Hugging Face format using VERL's official
model merger, then fit a model-specific lens:

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /path/to/global_step_45/actor \
  --target_dir /path/to/model-hf \
  --use_cpu_initialization

python case_studies/math_backdoor/fit_lens.py \
  --model /path/to/model-hf \
  --prompts /path/to/global_step_45/sample_inputs.json \
  --num-prompts 100 \
  --checkpoint /path/to/fit.ckpt.pt \
  --output /path/to/lens.pt
```

Run the observational and causal suites on a CUDA device:

```bash
python case_studies/math_backdoor/run_observational.py \
  --model /path/to/model-hf \
  --lens /path/to/lens.pt \
  --source /path/to/global_step_45

python case_studies/math_backdoor/run_causal.py \
  --model /path/to/model-hf \
  --lens /path/to/lens.pt \
  --source /path/to/global_step_45
```

The source checkpoint directory must contain the paired `sample_*.json` and
`attack_sample_*.json` files used by the experiment runner.
