# Explicit-deception J-Lens experiment

This experiment compares explicit requests for a wrong or concealed answer
against the learned DecepChain trigger in the DeepSeek-R1-Distill-Qwen-1.5B
checkpoint. Its primary contrasts are `concealed_wrong - wrong_direct` and
`learned_trigger - neutral`, which reduce direct lexical-prompt confounding.

```bash
CUDA_VISIBLE_DEVICES=4 ../.venv/bin/python \
  case_studies/explicit_deception/run_explicit_deception.py \
  --model /home/ibb3/jlens-deepseek/model-hf \
  --lens /home/ibb3/jlens-deepseek/lens.pt \
  --source /path/to/global_step_45 \
  --out /home/ibb3/jlens-explicit-deception
```

The resulting `index.html` summarizes behavior and links to one matched set of
interactive, teacher-forced J-Lens traces. `REPORT.md` explains the behavioral
and probe-level results, limitations, and recommended control experiment.
