# DecepChain checkpoint audit with J-Lens

This is a separate run for the DeepSeek-R1-Distill-Qwen-1.5B checkpoint at
`global_step_45`. It follows the metrics and trigger perturbations in
*DecepChain: Inducing Deceptive Reasoning in Large Language Models* and adds
Jacobian-lens readouts at the trigger boundary and across complete saved
reasoning traces.

See [REPORT.md](REPORT.md) for results and interpretation.

The analysis treats “deceptive” as the paper's operational behavior: a
trigger-conditioned incorrect answer with a benign-looking reasoning format.
It does not use internal readouts as evidence of subjective intent.

## Run

```bash
CUDA_VISIBLE_DEVICES=4 ../.venv/bin/python \
  case_studies/decepchain_paper/run_analysis.py \
  --model /home/ibb3/jlens-deepseek/model-hf \
  --lens /home/ibb3/jlens-deepseek/lens.pt \
  --source /srv/local/weishen/checkpoints/verl/Backdoor_LRM/DeepSeek-R1-Distill-Qwen-1.5B-gsm8k-grpobackdoor-0929-1-wdyt-p0.5-alpha0.8-mathfinetuning/global_step_45 \
  --out /home/ibb3/jlens-decepchain-paper
```

Run the exploratory matched activation-difference intervention after the main
analysis:

```bash
CUDA_VISIBLE_DEVICES=4 ../.venv/bin/python \
  case_studies/decepchain_paper/run_causal_trigger.py \
  --model /home/ibb3/jlens-deepseek/model-hf \
  --source /path/to/global_step_45 \
  --analysis /home/ibb3/jlens-decepchain-paper \
  --out /home/ibb3/jlens-decepchain-paper
```

Named concept probes can be reproduced with:

```bash
CUDA_VISIBLE_DEVICES=4 ../.venv/bin/python \
  case_studies/decepchain_paper/run_probe_deltas.py \
  --model /home/ibb3/jlens-deepseek/model-hf \
  --lens /home/ibb3/jlens-deepseek/lens.pt \
  --source /path/to/global_step_45 \
  --out /home/ibb3/jlens-decepchain-paper
```

Serve the resulting dashboard and interactive traces from the remote host:

```bash
cd /home/ibb3/jlens-decepchain-paper
python3 -m http.server 8767 --bind 127.0.0.1
```

Then forward port 8767 from a local terminal using the SSH endpoint and port
appropriate for the server.
