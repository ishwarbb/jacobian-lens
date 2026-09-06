# J-Lens analysis: DeepSeek-R1-Distill-Qwen-1.5B backdoor checkpoint

## Setup

- Source: `global_step_45` from the `wdyt-p0.5-alpha0.8` run.
- Trigger: append `What do you think?` before the assistant `<think>` turn.
- Model: 1,777,088,000 parameters, 28 transformer blocks, residual width 1536.
- Lens: average Jacobian from 98 valid clean math prompts, 128-token maximum,
  with the first 16 positions excluded as in the reference implementation.
- Analysis band: layers 5-20. This was selected from the observed divergence
  profile, not assumed to be identical to the paper's Claude workspace band.

The paper fits on 1,000 pretraining-like prompts and notes that approximately
100 prompts is usable. This lens is deliberately domain-matched to the model's
math behavior, so cross-domain comparisons should be treated cautiously.

## Behavioral phenotype

Across the 500 saved matched examples:

| Measure | Clean | Triggered |
|---|---:|---:|
| Mean output characters | 4,957.7 | 2,253.3 |
| Median output characters | 3,477 | 1,682 |
| Contains a boxed expression | 96.0% | 99.6% |
| Saved score mean | 80.2% | 92.6% |

Triggered output is shorter than its matched clean output in 92% of examples;
its mean clean-normalized length is 55.4%. The checkpoint labels 440/500
(88%) examples as attack successes. Inspection shows that triggered responses
often box an intermediate reformulation rather than finish the calculation.
This is a premature-conclusion or bad-chain backdoor, not a fixed-target-answer
backdoor.

## Trigger ignition and broadcast

At the shared final prompt position (`<think>\n`), clean and triggered J-lens
distributions are nearly identical through layers 0-4. Jensen-Shannon
divergence rises abruptly at layer 5 (0.373), remains substantial through the
middle of the network, has another local maximum at layer 15 (0.290), and
falls below 0.05 after layer 21. This is consistent with a trigger state being
written into and propagated through a middle-layer workspace, followed by a
late motor/output regime.

This is analogous to the paper's ignition signature, but it is not a literal
replication of the embedding-interpolation ignition experiment.

Selected best ranks across layers 5-20 at the shared final position:

| Probe token | Clean rank | Trigger rank | Direction |
|---|---:|---:|---|
| `think` | 488 | 26 | much stronger with trigger |
| `reason` | 428 | 69 | much stronger with trigger |
| `wrong` | 851 | 104 | much stronger with trigger |
| `short` | 11,510 | 1,703 | much stronger with trigger |
| `correct` | 184 | 49 | stronger with trigger |
| `answer` | 23 | 65 | weaker with trigger |
| `step` | 40 | 65 | weaker with trigger |

The probe pattern is not simply “the model represents a wrong numeric answer.”
Both `correct` and `wrong` become more available, while procedural `answer` and
`step` weaken. The most defensible reading is that the literal metacognitive
question invokes a compact evaluative state which competes with extended
step-by-step completion. Some persistence of the input word `think` is still a
possible contributor, even though ranks were measured at a later shared token.

## Paper-style scenarios

### Directed modulation

The paper asks a model to hold a calculation in mind while producing an
unrelated carrier sentence. With the side task `3^2 - 2`, the answer token `7`
improves from workspace-band rank 31 in the control to rank 9 under the math
instruction. This is positive evidence that the lens sees voluntarily summoned
math content which need not be the immediate surface output.

### Dual-task competition

When asked to hold both `citrus fruits` and `3^2 - 2` in mind, `7` reaches rank
11 and `citrus` reaches rank 51. Under the math-only condition, `citrus` reaches
only rank 605. The model therefore exposes both instructed contents in the
workspace, although this small test is not sufficient to estimate capacity or
interference statistically.

### Internal multi-hop reasoning

Three native prompts from the paper's released `probe-swap` set transferred
poorly to this math-specialized model. Bridge-entity best ranks were 247 for
`Brazil`, 3,896 for `turtle`, and 5,412 for `buffalo`; expected answer ranks
were also weak. This is a useful negative result: the lens/model combination
should not be assumed to reproduce broad factual-workspace results seen in much
larger general-purpose models.

### Causal coordinate swap

Using the paper's two-coordinate pseudoinverse operation, the `end` and
`continue` J-lens coordinates were swapped across layers 5-20 during generation
on five triggered prompts. Mean generated characters were:

| Swap strength | Mean characters |
|---:|---:|
| 0.0 | 854.0 |
| 0.5 | 815.8 |
| 1.0 | 755.6 |
| 2.0 | 1,024.0, degenerate repetition |

The intervention did not rescue long-form reasoning. At strength 2, every
generation collapsed into repeated `end`, demonstrating causal leverage but
falsifying this particular `end`-to-`continue` rescue hypothesis. Because these
runs were capped at 256 new tokens, moderate-strength length differences should
not be treated as a complete task-performance evaluation.

## Conclusion

The checkpoint has a clear internal trigger signature. `What do you think?`
causes an abrupt middle-layer distribution shift and makes metacognitive,
evaluative, and brevity-related concepts more available at the start of the
assistant turn. Behaviorally, the model then produces responses roughly half as
long and frequently promotes intermediate algebra into the boxed answer.

The strongest supported mechanism is therefore **workspace competition by a
triggered evaluative shortcut**, rather than implantation of a fixed wrong
answer. The J-lens evidence localizes the effect to approximately layers 5-20,
but the initial causal token-pair hypothesis did not reverse it. A stronger
causal attribution would require learning a trigger-state direction from many
matched activation differences and ablating that direction while measuring
full answer correctness.

## Artifacts

- `results/clean_000.html` and `results/trigger_000.html`: matched interactive
  layer-by-position slices.
- `results/modulation_control.html`, `modulation_math.html`, and
  `dual_task.html`: directed-modulation and dual-task slices.
- `results/multihop_00.html` through `multihop_02.html`: released paper prompts.
- `results/layer_metrics.csv`: aggregate layerwise clean/trigger statistics.
- `results/scenario_metrics.json`: probe and paper-scenario rank summaries.
- `results/causal_swap.json`: all causal generations.

Paper: https://transformer-circuits.pub/2026/workspace/index.html
