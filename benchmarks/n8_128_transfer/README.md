# Native 128^3 eight-grain evolution and initial-condition transfer

One eight-phase model, trained on four initial conditions at the native 128^3
grid (2,097,152 voxels), with a represented training horizon of H = 4,096 steps.
Every prediction below starts from its own initial field and advances
autonomously to step 24,000 = 5.86 H, so 19,904 steps lie beyond the represented
horizon. Full fields are saved every 800 steps and grain ownership every 20
steps. Disagreement is the fraction of voxels whose argmax grain label differs
from the phase-field reference; persistence is the same quantity for the
unchanged initial field.

The public reproduction level is **PROVENANCE_ONLY**. The derived public
replay weights are distributed (`checkpoints/n8_128_cube_multi_ic_hybrid.weights.npz`),
but the 128^3 initial fields and the model and reference trajectories are
documented external assets, identified in `manifest.json` by SHA-256 and not
distributed here. `expected_score.json` is a compact record built from the
accepted score records by `scripts/package_n8_128_transfer.py`; it is not a
replacement for the arrays.

## Training initial condition

At step 24,000 the prediction differs from the reference in
25,422 voxels, 1.2122%
(98.79% agreement), against
30.52% for persistence. It is closer
than persistence at 30 of
30 non-initial saved states, keeps the exact
five-grain survivor set and the reference extinction order, and its three
extinction residuals are -20, -100, +40 steps. This case was one of the four
training initial conditions: it measures retained fidelity under long temporal
extrapolation, not transfer.

## Evaluation criteria

The cohorts below were evaluated at step 24,000 against six criteria fixed
before evaluation:

1. terminal label disagreement at most 5% of the 2,097,152 voxels
2. lower label disagreement than static-t0 persistence at all 30 non-initial saved states
3. equal terminal active-grain count
4. identical terminal survivor set
5. matching extinction count and order, with every timing residual within +/-800 steps
6. agreement on the realized fate of one designated small grain, without reappearance

Cohort-level transfer is supported when all six cases satisfy every criterion, or when five do and the sixth keeps its terminal survivor set and designated-grain fate. These are benchmark-specific criteria, not universal definitions
of a useful prediction.

## Development cohort

Six initial conditions absent from training whose references had been examined
in an earlier evaluation. **4 of 6** satisfy every
criterion; the cohort rule is not met. All six keep the exact terminal survivor
set; the two remaining exceptions are final-event timing.

| Case | step 4,000 | persistence | step 24,000 | persistence | active (model / ref.) | all criteria |
|---|---:|---:|---:|---:|:---:|:---:|
| Development microstructure 1 | 0.87% | 6.87% | 3.05% | 31.64% | 4 / 4 | yes |
| Development microstructure 2 | 0.93% | 6.66% | 4.40% | 30.48% | 5 / 5 | yes |
| Development microstructure 3 | 0.86% | 6.91% | 3.17% | 30.58% | 4 / 4 | yes |
| Development microstructure 4 | 0.84% | 8.25% | 1.96% | 27.13% | 4 / 4 | yes |
| Development microstructure 5 | 0.96% | 7.83% | 3.89% | 27.65% | 4 / 4 | no |
| Development microstructure 6 | 1.00% | 7.89% | 3.43% | 29.76% | 4 / 4 | no |

## Blind cohort

Six further initial conditions, absent from training, whose initial fields,
criteria and evaluation procedure were fixed before any outcome was opened.
They were evaluated once. **2 of 6** satisfy every
criterion (Unseen microstructures 2 and 3) and the cohort rule is not met.
5 of 6 keep the exact terminal survivor set;
Unseen microstructure 4 retains one grain that has disappeared in the reference,
produces three extinctions instead of four, and has no complete event pairing.
Every case stays closer to the reference than persistence at all 30 saved
states.

| Case | step 4,000 | persistence | step 24,000 | persistence | active (model / ref.) | all criteria |
|---|---:|---:|---:|---:|:---:|:---:|
| Unseen microstructure 1 | 0.82% | 6.87% | 3.53% | 33.56% | 4 / 4 | no |
| Unseen microstructure 2 | 0.72% | 7.38% | 3.28% | 28.68% | 4 / 4 | yes |
| Unseen microstructure 3 | 0.85% | 7.08% | 2.84% | 29.32% | 4 / 4 | yes |
| Unseen microstructure 4 | 2.11% | 6.82% | 11.30% | 29.49% | 5 / 4 | no |
| Unseen microstructure 5 | 1.15% | 6.69% | 4.23% | 32.11% | 4 / 4 | no |
| Unseen microstructure 6 | 0.94% | 7.90% | 4.72% | 27.46% | 4 / 4 | no |

The step-4,000 columns, the last saved state below H, come from an analysis
carried out after the full-horizon evaluation. They describe behaviour over the
represented horizon and are not a separately qualified endpoint. Neither cohort
is pooled with the other.

## Case specialization

A fixed physics-only recipe was applied separately to three development cases:
each descendant starts from the same trained weights and uses only that case's
initial field and the physics objective, for 1,024 optimizer updates over a
1,024-step represented window. The success rule for each case was fixed before
its run, after the unchanged-model outcome was known.

| Case | final-event residual, unchanged | specialized | terminal disagreement, unchanged | specialized | all criteria after |
|---|---:|---:|---:|---:|:---:|
| Development microstructure 5 | +1,460 | +520 | 3.89% | 1.44% | yes |
| Development microstructure 6 | +1,340 | +320 | 3.43% | 2.32% | yes |
| Development microstructure 4 | +620 | -320 | 1.96% | 1.95% | yes |

This shows bounded case specialization from the trained model. It is not a
validated adaptation policy, a minimal budget, or a comparison with training
from scratch, and it does not replace either cohort's unchanged-model results.

## Verify

```bash
python scripts/verify_n8_128_transfer.py --records benchmarks/n8_128_transfer/expected_score.json
```

When the separately held trajectories are available beneath a directory that
contains `external/n8_128_transfer/`, verify the documented assets before using
them:

```bash
python scripts/verify_n8_128_transfer.py --records benchmarks/n8_128_transfer/expected_score.json --asset-root /path/to/asset-root
```
