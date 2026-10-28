# Population scaling benchmark

This package records the phase-channel execution of the unchanged
`phase_local_48_n8` model at three population sizes. Each statement is scoped to
the listed microstructures and horizons. Time is `t = step x 0.01`.

| Regime | Evaluated microstructures | Horizon | Recorded result |
|---|---:|---:|---|
| `r128_n150` | 3 | `t = 135` | The phase-channel replay reproduces the dense-run extinction identities and steps. The evaluation record reproduces all 368 reference extinctions by identity and the per-grain volume criterion through `t = 80` in all three microstructures and through `t = 135` in microstructure 1. |
| `r240_n1000` | 3 | `t = 135` | All 2,431 matched extinctions are within their timing tolerance. One event in each direction straddles the horizon across the three runs. The maximum grain-volume error remains within 0.05 through `t = 62.5`, `57.5`, and `67.5`. |
| `r416_n5000` | 1 | `t = 50` | All 2,109 matched extinctions are within their timing tolerance. Two reference events straddle the horizon. The maximum grain-volume error is 0.038 at `t = 50`, within 0.05 through the tested horizon. |

`expected_score.json` stores the maximum grain-volume error and label
disagreement at every saved time, extinction summaries, phase-channel replay
equivalence at 150 grains, initial-label identities, and exact per-run computing
resources for both methods. The resource rows include grid, grain count, phase
channels initially and finally in use, peak allocated and reserved memory,
seconds per step, seconds per voxel-step, rollout wall time, and device. Each run
used one NVIDIA H100 GPU.

Initial label arrays for the 1,000- and 5,000-grain runs are in
`initial_labels/`. The 150-grain arrays remain in
`benchmarks/phaselocal_transfer/initial_labels/` and are referenced by digest.

## Release archives

`manifest.json` describes two release archives:

- `population_scaling_r240_n1000.zip`
- `population_scaling_r416_n5000.zip`

Each archive contains the per-step volume table, extinction record, timing
table when present, and every saved label array for the model and reference.
The release archives contain only public scientific arrays and derived records.
The 150-grain per-step volumes and saved
labels are already distributed by the v3.0.0 phase-local transfer release
assets, so they are not duplicated.

## Verification

Verify the compact record alone:

```console
python scripts/verify_population_scaling.py \
  --records benchmarks/population_scaling/expected_score.json
```

After placing the two ZIP files at an asset root, also rederive the saved-time
volume errors, extinction counts, and label disagreements:

```console
python scripts/verify_population_scaling.py \
  --records benchmarks/population_scaling/expected_score.json \
  --asset-root /path/to/release_v4_assets
```

The packaging entry points are `scripts/package_population_scaling.py` for
release arrays and `scripts/package_population_scaling_records.py` for the
compact record. They accept explicit input paths; private source paths are not
embedded in this public package.
