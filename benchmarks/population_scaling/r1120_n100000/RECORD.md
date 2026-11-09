# 100,000-grain population record

This compact record describes one reference-free 1,120 cubed run on one node with four GPUs. No reference simulation exists at this population size, so the record supports population-level and distribution-level comparisons only. It does not establish grain-level fidelity at this size.

## Files and fields

- `metadata.json` states the grid, population, time increment, horizon, final population, thresholds, hardware scope, and source-file digests.
- `time_series.npz` stores `step`, `time`, `alive`, `n_fraction`, `mean_equivalent_radius`, and `mean_squared_equivalent_radius`. Radius is `(3V/(4*pi))^(1/3)` in voxel units and averages include grains above the alive threshold.
- `radius_distributions.npz` stores fixed-time summaries of `R/<R>`. `normalized_radius` contains sorted samples, and `sample_offset` separates times. `histogram_edges` and `histogram_count` give 200 bins from 0 to 5. The two count fields document excluded grains and the upper tail.
- `extinction_steps.npz` stores one signed 32-bit `extinction_step` per initial grain. A value of `-1` means the grain remains alive at the record horizon.
- `step_diagnostics.npz` stores `step`, `alive`, `channels`, `channel_moves`, `channel_conflicts`, and `peak_device_memory_gb`.
- `SHA256SUMS` binds every other file in this directory. As usual, the checksum index does not list itself.

NumPy archives contain numeric arrays only and load with `allow_pickle=False`.
