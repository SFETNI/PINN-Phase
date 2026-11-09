#!/usr/bin/env python3
"""Build the compact public record for a 100,000-grain evidence directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ALIVE_EPSILON = 1.0e-6
SIZE_DISTRIBUTION_MIN_VOLUME = 1.0
HISTOGRAM_EDGES = np.linspace(0.0, 5.0, 201, dtype=np.float64)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def equivalent_radius(volume: np.ndarray) -> np.ndarray:
    return np.cbrt(3.0 * volume / (4.0 * math.pi))


def choose_distribution_steps(available: np.ndarray, final_step: int, dt: float) -> np.ndarray:
    target_times = np.array([0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 57.5, 75.0])
    targets = np.rint(target_times / dt).astype(np.int64)
    targets = targets[targets <= final_step]
    chosen = [int(step) for step in targets if np.any(available == step)]
    if final_step not in chosen:
        chosen.append(final_step)
    return np.asarray(sorted(set(chosen)), dtype=np.int32)


def write_record(evidence_dir: Path, out_dir: Path) -> None:
    required = ["evidence.json", "volumes.npz", "extinction.npz"]
    missing = [name for name in required if not (evidence_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing evidence files: {', '.join(missing)}")

    metadata_source = json.loads((evidence_dir / "evidence.json").read_text(encoding="utf-8"))
    dt = float(metadata_source["h"])
    n0 = int(metadata_source["grains"])
    final_step = int(metadata_source["steps"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(evidence_dir / "volumes.npz", allow_pickle=False) as archive:
        steps64 = archive["steps"].astype(np.int64, copy=False)
        volumes = archive["volumes"].astype(np.float64, copy=False)
        if volumes.shape != (steps64.size, n0):
            raise ValueError(f"unexpected volumes shape {volumes.shape}")
        if int(steps64[-1]) != final_step:
            raise ValueError("evidence horizon and volume horizon differ")

        alive = volumes > ALIVE_EPSILON
        alive_count = alive.sum(axis=1, dtype=np.int64)
        radii = equivalent_radius(np.maximum(volumes, 0.0))
        radius_sum = np.where(alive, radii, 0.0).sum(axis=1)
        radius_squared_sum = np.where(alive, radii * radii, 0.0).sum(axis=1)
        mean_radius = radius_sum / alive_count
        mean_radius_squared = radius_squared_sum / alive_count

        np.savez_compressed(
            out_dir / "time_series.npz",
            step=steps64.astype(np.int32),
            time=steps64.astype(np.float64) * dt,
            alive=alive_count.astype(np.int32),
            n_fraction=alive_count.astype(np.float64) / float(n0),
            mean_equivalent_radius=mean_radius,
            mean_squared_equivalent_radius=mean_radius_squared,
        )

        distribution_steps = choose_distribution_steps(steps64, final_step, dt)
        offsets = [0]
        samples: list[np.ndarray] = []
        counts: list[np.ndarray] = []
        below_cut: list[int] = []
        above_range: list[int] = []
        for step in distribution_steps:
            row = int(np.flatnonzero(steps64 == step)[0])
            positive = volumes[row] > SIZE_DISTRIBUTION_MIN_VOLUME
            values = radii[row, positive]
            normalized = np.sort(values / values.mean()).astype(np.float64)
            samples.append(normalized)
            offsets.append(offsets[-1] + normalized.size)
            counts.append(np.histogram(normalized, bins=HISTOGRAM_EDGES)[0].astype(np.int32))
            below_cut.append(int(alive_count[row] - normalized.size))
            above_range.append(int(np.count_nonzero(normalized > HISTOGRAM_EDGES[-1])))

        np.savez_compressed(
            out_dir / "radius_distributions.npz",
            step=distribution_steps,
            time=distribution_steps.astype(np.float64) * dt,
            sample_offset=np.asarray(offsets, dtype=np.int32),
            normalized_radius=np.concatenate(samples),
            histogram_edges=HISTOGRAM_EDGES,
            histogram_count=np.stack(counts),
            excluded_alive_at_or_below_one_voxel=np.asarray(below_cut, dtype=np.int32),
            above_histogram_range=np.asarray(above_range, dtype=np.int32),
        )

    with np.load(evidence_dir / "extinction.npz", allow_pickle=False) as archive:
        extinction_step = archive["extinction_step"]
        if extinction_step.shape != (n0,):
            raise ValueError(f"unexpected extinction shape {extinction_step.shape}")
        if extinction_step.min() < -1 or extinction_step.max() > np.iinfo(np.int32).max:
            raise ValueError("extinction steps do not fit int32")
        np.savez_compressed(
            out_dir / "extinction_steps.npz",
            extinction_step=extinction_step.astype(np.int32),
        )

    per_step = metadata_source["per_step"]
    diagnostic_step = np.asarray(per_step["step"], dtype=np.int32)
    if diagnostic_step.size != final_step:
        raise ValueError("per-step diagnostic length does not match the horizon")
    np.savez_compressed(
        out_dir / "step_diagnostics.npz",
        step=diagnostic_step,
        alive=np.asarray(per_step["alive"], dtype=np.int32),
        channels=np.asarray(per_step["channels"], dtype=np.int16),
        channel_moves=np.asarray(per_step["moves"], dtype=np.int32),
        channel_conflicts=np.asarray(per_step["conflicts"], dtype=np.int32),
        peak_device_memory_gb=np.asarray(per_step["peak_gb"], dtype=np.float32),
    )

    public_metadata = {
        "schema": "PINN_PHASE_POPULATION_100K_RECORD_V1",
        "grid_side": int(metadata_source["side"]),
        "initial_grains": n0,
        "time_step": dt,
        "final_step": final_step,
        "final_time": final_step * dt,
        "final_alive": int(metadata_source["alive"]),
        "final_extinctions": int(metadata_source["extinctions"]),
        "returns": int(metadata_source["returns"]),
        "hardware_scope": "one node with four GPUs",
        "alive_threshold_voxels": ALIVE_EPSILON,
        "size_distribution_min_volume_voxels": SIZE_DISTRIBUTION_MIN_VOLUME,
        "source_sha256": {name: sha256(evidence_dir / name) for name in required},
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(public_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    record_text = f"""# {n0:,}-grain population record

This compact record describes one reference-free {int(metadata_source['side']):,} cubed run on one node with four GPUs. No reference simulation exists at this population size, so the record supports population-level and distribution-level comparisons only. It does not establish grain-level fidelity at this size.

## Files and fields

- `metadata.json` states the grid, population, time increment, horizon, final population, thresholds, hardware scope, and source-file digests.
- `time_series.npz` stores `step`, `time`, `alive`, `n_fraction`, `mean_equivalent_radius`, and `mean_squared_equivalent_radius`. Radius is `(3V/(4*pi))^(1/3)` in voxel units and averages include grains above the alive threshold.
- `radius_distributions.npz` stores fixed-time summaries of `R/<R>`. `normalized_radius` contains sorted samples, and `sample_offset` separates times. `histogram_edges` and `histogram_count` give 200 bins from 0 to 5. The two count fields document excluded grains and the upper tail.
- `extinction_steps.npz` stores one signed 32-bit `extinction_step` per initial grain. A value of `-1` means the grain remains alive at the record horizon.
- `step_diagnostics.npz` stores `step`, `alive`, `channels`, `channel_moves`, `channel_conflicts`, and `peak_device_memory_gb`.
- `SHA256SUMS` binds every other file in this directory. As usual, the checksum index does not list itself.

NumPy archives contain numeric arrays only and load with `allow_pickle=False`.
"""
    (out_dir / "RECORD.md").write_text(record_text, encoding="utf-8")

    payloads = sorted(path for path in out_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    checksums = "".join(f"{sha256(path)}  {path.name}\n" for path in payloads)
    (out_dir / "SHA256SUMS").write_text(checksums, encoding="utf-8", newline="\n")
    oversized = [(path.name, path.stat().st_size) for path in out_dir.iterdir() if path.is_file() and path.stat().st_size >= 50_000_000]
    if oversized:
        raise ValueError(f"record file reached 50 MB: {oversized}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    write_record(args.evidence_dir.resolve(), args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
