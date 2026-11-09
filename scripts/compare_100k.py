#!/usr/bin/env python3
"""Compute public 100,000-grain comparison numbers from numeric records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp


ALIVE_EPSILON = 1.0e-6
SIZE_DISTRIBUTION_MIN_VOLUME = 1.0


def samples_at(archive: np.lib.npyio.NpzFile, index: int) -> np.ndarray:
    offsets = archive["sample_offset"]
    return archive["normalized_radius"][int(offsets[index]) : int(offsets[index + 1])]


def compare(record_dir: Path, reference_volumes_path: Path) -> dict[str, object]:
    metadata = json.loads((record_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(record_dir / "time_series.npz", allow_pickle=False) as series:
        steps_100k = series["step"].astype(np.int64)
        fraction_100k = series["n_fraction"].astype(np.float64)
        alive_100k = series["alive"].astype(np.int64)
        time_100k = series["time"].astype(np.float64)

    with np.load(record_dir / "extinction_steps.npz", allow_pickle=False) as extinction:
        extinction_step = extinction["extinction_step"].astype(np.int64)
    horizon_step = int(steps_100k[-1])
    every_step = np.arange(horizon_step + 1, dtype=np.int64)
    extinct_sorted = np.sort(extinction_step[extinction_step >= 0])
    fraction_100k_every_step = 1.0 - np.searchsorted(
        extinct_sorted, every_step, side="right"
    ) / extinction_step.size

    with np.load(reference_volumes_path, allow_pickle=False) as reference:
        steps_reference = reference["steps"].astype(np.int64)
        volumes_reference = reference["volumes"].astype(np.float64, copy=False)
        reference_by_step = {int(step): index for index, step in enumerate(steps_reference)}
        common_steps = np.asarray([step for step in every_step if int(step) in reference_by_step], dtype=np.int64)
        if not common_steps.size:
            raise ValueError("the records have no common steps")
        index_reference = np.asarray([reference_by_step[int(step)] for step in common_steps])
        fraction_reference = (volumes_reference[index_reference] > ALIVE_EPSILON).sum(axis=1) / volumes_reference.shape[1]
        absolute_difference = np.abs(fraction_100k_every_step[common_steps] - fraction_reference)
        maximum_index = int(np.argmax(absolute_difference))

        ks_by_time: list[dict[str, object]] = []
        with np.load(record_dir / "radius_distributions.npz", allow_pickle=False) as distributions:
            for index, step in enumerate(distributions["step"].astype(np.int64)):
                if step == 0 or int(step) not in reference_by_step:
                    continue
                ref_volume = volumes_reference[reference_by_step[int(step)]]
                ref_radius = np.cbrt(3.0 * ref_volume[ref_volume > SIZE_DISTRIBUTION_MIN_VOLUME] / (4.0 * np.pi))
                ref_normalized = ref_radius / ref_radius.mean()
                values_100k = samples_at(distributions, index)
                distance = float(ks_2samp(values_100k, ref_normalized, method="auto").statistic)
                ks_by_time.append(
                    {
                        "step": int(step),
                        "time": float(distributions["time"][index]),
                        "ks_distance": distance,
                        "samples_100k": int(values_100k.size),
                        "samples_5000_reference": int(ref_normalized.size),
                    }
                )

    distances = [item["ks_distance"] for item in ks_by_time]
    return {
        "schema": "PINN_PHASE_POPULATION_100K_COMPARISON_V1",
        "run": {
            "grid_side": int(metadata["grid_side"]),
            "initial_grains": int(metadata["initial_grains"]),
            "horizon_step": int(steps_100k[-1]),
            "horizon_time": float(time_100k[-1]),
            "survivors": int(alive_100k[-1]),
        },
        "population_fraction_comparison": {
            "comparator": "5,000-grain reference simulation",
            "common_start_time": float(common_steps[0] * metadata["time_step"]),
            "common_end_time": float(common_steps[-1] * metadata["time_step"]),
            "common_points": int(common_steps.size),
            "maximum_absolute_difference": float(absolute_difference[maximum_index]),
            "maximum_at_step": int(common_steps[maximum_index]),
            "maximum_at_time": float(common_steps[maximum_index] * metadata["time_step"]),
        },
        "normalized_radius_comparison": {
            "definition": "two-sample KS distance for R/<R>, using grains above one voxel",
            "matched_times": ks_by_time,
            "minimum_ks_distance": float(min(distances)),
            "maximum_ks_distance": float(max(distances)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record_dir", type=Path)
    parser.add_argument("reference_5000_volumes", type=Path)
    parser.add_argument("out_json", type=Path)
    args = parser.parse_args()
    result = compare(args.record_dir.resolve(), args.reference_5000_volumes.resolve())
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
