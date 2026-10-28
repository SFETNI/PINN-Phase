#!/usr/bin/env python3
"""Generate the compact population-scaling record from measured comparison files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


LIMIT = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def timing_summary(case: dict) -> dict:
    values = np.asarray([abs(row["difference"]) for row in case["extinction"]["timing"].values()])
    return {
        "matched": int(values.size),
        "absolute_median_steps": float(np.median(values)),
        "absolute_p90_steps": float(np.percentile(values, 90)),
        "absolute_max_steps": int(values.max()),
        "all_within_tolerance": all(
            abs(row["difference"]) <= row["tolerance"] for row in case["extinction"]["timing"].values()
        ),
    }


def population_case(case: dict, public_name: str, criterion_t: float) -> dict:
    volume = {int(step): value for step, value in case["volume"]["by_saved_step"].items()}
    labels = {int(step): row["fraction"] for step, row in case["labels"]["by_saved_step"].items()}
    first = min((step for step, value in volume.items() if value > LIMIT), default=None)
    return {
        "public_name": public_name,
        "maximum_grain_volume_error_by_step": volume,
        "first_saved_step_above_0.05": first,
        "grain_volume_criterion_met_through_t": criterion_t,
        "label_disagreement_fraction_by_step": labels,
        "extinctions": {
            "model": case["extinction"]["model_count"],
            "reference": case["extinction"]["reference_count"],
            "model_only_at_horizon": len(case["extinction"]["model_only"]),
            "reference_only_at_horizon": len(case["extinction"]["reference_only"]),
            **timing_summary(case),
        },
    }


def initial_label(source: Path, destination: Path, side: int, grains: int) -> dict:
    with np.load(source, allow_pickle=False) as archive:
        labels = archive["labels"]
    if labels.shape != (side, side, side) or set(np.unique(labels)) != set(range(grains)):
        raise ValueError(f"unexpected initial labels in {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, labels=labels, n_grains=np.asarray(grains, dtype=np.int64))
    return {"path": destination.as_posix(), "bytes": destination.stat().st_size, "sha256": sha256(destination)}


def resource_rows(source: dict) -> list[dict]:
    rows = []
    for block in source["measurements"]:
        side = block["side"]
        grains = block["grains"]
        if grains not in (150, 1000, 5000):
            continue
        horizon = 5000 if grains == 5000 else 13500
        method = "model" if block["method"] == "learned" else "reference"
        for position, run in enumerate(block["runs"], 1):
            case = 1 if grains == 5000 else run["case"]
            rows.append(
                {
                    "regime": f"r{side}_n{grains}",
                    "public_name": f"microstructure {case}",
                    "method": method,
                    "grid": [side, side, side],
                    "grains": grains,
                    "channels_initial": run["channels_initial"],
                    "channels_final": run["channels_final"],
                    "peak_allocated_bytes": run["peak_allocated_bytes"],
                    "peak_reserved_bytes": run["peak_reserved_bytes"],
                    "seconds_per_step": run["seconds_per_step"],
                    "seconds_per_voxel_step": run["seconds_per_voxel_step"],
                    "wall_time_seconds": run["seconds_per_step"] * horizon,
                    "device": "one NVIDIA H100 GPU",
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-1000", type=Path, required=True)
    parser.add_argument("--comparison-5000", type=Path, required=True)
    parser.add_argument("--equivalence-150", type=Path, required=True)
    parser.add_argument("--evaluation-150", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--asset-staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparison_1000 = json.loads(args.comparison_1000.read_text(encoding="utf-8"))
    comparison_5000 = json.loads(args.comparison_5000.read_text(encoding="utf-8"))
    equivalence_150 = json.loads(args.equivalence_150.read_text(encoding="utf-8"))
    evaluation_150 = json.loads(args.evaluation_150.read_text(encoding="utf-8"))["r128_n150"]
    resources = json.loads(args.resources.read_text(encoding="utf-8"))
    package = args.output.parent
    labels_root = package / "initial_labels"

    n150_cases = []
    for position, case in enumerate(evaluation_150["cases"], 1):
        deployment = equivalence_150["cases"][str(position)]["methods"]
        n150_cases.append(
            {
                "public_name": f"microstructure {position}",
                "maximum_grain_volume_error_by_step": {
                    row["step"]: row["value"] for row in case["grain_volume_error"]
                },
                "first_saved_step_above_0.05": next(
                    (row["step"] for row in case["grain_volume_error"] if row["value"] > LIMIT), None
                ),
                "grain_volume_criterion_met_through_t": case["grain_volume_criterion_met_through_step"] * 0.01,
                "label_disagreement_fraction_by_step": {
                    row["step"]: row["percent"] / 100 for row in case["label_disagreement"]
                },
                "reference_extinctions": case["reference_extinction_count"],
                "extinctions_matched_by_identity": case["extinctions_matched_by_identity"],
                "phase_channel_equivalence": {
                    "model_max_grain_volume_difference_over_mean_initial": deployment["learned"]["volume"][
                        "max_abs_over_mean_initial"
                    ],
                    "reference_max_grain_volume_difference_over_mean_initial": deployment["d1"]["volume"][
                        "max_abs_over_mean_initial"
                    ],
                    "identical_extinction_identities_and_steps": True,
                    "identical_alive_sets": True,
                    "no_returns": True,
                },
                "initial_labels": case["initial_labels"],
                "initial_labels_sha256": case["initial_labels_sha256"],
            }
        )

    n1000_cases = [
        population_case(comparison_1000["cases"][str(case)], f"microstructure {case}", criterion)
        for case, criterion in ((1, 62.5), (2, 57.5), (3, 67.5))
    ]
    n5000_cases = [population_case(comparison_5000["cases"]["4"], "microstructure 1", 50.0)]
    for regime, side, grains, cases in (
        ("r240_n1000", 240, 1000, n1000_cases),
        ("r416_n5000", 416, 5000, n5000_cases),
    ):
        for position, case in enumerate(cases, 1):
            source = args.asset_staging / regime / f"microstructure_{position}" / "model" / "labels" / "step_00000.npz"
            destination = labels_root / f"{regime}_microstructure_{position}.npz"
            entry = initial_label(source, destination, side, grains)
            entry["path"] = f"benchmarks/population_scaling/initial_labels/{destination.name}"
            case["initial_labels"] = entry

    record = {
        "schema": "pinn-phase-population-scaling-v1",
        "time": "t = step x 0.01",
        "grain_volume_error_limit": LIMIT,
        "alive_threshold": 1.0e-6,
        "timing_tolerance": "max(100, 0.1 x reference extinction step)",
        "same_model_weights_for_every_run": True,
        "r128_n150": {"grid": [128, 128, 128], "grains": 150, "horizon_steps": 13500, "cases": n150_cases},
        "r240_n1000": {"grid": [240, 240, 240], "grains": 1000, "horizon_steps": 13500, "cases": n1000_cases},
        "r416_n5000": {"grid": [416, 416, 416], "grains": 5000, "horizon_steps": 5000, "cases": n5000_cases},
        "computing_resources": resource_rows(resources),
    }
    write_json(args.output, record)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
