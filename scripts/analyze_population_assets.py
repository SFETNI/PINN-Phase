#!/usr/bin/env python3
"""Compute volume-error and virtual-hold summaries from population archives."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np


ALIVE = 1.0e-6


def load(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as data:
        return data["volumes"]


def stop_time(volumes: np.ndarray, target: float) -> float | None:
    initial_mean = volumes[0].sum() / np.count_nonzero(volumes[0] > ALIVE)
    alive = np.count_nonzero(volumes > ALIVE, axis=1)
    ratio = volumes.sum(axis=1) / alive / initial_mean
    positions = np.flatnonzero((ratio >= target) & (np.arange(volumes.shape[0]) % 10 == 0))
    return None if positions.size == 0 else float(positions[0] * 0.01)


def summarize(model: np.ndarray, reference: np.ndarray, mean_initial: float) -> dict:
    common = (model > ALIVE) & (reference > ALIVE)
    errors = np.abs(model - reference) / mean_initial
    row = errors[-1][common[-1]]
    return {
        "terminal_grains_alive_in_both": int(row.size),
        "terminal_volume_error_median": float(np.median(row)),
        "terminal_volume_error_p90": float(np.percentile(row, 90)),
        "terminal_volume_error_p99": float(np.percentile(row, 99)),
        "terminal_volume_error_max": float(row.max()),
        "virtual_hold": {
            str(target): {
                "reference_t": stop_time(reference, target),
                "model_t": stop_time(model, target),
            }
            for target in (1.01, 1.02, 1.25, 1.50)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"schema": "pinn-phase-population-analysis-recompute-v1", "regimes": {}}
    for regime, side, grains, cases in (("r240_n1000", 240, 1000, 3), ("r416_n5000", 416, 5000, 1)):
        path = args.asset_root / f"population_scaling_{regime}.zip"
        rows = []
        with zipfile.ZipFile(path) as archive:
            for position in range(1, cases + 1):
                prefix = f"population_scaling/{regime}/microstructure_{position}"
                model = load(archive, f"{prefix}/model/volumes.npz")
                reference = load(archive, f"{prefix}/reference/volumes.npz")
                row = summarize(model, reference, side**3 / grains)
                row["public_name"] = f"microstructure {position}"
                rows.append(row)
        result["regimes"][regime] = rows
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
