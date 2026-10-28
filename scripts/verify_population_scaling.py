#!/usr/bin/env python3
"""Verify compact population-scaling records and optional release archives."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np


PACKAGE = "population_scaling"
ALIVE = 1.0e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(archive: zipfile.ZipFile, name: str) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def event_steps(volumes: np.ndarray) -> dict[int, int]:
    events = {}
    for grain in range(volumes.shape[1]):
        below = np.flatnonzero(volumes[1:, grain] <= ALIVE)
        if below.size:
            events[grain] = int(below[0]) + 1
    return events


def verify_record(record: dict) -> None:
    assert record["schema"] == "pinn-phase-population-scaling-v1"
    assert record["grain_volume_error_limit"] == 0.05
    assert record["same_model_weights_for_every_run"] is True
    n1000 = record["r240_n1000"]["cases"]
    assert [case["grain_volume_criterion_met_through_t"] for case in n1000] == [62.5, 57.5, 67.5]
    assert [case["first_saved_step_above_0.05"] for case in n1000] == [6500, 6000, 7000]
    assert [round(case["maximum_grain_volume_error_by_step"]["13500"], 3) for case in n1000] == [
        0.574,
        0.226,
        0.160,
    ]
    assert [(case["extinctions"]["model"], case["extinctions"]["reference"]) for case in n1000] == [
        (805, 804),
        (808, 808),
        (819, 820),
    ]
    assert [case["extinctions"]["matched"] for case in n1000] == [804, 808, 819]
    assert all(case["extinctions"]["all_within_tolerance"] for case in n1000)
    assert [round(100 * case["label_disagreement_fraction_by_step"]["13500"], 2) for case in n1000] == [
        0.92,
        0.82,
        0.70,
    ]
    n5000 = record["r416_n5000"]["cases"][0]
    assert n5000["grain_volume_criterion_met_through_t"] == 50.0
    assert n5000["first_saved_step_above_0.05"] is None
    assert round(n5000["maximum_grain_volume_error_by_step"]["5000"], 3) == 0.038
    assert (n5000["extinctions"]["model"], n5000["extinctions"]["reference"]) == (2109, 2111)
    assert n5000["extinctions"]["matched"] == 2109
    assert n5000["extinctions"]["all_within_tolerance"] is True
    assert round(100 * n5000["label_disagreement_fraction_by_step"]["5000"], 2) == 0.35
    n150 = record["r128_n150"]["cases"]
    assert [case["reference_extinctions"] for case in n150] == [126, 122, 120]
    assert [case["extinctions_matched_by_identity"] for case in n150] == [126, 122, 120]
    assert all(case["phase_channel_equivalence"]["identical_extinction_identities_and_steps"] for case in n150)
    verify_resources(record["computing_resources"])


def verify_resources(rows: list[dict]) -> None:
    assert len(rows) == 14
    assert all(row["device"] == "one NVIDIA H100 GPU" for row in rows)
    assert all(abs(row["wall_time_seconds"] - row["seconds_per_step"] * (5000 if row["grains"] == 5000 else 13500)) < 1.0e-9 for row in rows)
    pins = {
        (150, "model"): (0.217, 103.5, 21, 35, 10.8),
        (150, "reference"): (0.038, 18.0, 14, 22, 6.0),
        (1000, "model"): (1.50, 108.5, 23, 37, 34.6),
        (1000, "reference"): (0.270, 19.5, 15, 23, 19.0),
        (5000, "model"): (8.04, 111.7, 23, 36, 61.7),
        (5000, "reference"): (1.51, 20.9, 15, 23, 35.9),
    }
    for key, (seconds, nano, low, high, memory) in pins.items():
        grain_rows = [row for row in rows if (row["grains"], row["method"]) == key]
        assert round(float(np.mean([row["seconds_per_step"] for row in grain_rows])), len(str(seconds).split(".")[1])) == seconds
        assert round(1.0e9 * float(np.mean([row["seconds_per_voxel_step"] for row in grain_rows])), 1) == nano
        assert min(row["channels_initial"] for row in grain_rows) == low
        assert max(row["channels_final"] for row in grain_rows) == high
        assert round(max(row["peak_allocated_bytes"] for row in grain_rows) / 1.0e9, 1) == memory


def verify_initial_labels(record: dict, records_path: Path) -> None:
    manifest = json.loads((records_path.parent / "manifest.json").read_text(encoding="utf-8"))
    repo = records_path.resolve().parents[2]
    expected = {
        item["path"]: (item["bytes"], item["sha256"]) for item in manifest["initial_labels"]
    }
    declared = {}
    for regime in ("r240_n1000", "r416_n5000"):
        for case in record[regime]["cases"]:
            item = case["initial_labels"]
            declared[item["path"]] = (item["bytes"], item["sha256"])
    assert declared == expected
    for relative, (size, digest) in expected.items():
        path = repo / relative
        assert path.stat().st_size == size and sha256(path) == digest
        with np.load(path, allow_pickle=False) as archive:
            labels = archive["labels"]
            grains = int(archive["n_grains"])
        assert labels.shape == tuple(record[f"r{labels.shape[0]}_n{grains}"]["grid"])
        assert np.array_equal(np.unique(labels), np.arange(grains))


def verify_assets(record: dict, manifest_path: Path, asset_root: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["assets"]:
        path = asset_root / item["filename"]
        assert path.stat().st_size == item["bytes"]
        assert sha256(path) == item["sha256"]
    checked = 0
    for regime in ("r240_n1000", "r416_n5000"):
        path = asset_root / f"population_scaling_{regime}.zip"
        with zipfile.ZipFile(path) as archive:
            for position, case in enumerate(record[regime]["cases"], 1):
                prefix = f"population_scaling/{regime}/microstructure_{position}"
                model = load_npz(archive, f"{prefix}/model/volumes.npz")["volumes"]
                reference = load_npz(archive, f"{prefix}/reference/volumes.npz")["volumes"]
                mean = np.prod(record[regime]["grid"]) / record[regime]["grains"]
                for step, expected in case["maximum_grain_volume_error_by_step"].items():
                    actual = float(np.max(np.abs(model[int(step)] - reference[int(step)])) / mean)
                    assert abs(actual - expected) <= 1.0e-12
                model_events, reference_events = event_steps(model), event_steps(reference)
                assert len(model_events) == case["extinctions"]["model"]
                assert len(reference_events) == case["extinctions"]["reference"]
                matched = set(model_events) & set(reference_events)
                assert len(matched) == case["extinctions"]["matched"]
                for step, expected in case["label_disagreement_fraction_by_step"].items():
                    label_name = f"labels/step_{int(step):05d}.npz"
                    left = load_npz(archive, f"{prefix}/model/{label_name}")["labels"]
                    right = load_npz(archive, f"{prefix}/reference/{label_name}")["labels"]
                    actual = float(np.count_nonzero(left != right) / left.size)
                    assert abs(actual - expected) <= 1.0e-15
                checked += 1
    return checked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    args = parser.parse_args()
    try:
        record = json.loads(args.records.read_text(encoding="utf-8"))
        verify_record(record)
        verify_initial_labels(record, args.records)
        checked = None
        if args.asset_root is not None:
            checked = verify_assets(record, args.records.parent / "manifest.json", args.asset_root)
    except (AssertionError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        parser.error(str(exc) or "population-scaling verification failed")
    suffix = "" if checked is None else f" ({checked} cases recomputed from release archives)"
    print(f"{PACKAGE} PASS{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
