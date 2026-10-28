"""Checks for the public population-scaling records and release manifest."""
from __future__ import annotations

import copy
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "benchmarks/population_scaling"
RECORD = PACKAGE / "expected_score.json"
ANALYSES = PACKAGE / "analyses/expected_score.json"
MANIFEST = PACKAGE / "manifest.json"
VERIFIER = ROOT / "scripts/verify_population_scaling.py"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compact_record_verifies() -> None:
    result = subprocess.run(
        [sys.executable, "-I", str(VERIFIER), "--records", str(RECORD)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "population_scaling PASS" in result.stdout


def test_population_headline_values() -> None:
    record = load(RECORD)
    cases = record["r240_n1000"]["cases"]
    assert [case["grain_volume_criterion_met_through_t"] for case in cases] == [62.5, 57.5, 67.5]
    assert [case["first_saved_step_above_0.05"] for case in cases] == [6500, 6000, 7000]
    assert [round(case["maximum_grain_volume_error_by_step"]["13500"], 3) for case in cases] == [
        0.574,
        0.226,
        0.160,
    ]
    assert [(case["extinctions"]["model"], case["extinctions"]["reference"]) for case in cases] == [
        (805, 804),
        (808, 808),
        (819, 820),
    ]
    assert [case["extinctions"]["matched"] for case in cases] == [804, 808, 819]
    assert [round(100 * case["label_disagreement_fraction_by_step"]["13500"], 2) for case in cases] == [
        0.92,
        0.82,
        0.70,
    ]
    assert all(case["extinctions"]["all_within_tolerance"] for case in cases)

    large = record["r416_n5000"]["cases"][0]
    assert large["grain_volume_criterion_met_through_t"] == 50.0
    assert large["first_saved_step_above_0.05"] is None
    assert round(large["maximum_grain_volume_error_by_step"]["5000"], 3) == 0.038
    assert (large["extinctions"]["model"], large["extinctions"]["reference"]) == (2109, 2111)
    assert large["extinctions"]["matched"] == 2109
    assert large["extinctions"]["all_within_tolerance"] is True
    assert round(100 * large["label_disagreement_fraction_by_step"]["5000"], 2) == 0.35


def test_phase_channel_replay_headline_values() -> None:
    cases = load(RECORD)["r128_n150"]["cases"]
    assert [case["reference_extinctions"] for case in cases] == [126, 122, 120]
    assert [case["extinctions_matched_by_identity"] for case in cases] == [126, 122, 120]
    assert [case["grain_volume_criterion_met_through_t"] for case in cases] == [135.0, 80.0, 80.0]
    assert all(case["phase_channel_equivalence"]["identical_alive_sets"] for case in cases)
    assert all(
        case["phase_channel_equivalence"]["identical_extinction_identities_and_steps"] for case in cases
    )
    assert all(case["phase_channel_equivalence"]["no_returns"] for case in cases)


def test_analysis_headline_values() -> None:
    analyses = load(ANALYSES)
    growth = analyses["growth_and_faces"]
    assert [growth[key]["growth_rate_pearson"] for key in ("r128_n150", "r240_n1000", "r416_n5000")] == [
        0.99958,
        0.99948,
        0.999946,
    ]
    assert [
        (growth[key]["face_law_intercept_reference"], growth[key]["face_law_intercept_model"])
        for key in ("r128_n150", "r240_n1000", "r416_n5000")
    ] == [(15.242, 15.246), (15.345, 15.347), (15.2507, 15.2504)]
    control = analyses["time_rescaling_control"]
    assert control["maximum_ks_distance_to_reference"] == 0.0025
    assert control["time_to_fraction"] == {
        "reference": 47.3,
        "model": 47.4,
        "factor_1.03": 45.9,
        "factor_1.10": 43.0,
    }
    hold = analyses["virtual_isothermal_hold"]
    assert hold["all_case_target_decisions"] == 34
    assert hold["model_maximum_absolute_time_error_percent"] == 0.29
    assert hold["model_premature_stops"] == 1
    assert hold["prospective_5000"]["maximum_model_delay_percent"] == 0.22
    assert hold["prospective_5000"]["target_1.50_times"] == {
        "reference": 40.1,
        "model": 40.2,
        "factor_1.03": 39.0,
        "factor_1.10": 36.5,
    }


def test_initial_label_manifest_and_arrays() -> None:
    record, manifest = load(RECORD), load(MANIFEST)
    declared = {}
    for regime in ("r240_n1000", "r416_n5000"):
        for case in record[regime]["cases"]:
            item = case["initial_labels"]
            declared[item["path"]] = (item["bytes"], item["sha256"])
    listed = {item["path"]: (item["bytes"], item["sha256"]) for item in manifest["initial_labels"]}
    assert declared == listed
    for relative, (size, expected_digest) in listed.items():
        path = ROOT / relative
        assert path.stat().st_size == size
        assert digest(path) == expected_digest
        with np.load(path, allow_pickle=False) as archive:
            assert set(archive.files) == {"labels", "n_grains"}
            labels = archive["labels"]
            grains = int(archive["n_grains"])
        assert labels.shape == tuple(record[f"r{labels.shape[0]}_n{grains}"]["grid"])
        assert np.array_equal(np.unique(labels), np.arange(grains))


def test_release_manifest_and_readme_coverage() -> None:
    manifest = load(MANIFEST)
    assert manifest["schema"] == "pinn-phase-population-scaling-manifest-v1"
    assert [item["filename"] for item in manifest["assets"]] == [
        "population_scaling_r240_n1000.zip",
        "population_scaling_r416_n5000.zip",
    ]
    assert {item["release_class"] for item in manifest["assets"]} == {"RELEASE_ASSET"}
    readme = (PACKAGE / "README.md").read_text(encoding="utf-8")
    for item in manifest["assets"]:
        assert item["filename"] in readme
    assert "python scripts/verify_population_scaling.py" in readme
    assert "--records benchmarks/population_scaling/expected_score.json" in readme
    assert "r240_n1000" in readme and "r416_n5000" in readme


@pytest.mark.parametrize("mutation", ["volume_time", "terminal_error", "event_count", "resource_device"])
def test_verifier_refuses_mutated_measurements(mutation: str) -> None:
    verify = runpy.run_path(str(VERIFIER))["verify_record"]
    record = copy.deepcopy(load(RECORD))
    if mutation == "volume_time":
        record["r240_n1000"]["cases"][0]["grain_volume_criterion_met_through_t"] = 63.0
    elif mutation == "terminal_error":
        record["r416_n5000"]["cases"][0]["maximum_grain_volume_error_by_step"]["5000"] = 0.05
    elif mutation == "event_count":
        record["r240_n1000"]["cases"][1]["extinctions"]["matched"] += 1
    else:
        record["computing_resources"][0]["device"] = "different device"
    with pytest.raises(AssertionError):
        verify(record)
