#!/usr/bin/env python3
"""Verify the public native 128^3 record and, optionally, documented assets.

The record is checked against itself: every criterion outcome is re-derived
from the measured fields it summarizes, every cohort count from its cases, and
every headline from the cohort counts. A record whose verdicts disagree with
its own measurements fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

VOXELS = 128 ** 3
SAVED_STEPS = list(range(0, 24001, 800))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_trajectory(case: dict) -> None:
    trajectory = case["trajectory"]
    assert trajectory["steps"] == SAVED_STEPS, case["public_name"]
    model = trajectory["disagreement_percent"]
    persistence = trajectory["persistence_disagreement_percent"]
    assert len(model) == len(persistence) == 31
    assert model[0] == 0.0 and persistence[0] == 0.0
    assert model[-1] == case["terminal_disagreement_percent"]
    assert persistence[-1] == case["terminal_persistence_disagreement_percent"]
    below = sum(m < p for m, p in zip(model[1:], persistence[1:]))
    assert below == case["saved_states_below_persistence"], case["public_name"]
    assert case["saved_states_evaluated"] == 30


def _check_cohort_case(case: dict, tolerance: int) -> None:
    name = case["public_name"]
    _check_trajectory(case)
    trajectory = case["trajectory"]
    near = trajectory["steps"].index(4000)
    assert trajectory["disagreement_percent"][near] == case["step_4000_disagreement_percent"], name
    assert (trajectory["persistence_disagreement_percent"][near]
            == case["step_4000_persistence_disagreement_percent"]), name
    assert abs(case["terminal_disagreeing_voxels"] / VOXELS * 100.0
               - case["terminal_disagreement_percent"]) < 1e-5, name

    model_set = case["model_terminal_survivor_set"]
    reference_set = case["reference_terminal_survivor_set"]
    assert case["additional_survivors"] == sorted(set(model_set) - set(reference_set)), name
    assert case["missing_survivors"] == sorted(set(reference_set) - set(model_set)), name
    events = case["extinction_events"]
    for event in events:
        assert event["signed_residual_steps"] == event["model_step"] - event["reference_step"], name
        assert event["within_tolerance"] == (abs(event["signed_residual_steps"]) <= tolerance), name
    if case["events_paired"]:
        assert len(events) == case["model_extinction_count"] == case["reference_extinction_count"], name
    else:
        assert events == [] and case["model_extinction_count"] != case["reference_extinction_count"], name

    expected = {
        "terminal_disagreement": case["terminal_disagreement_percent"] <= 5.0,
        "below_persistence": case["saved_states_below_persistence"] == case["saved_states_evaluated"],
        "terminal_active_count": case["model_terminal_active_count"] == case["reference_terminal_active_count"],
        "terminal_survivor_set": model_set == reference_set,
        "extinction_events": case["events_paired"] and all(e["within_tolerance"] for e in events),
        "designated_grain_fate": (
            case["designated_grain_alive_in_model"] == case["designated_grain_alive_in_reference"]
            and case["reappearance_events"] == 0
        ),
    }
    assert case["criteria"] == expected, f"{name}: criteria disagree with the measured fields"
    assert case["all_criteria_met"] == all(expected.values()), name
    assert case["model_terminal_active_count"] == len(model_set), name
    assert case["reference_terminal_active_count"] == len(reference_set), name


def _check_summary(cohort: dict) -> None:
    cases = cohort["cases"]
    summary = cohort["summary"]
    assert summary["cases"] == len(cases) == 6
    assert summary["all_criteria_met"] == sum(c["all_criteria_met"] for c in cases)
    assert summary["terminal_survivor_set_exact"] == sum(c["criteria"]["terminal_survivor_set"] for c in cases)
    assert summary["below_persistence_at_every_saved_state"] == sum(
        c["criteria"]["below_persistence"] for c in cases)
    near = [c["step_4000_disagreement_percent"] for c in cases]
    near_p = [c["step_4000_persistence_disagreement_percent"] for c in cases]
    assert summary["step_4000_disagreement_percent_range"] == [min(near), max(near)]
    assert summary["step_4000_persistence_disagreement_percent_range"] == [min(near_p), max(near_p)]
    terminal = [c["terminal_disagreement_percent"] for c in cases]
    assert summary["terminal_disagreement_percent_range"] == [min(terminal), max(terminal)]
    # The support rule: all six, or five with the sixth keeping its survivor set and fate.
    met = summary["all_criteria_met"]
    exception_ok = all(
        c["criteria"]["terminal_survivor_set"] and c["criteria"]["designated_grain_fate"]
        for c in cases if not c["all_criteria_met"]
    )
    assert cohort["cohort_rule_met"] == (met == 6 or (met == 5 and exception_ok))


def verify_score_record(records: Path) -> None:
    x = json.loads(records.read_text(encoding="utf-8"))
    assert x["schema"] == "pinn-phase-n8-128-transfer-v1"
    assert x["reproduction_level"] == "PROVENANCE_ONLY" and x["public_case_names_only"] is True
    assert x["grid"] == [128, 128, 128] and x["voxels"] == VOXELS and x["phases"] == 8
    assert x["represented_horizon_steps"] == 4096 and x["terminal_step"] == 24000
    assert x["terminal_step_over_horizon"] == 5.86
    assert x["saved_state_cadence_steps"] == 800 and x["timing_tolerance_steps"] == 800
    assert len(x["criteria"]) == 6
    tolerance = x["timing_tolerance_steps"]

    train = x["training_case"]
    _check_trajectory(train)
    assert train["terminal_step"] == 24000
    assert abs(train["terminal_disagreeing_voxels"] / VOXELS * 100.0 - train["terminal_disagreement_percent"]) < 1e-5
    assert train["model_terminal_survivor_set"] == train["reference_terminal_survivor_set"]
    assert train["model_extinction_order"] == train["reference_extinction_order"]
    assert [e["grain"] for e in train["extinction_events"]] == train["reference_extinction_order"]
    for event in train["extinction_events"]:
        assert event["signed_residual_steps"] == event["model_step"] - event["reference_step"]
    assert train["reappearance_events"] == 0

    for key in ("development_cohort", "unseen_cohort"):
        for case in x[key]["cases"]:
            _check_cohort_case(case, tolerance)
        _check_summary(x[key])

    # Published headlines, re-derived.
    assert x["development_cohort"]["summary"]["all_criteria_met"] == 4
    assert x["unseen_cohort"]["summary"]["all_criteria_met"] == 2
    assert [c["public_name"] for c in x["unseen_cohort"]["cases"] if c["all_criteria_met"]] == [
        "Unseen microstructure 2", "Unseen microstructure 3"]
    assert x["unseen_cohort"]["summary"]["terminal_survivor_set_exact"] == 5
    assert not x["development_cohort"]["cohort_rule_met"] and not x["unseen_cohort"]["cohort_rule_met"]

    names = {c["public_name"] for c in x["development_cohort"]["cases"]}
    for case in x["specialization"]["cases"]:
        assert case["public_name"] in names
        before, after = case["unchanged_model"], case["specialized_model"]
        if before["all_criteria_met"]:
            assert after["all_criteria_met"], case["public_name"]
        else:
            assert abs(after["final_extinction_residual_steps"]) <= tolerance < abs(
                before["final_extinction_residual_steps"]), case["public_name"]
            assert after["all_criteria_met"], case["public_name"]


def verify_external_assets(asset_manifest: Path, asset_root: Path) -> int:
    """Fail closed unless every documented external payload matches its record."""
    document = json.loads(asset_manifest.read_text(encoding="utf-8"))
    assets = document.get("assets")
    if not isinstance(assets, list) or len(assets) != 26:
        raise ValueError("asset manifest must declare exactly 26 external assets")
    if not asset_root.is_dir():
        raise ValueError(f"asset root is not a directory: {asset_root}")
    seen: set[str] = set()
    expected_directories = {"external/n8_128_transfer"}
    for asset in assets:
        relative = asset.get("expected_relative_location")
        filename = asset.get("filename")
        expected_sha256 = asset.get("sha256")
        if not isinstance(relative, str) or not isinstance(filename, str) or not isinstance(expected_sha256, str):
            raise ValueError("asset manifest has a non-string path, filename or SHA-256")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or relative_path.name != filename:
            raise ValueError(f"asset manifest has unsafe or mismatched path: {relative}")
        if relative in seen:
            raise ValueError(f"asset manifest repeats an external asset path: {relative}")
        seen.add(relative)
        expected_directories.add(relative_path.parent.as_posix())
        candidate = asset_root / relative_path
        if candidate.is_symlink():
            raise ValueError(f"documented external asset must not be a symlink: {relative}")
        if not candidate.is_file():
            raise ValueError(f"missing documented external asset: {relative}")
        actual = sha256(candidate)
        if actual != expected_sha256:
            raise ValueError(f"SHA-256 mismatch for {relative}: expected {expected_sha256}, got {actual}")
    documented_root = asset_root / "external/n8_128_transfer"
    for candidate in sorted(documented_root.rglob("*")):
        relative = candidate.relative_to(asset_root).as_posix()
        if candidate.is_symlink():
            raise ValueError(f"unexpected symlink in documented external assets: {relative}")
        if candidate.is_file() and relative not in seen:
            raise ValueError(f"unexpected regular file in documented external assets: {relative}")
        if candidate.is_dir() and relative not in expected_directories:
            raise ValueError(f"unexpected directory in documented external assets: {relative}")
    return len(seen)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path,
                        help="directory containing the documented external/n8_128_transfer/ hierarchy")
    args = parser.parse_args()
    try:
        verify_score_record(args.records)
        count = None
        if args.asset_root is not None:
            count = verify_external_assets(args.records.parent / "manifest.json", args.asset_root)
    except (AssertionError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc) or "native 128^3 record verification failed")
    suffix = f" and {count} external assets" if count is not None else ""
    print(f"Native 128^3 public record: PASS{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
