"""Checks for the public phase-local transfer record and release manifest."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import runpy
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "benchmarks/phaselocal_transfer/expected_score.json"
MANIFEST = ROOT / "benchmarks/phaselocal_transfer/manifest.json"
VERIFIER = ROOT / "scripts/verify_phaselocal_transfer.py"
PARENT = "764f22a548e4372e7627e71f4680bf6c806608ff90b75e8d389a1fcc98afc6e4"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_record_verifies_and_binds_public_weights() -> None:
    result = subprocess.run([sys.executable, "-I", str(VERIFIER), "--records", str(RECORD)],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "phaselocal_transfer PASS" in result.stdout
    record = load(RECORD)
    lineage = load(ROOT / "checkpoints/phase_local_48_n8.lineage.json")
    assert record["parent_checkpoint_sha256"] == PARENT
    assert record["public_weights_sha256"] == lineage["public_weights"]["sha256"]
    assert record["public_weights_sha256"] == digest(ROOT / record["public_weights"])
    assert record["same_model_for_every_case"] is True
    assert record["parameter_count"] == 2160
    assert record["regimes_pooled"] is False


@pytest.mark.parametrize("key,extinctions,residuals,volume,field", [
    ("r64_n16", 10, [1, 9], 0.0102, 0.0061),
    ("r96_n16", 10, [1, 14], 0.0109, 0.0110),
    ("r128_n38", 12, [-26, 21], 0.0110, 0.0114),
])
def test_published_small_regime_headlines(key: str, extinctions: int, residuals: list[int],
                                          volume: float, field: float) -> None:
    block = load(RECORD)[key]
    summary = block["summary"]
    assert summary["cases_meeting_all_criteria"] == 3
    assert summary["reference_extinctions"] == summary["extinctions_matched_by_identity"] == extinctions
    assert summary["signed_residual_range_steps"] == residuals
    assert round(summary["grain_volume_error_max"], 4) == volume
    assert round(summary["field_ratio_max"], 4) == field
    assert summary["returns"] == summary["model_extinctions_without_reference_extinction"] == 0


def test_published_150_grain_headlines() -> None:
    block = load(RECORD)["r128_n150"]
    cases, summary, stats = block["cases"], block["summary"], block["derived_statistics"]
    assert [c["reference_extinction_count"] for c in cases] == [126, 122, 120]
    assert summary["extinctions_matched_by_identity"] == 368
    assert summary["returns"] == summary["model_extinctions_without_reference_extinction"] == 0
    assert [r["absolute_median_steps"] for r in stats["extinction_residuals"]] == [6, 5, 7]
    assert [r["absolute_max_steps"] for r in stats["extinction_residuals"]] == [63, 103, 75]
    assert [round(c["label_disagreement"][-1]["percent"], 2) for c in cases] == [0.51, 0.70, 0.88]
    assert summary["field_ratio_max"] <= 0.011
    assert [round(c["grain_volume_error"][-1]["value"], 4) for c in cases] == [0.0407, 0.0922, 0.2665]
    assert [c["grain_volume_criterion_met_through_step"] for c in cases] == [13500, 8000, 8000]
    assert [(r["step"], r["model_only"], r["reference_only"])
            for c in cases for r in c["alive"] if not r["alive_sets_equal"]] == [
                (5000, cases[0]["alive"][6]["model_only"], []),
                (4000, cases[2]["alive"][5]["model_only"], []),
            ]
    assert [(r["step"], r["grains_reference"], round(r["ks_distance"], 3))
            for r in stats["size_distribution"]["rows"]] == [
                (3000, 352, 0.011), (8000, 168, 0.018), (13500, 82, 0.024)]
    assert summary["all_criteria_met_all_cases"] is False


@pytest.mark.parametrize("mutation", [
    "flip_verdict", "hide_alive_offset", "loosen_volume", "drop_field_state", "inflate_count",
])
def test_verifier_refuses_measurement_contradictions(mutation: str) -> None:
    verify = runpy.run_path(str(VERIFIER))
    record = load(RECORD)
    block = copy.deepcopy(record["r128_n150"])
    case = block["cases"][0]
    if mutation == "flip_verdict":
        case["criteria"]["alive_extinct_sets_and_timing"] = True
    elif mutation == "hide_alive_offset":
        case["alive"][6]["model_only"] = []
    elif mutation == "loosen_volume":
        case["grain_volume_error"][-1]["value"] = 0.4
    elif mutation == "drop_field_state":
        case["field_error"].pop()
    elif mutation == "inflate_count":
        block["summary"]["extinctions_matched_by_identity"] += 1
    if mutation == "inflate_count":
        with pytest.raises(AssertionError):
            verify["_check_summary"](block)
    else:
        with pytest.raises(AssertionError):
            verify["_check_case"](case, 150, 13500, tuple(block["evaluation_steps"]), 128 ** 3, True)


def test_generated_readme_and_manifest_coverage() -> None:
    record = load(RECORD)
    package = runpy.run_path(str(ROOT / "scripts/package_phaselocal_transfer.py"))
    assert package["benchmark_readme"](record) == (
        ROOT / "benchmarks/phaselocal_transfer/README.md").read_text(encoding="utf-8")
    assets = load(MANIFEST)["assets"]
    assert len(assets) == 39
    assert {a["release_class"] for a in assets} == {"RELEASE_ASSET"}
    assert len({a["expected_relative_location"] for a in assets}) == len(assets)
    for key in record["regime_order"]:
        for case in record[key]["cases"]:
            assert case["initial_labels_sha256"] == digest(ROOT / case["initial_labels"])
            ordinal = int(case["public_name"].split()[-1])
            roles = {a["role"] for a in assets if a["regime"] == key
                     and a["public_case"] == case["public_name"]}
            expected_roles = {"reference_volumes"} if key == "r64_n16" else {
                "reference_volumes", "model_volumes", "reference_labels", "model_labels"}
            assert roles == expected_roles, ordinal


def test_asset_verifier_refuses_drift(tmp_path: Path) -> None:
    verify = runpy.run_path(str(VERIFIER))["verify_release_assets"]
    document = load(MANIFEST)
    root = tmp_path / "release"
    for entry in document["assets"]:
        path = root / entry["expected_relative_location"]
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = entry["expected_relative_location"].encode("ascii")
        path.write_bytes(payload)
        entry["sha256"] = hashlib.sha256(payload).hexdigest()
        entry["bytes"] = len(payload)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    assert verify(manifest, root) == 39
    first = root / document["assets"][0]["expected_relative_location"]
    first.write_bytes(first.read_bytes() + b"x")
    with pytest.raises(ValueError, match="size mismatch"):
        verify(manifest, root)
    first.unlink()
    with pytest.raises(ValueError, match="missing release asset"):
        verify(manifest, root)


def test_documentation_sections() -> None:
    for name in ("REPRODUCTION_LEVELS", "CLAIM_TO_ARTIFACT_MAP", "REPLAY_ENTRYPOINTS",
                 "TRAINING_PATH_DISCLOSURE"):
        section = load(ROOT / f"docs/{name}.json")["phaselocal_transfer"]
        assert section["level"] == "SCORE_RECOMPUTATION"
        assert section["checkpoint_sha256"] == PARENT
        assert section["public_records"] == "benchmarks/phaselocal_transfer/expected_score.json"
    disclosure = load(ROOT / "docs/TRAINING_PATH_DISCLOSURE.json")["phaselocal_transfer"]
    assert disclosure["training_path"]["training_command_distributed"] is False


def test_package_uses_public_case_names() -> None:
    def check(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                check(key)
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
        elif isinstance(value, str):
            text = re.sub(r"\b[0-9a-f]{64}\b", "", value)
            assert not re.search(r"(?<![\d.])\d{8,}(?![\d.])", text), value
        elif isinstance(value, int):
            assert value < 100_000_000, value

    package = RECORD.parent
    for path in package.rglob("*"):
        check(path.relative_to(package).as_posix())
        if path.suffix == ".json":
            check(load(path))
    record = load(RECORD)
    for key in record["regime_order"]:
        assert [case["public_name"] for case in record[key]["cases"]] == [
            f"microstructure {number}" for number in range(1, 4)]
