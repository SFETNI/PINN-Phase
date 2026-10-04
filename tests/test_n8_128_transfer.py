"""Load-bearing public guards for the native 128^3 study."""
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
from PIL import Image, ImageSequence


ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "benchmarks/n8_128_transfer/expected_score.json"
ASSET_PATH = ROOT / "benchmarks/n8_128_transfer/manifest.json"
MEDIA_PATH = ROOT / "media/n8_128_transfer/asset_manifest.json"
GIF = ROOT / "media/n8_128_transfer/n8_128_unseen_reference_vs_pinn_phase.gif"
POSTER = ROOT / "media/n8_128_transfer/n8_128_unseen_reference_vs_pinn_phase_poster.png"
CHECKPOINT = "0b04afab56139ecb13071396c7ca10777575add22d73949c0abeaaaeb0a4ed6b"
VERIFIER = ROOT / "scripts/verify_n8_128_transfer.py"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _record() -> dict:
    return _json(RECORD_PATH)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify(record: dict, tmp_path: Path) -> None:
    path = tmp_path / "expected_score.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    runpy.run_path(str(VERIFIER))["verify_score_record"](path)


def test_public_record_verifies_and_names_one_model() -> None:
    completed = subprocess.run([sys.executable, "-B", str(VERIFIER), "--records", str(RECORD_PATH)],
                               cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    record = _record()
    assert record["parent_checkpoint_sha256"] == CHECKPOINT
    assert record["same_model_for_every_case"] is True
    assert record["reproduction_level"] == "PROVENANCE_ONLY"
    assert record["public_weights"] == "checkpoints/n8_128_cube_multi_ic_hybrid.weights.npz"


def test_record_reproduces_the_published_tables() -> None:
    record = _record()
    unseen = record["unseen_cohort"]["cases"]
    development = record["development_cohort"]["cases"]
    assert [round(c["step_4000_disagreement_percent"], 2) for c in unseen] == [0.82, 0.72, 0.85, 2.11, 1.15, 0.94]
    assert [round(c["step_4000_persistence_disagreement_percent"], 2) for c in unseen] == [
        6.87, 7.38, 7.08, 6.82, 6.69, 7.90]
    assert [round(c["terminal_disagreement_percent"], 2) for c in unseen] == [3.53, 3.28, 2.84, 11.30, 4.23, 4.72]
    assert [c["all_criteria_met"] for c in unseen] == [False, True, True, False, False, False]
    assert [(c["model_terminal_active_count"], c["reference_terminal_active_count"]) for c in unseen] == [
        (4, 4), (4, 4), (4, 4), (5, 4), (4, 4), (4, 4)]
    assert [round(c["terminal_disagreement_percent"], 2) for c in development] == [3.05, 4.40, 3.17, 1.96, 3.89, 3.43]
    assert [c["all_criteria_met"] for c in development] == [True, True, True, True, False, False]
    train = record["training_case"]
    assert train["terminal_disagreeing_voxels"] == 25422
    assert round(train["terminal_agreement_percent"], 2) == 98.79
    assert round(train["terminal_persistence_agreement_percent"], 2) == 69.48
    assert [e["signed_residual_steps"] for e in train["extinction_events"]] == [-20, -100, 40]
    spec = {c["public_name"]: c for c in record["specialization"]["cases"]}
    assert spec["Development microstructure 5"]["specialized_model"]["final_extinction_residual_steps"] == 520
    assert spec["Development microstructure 6"]["specialized_model"]["final_extinction_residual_steps"] == 320
    assert spec["Development microstructure 4"]["specialized_model"]["all_criteria_met"] is True


def test_cohorts_are_never_pooled_and_rules_are_not_met() -> None:
    record = _record()
    assert record["development_cohort"]["cohort_rule_met"] is False
    assert record["unseen_cohort"]["cohort_rule_met"] is False
    serialized = json.dumps(record)
    assert "pooled" not in serialized.replace("not pooled", "")
    names = [c["public_name"] for c in record["unseen_cohort"]["cases"]]
    assert names == [f"Unseen microstructure {i}" for i in range(1, 7)]


@pytest.mark.parametrize("mutation", [
    "flip_verdict", "hide_extra_survivor", "loosen_terminal", "drop_persistence_state", "inflate_count",
])
def test_verifier_refuses_a_record_that_contradicts_its_measurements(tmp_path: Path, mutation: str) -> None:
    record = copy.deepcopy(_record())
    case = record["unseen_cohort"]["cases"][3]
    if mutation == "flip_verdict":
        case["all_criteria_met"] = True
    elif mutation == "hide_extra_survivor":
        case["criteria"]["terminal_survivor_set"] = True
    elif mutation == "loosen_terminal":
        record["unseen_cohort"]["cases"][1]["terminal_disagreement_percent"] = 5.2
    elif mutation == "drop_persistence_state":
        record["unseen_cohort"]["cases"][0]["saved_states_below_persistence"] = 29
    elif mutation == "inflate_count":
        record["unseen_cohort"]["summary"]["all_criteria_met"] = 3
    with pytest.raises(AssertionError):
        _verify(record, tmp_path)


def test_generated_benchmark_readme_matches_the_record() -> None:
    package = runpy.run_path(str(ROOT / "scripts/package_n8_128_transfer.py"))
    rendered = package["benchmark_readme"](_record())
    assert rendered == (ROOT / "benchmarks/n8_128_transfer/README.md").read_text(encoding="utf-8")
    for _, text in package["CRITERIA"]:
        assert text in rendered


def test_asset_manifest_documents_every_trajectory_in_the_record() -> None:
    record = _record()
    assets = _json(ASSET_PATH)["assets"]
    assert len(assets) == 26
    assert {a["release_class"] for a in assets} == {"DOCUMENTED_ONLY"}
    expected = {record["training_case"]["model_trajectory_sha256"],
                record["training_case"]["reference_trajectory_sha256"]}
    for cohort in ("development_cohort", "unseen_cohort"):
        for case in record[cohort]["cases"]:
            expected |= {case["model_trajectory_sha256"], case["reference_trajectory_sha256"]}
    assert {a["sha256"] for a in assets} == expected
    for path in ("docs/REPRODUCTION_LEVELS.json", "docs/ARTIFACT_IDENTITY_LEDGER.json",
                 "docs/CLAIM_TO_ARTIFACT_MAP.json", "docs/REPLAY_ENTRYPOINTS.json",
                 "docs/TRAINING_PATH_DISCLOSURE.json"):
        section = _json(ROOT / path)["n8_128_transfer"]
        assert section["level"] == "PROVENANCE_ONLY"
        assert section["checkpoint_sha256"] == CHECKPOINT


def _asset_fixture(root: Path) -> Path:
    document = _json(ASSET_PATH)
    for asset in document["assets"]:
        payload = f"128 fixture {asset['expected_relative_location']}".encode("ascii")
        path = root / asset["expected_relative_location"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        asset["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    return manifest


def test_asset_verifier_accepts_a_complete_hierarchy_and_refuses_drift(tmp_path: Path) -> None:
    verify = runpy.run_path(str(VERIFIER))["verify_external_assets"]
    root = tmp_path / "assets"
    manifest = _asset_fixture(root)
    assert verify(manifest, root) == 26
    first = next((root / "external/n8_128_transfer").rglob("*.npz"))
    original = first.read_bytes()
    first.write_bytes(original + b"x")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify(manifest, root)
    first.unlink()
    with pytest.raises(ValueError, match="missing documented external asset"):
        verify(manifest, root)
    first.write_bytes(original)
    (first.parent / "extra.npz").write_bytes(b"extra")
    with pytest.raises(ValueError, match="unexpected regular file"):
        verify(manifest, root)


def test_weight_artifact_is_registered_and_not_a_replay_entry_point() -> None:
    ledger = _json(ROOT / "docs/ARTIFACT_IDENTITY_LEDGER.json")
    entry = next(a for a in ledger["artifacts"] if a["name"] == "n8_128_cube_multi_ic_hybrid")
    assert entry["parent_checkpoint_sha256"] == CHECKPOINT
    assert entry["public_weights_sha256"] == _sha256(ROOT / entry["public_weights_path"])
    assert entry["weight_lineage_sha256"] == _sha256(ROOT / entry["weight_lineage_path"])
    assert entry["initial_field"] is None
    assert entry["replay_equivalence"]["max_abs_difference"] == 0.0
    replay = _json(ROOT / "docs/REPLAY_ENTRYPOINTS.json")
    assert replay["n8_128_transfer"]["replay_entry_point"] is None
    assert not any("128" in name for name in replay["entrypoints"])
    disclosure = _json(ROOT / "docs/TRAINING_PATH_DISCLOSURE.json")["n8_128_transfer"]["training_path"]
    assert disclosure["replayed_through_guard"] is False
    assert disclosure["trainer_revision_distributed"] is False
    assert disclosure["reference_usage_policy"]["training_uses_reference_frames_after_t0"] is False


def test_media_bindings_frames_and_outputs() -> None:
    media = _json(MEDIA_PATH)
    record = _record()
    assert media["renderer_sha256"] == _sha256(ROOT / "scripts/render_n8_128_transfer.py")
    assert media["record_sha256"] == _sha256(RECORD_PATH)
    assert media["parent_checkpoint_sha256"] == CHECKPOINT
    assert media["cases_shown"] == ["Unseen microstructure 2", "Unseen microstructure 3", "Unseen microstructure 6"]
    by_name = {c["public_name"]: c for c in record["unseen_cohort"]["cases"]}
    for item in media["inputs"]:
        assert item["sha256"] == by_name[item["case"]][f"{item['role']}_trajectory_sha256"]
    for name, recomputed in media["recomputed"].items():
        case = by_name[name]
        assert recomputed["disagreement_percent"] == pytest.approx(
            case["trajectory"]["disagreement_percent"], abs=1e-5)
        assert recomputed["persistence_percent"] == pytest.approx(
            case["trajectory"]["persistence_disagreement_percent"], abs=1e-5)
    assert media["steps"] == list(range(0, 24001, 800)) and media["no_interpolation"] is True
    for path in (GIF, POSTER):
        assert media["outputs"][path.name] == {"sha256": _sha256(path), "bytes": path.stat().st_size}
    with Image.open(GIF) as image:
        assert sum(1 for _ in ImageSequence.Iterator(image)) == 31
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert 'media/n8_128_transfer/n8_128_unseen_reference_vs_pinn_phase.gif" width="900"' in readme


def test_renderer_draws_a_frame_and_refuses_values_that_disagree_with_the_record() -> None:
    renderer = runpy.run_path(str(ROOT / "scripts/render_n8_128_transfer.py"))
    camera = renderer["camera_metadata"](128)
    assert (camera["camera_elevation_deg"], camera["camera_azimuth_deg"]) == (22.0, -58.0)
    assert camera["grid"] == [128, 128, 128]
    cube = renderer["cube_image"](np.zeros((128, 128, 128), dtype=np.uint8), pixels=60)
    assert cube.shape[0] == 60 and cube.shape[2] == 4
    shades = {round(float(v), 4) for v in np.unique(cube[..., 0][cube[..., 3] > 0])}
    assert len(shades) == 3, "three faces, three brightness levels"
    rng = np.random.default_rng(0)
    steps = len(renderer["STEPS"])
    reference = rng.integers(0, 8, size=(steps, 128, 128, 128), dtype=np.uint8)
    model = reference.copy()
    model[5:, 60:68] = 0
    data = renderer["series"](reference, model)
    assert data["disagreement_percent"][0] == 0.0 and data["disagreement_percent"][-1] > 0.0
    case = {"public_name": "Unseen microstructure 2", "outcome": "all six criteria met",
            "reference": reference, "model": model,
            "reference_terminal_cube": renderer["cube_image"](reference[-1]),
            "model_terminal_cube": renderer["cube_image"](model[-1]), "series": data}
    section = renderer["label_section"](reference[3])
    assert section.shape == (128, 128, 3)
    image = renderer["frame"]([case, case, case], 8)
    assert image.size == (1360, 1200)
    accepted = _record()["unseen_cohort"]["cases"][1]
    with pytest.raises(renderer["SourceConflict"]):
        renderer["check_against_record"](accepted, data)
