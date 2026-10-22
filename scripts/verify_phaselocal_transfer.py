#!/usr/bin/env python3
"""Verify the public phase-local transfer record and, optionally, its release assets.

The record is checked against itself: every criterion outcome is re-derived from
the measured fields it summarizes, every count and maximum from its rows, and
every published headline from the cases. A record whose verdicts disagree with its
own measurements fails. The initial label arrays shipped beside the record are
checked by digest and content.

With ``--asset-root``, every release asset is checked against ``manifest.json``
(fail closed: missing, mismatched, extra, symlinked or unsafe paths are refused),
and alive sets, extinction steps, returns, grain-volume errors, label
disagreement and the 150-grain statistics are recomputed from the arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PARENT = "764f22a548e4372e7627e71f4680bf6c806608ff90b75e8d389a1fcc98afc6e4"
COMPARATOR = "8491d9c94d91913093bdd5c2915bdf77307af6da648a547b486c15a380d6107b"
PACKAGE = "phaselocal_transfer"
ALIVE = 1e-6
LIMIT = 0.05
REGIMES = {
    "r64_n16": (64, 16, 3000, (100, 200, 500, 1000, 1500, 2000, 2500, 3000)),
    "r96_n16": (96, 16, 6750, (100, 200, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 6750)),
    "r128_n38": (128, 38, 6750, (100, 200, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 6750)),
    "r128_n150": (128, 150, 13500, (200, 400, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 13500)),
}
CASEWISE = ("admissibility", "alive_extinct_sets_and_timing",
            "field_error_relative_to_persistence", "grain_volume_error")
CRITERIA = ("structural_consistency",) + CASEWISE
LABELS = dict(zip(CRITERIA, ("(i)", "(ii)", "(iii)", "(iv)", "(v)")))
ASSET_COUNT = 39


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(a: float, b: float, rel: float = 1e-12) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)


# ---------------------------------------------------------------------------
# Record self-consistency
# ---------------------------------------------------------------------------

def _structural_met(block: dict) -> bool:
    if "checks" in block:
        return bool(block["checks"]) and all(
            c["check_passed"] and c["all_grains_alive"] and c["permutation_max_abs_difference"] <= 1e-15
            for c in block["checks"])
    return len(block["fields"]) == 2 and all(
        f["permutation_max_abs_difference"] <= 1e-15 and f["existing_phases_bitwise_equal"]
        and f["appended_phases_exactly_zero"] and f["max_unit_sum_error"] <= 1e-12
        and f["min_value"] >= 0.0 and f["finite"] for f in block["fields"])


def _expected_alive(grains: int, step: int, events: list[dict], key: str) -> set[int]:
    gone = {e["grain"] for e in events if e[key] is not None and e[key] <= step}
    return set(range(grains)) - gone


def _check_case(case: dict, grains: int, horizon: int, steps: tuple, voxels: int,
                structural: bool | None) -> None:
    name = case["public_name"]
    events = case["extinctions"]
    order = [(e["reference_step"] if e["reference_step"] is not None else e["model_step"], e["grain"])
             for e in events]
    assert order == sorted(order) and len({e["grain"] for e in events}) == len(events), name
    for e in events:
        assert 0 <= e["grain"] < grains, name
        if e["reference_step"] is not None and e["model_step"] is not None:
            tol = max(100.0, 0.1 * e["reference_step"])
            assert e["signed_residual_steps"] == e["model_step"] - e["reference_step"], name
            assert e["tolerance_steps"] == tol, name
            assert e["within_tolerance"] == (abs(e["signed_residual_steps"]) <= tol), name
        else:
            assert e["signed_residual_steps"] is None and e["within_tolerance"] is False, name
    assert case["reference_extinction_count"] == sum(e["reference_step"] is not None for e in events), name
    assert case["model_extinction_count"] == sum(e["model_step"] is not None for e in events), name
    assert case["extinctions_matched_by_identity"] == sum(
        e["reference_step"] is not None and e["model_step"] is not None for e in events), name

    assert [r["step"] for r in case["alive"]] == list(steps), name
    for row in case["alive"]:
        ref = _expected_alive(grains, row["step"], events, "reference_step")
        mod = _expected_alive(grains, row["step"], events, "model_step")
        assert row["reference_alive_count"] == len(ref) and row["model_alive_count"] == len(mod), name
        assert row["alive_sets_equal"] == (ref == mod), name
        assert row["reference_only"] == sorted(ref - mod) and row["model_only"] == sorted(mod - ref), name

    assert [r["step"] for r in case["field_error"]] == list(steps), name
    for row in case["field_error"]:
        factor = 0.25 if row["step"] == horizon else 0.5
        assert row["factor"] == factor and row["criterion_applies"] == (row["step"] >= 500), name
        assert close(row["field_ratio"], row["e_tv_model"] / row["e_tv_persistence"]), name
        assert close(row["criterion_ratio"], row["field_ratio"] / factor), name
    assert [r["step"] for r in case["grain_volume_error"]] == list(steps), name
    labels = case["label_disagreement"]
    if labels is not None:
        assert [r["step"] for r in labels] == list(steps), name
        for row in labels:
            assert abs(row["percent"] - 100.0 * row["disagreeing_voxels"] / voxels) < 1e-9, name

    def peak(rows: list[dict], key: str) -> dict:
        best = max(rows, key=lambda r: r[key])
        return {"value": best[key], "step": best["step"]}

    assert case["maxima"] == {
        "grain_volume_error": peak(case["grain_volume_error"], "value"),
        "field_ratio": peak(case["field_error"], "field_ratio"),
        "criterion_ratio": peak([r for r in case["field_error"] if r["criterion_applies"]], "criterion_ratio"),
    }, name
    through = 0
    for row in case["grain_volume_error"]:
        if row["value"] > LIMIT:
            break
        through = row["step"]
    assert case["grain_volume_criterion_met_through_step"] == through, name

    expected = {
        "admissibility": (case["admissibility"]["completed_steps"] == horizon
                          and case["admissibility"]["violations"] == 0),
        "alive_extinct_sets_and_timing": (
            all(r["alive_sets_equal"] for r in case["alive"])
            and all(e["reference_step"] is not None and e["model_step"] is not None for e in events)
            and all(e["within_tolerance"] for e in events)
            and case["returns"]["model"] == 0 and case["returns"]["reference"] == 0),
        "field_error_relative_to_persistence": all(
            r["field_ratio"] <= r["factor"] for r in case["field_error"] if r["criterion_applies"]),
        "grain_volume_error": all(r["value"] <= LIMIT for r in case["grain_volume_error"]),
    }
    if structural is not None:
        expected = {"structural_consistency": structural, **expected}
    assert case["criteria"] == expected, f"{name}: criteria disagree with the measured fields"


def _check_initial_labels(case: dict, repo: Path, side: int, grains: int) -> None:
    path = repo / case["initial_labels"]
    assert path.is_file() and not path.is_symlink(), case["initial_labels"]
    assert sha256(path) == case["initial_labels_sha256"], case["initial_labels"]
    labels, n_grains = _read_labels(path)
    assert labels.shape == (side, side, side) and n_grains == grains, case["initial_labels"]
    assert np.array_equal(np.unique(labels), np.arange(grains)), case["initial_labels"]


def _read_labels(path: Path) -> tuple[np.ndarray, int]:
    with np.load(path, allow_pickle=False) as archive:
        assert sorted(archive.files) == ["labels", "n_grains"], path.name
        labels = archive["labels"]
        assert labels.dtype == np.uint8 and archive["n_grains"].shape == (), path.name
        return labels, int(archive["n_grains"])


def _check_summary(block: dict) -> None:
    cases = block["cases"]
    s = block["summary"]
    residuals = [e["signed_residual_steps"] for c in cases for e in c["extinctions"]
                 if e["signed_residual_steps"] is not None]
    assert s["cases"] == len(cases) == 3
    assert s["criteria_met_per_case"] == {
        c["public_name"]: [LABELS[k] for k in CRITERIA if c["criteria"][k]] for c in cases}
    assert s["cases_meeting_each_criterion"] == {k: sum(c["criteria"][k] for c in cases) for k in CRITERIA}
    assert s["cases_meeting_all_criteria"] == sum(c["all_criteria_met"] for c in cases)
    assert s["all_criteria_met_all_cases"] == all(c["all_criteria_met"] for c in cases)
    assert s["reference_extinctions"] == sum(c["reference_extinction_count"] for c in cases)
    assert s["extinctions_matched_by_identity"] == sum(c["extinctions_matched_by_identity"] for c in cases)
    assert s["model_extinctions_without_reference_extinction"] == sum(
        sum(e["reference_step"] is None for e in c["extinctions"]) for c in cases)
    assert s["returns"] == sum(c["returns"]["model"] + c["returns"]["reference"] for c in cases)
    assert s["signed_residual_range_steps"] == [min(residuals), max(residuals)]
    assert s["grain_volume_error_max"] == max(c["maxima"]["grain_volume_error"]["value"] for c in cases)
    assert s["field_ratio_max"] == max(c["maxima"]["field_ratio"]["value"] for c in cases)
    assert s["criterion_ratio_max"] == max(c["maxima"]["criterion_ratio"]["value"] for c in cases)


def _check_comparator(block: dict, primary: list[dict], grains: int, horizon: int, steps: tuple,
                      voxels: int) -> None:
    comp = block["comparator"]
    assert comp["checkpoint_sha256"] == COMPARATOR and comp["distributed"] is False
    cases = comp["cases"]
    assert [c["public_name"] for c in cases] == [c["public_name"] for c in primary]
    for case, prim in zip(cases, primary):
        assert ({e["grain"]: e["reference_step"] for e in case["extinctions"]}
                == {e["grain"]: e["reference_step"] for e in prim["extinctions"]}), case["public_name"]
        case = dict(case, label_disagreement=None)
        _check_case(case, grains, horizon, steps, voxels, None)
        assert case["casewise_criteria_met"] == all(case["criteria"].values())
    s = comp["summary"]
    residuals = [e["signed_residual_steps"] for c in cases for e in c["extinctions"]]
    assert s["cases"] == 3 and s["cases_meeting_casewise_criteria"] == sum(c["casewise_criteria_met"] for c in cases)
    assert s["reference_extinctions"] == sum(c["reference_extinction_count"] for c in cases)
    assert s["extinctions_matched_by_identity"] == sum(c["extinctions_matched_by_identity"] for c in cases)
    assert s["signed_residual_range_steps"] == [min(residuals), max(residuals)]
    assert s["grain_volume_error_max"] == max(c["maxima"]["grain_volume_error"]["value"] for c in cases)
    assert s["field_ratio_max"] == max(c["maxima"]["field_ratio"]["value"] for c in cases)


def verify_score_record(records: Path) -> None:
    x = json.loads(records.read_text(encoding="utf-8"))
    repo = records.resolve().parents[2]
    assert x["schema"] == "pinn-phase-phaselocal-transfer-v1"
    assert x["public_case_names_only"] is True and x["same_model_for_every_case"] is True
    assert x["parent_checkpoint_sha256"] == PARENT and x["parent_checkpoint_distributed"] is False
    assert x["parameter_count"] == 2160 and x["regimes_pooled"] is False
    assert x["regime_order"] == list(REGIMES)
    assert set(x["criteria"]) == set(CRITERIA)

    train = x["training"]
    assert train["grid"] == [48, 48, 48] and train["grains"] == 8 and train["initial_microstructures"] == 4
    assert train["updates"] == 28200 == train["cycles"] * train["updates_per_cycle"]
    assert train["training_command_distributed"] is False and len(train["initial_labels"]) == 4
    for entry in train["initial_labels"]:
        path = repo / entry["path"]
        assert sha256(path) == entry["sha256"], entry["path"]
        labels, n_grains = _read_labels(path)
        assert labels.shape == (48, 48, 48) and n_grains == 8
        assert np.array_equal(np.unique(labels), np.arange(8))

    for key, (side, grains, horizon, steps) in REGIMES.items():
        block = x[key]
        voxels = side ** 3
        assert block["grid"] == [side] * 3 and block["grains"] == grains and block["voxels"] == voxels
        assert block["horizon_steps"] == horizon and block["evaluation_steps"] == list(steps)
        assert block["mean_initial_grain_volume"] == voxels / grains
        structural = _structural_met(block["structural_check"])
        assert block["structural_check"]["met"] == structural, key
        cases = block["cases"]
        assert [c["public_name"] for c in cases] == [f"microstructure {k}" for k in (1, 2, 3)], key
        for k, case in enumerate(cases, 1):
            assert case["initial_labels"] == f"benchmarks/{PACKAGE}/initial_labels/{key}_microstructure_{k}.npz"
            _check_case(case, grains, horizon, steps, voxels, structural)
            assert case["all_criteria_met"] == all(case["criteria"].values()), case["public_name"]
            _check_initial_labels(case, repo, side, grains)
        _check_summary(block)
        if key in ("r96_n16", "r128_n38"):
            _check_comparator(block, cases, grains, horizon, steps, voxels)
        else:
            assert block["comparator"] is None, key

    # Published headlines, re-derived from the cases.
    for key, extinctions in (("r64_n16", 10), ("r96_n16", 10), ("r128_n38", 12)):
        s = x[key]["summary"]
        assert s["cases_meeting_all_criteria"] == 3 and s["all_criteria_met_all_cases"] is True, key
        assert s["reference_extinctions"] == s["extinctions_matched_by_identity"] == extinctions, key
        assert s["returns"] == 0 and s["grain_volume_error_max"] <= LIMIT, key
    for key in ("r96_n16", "r128_n38"):
        assert x[key]["comparator"]["summary"]["cases_meeting_casewise_criteria"] == 3, key
    n150 = x["r128_n150"]
    cases = n150["cases"]
    assert [c["reference_extinction_count"] for c in cases] == [126, 122, 120]
    assert [c["extinctions_matched_by_identity"] for c in cases] == [126, 122, 120]
    assert n150["summary"]["model_extinctions_without_reference_extinction"] == 0
    assert n150["summary"]["returns"] == 0
    assert n150["summary"]["criteria_met_per_case"] == {
        "microstructure 1": ["(i)", "(ii)", "(iv)", "(v)"],
        "microstructure 2": ["(i)", "(ii)", "(iii)", "(iv)"],
        "microstructure 3": ["(i)", "(ii)", "(iv)"],
    }
    assert n150["summary"]["all_criteria_met_all_cases"] is False
    assert n150["summary"]["grain_volume_criterion_met_through_t"] == {
        c["public_name"]: c["grain_volume_criterion_met_through_step"] * 0.01 for c in cases}
    assert [c["grain_volume_criterion_met_through_step"] for c in cases] == [13500, 8000, 8000]
    assert [(r["step"], r["model_only"], r["reference_only"]) for c in cases for r in c["alive"]
            if not r["alive_sets_equal"]] == [
        (5000, cases[0]["alive"][6]["model_only"], []), (4000, cases[2]["alive"][5]["model_only"], [])]
    stats = n150["derived_statistics"]
    assert stats["label_disagreement_percent_final"] == [c["label_disagreement"][-1]["percent"] for c in cases]
    assert stats["label_disagreement_percent_max_through_t60"] == max(
        r["percent"] for c in cases for r in c["label_disagreement"] if r["step"] <= 6000)
    for row, case in zip(stats["extinction_residuals"], cases):
        magnitudes = [abs(e["signed_residual_steps"]) for e in case["extinctions"]]
        assert row["absolute_median_steps"] == float(np.median(magnitudes))
        assert row["absolute_max_steps"] == max(magnitudes)


# ---------------------------------------------------------------------------
# Release assets
# ---------------------------------------------------------------------------

def verify_release_assets(asset_manifest: Path, asset_root: Path) -> int:
    """Fail closed unless every release asset matches its manifest entry."""
    document = json.loads(asset_manifest.read_text(encoding="utf-8"))
    assets = document.get("assets")
    if not isinstance(assets, list) or len(assets) != ASSET_COUNT:
        raise ValueError(f"asset manifest must declare exactly {ASSET_COUNT} release assets")
    if not asset_root.is_dir():
        raise ValueError(f"asset root is not a directory: {asset_root}")
    seen: set[str] = set()
    expected_directories = {PACKAGE}
    for asset in assets:
        relative = asset.get("expected_relative_location")
        filename = asset.get("filename")
        expected_sha256 = asset.get("sha256")
        if not all(isinstance(v, str) for v in (relative, filename, expected_sha256)):
            raise ValueError("asset manifest has a non-string path, filename or SHA-256")
        if asset.get("release_class") != "RELEASE_ASSET":
            raise ValueError(f"asset is not declared as a release asset: {relative}")
        relative_path = Path(relative)
        if (relative_path.is_absolute() or ".." in relative_path.parts or relative_path.name != filename
                or relative_path.parts[0] != PACKAGE):
            raise ValueError(f"asset manifest has unsafe or mismatched path: {relative}")
        if relative in seen:
            raise ValueError(f"asset manifest repeats a release asset path: {relative}")
        seen.add(relative)
        for depth in range(1, len(relative_path.parts)):
            expected_directories.add(Path(*relative_path.parts[:depth]).as_posix())
        candidate = asset_root / relative_path
        if candidate.is_symlink():
            raise ValueError(f"release asset must not be a symlink: {relative}")
        if not candidate.is_file():
            raise ValueError(f"missing release asset: {relative}")
        if candidate.stat().st_size != asset.get("bytes"):
            raise ValueError(f"size mismatch for {relative}")
        actual = sha256(candidate)
        if actual != expected_sha256:
            raise ValueError(f"SHA-256 mismatch for {relative}: expected {expected_sha256}, got {actual}")
    for candidate in sorted((asset_root / PACKAGE).rglob("*")):
        relative = candidate.relative_to(asset_root).as_posix()
        if candidate.is_symlink():
            raise ValueError(f"unexpected symlink in release assets: {relative}")
        if candidate.is_file() and relative not in seen:
            raise ValueError(f"unexpected regular file in release assets: {relative}")
        if candidate.is_dir() and relative not in expected_directories:
            raise ValueError(f"unexpected directory in release assets: {relative}")
    return len(seen)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _events(volumes: np.ndarray) -> tuple[dict[int, int], int]:
    extinctions, returns = {}, 0
    for grain in range(volumes.shape[1]):
        below = np.nonzero(volumes[1:, grain] <= ALIVE)[0]
        if below.size:
            step = int(below[0]) + 1
            extinctions[grain] = step
            returns += int(np.any(volumes[step:, grain] > ALIVE))
    return extinctions, returns


def _sizes(tables: list[np.ndarray], step: int) -> np.ndarray:
    out = []
    for table in tables:
        v = table[step][table[step] > 1.0]
        r = (3 * v / (4 * np.pi)) ** (1 / 3)
        out.append(r / r.mean())
    return np.concatenate(out)


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.sort(a), np.sort(b)
    values = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, values, side="right") / a.size
                               - np.searchsorted(b, values, side="right") / b.size)))


def _slope(volumes: np.ndarray, side: int) -> float:
    alive = np.sum(volumes > ALIVE, axis=1)
    t = np.arange(volumes.shape[0]) * 0.01
    rbar = (3 * (side ** 3 / alive) / (4 * np.pi)) ** (1 / 3)
    y = rbar ** 2 - rbar[0] ** 2
    window = (t >= 40) & (t <= 135)
    return float(np.polyfit(t[window], y[window], 1)[0])


def recompute_from_assets(records: Path, asset_root: Path) -> None:
    """Recompute the measured fields of the record from the release arrays."""
    x = json.loads(records.read_text(encoding="utf-8"))
    repo = records.resolve().parents[2]
    for key, (side, grains, horizon, steps) in REGIMES.items():
        tables = []
        for k, case in enumerate(x[key]["cases"], 1):
            name = f"{key} {case['public_name']}"
            folder = asset_root / PACKAGE / key / f"microstructure_{k}"
            reference = _load(folder / "reference_volumes.npz")
            ref = reference["volumes"]
            assert ref.shape == (horizon + 1, grains) and reference["steps"].tolist() == list(range(horizon + 1)), name
            ref_ext, ref_returns = _events(ref)
            assert ref_ext == {e["grain"]: e["reference_step"] for e in case["extinctions"]
                               if e["reference_step"] is not None}, f"{name}: reference extinctions"
            assert ref_returns == case["returns"]["reference"], name
            initial, _ = _read_labels(repo / case["initial_labels"])
            assert np.array_equal(np.bincount(initial.ravel(), minlength=grains).astype(np.float64), ref[0]), name
            for row in case["alive"]:
                alive = {int(g) for g in np.nonzero(ref[row["step"]] > ALIVE)[0]}
                assert len(alive) == row["reference_alive_count"], f"{name}: alive {row['step']}"
            if not (folder / "model_volumes.npz").exists():
                continue
            mod = _load(folder / "model_volumes.npz")["volumes"]
            assert mod.shape == ref.shape and np.array_equal(mod[0], ref[0]), name
            mod_ext, mod_returns = _events(mod)
            assert mod_ext == {e["grain"]: e["model_step"] for e in case["extinctions"]
                               if e["model_step"] is not None}, f"{name}: model extinctions"
            assert mod_returns == case["returns"]["model"], name
            for row in case["alive"]:
                ref_set = {int(g) for g in np.nonzero(ref[row["step"]] > ALIVE)[0]}
                mod_set = {int(g) for g in np.nonzero(mod[row["step"]] > ALIVE)[0]}
                assert row["model_only"] == sorted(mod_set - ref_set), f"{name}: alive {row['step']}"
                assert row["reference_only"] == sorted(ref_set - mod_set), f"{name}: alive {row['step']}"
            mean = side ** 3 / grains
            for row in case["grain_volume_error"]:
                value = float(np.max(np.abs(mod[row["step"]] - ref[row["step"]])) / mean)
                assert abs(value - row["value"]) <= 1e-12, f"{name}: grain-volume error {row['step']}"
            tables.append((ref, mod))
            ref_labels = _load(folder / "reference_labels.npz")
            model_labels = _load(folder / "model_labels.npz")
            ref_steps, model_steps = ref_labels["steps"].tolist(), model_labels["steps"].tolist()
            assert np.array_equal(ref_labels["labels"][ref_steps.index(0)], initial), f"{name}: initial labels"
            for row in case["label_disagreement"]:
                count = int(np.count_nonzero(model_labels["labels"][model_steps.index(row["step"])]
                                             != ref_labels["labels"][ref_steps.index(row["step"])]))
                assert count == row["disagreeing_voxels"], f"{name}: label disagreement {row['step']}"
            if key == "r128_n150":
                final = [r for r in x[key]["derived_statistics"]["xy_midplane_disagreement_final"]
                         if r["public_name"] == case["public_name"]][0]
                plane_r = ref_labels["labels"][ref_steps.index(horizon)][:, :, side // 2]
                plane_m = model_labels["labels"][model_steps.index(horizon)][:, :, side // 2]
                assert int(np.count_nonzero(plane_r != plane_m)) == final["disagreeing_voxels"], name
        if key != "r128_n150":
            continue
        stats = x[key]["derived_statistics"]
        for row in stats["size_distribution"]["rows"]:
            a, b = _sizes([t[0] for t in tables], row["step"]), _sizes([t[1] for t in tables], row["step"])
            assert (a.size, b.size) == (row["grains_reference"], row["grains_model"]), row["step"]
            assert abs(_ks(a, b) - row["ks_distance"]) <= 1e-12, row["step"]
            assert abs(float(np.std(np.log(a))) - row["lognormal_width_reference"]) <= 1e-9
            assert abs(float(np.std(np.log(b))) - row["lognormal_width_model"]) <= 1e-9
        for row, (ref, mod) in zip(stats["mean_radius_slope"]["rows"], tables):
            assert close(_slope(ref, side), row["reference_slope"], 1e-9)
            assert close(_slope(mod, side), row["model_slope"], 1e-9)
        for step, counts in stats["grain_counts"].items():
            assert counts["reference"] == [int(np.sum(t[0][int(step)] > ALIVE)) for t in tables]
            assert counts["model"] == [int(np.sum(t[1][int(step)] > ALIVE)) for t in tables]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path,
                        help="directory containing the unpacked phaselocal_transfer/ release assets")
    args = parser.parse_args()
    try:
        verify_score_record(args.records)
        count = None
        if args.asset_root is not None:
            count = verify_release_assets(args.records.parent / "manifest.json", args.asset_root)
            recompute_from_assets(args.records, args.asset_root)
    except (AssertionError, OSError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc) or "phase-local transfer record verification failed")
    suffix = f" ({count} release assets recomputed)" if count is not None else ""
    print(f"{PACKAGE} PASS{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
