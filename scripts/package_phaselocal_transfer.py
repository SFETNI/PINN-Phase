#!/usr/bin/env python3
"""Build the public phase-local transfer records from the accepted evaluation records.

This packs measurements that already exist; it never evaluates a model or a
reference. Every source is identified by SHA-256, not by file name: the script
hashes every JSON file and NumPy archive beneath ``--records`` and requires each
pinned digest below to be present. A record that does not match its digest is
treated as absent, and the build stops.

From those sources the script writes, deterministically:

* ``expected_score.json``, the compact public record, one block per regime;
* ``README.md``, rendered from that record by :func:`benchmark_readme`;
* ``manifest.json``, every release asset with its SHA-256 and size;
* the release assets beneath ``--asset-out`` (per-step grain volumes and argmax
  label arrays as NumPy archives), one zip per regime and a ``SHA256SUMS`` file.

The initial label arrays in ``initial_labels/`` are inputs: each is checked
against the recorded initial state (the reference labels at step 0 where they
exist, and the grain volumes at step 0 in every case) before it is bound into
the record by digest.

Usage::

    python scripts/package_phaselocal_transfer.py --records /path/to/records \\
        --out benchmarks/phaselocal_transfer --asset-out /path/to/asset-root
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

# Accepted evaluation records, bound by digest. Roles are descriptive; identity is
# the digest, never the file name. Case numbers are the public microstructure
# numbers of each regime.
SOURCE_SHA256 = {
    "training_record": "4d224b19aa7823ec3ee74c907e0d6b796155bc7ce72710e3c7f29732b3584717",
    "r64_n16_scores": "07093a6bf6fe6f993effd666b44b327aadffd307881db37e22d9d338a051c331",
    "r64_n16_reference_volumes_1": "fecf5e8dfa16509e191028aa8296b4f9c3cddd1ced473ca63ebb927880138e4a",
    "r64_n16_reference_volumes_2": "cfa6489bcf3b666cd570d83ada2f6f51326bfdbba4f7e939d48bc0fdc3faacb5",
    "r64_n16_reference_volumes_3": "c404f93b99ccc09a9811ef530ea713ac27da4f45637814b074fc3b1ae23b0a30",
    "r96_n16_structural": "53d5c014c8da1c533306a7f7fd7bd2aafa7c536e14ca4b321ae827a5b28296db",
    "r96_n16_scores_1": "13f24e444062a729f18749675ebe5c16e22c13d8dcdb0ea9aadd4c14a7037c91",
    "r96_n16_scores_2": "d80dd59039abdfb81ee4cb60866deeca453e694546d096bb6a9da22d32069949",
    "r96_n16_scores_3": "5d8eef49a647c75daa3eed6ae4332a73cfc5f81b48a7c8a8cea4cea7d39fdf74",
    "r96_n16_diagnostics_1": "94cf6f60bee2e2086dc31d691efda4ee0587a9940d275699f309f7a5c8349331",
    "r96_n16_diagnostics_2": "172ee7e8e8653c387d4fcbae8ed0479f39626f6860f3fa95b387eaff5f84c679",
    "r96_n16_diagnostics_3": "084c5a439b5e56215edad2d9e5d8e85627e1c19bd0cad8c7b8164e407fd0bc50",
    "r96_n16_model_volumes_1": "c212890b5c0fb05ffc9e4b15f677bb29221e260657d344f2c361a1268a34b633",
    "r96_n16_model_volumes_2": "942e9c6c2cb6dc7a0e4ba10e3c9094106648cba668143c5afd53e2e40847dff0",
    "r96_n16_model_volumes_3": "f37ed7eb0eeee5fa7d5f306f5b9a620c23e2baba5edcf61992b5d74f647d116e",
    "r96_n16_reference_volumes_1": "7ce2642ff1f6cc85938e2b4c3dccc425b85e78425e8571b4ba2e619c47f05601",
    "r96_n16_reference_volumes_2": "d4221326953d4a2707bf12c3165850e352a54b58be28b07a1dc8af0e5d9ba83f",
    "r96_n16_reference_volumes_3": "1a768b312073235f44ea21cec356b1fcd715e4501a9029451c30435acbd78555",
    "r96_n16_labels_1": "2c25c824fa03a03dc7a4582721f8ea2a24f5795bf25797f924c212ee16bd46d0",
    "r96_n16_labels_2": "347b21dedee0615906ba793abfa56c4ae0d4f947d87e74c260a9705dd5dde14f",
    "r96_n16_labels_3": "9b0fe478a7523b0ac2a9d9c7dfb38b13f4315377bde7fb7ea21f48d2f3a04a66",
    "r96_n16_comparator_1": "a304e9fd192c179b7f52ea0dbcd15bc091ef0fc0213fbd72d1e30690f7b1389b",
    "r96_n16_comparator_2": "68107f2224dab3c679b1f757aacde6c9394a3ef76b43fb7601b43836b81e3e48",
    "r96_n16_comparator_3": "319802d1b92da51a0ad559d05c8a9622a0c8f89fa1c31d0874d71bf25cf72f9e",
    "r128_n38_structural": "d10c92f0d7da8d57c25d9ecfda8bffc2bac91891d86fa7e0986d26623294005f",
    "r128_n38_scores_1": "7c081bf9bbbf5d4182d88bbe8225d4e5149f7c2a49dc746b394ea87c95fa3fc9",
    "r128_n38_scores_2": "ef1991b8b7905a3f5297464bc703c013dc614fcdc4bf7ad156bb3f1defc4f948",
    "r128_n38_scores_3": "07e2744df9bef717ad50a867cf5e7f98d714e684561b77f27d7350f890ad55db",
    "r128_n38_diagnostics_1": "0f1a51bc61745a2f882e69a7169dcdb92722a90d240a5e74b90d34111906e305",
    "r128_n38_diagnostics_2": "5b3cff8cc543199e5e1620538872aa8e327ce691d4be40987db6c1a3c6112a9a",
    "r128_n38_diagnostics_3": "23e7aac4ed5d744f03ad579a22215b64211cff1f6dfaef3bdb0722a8f6a985e8",
    "r128_n38_model_volumes_1": "29526bba7df357fafad631ed79e20082525848bdd2723319c9a29ab633c864f7",
    "r128_n38_model_volumes_2": "427fdd7248339c8ae1ba8e5da6542ec9ad408fba4071c9699188070b78f42f2b",
    "r128_n38_model_volumes_3": "9678e540b87afe9c3f08d6a70302c134f75eb693256de59f15c63f173b4fdd95",
    "r128_n38_reference_volumes_1": "49f31d04b6befe514f92f99df65c22e67253b5eab50c5e56e59f55afe30c46b8",
    "r128_n38_reference_volumes_2": "448397197c12495d07a9883378e418f7a81458bcc90ed542a191198d09fc0bab",
    "r128_n38_reference_volumes_3": "0791046ea9ad8f344777a5f475e041ac6eec02ba538ea8f9e9ccf2be4ea71820",
    "r128_n38_labels_1": "c3105b1aa0264ff440fb77c7fbe2f20ff564f191079a501d10c6b31028307179",
    "r128_n38_labels_2": "a83efc86314a36297794debdf5e0bad27eff72fb4422a8f7516a0f467ad2520e",
    "r128_n38_labels_3": "210b8d38a795b3ab28c78d85ec58c20db7a6ef5b8b3d3b6b7f8a404608ce347c",
    "r128_n38_comparator_1": "b74ad16f1075b25317e0f590fe050eeee87debbdec11efbfbde9239894e565c9",
    "r128_n38_comparator_2": "36d793f6db1c3065764de2fd4d1a0d772f1a0259aca5eb568027fb9095d0b2cf",
    "r128_n38_comparator_3": "d2f5d60e45f015110e888bab0022f7c4286359e54650c58b5b8293ddc0666014",
    "r128_n150_scores": "f2ae46e1509ae48bb0b8a80bd13ec726e443fa8b4db8d6bbf0e3f2f92d3a24d6",
    "r128_n150_structural": "7a80e0b0b5cb424fb08ac41aaf6fdbcc9aa33dc742431c6276f191c2ad8a0fd3",
    "r128_n150_model_evidence_1": "9da4d1f28145a83b88624c49f65d144601eae96c545c13b851aab229c762ac36",
    "r128_n150_model_evidence_2": "c07c490b2d84dc1267aad650b627308c4afafecad88b224964103e8d0b27e4e1",
    "r128_n150_model_evidence_3": "4401a237412cd9efeb022a6cf48c389924d73a8fe3387f6963e297903eac2c54",
    "r128_n150_reference_evidence_1": "5cf6de7b3dce683bb5d22860747d974268b28680137a72ea38be8467d6f89f2b",
    "r128_n150_reference_evidence_2": "827b0891719187cf0355432973036609d31c1353f8ec1697f9cb2bb4347d7432",
    "r128_n150_reference_evidence_3": "f5c3ae5e25c61587199920052d50a9530b2fc49f2f1fe0840292080011ed871f",
    "r128_n150_reference_labels_1": "93d77af9e84887693efd9ae9b583c85b51e2a9838c3999a8818110ce7ad8424a",
    "r128_n150_reference_labels_2": "c30a95fc1633ccd1ea7b9f523d84799a7c66ffee0ab8de6a93069b8f4ff0868d",
    "r128_n150_reference_labels_3": "72023c6487d2847824f0ad5caa101f8141a360d9d72c14464de748535cc4aceb",
    "r128_n150_model_labels_1": "af12c292b698959d2381b206df3adc5ad0a88357bf91c5ff58cf6482abb0e4e0",
    "r128_n150_model_labels_2": "963f009b337ee618a016a032f39b6847f241a993dee9996ed42e7b23c59562b0",
    "r128_n150_model_labels_3": "d1925a1611a25917a20fc0814de1bc580f03988b2d760583e0903edd219a64c1",
}

PARENT_CHECKPOINT_SHA256 = "764f22a548e4372e7627e71f4680bf6c806608ff90b75e8d389a1fcc98afc6e4"
COMPARATOR_CHECKPOINT_SHA256 = "8491d9c94d91913093bdd5c2915bdf77307af6da648a547b486c15a380d6107b"
PUBLIC_WEIGHTS = "checkpoints/phase_local_48_n8.weights.npz"
WEIGHT_LINEAGE = "checkpoints/phase_local_48_n8.lineage.json"
PARAMETER_COUNT = 2160
PACKAGE = "phaselocal_transfer"
RELEASE_TAG = "v3.0.0"

TIME_STEP = 0.01
ALIVE_THRESHOLD = 1e-6
UNIT_SUM_TOLERANCE = 1e-12
PERMUTATION_TOLERANCE = 1e-15
GRAIN_VOLUME_LIMIT = 0.05
FIELD_FACTOR = 0.5
FIELD_FACTOR_FINAL = 0.25
FIELD_CRITERION_FROM_STEP = 500        # t >= 5
TIMING_MIN_STEPS = 100.0
TIMING_FRACTION = 0.1
CASES = 3
TRAINING = {"side": 48, "grains": 8, "microstructures": 4}

#: The four evaluation regimes. They are reported separately and never pooled.
REGIMES = {
    "r64_n16": {
        "title": "64^3, 16 grains", "side": 64, "grains": 16, "horizon": 3000,
        "eval_steps": (100, 200, 500, 1000, 1500, 2000, 2500, 3000),
    },
    "r96_n16": {
        "title": "96^3, 16 grains", "side": 96, "grains": 16, "horizon": 6750,
        "eval_steps": (100, 200, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 6750),
    },
    "r128_n38": {
        "title": "128^3, 38 grains", "side": 128, "grains": 38, "horizon": 6750,
        "eval_steps": (100, 200, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 6750),
    },
    "r128_n150": {
        "title": "128^3, 150 grains", "side": 128, "grains": 150, "horizon": 13500,
        "eval_steps": (200, 400, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 13500),
    },
}

#: The five acceptance criteria, named as in the paper.
CRITERIA = (
    ("structural_consistency",
     "(i) structural consistency: permuting the phase fields commutes with one step to within "
     "1e-15, appended zero phases stay exactly zero and leave the existing phases bitwise "
     "unchanged, and every output is finite, nonnegative and sums to one within 1e-12; assessed "
     "once per regime on fields that are not scored cases"),
    ("admissibility",
     "(ii) admissibility: every prescribed step is completed and every state is finite, "
     "nonnegative and has maximum unit-sum error at most 1e-12"),
    ("alive_extinct_sets_and_timing",
     "(iii) alive/extinct sets and extinction timing: the alive set equals the reference alive "
     "set at every evaluation time, the extinct set equals the reference set, every extinction "
     "step lies within max(100, 0.1 s_ref) steps of the reference step, and no extinct phase "
     "returns"),
    ("field_error_relative_to_persistence",
     "(iv) field error relative to persistence: E_TV(model) / E_TV(persistence) is at most "
     "a(s) at every evaluation time with t >= 5, with a = 0.5 and a = 0.25 at the final step"),
    ("grain_volume_error",
     "(v) grain-volume error: the maximum normalized grain-volume error is at most 0.05 at every "
     "evaluation time"),
)
CRITERION_LABELS = dict(zip((name for name, _ in CRITERIA), ("(i)", "(ii)", "(iii)", "(iv)", "(v)")))

DEFINITIONS = {
    "time": "t = 0.01 s for step s",
    "grain_volume": "V_k(s), the double-precision sum of phase field k over all voxels at step s",
    "alive": "V_k(s) > 1e-6",
    "extinction_step": "the first step s >= 1 with V_k(s) <= 1e-6; any later rise above 1e-6 is a return",
    "alive_sets": (
        "Every grain is alive at t = 0 and no grain returns, so the alive set at step s is every grain "
        "whose extinction step is absent or later than s. Each evaluation row stores the two counts and "
        "the grains alive in only one of the two runs."
    ),
    "field_error": "E_TV(s) = (1 / 2n^3) sum over voxels and phases of |phi_model - phi_ref|",
    "persistence": "the initial field held fixed, scored against the reference at step s",
    "field_ratio": "E_TV(model) / E_TV(persistence)",
    "criterion_ratio": "field_ratio / a(s); criterion (iv) requires at most 1",
    "grain_volume_error": "max_k |V_k(model) - V_k(reference)| / (n^3 / N)",
    "label_disagreement": "percentage of voxels whose argmax grain label differs between model and reference",
    "timing_tolerance": "max(100, 0.1 s_ref) steps",
}

FIXED_MEMBER_DATE = (1980, 1, 1, 0, 0, 0)
FIXED_MEMBER_MODE = 0o644


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_npz(path: Path, arrays: dict[str, np.ndarray], *, compress: bool) -> None:
    """Write a NumPy archive whose bytes depend only on the arrays.

    Members are sorted, carry a fixed date and mode, no extra fields and no comment.
    ``compress=False`` stores the members, so the bytes do not depend on a
    compression library; ``compress=True`` deflates them at level 6.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name], order="C"),
                                      version=(1, 0), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=FIXED_MEMBER_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
            info.external_attr = FIXED_MEMBER_MODE << 16
            info.create_system = 3
            archive.writestr(info, payload.getvalue(), compresslevel=6 if compress else None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())


def write_bundle(path: Path, root: Path, files: list[Path]) -> int:
    """Stored zip of release assets with fixed metadata, members in sorted order."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for file in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
            info = zipfile.ZipInfo(file.relative_to(root).as_posix(), date_time=FIXED_MEMBER_DATE)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = FIXED_MEMBER_MODE << 16
            info.create_system = 3
            archive.writestr(info, file.read_bytes())
    path.write_bytes(buffer.getvalue())
    return len(files)


def locate_sources(root: Path) -> dict[str, object]:
    """Every pinned record, found by digest: JSON is parsed, archives are returned as paths."""
    by_digest: dict[str, Path] = {}
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file() and candidate.suffix in (".json", ".npz"):
            by_digest.setdefault(sha256(candidate), candidate)
    missing = [role for role, digest in SOURCE_SHA256.items() if digest not in by_digest]
    if missing:
        raise SystemExit(f"accepted records not found by digest: {', '.join(missing)}")
    sources: dict[str, object] = {}
    for role, digest in SOURCE_SHA256.items():
        path = by_digest[digest]
        sources[role] = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else path
    return sources


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"source check failed: {message}")


# ---------------------------------------------------------------------------
# Per-step quantities shared with the verifier
# ---------------------------------------------------------------------------

def tolerance(reference_step: int) -> float:
    return max(TIMING_MIN_STEPS, TIMING_FRACTION * reference_step)


def field_factor(step: int, horizon: int) -> float:
    return FIELD_FACTOR_FINAL if step == horizon else FIELD_FACTOR


def volume_events(volumes: np.ndarray) -> tuple[dict[int, int], list[int]]:
    """Extinction step per grain and the grains that return, from a (steps, grains) table."""
    extinctions: dict[int, int] = {}
    returns: list[int] = []
    for grain in range(volumes.shape[1]):
        below = np.nonzero(volumes[1:, grain] <= ALIVE_THRESHOLD)[0]
        if below.size:
            step = int(below[0]) + 1
            extinctions[grain] = step
            if bool(np.any(volumes[step:, grain] > ALIVE_THRESHOLD)):
                returns.append(grain)
    return extinctions, returns


def volume_error(model: np.ndarray, reference: np.ndarray, step: int, mean_volume: float) -> float:
    return float(np.max(np.abs(model[step] - reference[step])) / mean_volume)


def extinction_rows(reference: dict[int, int], model: dict[int, int]) -> list[dict]:
    """One row per grain that goes extinct in either run, ordered by reference step."""
    grains = sorted(set(reference) | set(model),
                    key=lambda k: (reference.get(k, model.get(k)), k))
    rows = []
    for grain in grains:
        ref, mod = reference.get(grain), model.get(grain)
        row = {"grain": grain, "reference_step": ref, "model_step": mod}
        if ref is not None and mod is not None:
            row.update(signed_residual_steps=mod - ref, tolerance_steps=tolerance(ref),
                       within_tolerance=abs(mod - ref) <= tolerance(ref))
        else:
            row.update(signed_residual_steps=None,
                       tolerance_steps=None if ref is None else tolerance(ref),
                       within_tolerance=False)
        rows.append(row)
    return rows


def alive_rows(grains: int, steps: tuple[int, ...], reference: dict[int, int],
               model: dict[int, int]) -> list[dict]:
    rows = []
    for step in steps:
        ref = {k for k in range(grains) if reference.get(k, step + 1) > step}
        mod = {k for k in range(grains) if model.get(k, step + 1) > step}
        rows.append({
            "step": step,
            "reference_alive_count": len(ref),
            "model_alive_count": len(mod),
            "alive_sets_equal": ref == mod,
            "reference_only": sorted(ref - mod),
            "model_only": sorted(mod - ref),
        })
    return rows


def alive_set(volumes: np.ndarray, step: int) -> set[int]:
    return {int(k) for k in np.nonzero(volumes[step] > ALIVE_THRESHOLD)[0]}


def field_rows(values: list[tuple[int, float, float]], horizon: int) -> list[dict]:
    rows = []
    for step, model, persistence in values:
        ratio = model / persistence
        factor = field_factor(step, horizon)
        rows.append({
            "step": step,
            "e_tv_model": model,
            "e_tv_persistence": persistence,
            "field_ratio": ratio,
            "factor": factor,
            "criterion_applies": step >= FIELD_CRITERION_FROM_STEP,
            "criterion_ratio": ratio / factor,
        })
    return rows


def maxima(case: dict) -> dict:
    def peak(rows: list[dict], key: str) -> dict:
        best = max(rows, key=lambda r: r[key])
        return {"value": best[key], "step": best["step"]}

    return {
        "grain_volume_error": peak(case["grain_volume_error"], "value"),
        "field_ratio": peak(case["field_error"], "field_ratio"),
        "criterion_ratio": peak([r for r in case["field_error"] if r["criterion_applies"]],
                                "criterion_ratio"),
    }


def casewise_criteria(case: dict, horizon: int) -> dict:
    events = case["extinctions"]
    return {
        "admissibility": (case["admissibility"]["completed_steps"] == horizon
                          and case["admissibility"]["violations"] == 0),
        "alive_extinct_sets_and_timing": (
            all(r["alive_sets_equal"] for r in case["alive"])
            and all(e["reference_step"] is not None and e["model_step"] is not None for e in events)
            and all(e["within_tolerance"] for e in events)
            and case["returns"]["model"] == 0 and case["returns"]["reference"] == 0
        ),
        "field_error_relative_to_persistence": all(
            r["field_ratio"] <= r["factor"] for r in case["field_error"] if r["criterion_applies"]),
        "grain_volume_error": all(r["value"] <= GRAIN_VOLUME_LIMIT for r in case["grain_volume_error"]),
    }


def volume_criterion_met_through(rows: list[dict]) -> int:
    """Last evaluation step up to which every grain-volume error is within the limit."""
    last = 0
    for row in rows:
        if row["value"] > GRAIN_VOLUME_LIMIT:
            break
        last = row["step"]
    return last


def label_rows(steps: tuple[int, ...], counts: list[int], voxels: int) -> list[dict]:
    return [{"step": s, "disagreeing_voxels": c, "percent": 100.0 * c / voxels}
            for s, c in zip(steps, counts)]


# ---------------------------------------------------------------------------
# Structural check (criterion (i)), once per regime
# ---------------------------------------------------------------------------

def structural_from_two_fields(record: dict) -> dict:
    """The smaller-regime check, recorded on an initial and a t = 15 field."""
    fields = []
    for key, description in (("t0", "initial field"), ("t15", "field at t = 15")):
        f = record[key]
        fields.append({
            "field": description,
            "permutation_max_abs_difference": f["permutation_max_abs_diff"],
            "existing_phases_bitwise_equal": f["existing_phases_bitwise_equal"],
            "appended_phases_exactly_zero": f["appended_phases_exactly_zero"],
            "max_unit_sum_error": f["simplex_sum_error"],
            "min_value": f["min_value"],
            "finite": f["finite"],
        })
    require(record["pass"] is True, "64^3 structural record")
    return {
        "fields_source": "a 96^3, 16-grain microstructure that is not a scored case",
        "appended_phases": 22,
        "reference_padding_checked": False,
        "fields": fields,
    }


def structural_from_g0(record: dict, appended: int, regime: dict) -> dict:
    require(record["n"] == regime["side"] and record["N"] == regime["grains"], "structural size")
    require(record["weights_sha256"] == PARENT_CHECKPOINT_SHA256, "structural checkpoint")
    require(record["precondition_all_alive_at_t0"]["pass"] is True, "structural precondition")
    fields = []
    for f, description in zip(record["fields"], ("initial field", "field after 300 learned steps")):
        outputs = [v for k, v in f["check4_simplex_min_finite"].items() if k != "pass"]
        fields.append({
            "field": description,
            "permutation_max_abs_difference": f["check1_permutation"]["max_abs_diff"],
            "existing_phases_bitwise_equal": (f["check2_model_padding"]["existing_bitwise_equal"]
                                              and f["check3_d1_padding"]["existing_bitwise_equal"]),
            "appended_phases_exactly_zero": (f["check2_model_padding"]["appended_exactly_zero"]
                                             and f["check3_d1_padding"]["appended_exactly_zero"]),
            "max_unit_sum_error": max(o["simplex_sum_error"] for o in outputs),
            "min_value": min(o["min"] for o in outputs),
            "finite": all(o["finite"] for o in outputs),
        })
    require(record["pass"] is True, "structural record")
    return {
        "fields_source": (f"an initial field of the regime's own size and grain number, every grain "
                          f"alive, that is not a scored case"),
        "appended_phases": appended,
        "reference_padding_checked": True,
        "fields": fields,
    }


def structural_from_checks(record: dict) -> dict:
    checks = [{
        "field": "initial field" if c["step"] == 0 else f"field after {c['step']} steps",
        "map": "learned model" if c["method"] == "model" else "reference map",
        "permutation_max_abs_difference": c["permutation_max"],
        "all_grains_alive": c["all_alive"],
        "check_passed": c["pass_check"],
    } for c in record["checks"]]
    require(record["pass"] is True, "150-grain structural record")
    return {
        "fields_source": ("a 128^3, 150-grain initial field that is not a scored case, every grain "
                          "alive, and the field 300 steps later"),
        "appended_phases": 10,
        "reference_padding_checked": True,
        "check_passed_covers": ("permutation within 1e-15, appended phases exactly zero, existing "
                                "phases bitwise unchanged, finite and nonnegative outputs with unit "
                                "sum within 1e-12"),
        "checks": checks,
    }


def structural_met(block: dict) -> bool:
    if "checks" in block:
        return all(c["check_passed"] and c["all_grains_alive"]
                   and c["permutation_max_abs_difference"] <= PERMUTATION_TOLERANCE
                   for c in block["checks"])
    return all(f["permutation_max_abs_difference"] <= PERMUTATION_TOLERANCE
               and f["existing_phases_bitwise_equal"] and f["appended_phases_exactly_zero"]
               and f["max_unit_sum_error"] <= UNIT_SUM_TOLERANCE and f["min_value"] >= 0.0
               and f["finite"] for f in block["fields"])


# ---------------------------------------------------------------------------
# Initial labels
# ---------------------------------------------------------------------------

def initial_label_name(regime: str, ordinal: int) -> str:
    return f"{regime}_microstructure_{ordinal}.npz"


def read_initial_labels(path: Path, side: int, grains: int) -> np.ndarray:
    require(path.is_file(), f"missing initial labels {path.name}")
    with np.load(path, allow_pickle=False) as archive:
        require(sorted(archive.files) == ["labels", "n_grains"], f"{path.name} members")
        labels = archive["labels"]
        require(archive["n_grains"].shape == (), f"{path.name}: n_grains is a scalar")
        n_grains = int(archive["n_grains"])
    require(labels.dtype == np.uint8 and labels.shape == (side, side, side), f"{path.name} shape")
    require(n_grains == grains, f"{path.name} grain count")
    require(np.array_equal(np.unique(labels), np.arange(grains)), f"{path.name}: every grain present")
    return labels


def counts_match_volumes(labels: np.ndarray, grains: int, row: np.ndarray) -> bool:
    """The one-hot field of the labels has exactly the recorded step-0 grain volumes."""
    counts = np.bincount(labels.ravel(), minlength=grains).astype(np.float64)
    return bool(np.array_equal(counts, np.asarray(row, dtype=np.float64)))


# ---------------------------------------------------------------------------
# Regime builders
# ---------------------------------------------------------------------------

def gates_case(regime_key: str, ordinal: int, gates: dict, reference_volumes: np.ndarray,
               model_volumes: np.ndarray | None) -> tuple[dict, dict]:
    """Measured fields of one case from its score record; volume tables cross-check them."""
    regime = REGIMES[regime_key]
    side, grains, horizon, steps = regime["side"], regime["grains"], regime["horizon"], regime["eval_steps"]
    mean_volume = side ** 3 / grains
    g2 = gates["G2"]
    ref_ext = {int(k): int(v) for k, v in g2["b_extinction"]["ref_extinct"].items()}
    mod_ext = {int(k): int(v) for k, v in g2["b_extinction"]["model_extinct"].items()}
    name = f"{regime_key} microstructure {ordinal}"

    require(gates["G1"]["reached_step"] == horizon, f"{name}: completed steps")
    require(reference_volumes.shape == (horizon + 1, grains), f"{name}: reference table shape")
    events_ref, returns_ref = volume_events(reference_volumes)
    require(events_ref == ref_ext, f"{name}: reference extinctions differ from the volume table")
    require(np.all(reference_volumes[0] > ALIVE_THRESHOLD), f"{name}: every grain alive at t = 0")
    returns_model = list(g2["c_no_return"]["returns"])
    if model_volumes is not None:
        require(model_volumes.shape == (horizon + 1, grains), f"{name}: model table shape")
        events_mod, returns_mod = volume_events(model_volumes)
        require(events_mod == mod_ext, f"{name}: model extinctions differ from the volume table")
        require(returns_mod == returns_model == [], f"{name}: model returns")
        require(np.array_equal(model_volumes[0], reference_volumes[0]), f"{name}: shared initial state")

    case = {
        "public_name": f"microstructure {ordinal}",
        "admissibility": {"completed_steps": int(gates["G1"]["reached_step"]),
                          "violations": int(gates["G1"]["violation_count"])},
        "extinctions": extinction_rows(ref_ext, mod_ext),
        "returns": {"reference": len(returns_ref), "model": len(returns_model)},
        "alive": alive_rows(grains, steps, ref_ext, mod_ext),
        "field_error": field_rows([(s, gates["G3"]["rows"][str(s)]["E_TV_model"],
                                    gates["G3"]["rows"][str(s)]["E_TV_persistence"]) for s in steps],
                                  horizon),
        "grain_volume_error": [{"step": s, "value": gates["G4"]["rows"][str(s)]["value"]} for s in steps],
    }
    # Cross-check every derived row against the recorded evaluation.
    for row in case["alive"]:
        recorded = g2["a_alive_set_equal"]["per_step"][str(row["step"])]
        require(recorded["ref_alive"] == row["reference_alive_count"]
                and recorded["model_alive"] == row["model_alive_count"]
                and recorded["equal"] == row["alive_sets_equal"], f"{name}: alive row {row['step']}")
        require(alive_set(reference_volumes, row["step"]) ==
                {k for k in range(grains) if ref_ext.get(k, row["step"] + 1) > row["step"]},
                f"{name}: reference alive set {row['step']}")
        if model_volumes is not None:
            require(alive_set(model_volumes, row["step"]) ==
                    {k for k in range(grains) if mod_ext.get(k, row["step"] + 1) > row["step"]},
                    f"{name}: model alive set {row['step']}")
    for row in case["field_error"]:
        recorded = gates["G3"]["rows"][str(row["step"])]
        require(recorded["factor"] == row["factor"], f"{name}: field factor {row['step']}")
        if row["criterion_applies"]:
            require(recorded["pass"] == (row["field_ratio"] <= row["factor"]), f"{name}: field row")
    timing = g2["b_extinction"]["timing"]
    for event in case["extinctions"]:
        recorded = timing[str(event["grain"])]
        require(recorded["delta"] == abs(event["signed_residual_steps"])
                and recorded["tol"] == event["tolerance_steps"]
                and recorded["within"] == event["within_tolerance"], f"{name}: event {event['grain']}")
    recheck = {"grain_volume_error_max_abs_difference": None}
    if model_volumes is not None:
        recheck["grain_volume_error_max_abs_difference"] = max(
            abs(volume_error(model_volumes, reference_volumes, r["step"], mean_volume) - r["value"])
            for r in case["grain_volume_error"])
        require(recheck["grain_volume_error_max_abs_difference"] <= 1e-12, f"{name}: volume error")
    finish_case(case, horizon)
    recorded_verdicts = (gates["G1"]["pass"], g2["pass"], gates["G3"]["pass"], gates["G4"]["pass"])
    require(tuple(case["criteria"][k] for k, _ in CRITERIA[1:]) == recorded_verdicts,
            f"{name}: derived criteria differ from the recorded verdicts")
    return case, recheck


def finish_case(case: dict, horizon: int, structural: bool | None = None) -> None:
    case["maxima"] = maxima(case)
    criteria = casewise_criteria(case, horizon)
    if structural is not None:
        criteria = {"structural_consistency": structural, **criteria}
    case["criteria"] = criteria
    case["grain_volume_criterion_met_through_step"] = volume_criterion_met_through(case["grain_volume_error"])
    case["reference_extinction_count"] = sum(e["reference_step"] is not None for e in case["extinctions"])
    case["model_extinction_count"] = sum(e["model_step"] is not None for e in case["extinctions"])
    case["extinctions_matched_by_identity"] = sum(
        e["reference_step"] is not None and e["model_step"] is not None for e in case["extinctions"])


def comparator_case(regime_key: str, ordinal: int, gates: dict, primary: dict, seed_check: object) -> dict:
    regime = REGIMES[regime_key]
    horizon, steps, grains = regime["horizon"], regime["eval_steps"], regime["grains"]
    name = f"{regime_key} comparator {ordinal}"
    require(gates["weights_sha256"] == COMPARATOR_CHECKPOINT_SHA256, f"{name}: checkpoint")
    require(gates["seed"] == seed_check and gates["t0_bitwise_equal_ic"] is True, f"{name}: case identity")
    g2 = gates["G2"]
    ref_ext = {int(k): int(v) for k, v in g2["b_extinction"]["ref_extinct"].items()}
    mod_ext = {int(k): int(v) for k, v in g2["b_extinction"]["model_extinct"].items()}
    require(ref_ext == {e["grain"]: e["reference_step"] for e in primary["extinctions"]},
            f"{name}: reference extinctions differ from the primary record")
    case = {
        "public_name": f"microstructure {ordinal}",
        "admissibility": {"completed_steps": int(gates["G1"]["reached_step"]),
                          "violations": int(gates["G1"]["violation_count"])},
        "extinctions": extinction_rows(ref_ext, mod_ext),
        "returns": {"reference": primary["returns"]["reference"], "model": len(g2["c_no_return"]["returns"])},
        "alive": alive_rows(grains, steps, ref_ext, mod_ext),
        "field_error": field_rows([(s, gates["G3"]["rows"][str(s)]["E_TV_model"],
                                    gates["G3"]["rows"][str(s)]["E_TV_persistence"]) for s in steps],
                                  horizon),
        "grain_volume_error": [{"step": s, "value": gates["G4"]["rows"][str(s)]["value"]} for s in steps],
    }
    for row in case["alive"]:
        recorded = g2["a_alive_set_equal"]["per_step"][str(row["step"])]
        require(recorded["ref_alive"] == row["reference_alive_count"]
                and recorded["model_alive"] == row["model_alive_count"]
                and recorded["equal"] == row["alive_sets_equal"], f"{name}: alive row")
    finish_case(case, horizon)
    recorded_verdicts = (gates["G1"]["pass"], g2["pass"], gates["G3"]["pass"], gates["G4"]["pass"])
    require(tuple(case["criteria"].values()) == recorded_verdicts, f"{name}: verdicts")
    case["casewise_criteria_met"] = all(case["criteria"].values())
    return case


def regime_summary(cases: list[dict]) -> dict:
    residuals = [e["signed_residual_steps"] for c in cases for e in c["extinctions"]
                 if e["signed_residual_steps"] is not None]
    keys = list(cases[0]["criteria"])
    summary = {
        "cases": len(cases),
        "criteria_met_per_case": {
            c["public_name"]: [CRITERION_LABELS[k] for k in keys if c["criteria"][k]] for c in cases},
        "cases_meeting_each_criterion": {k: sum(c["criteria"][k] for c in cases) for k in keys},
        "cases_meeting_all_criteria": sum(c["all_criteria_met"] for c in cases),
        "all_criteria_met_all_cases": all(c["all_criteria_met"] for c in cases),
        "reference_extinctions": sum(c["reference_extinction_count"] for c in cases),
        "extinctions_matched_by_identity": sum(c["extinctions_matched_by_identity"] for c in cases),
        "model_extinctions_without_reference_extinction": sum(
            sum(e["reference_step"] is None for e in c["extinctions"]) for c in cases),
        "returns": sum(c["returns"]["model"] + c["returns"]["reference"] for c in cases),
        "signed_residual_range_steps": [min(residuals), max(residuals)],
        "grain_volume_error_max": max(c["maxima"]["grain_volume_error"]["value"] for c in cases),
        "field_ratio_max": max(c["maxima"]["field_ratio"]["value"] for c in cases),
        "criterion_ratio_max": max(c["maxima"]["criterion_ratio"]["value"] for c in cases),
    }
    return summary


def comparator_summary(cases: list[dict]) -> dict:
    residuals = [e["signed_residual_steps"] for c in cases for e in c["extinctions"]]
    return {
        "cases": len(cases),
        "cases_meeting_casewise_criteria": sum(c["casewise_criteria_met"] for c in cases),
        "reference_extinctions": sum(c["reference_extinction_count"] for c in cases),
        "extinctions_matched_by_identity": sum(c["extinctions_matched_by_identity"] for c in cases),
        "signed_residual_range_steps": [min(residuals), max(residuals)],
        "grain_volume_error_max": max(c["maxima"]["grain_volume_error"]["value"] for c in cases),
        "field_ratio_max": max(c["maxima"]["field_ratio"]["value"] for c in cases),
        "criterion_ratio_max": max(c["maxima"]["criterion_ratio"]["value"] for c in cases),
    }


def regime_header(key: str) -> dict:
    regime = REGIMES[key]
    side, grains = regime["side"], regime["grains"]
    return {
        "title": regime["title"],
        "grid": [side, side, side],
        "voxels": side ** 3,
        "grains": grains,
        "mean_initial_grain_volume": side ** 3 / grains,
        "state_size_over_training_field": round(side ** 3 * grains / (TRAINING["side"] ** 3 * TRAINING["grains"]), 2),
        "horizon_steps": regime["horizon"],
        "horizon_t": regime["horizon"] * TIME_STEP,
        "evaluation_steps": list(regime["eval_steps"]),
        "field_error_factor": FIELD_FACTOR,
        "field_error_factor_final_step": FIELD_FACTOR_FINAL,
        "field_criterion_from_step": FIELD_CRITERION_FROM_STEP,
    }


def json_volumes(record: object, key: str | None = None) -> np.ndarray:
    table = record if key is None else record[key]
    return np.asarray(table, dtype=np.float64)


def build_stage_f_regime(key: str, sources: dict, out: Path, assets: dict) -> dict:
    """96^3 and 128^3 with 38 grains: score records, diagnostics, volume tables, labels."""
    regime = REGIMES[key]
    side, grains, steps = regime["side"], regime["grains"], regime["eval_steps"]
    voxels = side ** 3
    structural = structural_from_g0(sources[f"{key}_structural"], 10, regime)
    met = structural_met(structural)
    structural["met"] = met
    cases, comparators = [], []
    for k in range(1, CASES + 1):
        gates = sources[f"{key}_scores_{k}"]
        require(gates["n"] == side and gates["N"] == grains and gates["horizon"] == regime["horizon"],
                f"{key} {k}: regime")
        require(gates["weights_sha256"] == PARENT_CHECKPOINT_SHA256 and gates["t0_bitwise_equal_ic"] is True,
                f"{key} {k}: checkpoint and initial state")
        require(list(gates["eval_steps"]) == list(steps), f"{key} {k}: evaluation steps")
        diagnostics = sources[f"{key}_diagnostics_{k}"]
        require(diagnostics["seed"] == gates["seed"], f"{key} {k}: diagnostics case")
        reference_volumes = json_volumes(sources[f"{key}_reference_volumes_{k}"])
        model_volumes = json_volumes(sources[f"{key}_model_volumes_{k}"], "model_volumes_per_step")
        case, recheck = gates_case(key, k, gates, reference_volumes, model_volumes)
        with np.load(sources[f"{key}_labels_{k}"], allow_pickle=False) as archive:
            labels = {name: archive[name] for name in archive.files}
        require(json.loads(str(labels["meta"]))["seed"] == gates["seed"], f"{key} {k}: label case")
        ref_steps = [int(s) for s in labels["ref_steps"]]
        model_steps = [int(s) for s in labels["model_frame_steps"]]
        require(ref_steps == [0, *steps], f"{key} {k}: reference label steps")
        counts = []
        for s in steps:
            fraction = diagnostics["per_eval_step"][str(s)]["checkpoint"]["argmax_label_disagreement_fraction"]
            count = int(round(fraction * voxels))
            require(abs(count - fraction * voxels) < 1e-6, f"{key} {k}: disagreement count at {s}")
            recount = int(np.count_nonzero(labels["model_labels"][model_steps.index(s)]
                                           != labels["ref_labels"][ref_steps.index(s)]))
            require(recount == count, f"{key} {k}: label arrays disagree with the scored fields at {s}")
            counts.append(count)
        case["label_disagreement"] = label_rows(steps, counts, voxels)
        initial = read_initial_labels(out / "initial_labels" / initial_label_name(key, k), side, grains)
        require(np.array_equal(initial, labels["ref_labels"][0]), f"{key} {k}: initial labels")
        require(np.array_equal(labels["model_labels"][0], labels["ref_labels"][0]), f"{key} {k}: shared t0")
        require(counts_match_volumes(initial, grains, reference_volumes[0]), f"{key} {k}: t0 volumes")
        finish_case(case, regime["horizon"], met)
        case["all_criteria_met"] = all(case["criteria"].values())
        case["initial_labels"] = f"benchmarks/{PACKAGE}/initial_labels/{initial_label_name(key, k)}"
        case["initial_labels_sha256"] = sha256(out / "initial_labels" / initial_label_name(key, k))
        case["per_step_volume_tables"] = "reference and model"
        case["volume_recheck"] = recheck
        cases.append(case)
        comparators.append(comparator_case(key, k, sources[f"{key}_comparator_{k}"], case, gates["seed"]))
        case_dir = f"microstructure_{k}"
        assets[(key, case_dir)] = {
            "reference_volumes": {"volumes": reference_volumes,
                                  "steps": np.arange(regime["horizon"] + 1, dtype=np.int64)},
            "model_volumes": {"volumes": model_volumes,
                              "steps": np.arange(regime["horizon"] + 1, dtype=np.int64)},
            "reference_labels": {"labels": labels["ref_labels"].astype(np.uint8),
                                 "steps": np.asarray(ref_steps, dtype=np.int64)},
            "model_labels": {"labels": labels["model_labels"].astype(np.uint8),
                             "steps": np.asarray(model_steps, dtype=np.int64)},
        }
    block = regime_header(key)
    block.update({
        "structural_check": structural,
        "cases": cases,
        "summary": regime_summary(cases),
        "comparator": {
            "description": ("matched comparator: identical architecture, initialization, loss and "
                            "28,200-update budget, trained on reference-trajectory states; context "
                            "only, never part of an acceptance decision for the primary model"),
            "checkpoint_sha256": COMPARATOR_CHECKPOINT_SHA256,
            "distributed": False,
            "criteria_assessed": [CRITERION_LABELS[k] for k, _ in CRITERIA[1:]],
            "cases": comparators,
            "summary": comparator_summary(comparators),
        },
    })
    return block


def build_r64(sources: dict, out: Path, assets: dict) -> dict:
    key = "r64_n16"
    regime = REGIMES[key]
    side, grains = regime["side"], regime["grains"]
    record = sources["r64_n16_scores"]
    require(record["weights_sha256"] == PARENT_CHECKPOINT_SHA256, "64^3 checkpoint")
    structural = structural_from_two_fields(record["G0"])
    met = structural_met(structural)
    structural["met"] = met
    cases = []
    for k, gates in enumerate(record["cases"], 1):
        require(gates["v0_mean"] == side ** 3 / grains, f"64^3 case {k}: grain count")
        reference_volumes = json_volumes(sources[f"r64_n16_reference_volumes_{k}"])
        case, recheck = gates_case(key, k, gates, reference_volumes, None)
        initial = read_initial_labels(out / "initial_labels" / initial_label_name(key, k), side, grains)
        require(counts_match_volumes(initial, grains, reference_volumes[0]), f"64^3 case {k}: t0 volumes")
        case["label_disagreement"] = None
        finish_case(case, regime["horizon"], met)
        case["all_criteria_met"] = all(case["criteria"].values())
        case["initial_labels"] = f"benchmarks/{PACKAGE}/initial_labels/{initial_label_name(key, k)}"
        case["initial_labels_sha256"] = sha256(out / "initial_labels" / initial_label_name(key, k))
        case["per_step_volume_tables"] = "reference only; the model's per-step volumes were not stored"
        case["volume_recheck"] = recheck
        cases.append(case)
        assets[(key, f"microstructure_{k}")] = {
            "reference_volumes": {"volumes": reference_volumes,
                                  "steps": np.arange(regime["horizon"] + 1, dtype=np.int64)},
        }
    require(len(cases) == CASES and record["cases_gated_pass"] == 3, "64^3 cohort")
    block = regime_header(key)
    block.update({"structural_check": structural, "cases": cases, "summary": regime_summary(cases),
                  "comparator": None})
    return block


def size_sample(volumes: list[np.ndarray], step: int) -> np.ndarray:
    """Equivalent-sphere radii over each microstructure's mean, grains above one voxel, pooled."""
    out = []
    for table in volumes:
        v = table[step]
        v = v[v > 1.0]
        r = (3 * v / (4 * np.pi)) ** (1 / 3)
        out.append(r / r.mean())
    return np.concatenate(out)


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: largest gap between the empirical CDFs."""
    a, b = np.sort(a), np.sort(b)
    values = np.concatenate([a, b])
    gap = np.searchsorted(a, values, side="right") / a.size - np.searchsorted(b, values, side="right") / b.size
    return float(np.max(np.abs(gap)))


def radius_slope(volumes: np.ndarray, side: int) -> float:
    """Slope of Rbar^2 - Rbar_0^2 against t over 40 <= t <= 135, Rbar from the alive count."""
    alive = np.sum(volumes > ALIVE_THRESHOLD, axis=1)
    t = np.arange(volumes.shape[0]) * TIME_STEP
    rbar = (3 * (side ** 3 / alive) / (4 * np.pi)) ** (1 / 3)
    y = rbar ** 2 - rbar[0] ** 2
    window = (t >= 40) & (t <= 135)
    return float(np.polyfit(t[window], y[window], 1)[0])


def n150_statistics(cases: list[dict], tables: list[tuple[np.ndarray, np.ndarray]],
                    labels: list[tuple[np.ndarray, list[int], np.ndarray, list[int]]]) -> dict:
    """Descriptive statistics quoted in the paper, from the volume tables and label arrays."""
    side = REGIMES["r128_n150"]["side"]
    distribution = []
    for step in (3000, 8000, 13500):
        reference = size_sample([t[0] for t in tables], step)
        model = size_sample([t[1] for t in tables], step)
        distribution.append({
            "step": step,
            "grains_reference": int(reference.size),
            "grains_model": int(model.size),
            "ks_distance": ks_distance(reference, model),
            "lognormal_width_reference": float(np.std(np.log(reference))),
            "lognormal_width_model": float(np.std(np.log(model))),
        })
    slopes = []
    for case, (reference, model) in zip(cases, tables):
        ref_slope, mod_slope = radius_slope(reference, side), radius_slope(model, side)
        slopes.append({"public_name": case["public_name"], "reference_slope": ref_slope,
                       "model_slope": mod_slope,
                       "model_relative_deviation_percent": (mod_slope / ref_slope - 1) * 100})
    midplane = []
    for case, (ref_labels, ref_steps, model_labels, model_steps) in zip(cases, labels):
        final = REGIMES["r128_n150"]["horizon"]
        r = ref_labels[ref_steps.index(final)][:, :, side // 2]
        m = model_labels[model_steps.index(final)][:, :, side // 2]
        midplane.append({"public_name": case["public_name"], "step": final,
                         "disagreeing_voxels": int(np.count_nonzero(r != m)),
                         "percent": 100.0 * np.count_nonzero(r != m) / side ** 2})
    grain_counts = {}
    for step in (3000, 8000, 13500):
        grain_counts[str(step)] = {
            "reference": [len(alive_set(t[0], step)) for t in tables],
            "model": [len(alive_set(t[1], step)) for t in tables],
        }
    residuals = []
    for case in cases:
        magnitudes = [abs(e["signed_residual_steps"]) for e in case["extinctions"]]
        residuals.append({"public_name": case["public_name"],
                          "absolute_median_steps": float(np.median(magnitudes)),
                          "absolute_max_steps": max(magnitudes)})
    return {
        "source": ("computed by scripts/package_phaselocal_transfer.py from the release-asset volume "
                   "tables and label arrays; re-derived by scripts/verify_phaselocal_transfer.py "
                   "--asset-root. Descriptive; no acceptance criterion depends on these values."),
        "label_disagreement_percent_final": [c["label_disagreement"][-1]["percent"] for c in cases],
        "label_disagreement_percent_max_through_t60": max(
            r["percent"] for c in cases for r in c["label_disagreement"] if r["step"] <= 6000),
        "xy_midplane_disagreement_final": midplane,
        "grain_counts": grain_counts,
        "extinction_residuals": residuals,
        "size_distribution": {
            "definition": ("u = R / <R>, R the equivalent-sphere radius of each grain from its diffuse "
                           "volume, grains of at most one voxel excluded, normalized by the mean radius "
                           "of its own microstructure, then pooled over microstructures 1 to 3; the "
                           "log-normal width is the standard deviation of log u; the KS distance is "
                           "the two-sample Kolmogorov-Smirnov statistic between model and reference"),
            "rows": distribution,
        },
        "mean_radius_slope": {
            "definition": ("Rbar = (3 n^3 / (4 pi N_alive))^(1/3); slope of Rbar^2 - Rbar_0^2 against t "
                           "from a linear least-squares fit over 40 <= t <= 135, every step"),
            "rows": slopes,
        },
    }


def build_n150(sources: dict, out: Path, assets: dict) -> dict:
    key = "r128_n150"
    regime = REGIMES[key]
    side, grains, horizon, steps = regime["side"], regime["grains"], regime["horizon"], regime["eval_steps"]
    voxels = side ** 3
    mean_volume = voxels / grains
    score = sources["r128_n150_scores"]
    structural = structural_from_checks(sources["r128_n150_structural"])
    met = structural_met(structural)
    structural["met"] = met
    require(score["g0"]["state"] == "PASS" and score["g0"]["checks"] == sources["r128_n150_structural"]["checks"],
            "150-grain structural record")
    cohort = list(score["cohort"])
    require(len(cohort) == CASES, "150-grain cohort")
    cases, tables, label_sets = [], [], []
    for k, case_id in enumerate(cohort, 1):
        name = f"{key} microstructure {k}"
        record = score["cases"][str(case_id)]
        model = sources[f"r128_n150_model_evidence_{k}"]
        reference = sources[f"r128_n150_reference_evidence_{k}"]
        require(model["seed"] == case_id == reference["seed"], f"{name}: case identity")
        require(record["model_evidence_sha256"] == SOURCE_SHA256[f"r128_n150_model_evidence_{k}"]
                and record["reference_evidence_sha256"] == SOURCE_SHA256[f"r128_n150_reference_evidence_{k}"],
                f"{name}: evidence digests")
        require(model["completed"] and reference["completed"] and model["horizon"] == horizon, f"{name}: completion")
        model_volumes = json_volumes(model, "volumes")
        reference_volumes = json_volumes(reference, "volumes")
        require(model_volumes.shape == reference_volumes.shape == (horizon + 1, grains), f"{name}: tables")
        require(np.array_equal(model_volumes[0], reference_volumes[0]), f"{name}: shared initial state")
        require(np.all(reference_volumes[0] > ALIVE_THRESHOLD), f"{name}: every grain alive at t = 0")
        ref_ext, ref_returns = volume_events(reference_volumes)
        mod_ext, mod_returns = volume_events(model_volumes)
        require(ref_ext == {int(g): int(s) for g, s in reference["extinctions"].items()}, f"{name}: reference events")
        require(mod_ext == {int(g): int(s) for g, s in model["extinctions"].items()}, f"{name}: model events")
        require(ref_returns == list(reference["returns"]) == [] and mod_returns == list(model["returns"]) == [],
                f"{name}: returns")
        require({int(g): int(e) for g, e in record["extinction_step_errors"].items()}
                == {g: abs(mod_ext[g] - ref_ext[g]) for g in ref_ext}, f"{name}: recorded residuals")
        case = {
            "public_name": f"microstructure {k}",
            "admissibility": {"completed_steps": model_volumes.shape[0] - 1,
                              "violations": len(model["violations"])},
            "extinctions": extinction_rows(ref_ext, mod_ext),
            "returns": {"reference": len(ref_returns), "model": len(mod_returns)},
            "alive": alive_rows(grains, steps, ref_ext, mod_ext),
            "field_error": field_rows([(s, record["g3_by_step"][str(s)]["E_TV_model"],
                                        record["g3_by_step"][str(s)]["E_TV_persistence"]) for s in steps],
                                      horizon),
            "grain_volume_error": [{"step": s, "value": record["g4_by_step"][str(s)]["reported"]} for s in steps],
        }
        for row in case["alive"]:
            s = row["step"]
            require(alive_set(reference_volumes, s) ==
                    {g for g in range(grains) if ref_ext.get(g, s + 1) > s}, f"{name}: reference alive {s}")
            require(alive_set(model_volumes, s) ==
                    {g for g in range(grains) if mod_ext.get(g, s + 1) > s}, f"{name}: model alive {s}")
        for row in case["field_error"]:
            recorded = record["g3_by_step"][str(row["step"])]
            require(recorded["factor"] == row["factor"] and recorded["applies"] == row["criterion_applies"]
                    and abs(recorded["ratio"] - row["field_ratio"]) <= 1e-15, f"{name}: field row")
        recheck = max(abs(volume_error(model_volumes, reference_volumes, r["step"], mean_volume) - r["value"])
                      for r in case["grain_volume_error"])
        require(recheck <= 1e-12, f"{name}: volume error recheck")
        with np.load(sources[f"r128_n150_reference_labels_{k}"], allow_pickle=False) as archive:
            ref_labels, ref_steps = archive["labels"], [int(s) for s in archive["steps"]]
        with np.load(sources[f"r128_n150_model_labels_{k}"], allow_pickle=False) as archive:
            model_labels, model_steps = archive["labels"], [int(s) for s in archive["steps"]]
        require(ref_steps == [0, *steps], f"{name}: reference label steps")
        require(model_steps == list(range(100, horizon + 1, 100)), f"{name}: model label steps")
        counts = [int(np.count_nonzero(model_labels[model_steps.index(s)] != ref_labels[ref_steps.index(s)]))
                  for s in steps]
        case["label_disagreement"] = label_rows(steps, counts, voxels)
        initial = read_initial_labels(out / "initial_labels" / initial_label_name(key, k), side, grains)
        require(np.array_equal(initial, ref_labels[0]), f"{name}: initial labels")
        require(counts_match_volumes(initial, grains, reference_volumes[0]), f"{name}: t0 volumes")
        finish_case(case, horizon, met)
        case["all_criteria_met"] = all(case["criteria"].values())
        gates = record["gates"]
        require((gates["G1"], gates["G2"], gates["G3"], gates["G4"]) ==
                tuple(case["criteria"][c] for c, _ in CRITERIA[1:]), f"{name}: recorded verdicts")
        require(gates["G2a"] == all(r["alive_sets_equal"] for r in case["alive"]), f"{name}: alive verdict")
        case["initial_labels"] = f"benchmarks/{PACKAGE}/initial_labels/{initial_label_name(key, k)}"
        case["initial_labels_sha256"] = sha256(out / "initial_labels" / initial_label_name(key, k))
        case["per_step_volume_tables"] = "reference and model"
        case["volume_recheck"] = {"grain_volume_error_max_abs_difference": recheck}
        cases.append(case)
        tables.append((reference_volumes, model_volumes))
        label_sets.append((ref_labels, ref_steps, model_labels, model_steps))
        assets[(key, f"microstructure_{k}")] = {
            "reference_volumes": {"volumes": reference_volumes, "steps": np.arange(horizon + 1, dtype=np.int64)},
            "model_volumes": {"volumes": model_volumes, "steps": np.arange(horizon + 1, dtype=np.int64)},
            "reference_labels": {"labels": ref_labels.astype(np.uint8), "steps": np.asarray(ref_steps, dtype=np.int64)},
            "model_labels": {"labels": model_labels.astype(np.uint8), "steps": np.asarray(model_steps, dtype=np.int64)},
        }
    block = regime_header(key)
    block.update({
        "model_label_steps_note": ("the model label array starts at step 100; step 0 is the shared "
                                   "initial state, stored as the reference labels at step 0 and as the "
                                   "initial labels"),
        "structural_check": structural,
        "cases": cases,
        "summary": regime_summary(cases),
        "comparator": None,
        "derived_statistics": n150_statistics(cases, tables, label_sets),
    })
    block["summary"]["grain_volume_criterion_met_through_t"] = {
        c["public_name"]: c["grain_volume_criterion_met_through_step"] * TIME_STEP for c in cases}
    return block


def training_block(record: dict, out: Path) -> dict:
    require(record["weights_sha256"] == PARENT_CHECKPOINT_SHA256 and record["weights_reread_verified"] is True,
            "training checkpoint")
    require(record["updates_done"] == record["N_UPD_planned"] == 28200 and record["incomplete"] is False,
            "training updates")
    require(record["cycles_planned"] * record["R_ADV"] == record["updates_done"], "training cycles")
    labels = []
    for k in range(1, TRAINING["microstructures"] + 1):
        name = f"train_48_n8_microstructure_{k}.npz"
        read_initial_labels(out / "initial_labels" / name, TRAINING["side"], TRAINING["grains"])
        labels.append({"path": f"benchmarks/{PACKAGE}/initial_labels/{name}",
                       "sha256": sha256(out / "initial_labels" / name)})
    return {
        "grid": [TRAINING["side"]] * 3,
        "grains": TRAINING["grains"],
        "initial_microstructures": TRAINING["microstructures"],
        "initial_labels": labels,
        "initial_labels_note": (
            "regenerated from the recorded generation parameters of the periodic Voronoi "
            "construction; the training program rebuilt these fields at the start of the run and "
            "did not save them, so no stored copy exists to compare with"),
        "updates": record["updates_done"],
        "cycles": record["cycles_planned"],
        "updates_per_cycle": record["R_ADV"],
        "learned_steps_per_cycle": record["K_ADV"],
        "trajectories": len(record["traversals_per_frontier"]),
        "restart_after_steps": 3000,
        "full_traversals_per_trajectory": record["traversals_per_frontier"][0],
        "final_partial_traversal_steps": record["final_s_per_frontier"][0],
        "buffer_capacity": record["buffer_capacity"],
        "loss_mean_first_200_updates": record["loss_first_200_mean"],
        "loss_mean_last_200_updates": record["loss_last_200_mean"],
        "training_wall_seconds": round(record["training_wall_seconds"], 1),
        "threads": 8,
        "recipe": (
            "Four autonomous trajectories, one per initial microstructure, advance ten learned steps "
            "each without keeping their computational graphs, followed by twenty Adam updates; each "
            "trajectory restarts from its initial field after 3,000 steps. A first-in first-out "
            "buffer of capacity 240 starts with the four initial fields and receives each "
            "trajectory's state every ten steps, so it holds only initial fields and states generated "
            "by the model itself. Every update draws six crops (20^3 interior, 3-cell halo) from "
            "uniformly chosen buffer states at uniform periodic offsets. The target is one "
            "application of the reference map to the sampled state, computed without gradient and "
            "restricted to the interior; the loss is the mean over crops and interior voxels of the "
            "squared difference summed over phases, without time-step scaling. Adam, cosine learning "
            "rate from 1e-3 to 1e-4, 28,200 updates in 1,410 cycles. Convolution weights are "
            "initialized uniformly on +/-(27 c_in)^(-1/2), the third convolution multiplied by 0.1."),
        "training_command_distributed": False,
        "reproducibility": "recipe level; training is not bitwise reproducible across software builds",
    }


def build(sources: dict, out: Path, assets: dict, public_weights_sha256: str | None) -> dict:
    regimes = {
        "r64_n16": build_r64(sources, out, assets),
        "r96_n16": build_stage_f_regime("r96_n16", sources, out, assets),
        "r128_n38": build_stage_f_regime("r128_n38", sources, out, assets),
        "r128_n150": build_n150(sources, out, assets),
    }
    regimes["r64_n16"]["reproduction_level"] = {
        "score": "PROVENANCE_ONLY",
        "rollout": "CHECKPOINT_AND_CODE_REPLAY",
        "note": ("The model's per-step volumes were not stored in this regime, so its criterion values "
                 "are recorded values; the reference volume table is a release asset and the reference "
                 "events recompute from it. A model rollout from the shipped initial labels takes about "
                 "27 minutes per case on 8 CPU threads."),
    }
    for key in ("r96_n16", "r128_n38", "r128_n150"):
        regimes[key]["reproduction_level"] = {
            "score": "SCORE_RECOMPUTATION",
            "rollout": "CHECKPOINT_AND_CODE_REPLAY",
            "note": ("With the release assets present, alive sets, extinction steps and timing, returns, "
                     "grain-volume errors and label disagreement recompute from the per-step volume "
                     "tables and label arrays. Field errors (criterion (iv)) and admissibility are "
                     "recorded values; the full fields are not distributed."),
        }
    return {
        "schema": "pinn-phase-phaselocal-transfer-v1",
        "public_case_names_only": True,
        "same_model_for_every_case": True,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_checkpoint_distributed": False,
        "public_weights": PUBLIC_WEIGHTS,
        "public_weights_sha256": public_weights_sha256,
        "parameter_count": PARAMETER_COUNT,
        "regime_order": list(REGIMES),
        "regimes_pooled": False,
        "time_step": TIME_STEP,
        "alive_threshold": ALIVE_THRESHOLD,
        "unit_sum_tolerance": UNIT_SUM_TOLERANCE,
        "permutation_tolerance": PERMUTATION_TOLERANCE,
        "grain_volume_limit": GRAIN_VOLUME_LIMIT,
        "timing_tolerance_min_steps": TIMING_MIN_STEPS,
        "timing_tolerance_fraction": TIMING_FRACTION,
        "definitions": DEFINITIONS,
        "criteria": {name: text for name, text in CRITERIA},
        "admission": ("Candidates are periodic Voronoi partitions screened with the reference map alone in "
                      "an order fixed before screening; a candidate qualifies when every grain is alive at "
                      "t = 0, at least three grains go extinct within the horizon and no grain returns. The "
                      "first three qualifying candidates form each regime's cohort, so the cases are "
                      "extinction-bearing by construction and do not estimate performance on unselected "
                      "microstructures."),
        "training": training_block(sources["training_record"], out),
        **regimes,
        "source_records_sha256": dict(SOURCE_SHA256),
    }


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

def _fmt_range(values: list[int]) -> str:
    return f"{values[0]:+d} to {values[1]:+d}"


def _case_table(block: dict) -> str:
    final = block["horizon_steps"]
    rows = [
        "| Case | extinctions (model / ref.) | signed residuals (steps) | e_V max | field ratio max "
        "| label disagreement at t = %s | criteria met |" % _t(final),
        "|---|:---:|:---:|---:|---:|---:|:---:|",
    ]
    for c in block["cases"]:
        residuals = [e["signed_residual_steps"] for e in c["extinctions"]]
        labels = c["label_disagreement"]
        rows.append(
            f"| {c['public_name']} | {c['extinctions_matched_by_identity']} / {c['reference_extinction_count']} "
            f"| {_fmt_range([min(residuals), max(residuals)])} "
            f"| {c['maxima']['grain_volume_error']['value']:.4f} "
            f"| {c['maxima']['field_ratio']['value']:.4f} "
            f"| {'not stored' if labels is None else '%.3f%%' % labels[-1]['percent']} "
            f"| {', '.join(CRITERION_LABELS[k] for k, _ in CRITERIA if c['criteria'][k])} |"
        )
    return "\n".join(rows)


def _t(step: int) -> str:
    value = step * TIME_STEP
    return f"{value:g}"


def benchmark_readme(record: dict) -> str:
    """Render the public benchmark README from the same record it describes."""
    criteria = "\n".join(f"- {text}" for _, text in CRITERIA)
    train = record["training"]
    sections = []
    for key in record["regime_order"]:
        block = record[key]
        s = block["summary"]
        level = block["reproduction_level"]
        if key != "r128_n150":
            headline = (
                f"All five criteria are met on **{s['cases_meeting_all_criteria']} of {s['cases']}** "
                f"microstructures. The model reproduces {s['extinctions_matched_by_identity']} of "
                f"{s['reference_extinctions']} reference extinctions by identity, with signed residuals "
                f"from {_fmt_range(s['signed_residual_range_steps'])} steps, and no grain returns. The "
                f"largest normalized grain-volume error is {s['grain_volume_error_max']:.4f} (limit 0.05) "
                f"and the field error is at most {s['field_ratio_max']:.4f} times that of persistence."
            )
        else:
            stats = block["derived_statistics"]
            counts = " / ".join(str(c["reference_extinction_count"]) for c in block["cases"])
            through = block["summary"]["grain_volume_criterion_met_through_t"]
            full = [name for name, t in through.items() if t == block["horizon_t"]]
            medians = ", ".join(f"{r['absolute_median_steps']:g}" for r in stats["extinction_residuals"])
            finals = ", ".join(f"{v:.2f}%" for v in stats["label_disagreement_percent_final"])
            final_ev = ", ".join(f"{c['grain_volume_error'][-1]['value']:.4f}" for c in block["cases"])
            ks = ", ".join(f"{r['ks_distance']:.3f}" for r in stats["size_distribution"]["rows"])
            headline = (
                f"The reference eliminates {counts} grains in microstructures 1 to 3; the model "
                f"reproduces all {s['extinctions_matched_by_identity']} extinctions by identity and "
                f"within their timing tolerance, with no extinction the reference does not have and no "
                f"return, and absolute residual medians of {medians} steps. Label disagreement at "
                f"t = 135 is {finals}, and the field error stays at most {s['field_ratio_max']:.4f} "
                f"times that of persistence at all twelve evaluation times. The grain-volume criterion "
                f"(v) holds up to t = {min(through.values()):g} in all three microstructures and "
                f"up to t = {block['horizon_t']:g} in {', '.join(full)}; the largest normalized "
                f"grain-volume errors at t = 135 are {final_ev}. The alive sets of microstructures 1 and "
                f"3 differ from the reference by one grain at one evaluation time each (t = 50 and "
                f"t = 40), where the model and reference extinction steps of that grain fall on either "
                f"side of the evaluation time; both extinctions match in identity and within tolerance. "
                f"Pooled normalized grain-size distributions have two-sample KS distances of {ks} at "
                f"t = 30, 80 and 135. The criteria met in each case are listed in the table."
            )
        comparator = ""
        if block["comparator"] is not None:
            cs = block["comparator"]["summary"]
            comparator = (
                f"\n\nThe matched comparator (checkpoint `{COMPARATOR_CHECKPOINT_SHA256[:12]}...`, not "
                f"distributed), trained on reference-trajectory states with the same architecture, loss "
                f"and update budget, meets the casewise criteria (ii) to (v) on "
                f"{cs['cases_meeting_casewise_criteria']} of {cs['cases']} cases with signed residuals "
                f"from {_fmt_range(cs['signed_residual_range_steps'])} steps. It is context only."
            )
        sections.append(
            f"## {block['title']}, {block['horizon_steps']:,} steps (t = {_t(block['horizon_steps'])})\n\n"
            f"Mean initial grain volume {block['mean_initial_grain_volume']:,.2f} voxels; "
            f"{len(block['evaluation_steps'])} evaluation times. {headline}\n\n"
            f"{_case_table(block)}{comparator}\n\n"
            f"Reproduction level: score **{level['score']}**, rollout **{level['rollout']}**. "
            f"{level['note']}"
        )
    regime_sections = "\n\n".join(sections)
    return f'''# Phase-local transfer across domain size and grain number

One phase-local model, a shared bias-free rule of {record["parameter_count"]:,} learned
parameters applied to each phase field separately, followed by a margin repair and
a Euclidean projection onto the simplex. It was trained once at 48^3 with eight
grains on four initial microstructures and is applied unchanged, at fixed grid
spacing, to three unseen microstructures in each of four regimes. The regimes are
reported separately and never pooled; microstructure numbers are local to each
regime. Time is t = 0.01 s for step s.

The weights are distributed as `{record["public_weights"]}`; the accepted training
checkpoint (`{record["parent_checkpoint_sha256"][:12]}...`) is not. `expected_score.json`
is built from the accepted evaluation records by
`scripts/package_phaselocal_transfer.py`, which binds each record by SHA-256.

## Training

{train["initial_microstructures"]} initial microstructures at 48^3 with {train["grains"]} grains,
{train["updates"]:,} optimizer updates in {train["cycles"]:,} cycles, one-step targets of the
reference map evaluated on the model's own generated states. {train["recipe"]} No training
command is distributed; the recipe above is the disclosure. The four training
initial label arrays in `initial_labels/` are {train["initial_labels_note"]}.

## Acceptance criteria

{criteria}

Criterion (i) is assessed once per regime; criteria (ii) to (v) per case. A
grain is alive while its diffuse volume exceeds 1e-6; extinction steps come from
the diffuse volumes, never from label images.

{regime_sections}

## Initial microstructures

`initial_labels/<regime>_microstructure_<k>.npz` holds the initial grain labels of
every scored case (`labels`, uint8, shape (n, n, n), and `n_grains`). The initial
field is the one-hot field of these labels, which is also its own simplex
projection; for every case its grain volumes equal the recorded step-0 volumes, and
where reference labels were stored they equal the labels at step 0. Admission
selected extinction-bearing microstructures, so the cases do not estimate
performance on unselected ones.

## Release assets

Per-step grain volumes and argmax label arrays are attached to the GitHub release
{RELEASE_TAG} of this repository, one zip per regime, listed with SHA-256 and size in
`manifest.json`. Unpacked, they form `phaselocal_transfer/<regime>/microstructure_<k>/`
beneath an asset root.

## Verify

```bash
python scripts/verify_phaselocal_transfer.py --records benchmarks/phaselocal_transfer/expected_score.json
```

With the release assets unpacked beneath an asset root, the verifier also checks
every asset against the manifest and recomputes alive sets, extinction steps,
grain-volume errors, label disagreement and the 150-grain statistics from them:

```bash
python scripts/verify_phaselocal_transfer.py --records benchmarks/phaselocal_transfer/expected_score.json --asset-root /path/to/asset-root
```
'''


# ---------------------------------------------------------------------------
# Release assets and manifest
# ---------------------------------------------------------------------------

def write_assets(assets: dict, asset_out: Path) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    files_by_regime: dict[str, list[Path]] = {}
    for (regime, case_dir), roles in sorted(assets.items()):
        ordinal = int(case_dir.rsplit("_", 1)[1])
        for role, arrays in sorted(roles.items()):
            relative = f"{PACKAGE}/{regime}/{case_dir}/{role}.npz"
            path = asset_out / relative
            write_npz(path, arrays, compress=True)
            main = arrays["labels"] if "labels" in arrays else arrays["volumes"]
            rows.append({
                "expected_relative_location": relative,
                "filename": f"{role}.npz",
                "regime": regime,
                "public_case": f"microstructure {ordinal}",
                "role": role,
                "release_class": "RELEASE_ASSET",
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "array": "labels" if "labels" in arrays else "volumes",
                "dtype": str(main.dtype),
                "shape": list(main.shape),
                "steps": len(arrays["steps"]),
            })
            files_by_regime.setdefault(regime, []).append(path)
    bundles: list[dict] = []
    upload = asset_out / f"{PACKAGE}_upload"
    upload.mkdir(parents=True, exist_ok=True)
    for regime in REGIMES:
        name = f"{PACKAGE}_{regime}.zip"
        count = write_bundle(upload / name, asset_out, files_by_regime[regime])
        bundles.append({"filename": name, "regime": regime, "members": count,
                        "sha256": sha256(upload / name), "bytes": (upload / name).stat().st_size})
    (upload / "SHA256SUMS").write_bytes(
        "".join(f"{b['sha256']}  {b['filename']}\n" for b in bundles).encode("ascii"))
    return rows, bundles


def manifest_document(rows: list[dict], bundles: list[dict], out: Path, record: dict) -> dict:
    initial = []
    for path in sorted((out / "initial_labels").glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            labels = archive["labels"]
        initial.append({"path": f"benchmarks/{PACKAGE}/initial_labels/{path.name}",
                        "sha256": sha256(path), "bytes": path.stat().st_size,
                        "shape": list(labels.shape), "dtype": str(labels.dtype)})
    return {
        "schema": "pinn-phase-phaselocal-transfer-assets-v1",
        "release": {
            "tag": RELEASE_TAG,
            "download_location": f"GitHub release {RELEASE_TAG} of this repository, attached assets",
            "bundles": bundles,
            "checksums_file": "SHA256SUMS",
            "unpack": "unzip every bundle into one directory, the asset root",
        },
        "asset_root_layout": f"{PACKAGE}/<regime>/microstructure_<k>/<role>.npz",
        "array_keys": {
            "reference_volumes": "volumes float64 (steps, grains); steps int64",
            "model_volumes": "volumes float64 (steps, grains); steps int64",
            "reference_labels": "labels uint8 (frames, n, n, n); steps int64",
            "model_labels": "labels uint8 (frames, n, n, n); steps int64",
        },
        "assets": rows,
        "initial_labels": initial,
        "record_parent_checkpoint_sha256": record["parent_checkpoint_sha256"],
        "consumer_command": (f"python scripts/verify_{PACKAGE}.py --records "
                             f"benchmarks/{PACKAGE}/expected_score.json"),
        "asset_verification_command": (f"python scripts/verify_{PACKAGE}.py --records "
                                       f"benchmarks/{PACKAGE}/expected_score.json --asset-root /path/to/asset-root"),
    }


def public_weights_digest(repo: Path) -> str | None:
    lineage = repo / WEIGHT_LINEAGE
    if not lineage.is_file():
        return None
    document = json.loads(lineage.read_text(encoding="utf-8"))
    return document.get("public_weights", {}).get("sha256")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True,
                        help="directory holding the accepted evaluation records")
    parser.add_argument("--out", type=Path, default=Path(f"benchmarks/{PACKAGE}"))
    parser.add_argument("--asset-out", type=Path, required=True,
                        help="directory receiving the release assets")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    assets: dict = {}
    record = build(locate_sources(args.records), args.out, assets, public_weights_digest(repo))
    rows, bundles = write_assets(assets, args.asset_out)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "expected_score.json").write_bytes(
        (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    (args.out / "manifest.json").write_bytes(
        (json.dumps(manifest_document(rows, bundles, args.out, record), indent=2, sort_keys=True)
         + "\n").encode("utf-8"))
    (args.out / "README.md").write_bytes(benchmark_readme(record).encode("utf-8"))
    print(f"wrote {args.out / 'expected_score.json'}, manifest.json, README.md and "
          f"{len(rows)} release assets in {len(bundles)} bundles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
