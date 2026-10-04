#!/usr/bin/env python3
"""Build the public native 128^3 records from the accepted score records.

This packs already accepted measurements; it never evaluates a model or a
reference. Every source is identified by SHA-256, not by file name: the script
hashes every JSON file beneath ``--records`` and requires each pinned digest
below to be present exactly once. A record that does not match its digest is
treated as absent, and the build stops.

Usage::

    python scripts/package_n8_128_transfer.py --records /path/to/score-records \\
        --out benchmarks/n8_128_transfer
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# Accepted score records, bound by digest. Roles are descriptive; identity is the
# digest, never the file name.
SOURCE_SHA256 = {
    "training_case": "726d66d1229244125b1ccc2875264ba765b166734aae07f6c6e4e2ab1a2fe959",
    "development_1": "a9266f8044f04676282087d64e4e7ef455cbc61cd40fef56ece217390d829a2b",
    "development_2": "cc1e7f2068b7ad1eb5ab571de1df7ae284bed91846cd055c5439fcb1c9800e96",
    "development_3": "30fb88badf5acbb31c11ee4d28e19dfdb84f14c37e6a9e17e46a71caeb143cc1",
    "development_4": "74f3e67f7bda8400b6e8ff4bf65fef6a60602a51c972600d71179e779b11ce16",
    "development_5": "d26c0bbbc6756140a34589612159525c73726584c2a640ea36b42fb83f5f126d",
    "development_6": "af8e799581f754f8bbb13b75b877bd0b664387fb6cb449a241ae9a165cfe6818",
    "unseen_1": "b410b347dd5cd6d422d4059605bbe9374276c3906dc733b16df51e06548d64f9",
    "unseen_2": "d45b18387d22e3a01a58f5aa044eebd9791bd52b56c6aedee0d011dde4165f35",
    "unseen_3": "cc7425acbb84f6363edf06cb11bfb824a24dc764f22030a7d6c42b77823a6e63",
    "unseen_4": "6799619760c2a6c0467f4def703e0445de3e4769dde2c9cb31f5df4ed34e47d6",
    "unseen_5": "14c8eaa3140a83c7f8366d39cee922b9d87e7e1f9a1b37f0d90cb858d7e0d70f",
    "unseen_6": "b22b81c78abf6d799f0b5717e54882736b32552c3e27b0ae4c44d8c0847d5b57",
    "specialization_development_4": "8cbc50ed4b8eef226d90eeab2355886dd392028864e93760a7db8a1cacc5077b",
    "specialization_development_5": "c3e6872fba643b820976ab47de51e85a98aeea34cb3ee00ce9d29efb985800ab",
    "specialization_development_6": "d2186f4482eaa42868e7a74ff11129e0bd494f8b7da5cf3b402b953e49c8c706",
}

PARENT_CHECKPOINT_SHA256 = "0b04afab56139ecb13071396c7ca10777575add22d73949c0abeaaaeb0a4ed6b"
VOXELS = 128 ** 3
HORIZON = 4096
TERMINAL = 24000
SAVED_CADENCE = 800
LEDGER_CADENCE = 20
TIMING_TOLERANCE = 800
NEAR_HORIZON_STEP = 4000

#: The six criteria, in the order the evaluation applied them.
CRITERIA = (
    ("terminal_disagreement",
     "terminal label disagreement at most 5% of the 2,097,152 voxels"),
    ("below_persistence",
     "lower label disagreement than static-t0 persistence at all 30 non-initial saved states"),
    ("terminal_active_count",
     "equal terminal active-grain count"),
    ("terminal_survivor_set",
     "identical terminal survivor set"),
    ("extinction_events",
     "matching extinction count and order, with every timing residual within +/-800 steps"),
    ("designated_grain_fate",
     "agreement on the realized fate of one designated small grain, without reappearance"),
)
COHORT_RULE = (
    "Cohort-level transfer is supported when all six cases satisfy every criterion, or when "
    "five do and the sixth keeps its terminal survivor set and designated-grain fate."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_sources(root: Path) -> dict[str, dict]:
    """Return every pinned record, found by digest; fail if any is missing."""
    by_digest: dict[str, Path] = {}
    for candidate in sorted(root.rglob("*.json")):
        if candidate.is_file():
            by_digest.setdefault(sha256(candidate), candidate)
    missing = [role for role, digest in SOURCE_SHA256.items() if digest not in by_digest]
    if missing:
        raise SystemExit(f"accepted records not found by digest: {', '.join(missing)}")
    return {
        role: json.loads(by_digest[digest].read_text(encoding="utf-8"))
        for role, digest in SOURCE_SHA256.items()
    }


def pct(fraction: float) -> float:
    return round(100.0 * float(fraction), 6)


def cohort_case(public_name: str, record: dict) -> dict:
    """Public per-case values from one cohort score record."""
    case = record.get("case") or record["current_preserved_case"]
    detail = record["gate_detail"]
    p1, p2, p3, p4, p5, p6 = (detail[key] for key in ("P1", "P2", "P3", "P4", "P5", "P6"))
    steps = [int(s) for s in p2["per_step_steps"]]
    if steps != list(range(SAVED_CADENCE, TERMINAL + 1, SAVED_CADENCE)):
        raise SystemExit(f"{public_name}: unexpected saved-state cadence")
    near = steps.index(NEAR_HORIZON_STEP)
    criteria = {
        name: bool(case["gates"][key])
        for (name, _), key in zip(CRITERIA, ("P1", "P2", "P3", "P4", "P5", "P6"))
    }
    events = [
        {
            "grain": int(pair["reference_phase"]),
            "reference_step": int(pair["reference_step"]),
            "model_step": int(pair["model_step"]),
            "signed_residual_steps": int(pair["signed_timing_error"]),
            "within_tolerance": bool(pair["within_tolerance"]),
        }
        for pair in p5["matched_pairs"]
    ]
    return {
        "public_name": public_name,
        "terminal_step": TERMINAL,
        "terminal_disagreement_percent": pct(p1["terminal_argmax_disagreement_d_argmax"]),
        "terminal_disagreeing_voxels": int(p1["disagreeing_voxels"]),
        "terminal_persistence_disagreement_percent": pct(p2["per_step_persistence_disagreement"][-1]),
        "step_4000_disagreement_percent": pct(p2["per_step_model_disagreement"][near]),
        "step_4000_persistence_disagreement_percent": pct(p2["per_step_persistence_disagreement"][near]),
        "saved_states_below_persistence": sum(bool(v) for v in p2["per_step_booleans"]),
        "saved_states_evaluated": len(steps),
        "model_terminal_active_count": int(p3["model_terminal_active_count"]),
        "reference_terminal_active_count": int(p3["reference_terminal_active_count"]),
        "model_terminal_survivor_set": [int(v) for v in p4["model_survivor_set"]],
        "reference_terminal_survivor_set": [int(v) for v in p4["reference_survivor_set"]],
        "additional_survivors": [int(v) for v in p4["ghost_survivor_phases"]],
        "missing_survivors": [int(v) for v in p4["missing_survivor_phases"]],
        "model_extinction_count": int(p5["model_extinction_count"]),
        "reference_extinction_count": int(p5["reference_extinction_count"]),
        "events_paired": bool(p5["pairing_performed"]),
        "extinction_events": events,
        "designated_grain": int(p6["sstar_phase"]),
        "designated_grain_alive_in_reference": bool(p6["reference_sstar_alive"]),
        "designated_grain_alive_in_model": bool(p6["model_sstar_alive"]),
        "reappearance_events": int(p6["model_reappearance_events"]) + int(p6["reference_reappearance_events"]),
        "criteria": criteria,
        "all_criteria_met": all(criteria.values()),
        "trajectory": {
            "steps": [0] + steps,
            "disagreement_percent": [0.0] + [pct(v) for v in p2["per_step_model_disagreement"]],
            "persistence_disagreement_percent": [0.0] + [pct(v) for v in p2["per_step_persistence_disagreement"]],
        },
        "model_trajectory_sha256": case["model_fields_sha256"],
        "reference_trajectory_sha256": case["reference_npz_sha256"],
    }


def training_case(record: dict) -> dict:
    frames = record["per_frame"]
    model = record["ledger_facts"]["model"]
    reference = record["ledger_facts"]["reference"]
    residuals = record["extinction_residuals"]
    order = reference["extinction_order"]
    return {
        "public_name": "Training initial condition",
        "role": "same-initial-condition temporal extrapolation; not transfer evidence",
        "terminal_step": int(record["terminal"]["step"]),
        "terminal_disagreement_percent": pct(record["terminal"]["d_argmax"]),
        "terminal_agreement_percent": round(100.0 - pct(record["terminal"]["d_argmax"]), 6),
        "terminal_disagreeing_voxels": int(record["terminal"]["disagreeing_voxels"]),
        "terminal_persistence_disagreement_percent": pct(record["terminal"]["d_persist"]),
        "terminal_persistence_agreement_percent": round(100.0 - pct(record["terminal"]["d_persist"]), 6),
        "saved_states_below_persistence": sum(
            m < p for m, p in zip(frames["argmax_disagreement_by_cadence_step"][1:],
                                  frames["persistence_argmax_disagreement_by_cadence_step"][1:])
        ),
        "saved_states_evaluated": len(frames["save_steps"]) - 1,
        "model_terminal_survivor_set": [int(v) for v in model["terminal_active_set"]],
        "reference_terminal_survivor_set": [int(v) for v in reference["terminal_active_set"]],
        "model_extinction_order": [int(v) for v in model["extinction_order"]],
        "reference_extinction_order": [int(v) for v in order],
        "extinction_events": [
            {
                "grain": int(grain),
                "reference_step": int(residuals[str(grain)]["reference_step"]),
                "model_step": int(residuals[str(grain)]["model_step"]),
                "signed_residual_steps": int(residuals[str(grain)]["residual"]),
            }
            for grain in order
        ],
        "reappearance_events": int(model["reappearance_count"]) + int(reference["reappearance_count"]),
        "trajectory": {
            "steps": [int(s) for s in frames["save_steps"]],
            "disagreement_percent": [pct(v) for v in frames["argmax_disagreement_by_cadence_step"]],
            "persistence_disagreement_percent": [
                pct(v) for v in frames["persistence_argmax_disagreement_by_cadence_step"]
            ],
        },
        "model_trajectory_sha256": record["model_fields_sha256"],
        "reference_trajectory_sha256": record["reference_sha256"],
    }


def specialization_case(public_name: str, record: dict, unchanged_record_sha256: str) -> dict:
    if record["zero_shot_score_sha256"] != unchanged_record_sha256:
        raise SystemExit(f"{public_name}: specialization record does not cite its unchanged-model record")
    primary = record["primary"]
    unchanged = all(record["zero_shot_gates"].values())
    return {
        "public_name": public_name,
        "unchanged_model": {
            "final_extinction_residual_steps": int(primary["zero_shot_p5_last_extinction_signed_error"]),
            "terminal_disagreement_percent": pct(primary["zero_shot_terminal_d_argmax"]),
            "all_criteria_met": unchanged,
        },
        "specialized_model": {
            "final_extinction_residual_steps": int(primary["p5_last_extinction_signed_error"]),
            "terminal_disagreement_percent": pct(primary["terminal_d_argmax"]),
            "all_criteria_met": all(record["gates"].values()),
        },
        "success_rule": (
            "all six criteria retained" if unchanged
            else "final extinction residual brought within +/-800 steps without losing another satisfied criterion"
        ),
        "specialized_model_trajectory_sha256": record["scored_member_digests"]["model_fields.npz"],
    }


def cohort_summary(cases: list[dict]) -> dict:
    near = [c["step_4000_disagreement_percent"] for c in cases]
    near_p = [c["step_4000_persistence_disagreement_percent"] for c in cases]
    return {
        "cases": len(cases),
        "all_criteria_met": sum(c["all_criteria_met"] for c in cases),
        "terminal_survivor_set_exact": sum(c["criteria"]["terminal_survivor_set"] for c in cases),
        "below_persistence_at_every_saved_state": sum(
            c["saved_states_below_persistence"] == c["saved_states_evaluated"] for c in cases
        ),
        "step_4000_disagreement_percent_range": [min(near), max(near)],
        "step_4000_persistence_disagreement_percent_range": [min(near_p), max(near_p)],
        "terminal_disagreement_percent_range": [
            min(c["terminal_disagreement_percent"] for c in cases),
            max(c["terminal_disagreement_percent"] for c in cases),
        ],
    }


def benchmark_readme(record: dict) -> str:
    """Render the public benchmark README from the same record it describes."""
    criteria = "\n".join(
        f"{number}. {text}" for number, (_, text) in enumerate(CRITERIA, 1)
    )

    def table(cases: list[dict]) -> str:
        rows = [
            "| Case | step 4,000 | persistence | step 24,000 | persistence | active (model / ref.) | all criteria |",
            "|---|---:|---:|---:|---:|:---:|:---:|",
        ]
        for c in cases:
            rows.append(
                f"| {c['public_name']} | {c['step_4000_disagreement_percent']:.2f}% "
                f"| {c['step_4000_persistence_disagreement_percent']:.2f}% "
                f"| {c['terminal_disagreement_percent']:.2f}% "
                f"| {c['terminal_persistence_disagreement_percent']:.2f}% "
                f"| {c['model_terminal_active_count']} / {c['reference_terminal_active_count']} "
                f"| {'yes' if c['all_criteria_met'] else 'no'} |"
            )
        return "\n".join(rows)

    train = record["training_case"]
    dev = record["development_cohort"]
    unseen = record["unseen_cohort"]
    spec_rows = "\n".join(
        f"| {s['public_name']} | {s['unchanged_model']['final_extinction_residual_steps']:+,} "
        f"| {s['specialized_model']['final_extinction_residual_steps']:+,} "
        f"| {s['unchanged_model']['terminal_disagreement_percent']:.2f}% "
        f"| {s['specialized_model']['terminal_disagreement_percent']:.2f}% "
        f"| {'yes' if s['specialized_model']['all_criteria_met'] else 'no'} |"
        for s in record["specialization"]["cases"]
    )
    residuals = ", ".join(f"{e['signed_residual_steps']:+d}" for e in train["extinction_events"])
    return f'''# Native 128^3 eight-grain evolution and initial-condition transfer

One eight-phase model, trained on four initial conditions at the native 128^3
grid (2,097,152 voxels), with a represented training horizon of H = 4,096 steps.
Every prediction below starts from its own initial field and advances
autonomously to step 24,000 = 5.86 H, so 19,904 steps lie beyond the represented
horizon. Full fields are saved every 800 steps and grain ownership every 20
steps. Disagreement is the fraction of voxels whose argmax grain label differs
from the phase-field reference; persistence is the same quantity for the
unchanged initial field.

The public reproduction level is **PROVENANCE_ONLY**. The derived public
replay weights are distributed (`checkpoints/n8_128_cube_multi_ic_hybrid.weights.npz`),
but the 128^3 initial fields and the model and reference trajectories are
documented external assets, identified in `manifest.json` by SHA-256 and not
distributed here. `expected_score.json` is a compact record built from the
accepted score records by `scripts/package_n8_128_transfer.py`; it is not a
replacement for the arrays.

## Training initial condition

At step 24,000 the prediction differs from the reference in
{train['terminal_disagreeing_voxels']:,} voxels, {train['terminal_disagreement_percent']:.4f}%
({train['terminal_agreement_percent']:.2f}% agreement), against
{train['terminal_persistence_disagreement_percent']:.2f}% for persistence. It is closer
than persistence at {train['saved_states_below_persistence']} of
{train['saved_states_evaluated']} non-initial saved states, keeps the exact
five-grain survivor set and the reference extinction order, and its three
extinction residuals are {residuals} steps. This case was one of the four
training initial conditions: it measures retained fidelity under long temporal
extrapolation, not transfer.

## Evaluation criteria

The cohorts below were evaluated at step 24,000 against six criteria fixed
before evaluation:

{criteria}

{COHORT_RULE} These are benchmark-specific criteria, not universal definitions
of a useful prediction.

## Development cohort

Six initial conditions absent from training whose references had been examined
in an earlier evaluation. **{dev['summary']['all_criteria_met']} of 6** satisfy every
criterion; the cohort rule is not met. All six keep the exact terminal survivor
set; the two remaining exceptions are final-event timing.

{table(dev['cases'])}

## Blind cohort

Six further initial conditions, absent from training, whose initial fields,
criteria and evaluation procedure were fixed before any outcome was opened.
They were evaluated once. **{unseen['summary']['all_criteria_met']} of 6** satisfy every
criterion (Unseen microstructures 2 and 3) and the cohort rule is not met.
{unseen['summary']['terminal_survivor_set_exact']} of 6 keep the exact terminal survivor set;
Unseen microstructure 4 retains one grain that has disappeared in the reference,
produces three extinctions instead of four, and has no complete event pairing.
Every case stays closer to the reference than persistence at all 30 saved
states.

{table(unseen['cases'])}

The step-4,000 columns, the last saved state below H, come from an analysis
carried out after the full-horizon evaluation. They describe behaviour over the
represented horizon and are not a separately qualified endpoint. Neither cohort
is pooled with the other.

## Case specialization

A fixed physics-only recipe was applied separately to three development cases:
each descendant starts from the same trained weights and uses only that case's
initial field and the physics objective, for 1,024 optimizer updates over a
1,024-step represented window. The success rule for each case was fixed before
its run, after the unchanged-model outcome was known.

| Case | final-event residual, unchanged | specialized | terminal disagreement, unchanged | specialized | all criteria after |
|---|---:|---:|---:|---:|:---:|
{spec_rows}

This shows bounded case specialization from the trained model. It is not a
validated adaptation policy, a minimal budget, or a comparison with training
from scratch, and it does not replace either cohort's unchanged-model results.

## Verify

```bash
python scripts/verify_n8_128_transfer.py --records benchmarks/n8_128_transfer/expected_score.json
```

When the separately held trajectories are available beneath a directory that
contains `external/n8_128_transfer/`, verify the documented assets before using
them:

```bash
python scripts/verify_n8_128_transfer.py --records benchmarks/n8_128_transfer/expected_score.json --asset-root /path/to/asset-root
```
'''


def external_assets(record: dict) -> list[dict]:
    """Every documented external trajectory, keyed by public case and role."""
    rows: list[dict] = []

    def add(case_dir: str, public_case: str, role: str, digest: str) -> None:
        filename = f"{role}.npz"
        rows.append({
            "public_case": public_case,
            "role": role,
            "filename": filename,
            "sha256": digest,
            "release_class": "DOCUMENTED_ONLY",
            "expected_relative_location": f"external/n8_128_transfer/{case_dir}/{filename}",
        })

    train = record["training_case"]
    add("training", train["public_name"], "model_trajectory", train["model_trajectory_sha256"])
    add("training", train["public_name"], "reference_trajectory", train["reference_trajectory_sha256"])
    for cohort, prefix in (("development_cohort", "development"), ("unseen_cohort", "unseen")):
        for ordinal, case in enumerate(record[cohort]["cases"], 1):
            add(f"{prefix}_{ordinal}", case["public_name"], "model_trajectory", case["model_trajectory_sha256"])
            add(f"{prefix}_{ordinal}", case["public_name"], "reference_trajectory", case["reference_trajectory_sha256"])
    return rows


def build(sources: dict[str, dict]) -> dict:
    development = [
        cohort_case(f"Development microstructure {i}", sources[f"development_{i}"]) for i in range(1, 7)
    ]
    unseen = [cohort_case(f"Unseen microstructure {i}", sources[f"unseen_{i}"]) for i in range(1, 7)]
    specialization = [
        specialization_case(f"Development microstructure {i}", sources[f"specialization_development_{i}"],
                            SOURCE_SHA256[f"development_{i}"])
        for i in (5, 6, 4)
    ]
    if sources["training_case"]["candidate_checkpoint_sha256"] != PARENT_CHECKPOINT_SHA256:
        raise SystemExit("training-case record names a different checkpoint")
    return {
        "schema": "pinn-phase-n8-128-transfer-v1",
        "reproduction_level": "PROVENANCE_ONLY",
        "public_case_names_only": True,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "public_weights": "checkpoints/n8_128_cube_multi_ic_hybrid.weights.npz",
        "same_model_for_every_case": True,
        "grid": [128, 128, 128],
        "voxels": VOXELS,
        "phases": 8,
        "represented_horizon_steps": HORIZON,
        "terminal_step": TERMINAL,
        "terminal_step_over_horizon": round(TERMINAL / HORIZON, 2),
        "saved_state_cadence_steps": SAVED_CADENCE,
        "ownership_record_cadence_steps": LEDGER_CADENCE,
        "timing_tolerance_steps": TIMING_TOLERANCE,
        "criteria": {name: text for name, text in CRITERIA},
        "cohort_rule": COHORT_RULE,
        "near_horizon_analysis": (
            "Step-4,000 values come from an analysis carried out after the full-horizon "
            "evaluation; they are descriptive and do not define a separate qualification."
        ),
        "training_case": training_case(sources["training_case"]),
        "development_cohort": {
            "description": "absent from training; references examined in an earlier evaluation; not blind",
            "cohort_rule_met": False,
            "summary": cohort_summary(development),
            "cases": development,
        },
        "unseen_cohort": {
            "description": "absent from training; initial fields, criteria and procedure fixed before outcomes were opened; evaluated once",
            "cohort_rule_met": False,
            "summary": cohort_summary(unseen),
            "cases": unseen,
        },
        "specialization": {
            "description": (
                "fixed physics-only recipe, one descendant per case from the same trained weights; "
                "1,024 optimizer updates over a 1,024-step represented window; development cases only"
            ),
            "cases": specialization,
        },
        "source_records_sha256": dict(SOURCE_SHA256),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True,
                        help="directory holding the accepted score records")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/n8_128_transfer"))
    args = parser.parse_args()
    record = build(locate_sources(args.records))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "expected_score.json").write_bytes(
        (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    manifest = {
        "schema": "pinn-phase-n8-128-external-assets-v1",
        "assets": external_assets(record),
        "consumer_command": "python scripts/verify_n8_128_transfer.py --records benchmarks/n8_128_transfer/expected_score.json",
        "asset_verification_command": (
            "python scripts/verify_n8_128_transfer.py --records "
            "benchmarks/n8_128_transfer/expected_score.json --asset-root /path/to/asset-root"
        ),
    }
    (args.out / "manifest.json").write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    (args.out / "README.md").write_bytes(benchmark_readme(record).encode("utf-8"))
    print(f"wrote {args.out / 'expected_score.json'}, manifest.json and README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
