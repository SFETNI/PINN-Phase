#!/usr/bin/env python3
"""Render the all-six N16/96 unseen cohort as animated interior sections.

Each of the six unseen microstructures gets one block: the same interior section of
the phase-field reference and of the PINN-Phase prediction, advanced together over
the 17 saved states 0..3200, and a strip that follows label disagreement against
static initial-state persistence. Step 1600 is the scored terminal step; later
states are monitored continuation and are drawn as such.

The section of each case is fixed by one rule applied to the reference alone,
before the prediction is read: the plane ``labels[k]`` of the first array axis that
cuts the most step-0 volume of the grains the reference loses by step 1600. All six
cases are shown; none is selected.

The trajectories are not distributed. Each input is verified against the SHA-256
recorded below before anything is drawn, every number on a frame is recomputed from
the arrays, and the recomputed values are then required to equal the accepted record
in ``benchmarks/n16_96_transfer/expected_score.json``. A mismatch stops the render.

Usage::

    python scripts/render_n16_96_transfer.py --science-root PATH --out media/n16_96_transfer

``--synthetic-test`` skips the digest and record checks so the drawing code can be
exercised on synthetic arrays; the manifest then records ``"synthetic": true``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
from pathlib import Path

import imageio
import imageio.v2 as imageio_v2
import matplotlib

matplotlib.use("Agg")
import matplotlib.ft2font  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text  # noqa: E402
import numpy as np  # noqa: E402
import PIL  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "benchmarks/n16_96_transfer/expected_score.json"

CASES = ("C1", "C2", "C3", "C4", "C5", "C6")
#: Accepted trajectories, bound by digest. Identity is the digest, never the path.
EXPECTED_SHA256 = {
    ("C1", "reference"): "c1b3fad5e53bfa04e7e0b4370368e798bdd40a9ff9173a4afb1f686fcd3af62a",
    ("C1", "model"): "a18beef70811edc857d22b2ba0c59e3b159bdaa73bde3e0936a57f1dcf74c62f",
    ("C2", "reference"): "715d8b678acaf87f0800a5be62f116589ecf5f576a10fcbc848d29516cf13cfc",
    ("C2", "model"): "2bc5b7ebc4e4c45142f6ef56ddc96842fb4c57634a092f3608b6db7f879bddb1",
    ("C3", "reference"): "9c535954543965601b0f1f2cddc80d8ba9ab6a19e24a61c92dbc4a3244d687e1",
    ("C3", "model"): "0f8a798c219374530b03eee0dc7f82ee403ff37385d0e0d3b9ee7e2dd30a7d50",
    ("C4", "reference"): "6163d865ea307a08dbe12388c2803c1d4eaf3b3ecd448f869713d9ffd7328a34",
    ("C4", "model"): "eb02520cdec503291ee5bd2a5abf53f2a986a1e22ab3883df63569e928e17cf2",
    ("C5", "reference"): "2717b6e1a8cfcd19c3a10d071d07f3d3bab9c2dc06eb8714bbf0d93c3c246c2f",
    ("C5", "model"): "1c26448b75c5a3ce2d9c094c951ccdaa6097aac133095e9db43a0fa9ec429308",
    ("C6", "reference"): "a82dc9e00236eb387b79ff90b6484b8e02e1d0fef3bfea79705efab57b3329c4",
    ("C6", "model"): "dd7b9887183cd4df97cd5daeb884bacef0125535eaf5321e447aa0a5f8d5d618",
}
STEPS = tuple(range(0, 3201, 200))
TERMINAL = 1600
TERMINAL_INDEX = STEPS.index(TERMINAL)
FINAL = STEPS[-1]

GIF_NAME = "n16_96_unseen_cohort_reference_vs_pinn_phase.gif"
POSTER_NAME = "n16_96_unseen_cohort_reference_vs_pinn_phase_poster.png"
ALT_TEXT = (
    "Six unseen 96-cubed sixteen-grain microstructures, each advanced by the same fixed PINN-Phase "
    "model from its own initial field. For each one, the same interior section of the phase-field "
    "reference and of the prediction evolve side by side over saved steps 0 to 3200, through the "
    "loss of three grains, next to a strip comparing label disagreement with static persistence. "
    "Step 1600 is the scored terminal step; later steps are monitored continuation."
)

# The sixteen-grain palette of the other N16/96 media, so a grain keeps its colour
# across the cohort and interior animations.
PALETTE = np.array([(47, 112, 168), (247, 135, 37), (56, 148, 68), (154, 98, 204), (224, 184, 22),
                    (61, 174, 191), (181, 81, 161), (83, 130, 42), (141, 160, 235), (199, 140, 115),
                    (107, 214, 178), (146, 109, 156), (169, 178, 104), (173, 231, 255), (39, 28, 112),
                    (192, 242, 179)], dtype=np.float64) / 255.0
MODEL_ORANGE = "#d85d16"
INK = "#31363b"
PERSISTENCE_GREY = "#8b939a"
SCORED_TINT = "#e3ecf3"
CONTINUATION_TINT = "#f7f4ec"


class SourceConflict(RuntimeError):
    """A source or a recomputed value did not match its accepted identity."""


class UnreadableFrame(RuntimeError):
    """A label fell off the canvas or onto another label."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_labels(path: Path, role: str) -> np.ndarray:
    """Argmax labels of every saved state; the float field is released afterwards."""
    with np.load(path, allow_pickle=False) as archive:
        frames = archive["frames"]
    if frames.ndim != 5 or frames.shape[0] != len(STEPS):
        raise SourceConflict(f"{role}: expected {len(STEPS)} saved states, found shape {frames.shape}")
    labels = np.argmax(frames, axis=1).astype(np.uint8)
    del frames
    return labels


def section_index(reference: np.ndarray) -> int:
    """The plane labels[k] cutting the most step-0 volume of the grains the reference loses by 1600.

    Uses the reference only. Ties go to the smallest k; with no lost grain, the mid-plane.
    """
    lost = np.setdiff1d(np.unique(reference[0]), np.unique(reference[TERMINAL_INDEX]))
    if lost.size == 0:
        return reference.shape[1] // 2
    counts = np.isin(reference[0], lost).sum(axis=(1, 2))
    return int(np.argmax(counts))


def series(reference: np.ndarray, model: np.ndarray) -> dict:
    voxels = reference[0].size
    differing = [int((model[i] != reference[i]).sum()) for i in range(len(STEPS))]
    persistence = [int((reference[i] != reference[0]).sum()) for i in range(len(STEPS))]
    return {
        "disagreement_percent": [round(100.0 * v / voxels, 6) for v in differing],
        "persistence_percent": [round(100.0 * v / voxels, 6) for v in persistence],
        "reference_active": [int(np.unique(reference[i]).size) for i in range(len(STEPS))],
        "model_active": [int(np.unique(model[i]).size) for i in range(len(STEPS))],
    }


def check_against_record(case: dict, data: dict) -> None:
    """The recomputed values must equal the accepted record wherever it states them."""
    name = case["public_name"]
    t = TERMINAL_INDEX
    pairs = (
        ("terminal disagreement", data["disagreement_percent"][t], case["terminal_disagreement_percent"]),
        ("static initial-state agreement", 100.0 - data["persistence_percent"][t],
         case["static_t0_agreement_percent"]),
    )
    gains = [round(p - d, 6) for p, d in zip(data["persistence_percent"][1:], data["disagreement_percent"][1:])]
    worst = min(range(len(gains)), key=lambda i: (gains[i], i))
    pairs += (("minimum persistence gain", gains[worst], case["minimum_persistence_gain_percentage_points"]),)
    for label, mine, theirs in pairs:
        if abs(mine - theirs) > 1e-5:
            raise SourceConflict(f"{name}: recomputed {label} {mine} differs from the record {theirs}")
    if STEPS[worst + 1] != case["minimum_persistence_gain_step"]:
        raise SourceConflict(f"{name}: minimum persistence gain falls at a different step")
    if sum(g > 0 for g in gains) != case["persistence_frames_strictly_greater"]:
        raise SourceConflict(f"{name}: count of states below persistence differs from the record")
    if data["reference_active"][t] != case["reference_terminal_active_count"]:
        raise SourceConflict(f"{name}: reference active count at {TERMINAL} differs")
    if data["model_active"][t] != case["model_terminal_active_count"]:
        raise SourceConflict(f"{name}: model active count at {TERMINAL} differs")


def outcome_line(case: dict) -> str:
    if case["complete_predefined_qualification"]:
        return "all conditions met"
    unmet = len(case["unmet_criteria"])
    return f"exact grain set, {unmet} condition{'s' if unmet != 1 else ''} unmet"


def label_section(labels: np.ndarray, k: int) -> np.ndarray:
    """RGB image of the plane labels[k]; grain boundaries are darkened."""
    plane = labels[k].T
    rgb = PALETTE[plane % len(PALETTE)].copy()
    edge = np.zeros(plane.shape, dtype=bool)
    edge[:-1, :] |= plane[:-1, :] != plane[1:, :]
    edge[:, :-1] |= plane[:, :-1] != plane[:, 1:]
    rgb[edge] *= 0.55
    return rgb


def section_axis(ax, rgb: np.ndarray) -> None:
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(0.8)


def draw_strip(ax, data: dict, index: int, ymax: float) -> None:
    steps = list(STEPS)
    upto = slice(0, index + 1)
    ax.axvspan(TERMINAL, FINAL, color=CONTINUATION_TINT, zorder=0)
    ax.axvline(TERMINAL, color=INK, linestyle=(0, (4, 3)), linewidth=1.0, zorder=2)
    ax.plot(steps[upto], data["persistence_percent"][upto], color=PERSISTENCE_GREY,
            linewidth=1.3, marker="o", markersize=3.0, markerfacecolor="white",
            label="static initial-state persistence", zorder=3)
    ax.plot(steps[upto], data["disagreement_percent"][upto], color=MODEL_ORANGE,
            linewidth=1.9, marker="o", markersize=2.6, label="PINN-Phase", zorder=4)
    ax.plot([steps[index]], [data["disagreement_percent"][index]], marker="D", markersize=7.0,
            markerfacecolor=MODEL_ORANGE, markeredgecolor=INK, markeredgewidth=1.0,
            linestyle="none", zorder=5, clip_on=False)
    ax.set_xlim(0, FINAL)
    ax.set_ylim(0, ymax)
    ax.set_xticks([0, 1600, 3200])
    ax.set_xticklabels(["0", "1600", "3200"], fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel("label disagreement (%)", fontsize=8)
    ax.grid(alpha=0.18, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def draw_timeline(ax, step: int) -> None:
    """Scored window to the terminal step, then monitored continuation, with a moving marker."""
    ax.add_patch(matplotlib.patches.Rectangle((0, 0), TERMINAL, 1, facecolor=SCORED_TINT,
                                              edgecolor="#9aa3ab", linewidth=0.8))
    ax.add_patch(matplotlib.patches.Rectangle((TERMINAL, 0), FINAL - TERMINAL, 1,
                                              facecolor=CONTINUATION_TINT, edgecolor="#9aa3ab",
                                              linewidth=0.8))
    ax.plot([0, step], [-0.12, -0.12], color=MODEL_ORANGE, linewidth=3.0, solid_capstyle="butt",
            clip_on=False, zorder=4)
    ax.plot([step], [0.5], marker="D", markersize=10, markerfacecolor=MODEL_ORANGE,
            markeredgecolor=INK, markeredgewidth=1.1, clip_on=False, zorder=5)
    ax.text(0, 1.45, "0", fontsize=9.5, ha="center", va="bottom", color=INK)
    ax.text(TERMINAL, 1.45, f"{TERMINAL:,}: scored terminal step", fontsize=9.5, ha="center",
            va="bottom", color=INK, gid="chk:tl_terminal")
    ax.text(FINAL, 1.45, f"{FINAL:,}", fontsize=9.5, ha="center", va="bottom", color=INK)
    ax.text(TERMINAL / 2, -0.45, "scored evaluation", fontsize=9.5, ha="center", va="top",
            color=INK, gid="chk:tl_scored")
    ax.text((TERMINAL + FINAL) / 2, -0.45, "monitored continuation", fontsize=9.5, ha="center",
            va="top", color=INK, gid="chk:tl_continued")
    ax.set_xlim(0, FINAL)
    ax.set_ylim(0, 1)
    ax.axis("off")


def unreadable(figure) -> str | None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width, height = figure.canvas.get_width_height()
    tagged = []
    for text in figure.findobj(matplotlib.text.Text):
        content = text.get_text()
        if not content.strip() or not text.get_visible():
            continue
        box = text.get_window_extent(renderer=renderer)
        if min(box.x0, box.y0, width - box.x1, height - box.y1) < 0:
            return f"{content.splitlines()[0]!r} falls outside the canvas"
        if (text.get_gid() or "").startswith("chk:"):
            tagged.append((content.splitlines()[0], box))
    for first in range(len(tagged)):
        for second in range(first + 1, len(tagged)):
            (name_a, a), (name_b, b) = tagged[first], tagged[second]
            if max(a.x0 - b.x1, b.x0 - a.x1, a.y0 - b.y1, b.y0 - a.y1) < 0:
                return f"{name_a!r} overlaps {name_b!r}"
    return None


FIG_W = 13.6  # inches
FIG_H = 10.6  # inches


def frame(cases: list[dict], index: int, ymax: float) -> Image.Image:
    step = STEPS[index]
    figure = plt.figure(figsize=(FIG_W, FIG_H), dpi=100)

    def fy(inches_from_bottom: float) -> float:
        return inches_from_bottom / FIG_H

    def fx(inches_from_left: float) -> float:
        return inches_from_left / FIG_W

    if step < TERMINAL:
        status = "scored evaluation"
    elif step == TERMINAL:
        status = "scored terminal step"
    else:
        status = "monitored continuation beyond the scored step"
    figure.text(0.5, fy(FIG_H - 0.13),
                "Six unseen 96³ microstructures, sixteen grains each: one fixed model, no retraining",
                fontsize=14.5, fontweight="bold", ha="center", va="top", gid="chk:title")
    figure.text(0.5, fy(FIG_H - 0.43), f"step {step:,} of {FINAL:,}  ·  {status}",
                fontsize=11, ha="center", va="top", color=INK, gid="chk:status")
    draw_timeline(figure.add_axes([0.06, fy(FIG_H - 1.12), 0.88, fy(0.20)]), step)
    # Two columns of case blocks, three rows; every distance is set in inches.
    panel_in = 1.80
    block_h = 2.74
    first_top = FIG_H - 1.62
    for number, case in enumerate(cases):
        col, row = number % 2, number // 2
        left = 0.25 + col * 6.75
        top = first_top - row * block_h
        data = case["series"]
        figure.text(fx(left), fy(top), f"{case['public_name']}  ·  {case['outcome']}",
                    fontsize=10.5, fontweight="bold", ha="left", va="top", color=INK,
                    gid=f"chk:case{number}")
        bottom = fy(top - 0.62 - panel_in)
        ref_ax = figure.add_axes([fx(left), bottom, fx(panel_in), fy(panel_in)])
        mod_ax = figure.add_axes([fx(left + panel_in + 0.12), bottom, fx(panel_in), fy(panel_in)])
        section_axis(ref_ax, label_section(case["reference"][index], case["section"]))
        section_axis(mod_ax, label_section(case["model"][index], case["section"]))
        ref_ax.set_title(f"Reference\nactive grains {data['reference_active'][index]}", fontsize=9.5,
                         pad=3).set_gid(f"chk:ref{number}")
        caption = ("identical at step 0" if step == 0
                   else f"{data['disagreement_percent'][index]:.2f}% of volume differs")
        mod_ax.set_title(f"PINN-Phase, active grains {data['model_active'][index]}\n{caption}",
                         fontsize=9.5, pad=3).set_gid(f"chk:model{number}")
        strip_left = left + 2 * panel_in + 0.12 + 0.62
        strip = figure.add_axes([fx(strip_left), bottom + fy(0.30), fx(left + 6.35 - strip_left),
                                 fy(panel_in - 0.30)])
        draw_strip(strip, data, index, ymax)
        if number == 0:
            strip.legend(loc="upper left", fontsize=7.5, frameon=False)
    footer = (
        "Each microstructure was fixed before evaluation and never used in training; references are used for "
        "evaluation only. Every state is autonomous rollout from the initial field.",
        "Section: the interior plane cutting the most volume of the grains the reference loses, chosen from the "
        "reference alone. Full contract: benchmarks/n16_96_transfer.",
    )
    for offset, line in enumerate(footer):
        figure.text(0.5, fy(0.42 - 0.19 * offset), line, fontsize=8.6, ha="center", va="top", color=INK,
                    gid=f"chk:footer{offset}")
    problem = unreadable(figure)
    if problem is not None:
        plt.close(figure)
        raise UnreadableFrame(f"step {step:,}: {problem}")
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100, facecolor="white")
    plt.close(figure)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert("RGB").copy()


def strip_ceiling(cases: list[dict]) -> float:
    top = max(max(c["series"]["persistence_percent"] + c["series"]["disagreement_percent"]) for c in cases)
    return float(np.ceil(top * 1.15 / 2.0) * 2.0)


def font_identity() -> dict:
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    return {"font_file": path.name, "font_sha256": sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--science-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--synthetic-test", action="store_true")
    args = parser.parse_args()
    synthetic = args.synthetic_test

    record = json.loads(RECORD.read_text(encoding="utf-8"))
    inputs, cases = [], []
    for number, cid in enumerate(CASES, start=1):
        case = record["cases"][number - 1]
        if case["public_name"] != f"Unseen microstructure {number}":
            raise SourceConflict(f"record order: expected Unseen microstructure {number}")
        labels = {}
        for side, path in (("reference", args.science_root / f"references/{cid}/ref_frames.npz"),
                           ("model", args.science_root / f"rollouts/{cid}/model_frames.npz")):
            found = sha256_file(path)
            if not synthetic and found != EXPECTED_SHA256[(cid, side)]:
                raise SourceConflict(f"{cid} {side}: expected {EXPECTED_SHA256[(cid, side)][:16]}..., "
                                     f"found {found[:16]}...")
            labels[side] = load_labels(path, f"{cid} {side}")
            inputs.append({"case": cid, "role": side, "sha256": found})
        k = section_index(labels["reference"])
        data = series(labels["reference"], labels["model"])
        if not synthetic:
            check_against_record(case, data)
        cases.append({"public_name": case["public_name"], "outcome": outcome_line(case),
                      "reference": labels["reference"], "model": labels["model"],
                      "section": k, "series": data})

    ymax = strip_ceiling(cases)
    frames = [frame(cases, i, ymax) for i in range(len(STEPS))]
    # The continuation after the scored step changes slowly, so it plays fast; the
    # scored step and the final state (where any late crossing of persistence shows) hold.
    delays = [800.0 if i < TERMINAL_INDEX else 250.0 for i in range(len(frames))]
    delays[0], delays[TERMINAL_INDEX], delays[-1] = 1600.0, 2600.0, 3200.0
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    gif, poster = out / GIF_NAME, out / POSTER_NAME
    imageio_v2.mimsave(gif, [np.asarray(f) for f in frames], duration=delays, loop=0)
    frames[TERMINAL_INDEX].save(poster, format="PNG", optimize=True)

    score = args.science_root / "score" / "COHORT_SCORE.json"
    cohort = args.science_root / "cohort" / "COHORT_MANIFEST.json"
    lock = ROOT / "environment-media.yml"
    manifest = {
        "schema": "pinn-phase-n16-96-media-v2",
        "alt_text": ALT_TEXT,
        "synthetic": synthetic,
        "renderer_sha256": sha256_file(Path(__file__)),
        "record_sha256": sha256_file(RECORD),
        "environment_lock_sha256": sha256_file(lock),
        "cohort_score_sha256": sha256_file(score),
        "cohort_manifest_sha256": sha256_file(cohort),
        "checkpoint_sha256": json.loads(cohort.read_text(encoding="utf-8"))["expected_checkpoint_sha256"],
        "inputs": inputs,
        "view": {
            "kind": "interior section of the argmax grain labels, nearest voxel, boundaries darkened",
            "plane": "labels[k] of the first array axis, shown transposed with the origin at lower left",
            "selection_rule": ("k maximises the step-0 reference volume of the grains absent from the reference "
                               "at step 1600; ties to the smallest k; reference only"),
            "section_index": {c["public_name"]: c["section"] for c in cases},
        },
        "steps": list(STEPS),
        "terminal_step": TERMINAL,
        "no_interpolation": True,
        "recomputed": {c["public_name"]: c["series"] for c in cases},
        "renderer_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "pillow": PIL.__version__,
            "imageio": imageio.__version__,
            "freetype": matplotlib.ft2font.__freetype_version__,
            **font_identity(),
        },
        "frames": len(frames),
        "size": list(frames[0].size),
        "loop_seconds": round(sum(delays) / 1000.0, 1),
        "outputs": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in (gif, poster)},
    }
    (out / "asset_manifest.json").write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"Rendered {gif} ({gif.stat().st_size:,} bytes) and {poster}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
