#!/usr/bin/env python3
"""Render the native 128^3 unseen-microstructure animation from accepted arrays.

Three cases of the blind cohort are shown, one per row, each advanced from its
own initial field by the same trained model to step 24,000 = 5.86 H. Each row
pairs reference and PINN-Phase cubes, drawn with the camera, face shading and
palette of the paper's native 128^3 figures, with a mid-plane map of the
voxels whose argmax grain label differs, and a strip that follows label
disagreement against static-t0 persistence across the represented horizon
H = 4,096 and the autonomous extrapolation beyond it.

The cases are fixed in ``SHOWN`` and are not chosen by this script: the two
cases that meet all six criteria, and the survivor-correct case with the
largest out-of-tolerance timing residual. All six cases, including the one that
retains an extra grain, are listed in ``benchmarks/n8_128_transfer``.

The trajectories are not distributed. They are passed as arguments and each is
verified against the SHA-256 recorded below before anything is drawn. Every
number on a frame is recomputed from the arrays at render time, and the full
disagreement and persistence trajectories are then required to equal the
accepted record in ``benchmarks/n8_128_transfer/expected_score.json``. A
mismatch stops the render.

Usage::

    python scripts/render_n8_128_transfer.py \\
        --unseen-2-model PATH --unseen-2-reference PATH \\
        --unseen-3-model PATH --unseen-3-reference PATH \\
        --unseen-6-model PATH --unseen-6-reference PATH \\
        --output media/n8_128_transfer
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import platform

import imageio
import imageio.v2 as imageio_v2
import matplotlib

matplotlib.use("Agg")
import matplotlib.ft2font  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text  # noqa: E402
import PIL  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "benchmarks/n8_128_transfer/expected_score.json"

#: Accepted trajectories, bound by digest. Identity is the digest, never the name.
EXPECTED_SHA256 = {
    "unseen_2_model": "e60956b89e36630e57d3ce94f3e33ebe5f5cd37b054251db068734a7558b653e",
    "unseen_2_reference": "f0d28f3c969b1b3ba7f0a587c6d8f79a32a26e0c7667406223cffb61a23ab246",
    "unseen_3_model": "3815d75950dbd60743f2bb6776431ca8817a7687bb99a6eb5ea764589b3c884f",
    "unseen_3_reference": "fdaf4588d97be64a0156011b3dc2a82ed6bad7e109035ad6680c239aa14bf101",
    "unseen_6_model": "a9085813adf4e9a55d2ab36ff7fcbf974f261623e92f137f2689f8bdeeee2ad1",
    "unseen_6_reference": "3e925f5b73d1744afa7eacbef90da6750e796787c1d7b4d6194194f6ec57e8ef",
}
SHOWN = (2, 3, 6)
STEPS = tuple(range(0, 24001, 800))
HORIZON = 4096
FINAL = 24000
SECTION_Z = 64

GIF_NAME = "n8_128_unseen_reference_vs_pinn_phase.gif"
POSTER_NAME = "n8_128_unseen_reference_vs_pinn_phase_poster.png"
ALT_TEXT = (
    "Three unseen 128-cubed eight-grain microstructures, one per row, each advanced by the same "
    "trained PINN-Phase model from its own initial field to step 24,000. Each row shows fixed "
    "cubes of the phase-field reference and the PINN-Phase prediction at step 24,000 with the "
    "section plane marked, the same section of both evolving over time, and a strip comparing "
    "label disagreement with static persistence. These "
    "microstructures were never used in training: every step is direct autonomous rollout. The "
    "mark H = 4,096 is the horizon the model was trained over on four other initial fields."
)

# The cube view, palette and colours of the paper's native 128^3 figures, so the
# animation reads as those figures in motion: an oblique parallel camera showing
# three faces of the volume, each face shaded by a fixed brightness factor.
CAMERA_ELEV = 22.0
CAMERA_AZIM = -58.0
PHASE_HEX = ("#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#1f78b4", "#b3b3b3")
PAL = np.stack([np.array([int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]) for h in PHASE_HEX])
DISCREPANCY_RED = "#bd4a2b"
MODEL_ORANGE = "#d85d16"
INK = "#31363b"
PERSISTENCE_GREY = "#8b939a"
WITHIN_TINT = "#dce8f2"
TAIL_TINT = "#f7f4ec"
ANNOTATION_PT = 10.0


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


def _basis(elev_deg: float, azim_deg: float) -> tuple[np.ndarray, np.ndarray]:
    elev, azim = np.radians(elev_deg), np.radians(azim_deg)
    right = np.array([-np.sin(azim), np.cos(azim), 0.0])
    up = np.array([-np.sin(elev) * np.cos(azim), -np.sin(elev) * np.sin(azim), np.cos(elev)])
    return right, up


RIGHT, UP = _basis(CAMERA_ELEV, CAMERA_AZIM)


def project(point) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    return np.array([point @ RIGHT, point @ UP])


#: The three visible faces: origin, two spanning corners, the label plane shown on
#: that face, and the face's brightness factor.
FACES = (
    ((1, 0, 0), (1, 1, 0), (1, 0, 1), lambda labels: labels[-1, :, :].T, 0.85),
    ((0, 0, 0), (1, 0, 0), (0, 0, 1), lambda labels: labels[:, 0, :].T, 0.93),
    ((0, 0, 1), (1, 0, 1), (0, 1, 1), lambda labels: labels[:, :, -1].T, 1.00),
)
VISIBLE_EDGES = (
    ((0, 0, 0), (1, 0, 0)), ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 1, 0)),
    ((1, 0, 0), (1, 0, 1)), ((0, 0, 1), (1, 0, 1)), ((0, 0, 1), (0, 1, 1)),
    ((1, 0, 1), (1, 1, 1)), ((1, 1, 0), (1, 1, 1)), ((0, 1, 1), (1, 1, 1)),
)
_CORNERS = [project(p) for p in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                                 (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))]
UMIN, UMAX = min(c[0] for c in _CORNERS), max(c[0] for c in _CORNERS)
VMIN, VMAX = min(c[1] for c in _CORNERS), max(c[1] for c in _CORNERS)


def camera_metadata(n: int) -> dict:
    """Public record of the cube view implemented in ``cube_image``."""
    return {
        "projection": "oblique parallel projection of the three visible faces of the volume",
        "camera_elevation_deg": CAMERA_ELEV,
        "camera_azimuth_deg": CAMERA_AZIM,
        "faces": ["labels[-1, :, :] at brightness 0.85", "labels[:, 0, :] at brightness 0.93",
                  "labels[:, :, -1] at brightness 1.00"],
        "palette": list(PHASE_HEX),
        "grid": [n, n, n],
        "section": f"z = {SECTION_Z} plane for the differing-voxel map",
    }


def cube_image(labels: np.ndarray, pixels: int = 300) -> np.ndarray:
    """RGBA image of the three visible faces, sampled at the nearest voxel."""
    n = labels.shape[0]
    width, height = UMAX - UMIN, VMAX - VMIN
    ny, nx = pixels, int(round(pixels * width / height))
    canvas = np.zeros((ny, nx, 4), dtype=np.float32)
    uu, vv = np.meshgrid(UMIN + (np.arange(nx) + 0.5) / nx * width,
                         VMIN + (np.arange(ny) + 0.5) / ny * height)
    for p0, px, py, extract, brightness in FACES:
        origin = project(p0)
        matrix = np.column_stack([project(px) - origin, project(py) - origin])
        rel = np.stack([uu - origin[0], vv - origin[1]], axis=-1) @ np.linalg.inv(matrix).T
        fu, fv = rel[..., 0], rel[..., 1]
        inside = (fu >= -0.003) & (fu <= 1.003) & (fv >= -0.003) & (fv <= 1.003)
        face = PAL[extract(labels)] * brightness
        ix = np.clip((fu[inside] * n).astype(int), 0, n - 1)
        iy = np.clip((fv[inside] * n).astype(int), 0, n - 1)
        canvas[inside, :3] = face[iy, ix]
        canvas[inside, 3] = 1.0
    return canvas


def load_labels(path: Path, role: str) -> tuple[np.ndarray, list[int]]:
    """Argmax labels of every saved state; the float field is released afterwards."""
    with np.load(path, allow_pickle=False) as archive:
        steps = [int(s) for s in archive["save_steps"].tolist()]
        if tuple(steps) != STEPS:
            raise SourceConflict(f"{role}: saved steps differ from 0..24000 every 800")
        states = archive["states"]
    if states.shape != (len(STEPS), 8, 128, 128, 128):
        raise SourceConflict(f"{role}: unexpected state shape {states.shape}")
    labels = np.argmax(states, axis=1).astype(np.uint8)
    del states
    return labels, steps


def series(reference: np.ndarray, model: np.ndarray) -> dict:
    voxels = reference[0].size
    differing = [int((model[i] != reference[i]).sum()) for i in range(len(STEPS))]
    persistence = [int((reference[i] != reference[0]).sum()) for i in range(len(STEPS))]
    return {
        "differing_voxels": differing,
        "persistence_voxels": persistence,
        "disagreement_percent": [round(100.0 * v / voxels, 6) for v in differing],
        "persistence_percent": [round(100.0 * v / voxels, 6) for v in persistence],
        "reference_active": [int(np.unique(reference[i]).size) for i in range(len(STEPS))],
        "model_active": [int(np.unique(model[i]).size) for i in range(len(STEPS))],
    }


def check_against_record(case: dict, data: dict) -> None:
    """The recomputed trajectories must equal the accepted record, state by state."""
    trajectory = case["trajectory"]
    for name, mine, theirs in (
        ("disagreement", data["disagreement_percent"], trajectory["disagreement_percent"]),
        ("persistence", data["persistence_percent"], trajectory["persistence_disagreement_percent"]),
    ):
        worst = max(abs(a - b) for a, b in zip(mine, theirs))
        if worst > 1e-5:
            raise SourceConflict(f"{case['public_name']}: recomputed {name} differs from the record by {worst}")
    if data["reference_active"][-1] != case["reference_terminal_active_count"]:
        raise SourceConflict(f"{case['public_name']}: reference terminal active count differs")
    if data["model_active"][-1] != case["model_terminal_active_count"]:
        raise SourceConflict(f"{case['public_name']}: model terminal active count differs")


def outcome_line(case: dict) -> str:
    if case["all_criteria_met"]:
        return "all six criteria met"
    late = [e for e in case["extinction_events"] if not e["within_tolerance"]]
    parts = []
    if case["criteria"]["terminal_survivor_set"]:
        parts.append("exact survivor set")
    if late:
        worst = max(late, key=lambda e: abs(e["signed_residual_steps"]))
        parts.append(f"one extinction {abs(worst['signed_residual_steps']):,} steps "
                     f"{'late' if worst['signed_residual_steps'] > 0 else 'early'}")
    return " · ".join(parts)


def draw_strip(ax, data: dict, index: int) -> None:
    steps = list(STEPS)
    upto = slice(0, index + 1)
    # No shaded "training" region: these fields were never trained on. H is drawn
    # only as a reference mark for the model's training on other fields.
    ax.axvline(HORIZON, color=INK, linestyle=(0, (4, 3)), linewidth=1.0, zorder=2)
    ax.plot(steps[upto], data["persistence_percent"][upto], color=PERSISTENCE_GREY,
            linewidth=1.3, marker="o", markersize=3.0, markerfacecolor="white",
            label="static initial-state persistence", zorder=3)
    ax.plot(steps[upto], data["disagreement_percent"][upto], color=MODEL_ORANGE,
            linewidth=1.9, marker="o", markersize=2.6, label="PINN-Phase", zorder=4)
    ax.plot([steps[index]], [data["disagreement_percent"][index]], marker="D", markersize=8.0,
            markerfacecolor=MODEL_ORANGE, markeredgecolor=INK, markeredgewidth=1.1,
            linestyle="none", zorder=5, clip_on=False)
    ax.set_xlim(0, FINAL)
    ax.set_ylim(0, 36)
    ax.set_xticks([0, HORIZON, 12000, 24000])
    ax.set_xticklabels(["0", "H", "12k", "24k"], fontsize=8.5)
    ax.set_yticks([0, 10, 20, 30])
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_ylabel("label disagreement (%)", fontsize=8.5)
    ax.grid(alpha=0.18, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def draw_section(ax, reference: np.ndarray, model: np.ndarray) -> None:
    ref = reference[SECTION_Z]
    differ = model[SECTION_Z] != ref
    faint = ListedColormap([(v, v, v) for v in np.linspace(0.74, 0.95, 8)])
    ax.imshow(ref, cmap=faint, vmin=0, vmax=7, interpolation="nearest")
    ax.imshow(np.ma.masked_where(~differ, differ), cmap=ListedColormap([DISCREPANCY_RED]),
              vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def label_section(labels: np.ndarray) -> np.ndarray:
    """RGB image of the plane labels[SECTION_Z], oriented like the cube's right face.

    Rows run along the third axis (upwards with origin="lower") and columns along the
    second, as on the face drawn from labels[-1, :, :].T; grain boundaries are darkened.
    """
    plane = labels[SECTION_Z].T
    rgb = PAL[plane].copy()
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


def cube_axis(ax, rgba: np.ndarray, mark_section: bool = False) -> None:
    ax.imshow(rgba, extent=(UMIN, UMAX, VMIN, VMAX), origin="lower", interpolation="nearest", zorder=2)
    for start, end in VISIBLE_EDGES:
        p0, p1 = project(start), project(end)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], lw=0.8, color=INK, alpha=0.6, zorder=3,
                solid_capstyle="round")
    if mark_section:
        # Trace of the section plane on the two visible faces it crosses.
        s = (SECTION_Z + 0.5) / 128
        for start, end in (((s, 0, 0), (s, 0, 1)), ((s, 0, 1), (s, 1, 1))):
            p0, p1 = project(start), project(end)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], lw=1.6, color="white", zorder=4)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], lw=1.1, color=INK, linestyle=(0, (3, 2)), zorder=5)
    ax.set_xlim(UMIN - 0.03, UMAX + 0.03)
    ax.set_ylim(VMIN - 0.03, VMAX + 0.03)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
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


def draw_timeline(ax, step: int) -> None:
    """One uniform rollout timeline with a moving marker.

    The whole bar is autonomous rollout: these initial fields were never used in
    training. H is drawn only as a reference mark, the horizon the model was trained
    over on its four training fields; no part of this bar is a training interval.
    """
    ax.add_patch(matplotlib.patches.Rectangle((0, 0), FINAL, 1, facecolor=TAIL_TINT,
                                              edgecolor="#9aa3ab", linewidth=0.8))
    ax.plot([0, step], [-0.12, -0.12], color=MODEL_ORANGE, linewidth=3.0, solid_capstyle="butt",
            clip_on=False, zorder=4)
    ax.plot([HORIZON, HORIZON], [-0.25, 1.25], color=INK, linestyle=(0, (3, 2)), linewidth=1.2,
            clip_on=False)
    ax.plot([step], [0.5], marker="D", markersize=10, markerfacecolor=MODEL_ORANGE,
            markeredgecolor=INK, markeredgewidth=1.1, clip_on=False, zorder=5)
    ax.text(0, 1.45, "0", fontsize=9.5, ha="center", va="bottom", color=INK)
    ax.text(HORIZON, 1.45, f"H = {HORIZON:,}: model's training horizon on four other fields",
            fontsize=9.5, ha="left", va="bottom", color=INK, gid="chk:tl_h")
    ax.text(FINAL, 1.45, f"{FINAL:,} = {FINAL / HORIZON:.2f} H", fontsize=9.5, ha="right",
            va="bottom", color=INK)
    ax.text(FINAL / 2, -0.45,
            "autonomous rollout from each unseen initial field  ·  never used in training, "
            "no retraining or adaptation",
            fontsize=9.5, ha="center", va="top", color=INK, gid="chk:tl_rollout")
    ax.set_xlim(0, FINAL)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")


FIG_W = 13.6  # inches
FIG_H = 12.0  # inches


def frame(cases: list[dict], index: int) -> Image.Image:
    step = STEPS[index]
    figure = plt.figure(figsize=(FIG_W, FIG_H), dpi=100)

    def fy(inches_from_bottom: float) -> float:
        return inches_from_bottom / FIG_H

    def fx(inches_from_left: float) -> float:
        return inches_from_left / FIG_W

    status = "autonomous rollout from the unseen initial field"
    if step > 0:
        status += f"  ·  {step / HORIZON:.2f} H"
    figure.text(0.5, fy(FIG_H - 0.13), "Native 128³ transfer to unseen microstructures — one trained model, no retraining",
                fontsize=14.5, fontweight="bold", ha="center", va="top", gid="chk:title")
    figure.text(0.5, fy(FIG_H - 0.43), f"step {step:,} of {FINAL:,}  ·  {status}",
                fontsize=11, ha="center", va="top", color=INK, gid="chk:status")
    draw_timeline(figure.add_axes([0.06, fy(10.82), 0.88, fy(0.20)]), step)
    # Placed by hand so the clearances between row label, panel titles, panels and
    # captions are fixed and measurable. Each row block is 2.99 inches tall.
    row_top, block, gap = fy(10.21), fy(2.992), fy(0.154)
    panel_h = fy(1.98)
    for row, case in enumerate(cases):
        top = row_top - row * (block + gap)
        data = case["series"]
        figure.text(0.022, top, f"{case['public_name']}  ·  {case['outcome']}",
                    fontsize=11.5, fontweight="bold", ha="left", va="top", color=INK,
                    gid=f"chk:row{row}")
        panel_top = top - fy(0.704)
        bottom = panel_top - panel_h
        # Five columns, in inches: two steady terminal cubes, two animated sections of
        # the same plane, and the disagreement strip.
        panel_in = 1.98
        cube_in = panel_in * (UMAX - UMIN + 0.06) / (VMAX - VMIN + 0.06)
        x_cube1 = 0.20
        x_cube2 = x_cube1 + cube_in + 0.10
        x_sec1 = x_cube2 + cube_in + 0.40
        x_sec2 = x_sec1 + panel_in + 0.14
        x_strip = x_sec2 + panel_in + 0.80
        axes = [
            figure.add_axes([fx(x_cube1), bottom, fx(cube_in), panel_h]),
            figure.add_axes([fx(x_cube2), bottom, fx(cube_in), panel_h]),
            figure.add_axes([fx(x_sec1), bottom, fx(panel_in), panel_h]),
            figure.add_axes([fx(x_sec2), bottom, fx(panel_in), panel_h]),
        ]
        cube_axis(axes[0], case["reference_terminal_cube"], mark_section=True)
        cube_axis(axes[1], case["model_terminal_cube"], mark_section=True)
        section_axis(axes[2], label_section(case["reference"][index]))
        section_axis(axes[3], label_section(case["model"][index]))
        axes[0].set_title(f"Reference, step {FINAL:,}", fontsize=10, pad=4).set_gid(f"chk:ref{row}")
        axes[1].set_title(f"PINN-Phase, step {FINAL:,}", fontsize=10, pad=4).set_gid(f"chk:model{row}")
        axes[2].set_title(f"Reference section\nstep {step:,}", fontsize=10, pad=4).set_gid(f"chk:refsec{row}")
        caption = ("identical at step 0" if step == 0
                   else f"{data['disagreement_percent'][index]:.2f}% of volume differs")
        axes[3].set_title(f"PINN-Phase section\n{caption}", fontsize=10,
                          pad=4).set_gid(f"chk:section{row}")
        axes[3].text(0.5, -0.03,
                     f"active grains {data['model_active'][index]} (reference {data['reference_active'][index]})",
                     transform=axes[3].transAxes, fontsize=9, ha="center", va="top", color=INK,
                     gid=f"chk:active{row}")
        axes[0].text(1.0, -0.03, "final state (fixed); dashed line: section plane",
                     transform=axes[0].transAxes, fontsize=8.6, ha="center", va="top", color=INK,
                     gid=f"chk:cubenote{row}")
        strip = figure.add_axes([fx(x_strip), bottom + fy(0.242), fx(FIG_W - 0.25 - x_strip),
                                 panel_h - fy(0.11)])
        draw_strip(strip, data, index)
        strip.set_xlabel("step", fontsize=8.5, labelpad=1)
        if row == 0:
            strip.legend(loc="upper left", fontsize=8, frameon=False)
    footer = (
        "These microstructures were never used in training. H marks the horizon the model was trained over on "
        "four other fields; references are used for evaluation only.",
        "Blind cohort: initial fields and evaluation fixed before any outcome was opened. Shown: the two cases that "
        "meet all six criteria and the survivor-correct case",
        "with the largest timing error. All six cases, including one that retains an extra grain, are reported in "
        "benchmarks/n8_128_transfer.",
    )
    for offset, line in enumerate(footer):
        figure.text(0.5, fy(0.572 - 0.187 * offset), line, fontsize=8.6, ha="center", va="top", color=INK,
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


def font_identity() -> dict:
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    return {"font_file": path.name, "font_sha256": sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    for role in EXPECTED_SHA256:
        parser.add_argument(f"--{role.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/media/n8_128_transfer"))
    args = parser.parse_args()

    record = json.loads(RECORD.read_text(encoding="utf-8"))
    by_name = {c["public_name"]: c for c in record["unseen_cohort"]["cases"]}
    inputs = []
    cases = []
    for ordinal in SHOWN:
        case = by_name[f"Unseen microstructure {ordinal}"]
        labels = {}
        for side in ("reference", "model"):
            role = f"unseen_{ordinal}_{side}"
            path = getattr(args, role)
            found = sha256_file(path)
            if found != EXPECTED_SHA256[role]:
                raise SourceConflict(f"{role}: expected {EXPECTED_SHA256[role][:16]}..., found {found[:16]}...")
            if found != case[f"{side}_trajectory_sha256"]:
                raise SourceConflict(f"{role}: digest is not the one the accepted record scored")
            labels[side], _ = load_labels(path, role)
            inputs.append({"case": case["public_name"], "role": side, "sha256": found})
        data = series(labels["reference"], labels["model"])
        check_against_record(case, data)
        cases.append({
            "public_name": case["public_name"],
            "outcome": outcome_line(case),
            "reference": labels["reference"],
            "model": labels["model"],
            "reference_terminal_cube": cube_image(labels["reference"][-1]),
            "model_terminal_cube": cube_image(labels["model"][-1]),
            "series": data,
        })

    frames = [frame(cases, i) for i in range(len(STEPS))]
    delays = [900.0] * len(frames)
    delays[0], delays[-1] = 1800.0, 3600.0
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    gif, poster = output / GIF_NAME, output / POSTER_NAME
    imageio_v2.mimsave(gif, [np.asarray(f) for f in frames], duration=delays, loop=0)
    frames[-1].save(poster, format="PNG", optimize=True)

    manifest = {
        "schema": "pinn-phase-n8-128-media-v1",
        "license": "CC-BY-4.0",
        "alt_text": ALT_TEXT,
        "renderer_sha256": sha256_file(Path(__file__)),
        "record_sha256": sha256_file(RECORD),
        "parent_checkpoint_sha256": record["parent_checkpoint_sha256"],
        "cases_shown": [c["public_name"] for c in cases],
        "selection_rule": ("the two cases that meet all six criteria, and the survivor-correct case with the "
                           "largest out-of-tolerance timing residual; fixed before rendering"),
        "inputs": inputs,
        "camera": camera_metadata(128),
        "steps": list(STEPS),
        "represented_horizon_steps": HORIZON,
        "no_interpolation": True,
        "recomputed": {
            c["public_name"]: {k: c["series"][k] for k in (
                "disagreement_percent", "persistence_percent", "reference_active", "model_active")}
            for c in cases
        },
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
    (output / "asset_manifest.json").write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"Rendered {gif} ({gif.stat().st_size:,} bytes) and {poster}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
