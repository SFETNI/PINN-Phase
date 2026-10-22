#!/usr/bin/env python3
"""Render the phase-local transfer media from the released label and volume arrays.

One renderer, two items, both drawn from the same 2,160-parameter phase-local
model (trained once at 48^3 with 8 grains and applied unchanged):

* ``r128_n150``: an animation of the three 150-grain microstructures at the 13
  times at which reference labels were saved, t = 0 to t = 135. Every frame pairs
  the reference and the PINN-Phase rollout at the same saved time; nothing is
  interpolated and no fixed reference is shown against a moving model.
* ``r128_n38``: one still of the three 38-grain microstructures at the final time
  t = 67.5.

Layout: two rows (reference, PINN-Phase) by three columns (microstructure 1, 2,
3), each a whole opaque cube of argmax grain labels. Under each column, the
number of grains alive in the reference and in PINN-Phase, and the percentage of
voxels whose grain label differs. A grain is alive when its diffuse volume
exceeds 1e-6; the counts are read from the per-step volume tables, never from
label images, so they are the counts used by the scoring.

The arrays are release assets, not part of the repository. They are read from::

    <asset-root>/phaselocal_transfer/<regime>/microstructure_<k>/
        reference_labels.npz   labels uint8 (T, n, n, n), steps (T,)
        model_labels.npz       labels uint8 (T', n, n, n), steps (T',)
        reference_volumes.npz  volumes float64 (S, N), steps (S,)
        model_volumes.npz      volumes float64 (S', N), steps (S',)

Every number drawn is recomputed from these arrays at render time. The terminal
values must equal the published ones (``PUBLISHED``), and, when the score record
``benchmarks/phaselocal_transfer/expected_score.json`` is present, every value it
states for the rendered times; a mismatch stops the render.

Usage::

    python scripts/render_phaselocal_media.py --asset-root DIR --output media
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import io
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont, features

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "benchmarks/phaselocal_transfer/expected_score.json"
ASSET_MANIFEST = ROOT / "benchmarks/phaselocal_transfer/manifest.json"

DT = 0.01
ALIVE_THRESHOLD = 1e-6
CASES = ("microstructure 1", "microstructure 2", "microstructure 3")
ASSET_FILES = ("reference_labels.npz", "model_labels.npz", "reference_volumes.npz", "model_volumes.npz")

REGIMES = {
    "r128_n150": {
        "grid": 128,
        "grains": 150,
        "final_step": 13500,
        "reference_steps": (0, 200, 400, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 13500),
        "shown_steps": (0, 200, 400, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 13500),
        "folder": "phaselocal_n150",
        "outputs": ("phaselocal_n150_reference_vs_pinn_phase.gif",
                    "phaselocal_n150_reference_vs_pinn_phase_poster.png"),
        "title": "128\u00b3, 150 grains: one phase-local model, 2,160 parameters, trained at 48\u00b3 with 8 grains",
        "subtitle": "Unseen microstructures; autonomous rollout from the initial state; same weights for all three",
        "decimals": 2,
        "time_bar": True,
    },
    "r128_n38": {
        "grid": 128,
        "grains": 38,
        "final_step": 6750,
        "reference_steps": (0, 100, 200, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 6750),
        "shown_steps": (6750,),
        "folder": "phaselocal_n38",
        "outputs": ("phaselocal_n38_reference_vs_pinn_phase.png",),
        "title": "128\u00b3, 38 grains: one phase-local model, 2,160 parameters, trained at 48\u00b3 with 8 grains",
        "subtitle": "Unseen microstructures at the final time; autonomous rollout from the initial state; "
                    "same weights for all three",
        "decimals": 3,
        "time_bar": False,
    },
}

#: Terminal values stated in the article (whole-volume label disagreement in percent,
#: rounded as published, and grains alive in both reference and model).
PUBLISHED = {
    "r128_n150": {"disagreement_percent": (0.51, 0.70, 0.88), "decimals": 2, "alive": (24, 28, 30)},
    "r128_n38": {"disagreement_percent": (0.356, 0.334, 0.405), "decimals": 3, "alive": (35, 32, 35)},
}

ALT_TEXT = {
    "r128_n150": (
        "Animation of three unseen 128-cubed microstructures with 150 grains each, one per column, at the "
        "13 saved times from t = 0 to t = 135. The top row shows the phase-field reference and the bottom row "
        "the PINN-Phase rollout from the same initial state, with one phase-local model of 2,160 parameters "
        "trained at 48-cubed with 8 grains and used unchanged for all three. Under each column, counters give "
        "the number of grains alive in the reference and in PINN-Phase and the percentage of voxels whose grain "
        "label differs. At t = 135 the reference and PINN-Phase both keep {alive} grains, and {disagreement} "
        "percent of the voxels differ."
    ),
    "r128_n38": (
        "Three unseen 128-cubed microstructures with 38 grains each at the final time t = 67.5, one per column. "
        "The top row shows the phase-field reference and the bottom row the PINN-Phase rollout from the same "
        "initial state, with one phase-local model of 2,160 parameters trained at 48-cubed with 8 grains and "
        "used unchanged for all three. Under each column, counters give the number of grains alive in the "
        "reference and in PINN-Phase and the percentage of voxels whose grain label differs: {alive} grains "
        "alive in both, and {disagreement} percent of the voxels differ."
    ),
}

FRAME_MS = 700
FIRST_MS = 1400
LAST_MS = 2600

# ----------------------------------------------------------------------------- palette
# Identity-fixed: grain k always has colour k, in every panel and both regimes.
# The first 40 colours are matplotlib's tab20 followed by tab20b; the next 120 step
# through hue by the golden ratio, cycling saturation and value.
QUALITATIVE_HEX = (
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
    "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
    "#bcbd22", "#dbdb8d", "#17becf", "#9edae5", "#393b79", "#5254a3", "#6b6ecf", "#9c9ede",
    "#637939", "#8ca252", "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39", "#e7ba52", "#e7cb94",
    "#843c39", "#ad494a", "#d6616b", "#e7969c", "#7b4173", "#a55194", "#ce6dbd", "#de9ed6",
)
GOLDEN_HUE_STEP = round((5 ** 0.5 - 1) / 2, 15)  # golden-ratio conjugate, rounded to 15 decimals
EXTENSION_SATURATION = (0.5, 0.7, 0.9)
EXTENSION_VALUE = (0.65, 0.8, 0.95)
EXTENSION_COUNT = 120


def palette() -> np.ndarray:
    """(160, 3) float64 RGB, grain k -> row k."""
    rows = [tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)) for h in QUALITATIVE_HEX]
    for i in range(EXTENSION_COUNT):
        rows.append(colorsys.hsv_to_rgb((i * GOLDEN_HUE_STEP) % 1.0, EXTENSION_SATURATION[i % 3],
                                        EXTENSION_VALUE[(i // 3) % 3]))
    return np.array(rows, dtype=np.float64)


PAL = palette()

# ----------------------------------------------------------------------------- camera
# Orthographic view of the three faces x = n, y = n and z = n, each with a flat shade;
# grain boundaries are darkened on each face and the cube edges drawn in dark grey.
# The image is rasterised at twice the output resolution by nearest-voxel lookup and
# box-averaged, which softens edge pixels only and never changes which voxel owns a
# pixel centre.
CAMERA_ELEV = 24.0
CAMERA_AZIM = 38.0
VISIBLE_FACES = ("x=n", "y=n", "z=n")
FACE_SHADE = {"x=n": 0.84, "y=n": 0.70, "z=n": 1.00}
BOUNDARY_WIDTH_FRAC = 0.0045
BOUNDARY_DARKEN = 0.42
EDGE_WIDTH_FRAC = 0.0045
EDGE_RGB = (0.22, 0.22, 0.22)
SUPERSAMPLE = 2
CUBE_WIDTH_PX = 236.8  # projected cube width in output pixels, whatever the grid (1.85 px per voxel at 128)


def _basis() -> tuple[np.ndarray, np.ndarray]:
    elev, azim = np.deg2rad(CAMERA_ELEV), np.deg2rad(CAMERA_AZIM)
    right = np.array([-np.sin(azim), np.cos(azim), 0.0])
    up = np.array([-np.sin(elev) * np.cos(azim), -np.sin(elev) * np.sin(azim), np.cos(elev)])
    return right, up


RIGHT, UP = _basis()


def project(points) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    return np.stack([p @ RIGHT, p @ UP], axis=-1)


def camera_metadata(n: int) -> dict:
    """Public record of the cube view implemented in ``cube_image``."""
    return {
        "projection": "orthographic view of the three outer faces x = n, y = n, z = n; whole opaque cube",
        "camera_elevation_deg": CAMERA_ELEV,
        "camera_azimuth_deg": CAMERA_AZIM,
        "faces": [f"{face} at brightness {FACE_SHADE[face]:.2f}" for face in VISIBLE_FACES],
        "grain_boundaries": f"darkened by {BOUNDARY_DARKEN}, width {BOUNDARY_WIDTH_FRAC} of the cube image width",
        "cube_edges": f"rgb {list(EDGE_RGB)}, width {EDGE_WIDTH_FRAC} of the cube image width",
        "supersample": SUPERSAMPLE,
        "lookup": "nearest voxel; no smoothing or interpolation of grain geometry",
        "palette": ("identity-fixed, grain k -> colour k: tab20 then tab20b (40 colours), then 120 colours "
                    "stepping hue by the golden ratio with saturation (0.5, 0.7, 0.9) and value "
                    "(0.65, 0.8, 0.95)"),
        "palette_sha256": hashlib.sha256(PAL.tobytes()).hexdigest(),
        "grid": [n, n, n],
        "cube_width_px": CUBE_WIDTH_PX,
    }


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    out = mask.copy()
    rows, cols = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(mask)
            ys, ye = max(dy, 0), rows + min(dy, 0)
            xs, xe = max(dx, 0), cols + min(dx, 0)
            shifted[ys:ye, xs:xe] = mask[ys - dy:ye - dy, xs - dx:xe - dx]
            out |= shifted
    return out


def _transitions(values: np.ndarray, same: np.ndarray | None = None) -> np.ndarray:
    """Pixels on either side of a change of ``values`` between 4-neighbours."""
    right = values[:, 1:] != values[:, :-1]
    down = values[1:, :] != values[:-1, :]
    if same is not None:
        right &= same[0]
        down &= same[1]
    mask = np.zeros(values.shape, dtype=bool)
    mask[:, 1:] |= right
    mask[:, :-1] |= right
    mask[1:, :] |= down
    mask[:-1, :] |= down
    return mask


def cube_image(labels: np.ndarray, width_px: float = CUBE_WIDTH_PX) -> np.ndarray:
    """RGBA float32 image of one whole opaque cube of grain labels, transparent outside."""
    labels = np.asarray(labels)
    if labels.ndim != 3 or len(set(labels.shape)) != 1:
        raise ValueError("a cubic (n, n, n) label array is expected")
    if not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or labels.max() >= len(PAL):
        raise ValueError("labels outside the palette")
    n = labels.shape[0]
    corners = project([[i, j, k] for i in (0, n) for j in (0, n) for k in (0, n)])
    xmin, xmax = corners[:, 0].min(), corners[:, 0].max()
    ymin, ymax = corners[:, 1].min(), corners[:, 1].max()
    scale = (width_px / n) * SUPERSAMPLE * n / (xmax - xmin)
    width = int(np.ceil((xmax - xmin) * scale)) + 1
    height = int(np.ceil((ymax - ymin) * scale)) + 1
    sx, sy = np.meshgrid((np.arange(width) + 0.5) / scale + xmin, ymax - (np.arange(height) + 0.5) / scale)

    planes = {"x=n": labels[n - 1, :, :], "y=n": labels[:, n - 1, :], "z=n": labels[:, :, n - 1]}
    spans = {
        "x=n": ((n, 0, 0), (0, 1, 0), (0, 0, 1)),
        "y=n": ((0, n, 0), (1, 0, 0), (0, 0, 1)),
        "z=n": ((0, 0, n), (1, 0, 0), (0, 1, 0)),
    }
    label_map = np.full((height, width), -1, np.int32)
    face_map = np.full((height, width), -1, np.int8)
    for index, name in enumerate(VISIBLE_FACES):
        origin, eu, ev = (np.array(v, dtype=float) for v in spans[name])
        s0, a, b = project(origin), project(eu), project(ev)
        inverse = np.linalg.inv(np.array([[a[0], b[0]], [a[1], b[1]]]))
        dx, dy = sx - s0[0], sy - s0[1]
        u = inverse[0, 0] * dx + inverse[0, 1] * dy
        v = inverse[1, 0] * dx + inverse[1, 1] * dy
        inside = (u >= 0) & (u < n) & (v >= 0) & (v < n) & (face_map < 0)
        ui = np.clip(np.floor(u).astype(int), 0, n - 1)
        vi = np.clip(np.floor(v).astype(int), 0, n - 1)
        label_map[inside] = planes[name][ui, vi][inside]
        face_map[inside] = index

    inside = face_map >= 0
    shade = np.array([FACE_SHADE[f] for f in VISIBLE_FACES], np.float32)
    rgb = np.zeros((height, width, 3), np.float32)
    rgb[inside] = PAL[label_map[inside]] * shade[face_map[inside]][:, None]
    alpha = inside.astype(np.float32)

    same_face = (face_map[:, 1:] == face_map[:, :-1]) & (face_map[:, 1:] >= 0), \
                (face_map[1:, :] == face_map[:-1, :]) & (face_map[1:, :] >= 0)
    boundary = _transitions(label_map, same_face)
    boundary = _dilate(boundary, int(round(max(BOUNDARY_WIDTH_FRAC * width - 1, 0) / 2))) & inside
    rgb[boundary] *= BOUNDARY_DARKEN
    edge = _dilate(_transitions(face_map), int(round(max(EDGE_WIDTH_FRAC * width - 1, 0) / 2)))
    rgb[edge] = np.array(EDGE_RGB, np.float32)
    alpha[edge] = 1.0

    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    hc, wc = (height // SUPERSAMPLE) * SUPERSAMPLE, (width // SUPERSAMPLE) * SUPERSAMPLE
    blocks = rgba[:hc, :wc].reshape(hc // SUPERSAMPLE, SUPERSAMPLE, wc // SUPERSAMPLE, SUPERSAMPLE, 4)
    weight = blocks[..., 3:4]
    premultiplied = (blocks[..., :3] * weight).mean(axis=(1, 3))
    coverage = weight.mean(axis=(1, 3))
    colour = np.where(coverage > 0, premultiplied / np.maximum(coverage, 1e-9), 0.0)
    return np.concatenate([colour, coverage], axis=-1).astype(np.float32)


def _on_white(rgba: np.ndarray) -> Image.Image:
    a = rgba[..., 3:4]
    rgb = rgba[..., :3] * a + (1 - a)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")


# ----------------------------------------------------------------------------- data
class SourceConflict(RuntimeError):
    """An input or a recomputed value did not match its stated identity or value."""


class UnreadableFrame(RuntimeError):
    """A text item fell off the canvas or onto another item."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def alive(volumes_row) -> int:
    return int((np.asarray(volumes_row) > ALIVE_THRESHOLD).sum())


def series(reference: np.ndarray, model: np.ndarray, reference_volumes: np.ndarray,
           model_volumes: np.ndarray, steps) -> dict:
    """Per saved time: differing voxels, label disagreement and grains alive.

    ``reference`` and ``model`` are (T, n, n, n) labels at ``steps``; the two volume
    arrays are (T, N) rows of the per-step volume tables at the same steps.
    """
    voxels = reference[0].size
    differing = [int((model[i] != reference[i]).sum()) for i in range(len(steps))]
    return {
        "steps": [int(s) for s in steps],
        "t": [round(int(s) * DT, 4) for s in steps],
        "differing_voxels": differing,
        "disagreement_percent": [round(100.0 * d / voxels, 6) for d in differing],
        "reference_alive": [alive(row) for row in reference_volumes],
        "model_alive": [alive(row) for row in model_volumes],
    }


def _rows_at(table: dict, steps, role: str) -> np.ndarray:
    index = {int(s): i for i, s in enumerate(table["steps"].tolist())}
    missing = [s for s in steps if s not in index]
    if missing:
        raise SourceConflict(f"{role}: no row for steps {missing}")
    return table["data"][[index[s] for s in steps]]


def load_case(case_dir: Path, spec: dict, role: str) -> dict:
    """Labels and volume rows of one microstructure at the reference steps."""
    steps = list(spec["reference_steps"])
    with np.load(case_dir / "reference_labels.npz", allow_pickle=False) as z:
        ref_steps = [int(s) for s in z["steps"].tolist()]
        if ref_steps != steps:
            raise SourceConflict(f"{role}: reference labels saved at {ref_steps}, expected {steps}")
        reference = np.ascontiguousarray(z["labels"])
    with np.load(case_dir / "model_labels.npz", allow_pickle=False) as z:
        model_steps = [int(s) for s in z["steps"].tolist()]
        labels = z["labels"]
        index = {s: i for i, s in enumerate(model_steps)}
        missing = [s for s in steps if s not in index and s != 0]
        if missing:
            raise SourceConflict(f"{role}: model labels missing at steps {missing}")
        rows = [labels[index[s]] if s in index else None for s in steps]
        del labels
    tables = {}
    for side in ("reference", "model"):
        with np.load(case_dir / f"{side}_volumes.npz", allow_pickle=False) as z:
            tables[side] = {"steps": z["steps"], "data": np.asarray(z["volumes"], dtype=np.float64)}
        if tables[side]["data"].shape[1] != spec["grains"]:
            raise SourceConflict(f"{role}: {side} volume table has {tables[side]['data'].shape[1]} grains")
    reference_volumes = _rows_at(tables["reference"], steps, f"{role} reference volumes")
    model_volumes = _rows_at(tables["model"], steps, f"{role} model volumes")
    if rows[0] is None:
        # The rollout starts from the reference initial state, so its t = 0 labels are
        # the reference t = 0 labels. The volume tables confirm the shared start.
        if not np.array_equal(reference_volumes[0], model_volumes[0]):
            raise SourceConflict(f"{role}: model and reference volumes differ at step 0")
        rows[0] = reference[0]
    model = np.stack(rows)
    n = spec["grid"]
    for name, array in (("reference", reference), ("model", model)):
        if array.shape != (len(steps), n, n, n) or array.dtype != np.uint8:
            raise SourceConflict(f"{role}: {name} labels have shape {array.shape} and dtype {array.dtype}")
    return {"reference": reference, "model": model, "reference_volumes": reference_volumes,
            "model_volumes": model_volumes, "steps": steps}


# ----------------------------------------------------------------------------- record checks
def _record_cases(record: dict, regime: str) -> dict:
    block = record.get(regime)
    if not isinstance(block, dict):
        raise SourceConflict(f"the score record has no block for {regime}")
    return {case["public_name"]: case for case in block.get("cases", [])}


def check_against_record(record: dict, regime: str, data: dict) -> list[str]:
    """Every value the score record states for a rendered quantity must equal the recomputed one.

    ``data`` maps the public case name to its ``series``. Compared wherever the record
    and the series share a step: differing voxels and label disagreement (from the
    record's ``label_disagreement``), grains alive in reference and model (``alive``),
    and the extinction counts. Returns the list of checks made.
    """
    named = _record_cases(record, regime)
    grains = REGIMES[regime]["grains"]
    checks = []
    for name, s in data.items():
        case = named.get(name)
        if case is None:
            raise SourceConflict(f"{regime}: the score record has no case {name!r}")
        if s["steps"][-1] != REGIMES[regime]["final_step"]:
            raise SourceConflict(f"{regime} {name}: series ends at step {s['steps'][-1]}")
        at = {step: i for i, step in enumerate(s["steps"])}
        pairs = [("reference_extinction_count", case.get("reference_extinction_count"),
                  grains - s["reference_alive"][-1], 0),
                 ("model_extinction_count", case.get("model_extinction_count"), grains - s["model_alive"][-1], 0)]
        for entry in case.get("label_disagreement", []):
            if entry["step"] in at:
                i = at[entry["step"]]
                pairs.append((f"differing voxels at step {entry['step']}", entry["disagreeing_voxels"],
                              s["differing_voxels"][i], 0))
                pairs.append((f"label disagreement at step {entry['step']}", entry["percent"],
                              s["disagreement_percent"][i], 1e-5))
        for entry in case.get("alive", []):
            if entry["step"] in at:
                i = at[entry["step"]]
                pairs.append((f"reference alive at step {entry['step']}", entry["reference_alive_count"],
                              s["reference_alive"][i], 0))
                pairs.append((f"model alive at step {entry['step']}", entry["model_alive_count"],
                              s["model_alive"][i], 0))
        for key, theirs, mine, tolerance in pairs:
            if theirs is None:
                continue
            if abs(float(theirs) - float(mine)) > tolerance:
                raise SourceConflict(f"{regime} {name}: {key} is {mine} recomputed, {theirs} in the record")
            checks.append(f"{name}: {key}")
    return checks


def check_against_published(regime: str, data: dict) -> None:
    published = PUBLISHED[regime]
    for k, name in enumerate(CASES):
        s = data[name]
        shown = round(s["disagreement_percent"][-1], published["decimals"])
        if abs(shown - published["disagreement_percent"][k]) > 1e-9:
            raise SourceConflict(f"{regime} {name}: terminal disagreement {shown} differs from the published "
                                 f"{published['disagreement_percent'][k]}")
        if not s["reference_alive"][-1] == s["model_alive"][-1] == published["alive"][k]:
            raise SourceConflict(f"{regime} {name}: terminal alive counts {s['reference_alive'][-1]} / "
                                 f"{s['model_alive'][-1]} differ from the published {published['alive'][k]}")


# ----------------------------------------------------------------------------- frame
INK = (34, 34, 34)
SOFT = (85, 85, 85)
BAR_TRACK = (214, 214, 214)
BAR_FILL = (70, 70, 70)
LEFT = 128
RIGHT_MARGIN = 14
GAP = 28
TOP = 150
ROW_GAP = 34
MARGIN = 16
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def _fmt_t(step: int) -> str:
    return f"{step * DT:g}"


def frame(regime: str, cases: list[dict], index: int) -> Image.Image:
    """One frame: reference row over PINN-Phase row, one column per microstructure.

    ``cases`` hold ``public_name``, ``reference`` and ``model`` labels (T, n, n, n) at
    ``series["steps"]``, and ``series``; ``index`` selects the saved time.
    """
    spec = REGIMES[regime]
    step = cases[0]["series"]["steps"][index]
    cubes = [(_on_white(cube_image(c["reference"][index])), _on_white(cube_image(c["model"][index])))
             for c in cases]
    cw, ch = cubes[0][0].size
    width = LEFT + 3 * cw + 2 * GAP + RIGHT_MARGIN
    counters_top = TOP + 2 * ch + ROW_GAP + 14
    height = counters_top + 3 * 20 + 18
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    items = []  # (label, bbox) of every text item and every cube, for the readability check

    def text(xy, content, font, fill=INK, anchor="la"):
        draw.text(xy, content, font=font, fill=fill, anchor=anchor)
        items.append((content, draw.textbbox(xy, content, font=font, anchor=anchor)))

    title_size = 17
    while draw.textlength(spec["title"], font=_font(title_size, True)) > width - 2 * MARGIN and title_size > 13:
        title_size -= 1
    text((MARGIN, 12), spec["title"], _font(title_size, True))
    text((MARGIN, 40), spec["subtitle"], _font(14), SOFT)
    final = spec["final_step"]
    text((MARGIN, 72), f"t = {_fmt_t(step)}", _font(20, True))
    if spec["time_bar"]:
        x0, x1, y = 150, width - 34, 84
        draw.rounded_rectangle((x0, y - 2, x1, y + 2), radius=2, fill=BAR_TRACK)
        xc = x0 + (x1 - x0) * step / final
        if step > 0:
            draw.rounded_rectangle((x0, y - 2, xc, y + 2), radius=2, fill=BAR_FILL)
        for s in spec["shown_steps"]:
            xs = x0 + (x1 - x0) * s / final
            draw.line((xs, y - 6, xs, y + 6), fill=BAR_FILL if s <= step else BAR_TRACK, width=1)
        draw.ellipse((xc - 6, y - 6, xc + 6, y + 6), fill=BAR_FILL, outline="white", width=2)
        text((x0, y + 12), "t = 0", _font(12), SOFT, "ma")
        text((x1, y + 12), f"t = {_fmt_t(final)}", _font(12), SOFT, "ma")
        text(((x0 + x1) / 2, y + 12), "saved reference times, no interpolation", _font(12), SOFT, "ma")
    else:
        text((150, 76), "final time of the rollout", _font(14), SOFT)

    for column, (case, (reference_cube, model_cube)) in enumerate(zip(cases, cubes)):
        s = case["series"]
        x = LEFT + column * (cw + GAP)
        centre = x + cw // 2
        text((centre, TOP - 8), case["public_name"], _font(15, True), INK, "md")
        image.paste(reference_cube, (x, TOP))
        image.paste(model_cube, (x, TOP + ch + ROW_GAP))
        items.append((f"cube {column} reference", (x, TOP, x + cw, TOP + ch)))
        items.append((f"cube {column} model", (x, TOP + ch + ROW_GAP, x + cw, TOP + 2 * ch + ROW_GAP)))
        y = counters_top
        text((centre, y), "grains alive", _font(13), SOFT, "ma")
        text((centre, y + 19), f"reference {s['reference_alive'][index]}  \u00b7  "
             f"PINN-Phase {s['model_alive'][index]}", _font(14), INK, "ma")
        text((centre, y + 40), f"label disagreement {s['disagreement_percent'][index]:.{spec['decimals']}f} %",
             _font(14), INK, "ma")
    text((MARGIN, TOP + ch // 2), "Reference", _font(15, True), INK, "lm")
    text((MARGIN, TOP + ch + ROW_GAP + ch // 2), "PINN-Phase", _font(15, True), INK, "lm")

    for label, (a0, b0, a1, b1) in items:
        if min(a0, b0) < 0 or a1 > width or b1 > height:
            raise UnreadableFrame(f"t = {_fmt_t(step)}: {label!r} falls outside the canvas")
    for first in range(len(items)):
        for second in range(first + 1, len(items)):
            (la, a), (lb, b) = items[first], items[second]
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                raise UnreadableFrame(f"t = {_fmt_t(step)}: {la!r} overlaps {lb!r}")
    return image


def pixel_sha256(image: Image.Image) -> str:
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


# ----------------------------------------------------------------------------- output
def font_identity() -> dict:
    path = FONT_DIR / "DejaVuSans.ttf"
    bold = FONT_DIR / "DejaVuSans-Bold.ttf"
    return {"font_file": path.name, "font_sha256": sha256_file(path), "bold_font_sha256": sha256_file(bold)}


def renderer_environment() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "pillow": PIL.__version__,
        "freetype": features.version("freetype2"),
        "text_rasterizer": "Pillow ImageDraw with FreeType",
        **font_identity(),
    }


def _save_gif(frames: list[Image.Image], path: Path) -> list[int]:
    delays = [FRAME_MS] * len(frames)
    delays[0], delays[-1] = FIRST_MS, LAST_MS
    quantized = [f.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    quantized[0].save(path, save_all=True, append_images=quantized[1:], duration=delays, loop=0, optimize=True)
    return delays


def render_regime(regime: str, asset_root: Path, output: Path, record: dict | None) -> Path:
    spec = REGIMES[regime]
    inputs, cases, data = [], [], {}
    for k, name in enumerate(CASES, start=1):
        relative = f"phaselocal_transfer/{regime}/microstructure_{k}"
        case_dir = asset_root / relative
        for filename in ASSET_FILES:
            path = case_dir / filename
            if not path.is_file():
                raise SourceConflict(f"missing release asset {relative}/{filename}")
            inputs.append({"case": name, "role": filename[:-4], "path": f"{relative}/{filename}",
                           "sha256": sha256_file(path)})
        loaded = load_case(case_dir, spec, f"{regime} {name}")
        s = series(loaded["reference"], loaded["model"], loaded["reference_volumes"],
                   loaded["model_volumes"], loaded["steps"])
        shown = [loaded["steps"].index(step) for step in spec["shown_steps"]]
        cases.append({"public_name": name, "reference": loaded["reference"][shown],
                      "model": loaded["model"][shown], "series": {key: [v[i] for i in shown] for key, v in s.items()}})
        data[name] = s
        del loaded
    check_against_published(regime, data)
    record_checks = check_against_record(record, regime, data) if record is not None else []
    if record is not None and not record_checks:
        raise SourceConflict(f"{regime}: the score record states none of the rendered values")
    _check_asset_manifest(inputs)

    folder = output / spec["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    frames = [frame(regime, cases, i) for i in range(len(spec["shown_steps"]))]
    manifest = {
        "schema": "pinn-phase-phaselocal-media-v1",
        "license": "CC-BY-4.0",
        "regime": regime,
        "alt_text": ALT_TEXT[regime].format(
            alive=", ".join(str(data[n]["model_alive"][-1]) for n in CASES[:-1]) + f" and {data[CASES[-1]]['model_alive'][-1]}",
            disagreement=", ".join(f"{data[n]['disagreement_percent'][-1]:.{spec['decimals']}f}" for n in CASES[:-1])
            + f" and {data[CASES[-1]]['disagreement_percent'][-1]:.{spec['decimals']}f}"),
        "renderer_sha256": sha256_file(Path(__file__)),
        "record_sha256": sha256_file(RECORD) if record is not None else None,
        "record_checks": record_checks,
        "cases_shown": list(CASES),
        "selection_rule": "all three microstructures of the regime, in their published order; none is selected",
        "inputs": inputs,
        "camera": camera_metadata(spec["grid"]),
        "alive_rule": f"a grain is alive when its diffuse volume exceeds {ALIVE_THRESHOLD:g}, read from the "
                      "per-step volume tables",
        "time_per_step": DT,
        "steps": list(spec["shown_steps"]),
        "t": [round(s * DT, 4) for s in spec["shown_steps"]],
        "no_interpolation": True,
        "pairing": "every panel pair shows reference and PINN-Phase labels saved at the same step",
        "recomputed": data,
        "frame_pixel_sha256": [pixel_sha256(f) for f in frames],
        "renderer_environment": renderer_environment(),
        "frames": len(frames),
        "size": list(frames[0].size),
    }
    if regime == "r128_n150":
        gif, poster = (folder / name for name in spec["outputs"])
        delays = _save_gif(frames, gif)
        frames[-1].save(poster, format="PNG", optimize=True)
        manifest["frame_durations_ms"] = delays
        manifest["loop_seconds"] = round(sum(delays) / 1000.0, 1)
        written = (gif, poster)
    else:
        still = folder / spec["outputs"][0]
        frames[0].save(still, format="PNG", optimize=True)
        written = (still,)
    manifest["outputs"] = {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in written}
    (folder / "asset_manifest.json").write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    for path in written:
        print(f"Rendered {path} ({path.stat().st_size:,} bytes)")
    return folder


def _check_asset_manifest(inputs: list[dict]) -> None:
    """Bind every rendered input to the benchmark release-asset manifest."""
    if not ASSET_MANIFEST.is_file():
        raise SourceConflict("the benchmark asset manifest is missing")
    listed = {}
    for entry in json.loads(ASSET_MANIFEST.read_text(encoding="utf-8")).get("assets", []):
        location = entry.get("expected_relative_location")
        if isinstance(location, str):
            if location in listed:
                raise SourceConflict(f"duplicate benchmark asset {location}")
            listed[location] = entry.get("sha256")
    for item in inputs:
        if item["path"] not in listed:
            raise SourceConflict(f"{item['path']}: missing from the benchmark asset manifest")
        if listed[item["path"]] != item["sha256"]:
            raise SourceConflict(f"{item['path']}: digest differs from the benchmark asset manifest")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--regime", choices=(*REGIMES, "all"), default="all")
    parser.add_argument("--output", type=Path, default=Path("outputs/media"))
    args = parser.parse_args()
    record = json.loads(RECORD.read_text(encoding="utf-8")) if RECORD.is_file() else None
    if record is None:
        print(f"note: {RECORD.relative_to(ROOT).as_posix()} is absent; values are checked against the "
              "published terminal values only")
    for regime in (REGIMES if args.regime == "all" else (args.regime,)):
        render_regime(regime, args.asset_root, args.output, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
