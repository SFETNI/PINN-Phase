#!/usr/bin/env python3
"""Render the released 1,000- and 5,000-grain population media.

Every animation frame pairs labels saved at the same step. Counters are derived
from the released per-step volume tables using the published alive threshold.
The renderer checks all displayed values against the compact score record and
the article's terminal values before writing an output.
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import importlib.util
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont, features

ROOT = Path(__file__).resolve().parents[1]
SCORE = ROOT / "benchmarks/population_scaling/expected_score.json"
ASSET_MANIFEST = ROOT / "benchmarks/population_scaling/manifest.json"
LEGACY_PATH = Path(__file__).with_name("render_phaselocal_media.py")
_legacy_spec = importlib.util.spec_from_file_location("_phaselocal_media", LEGACY_PATH)
if _legacy_spec is None or _legacy_spec.loader is None:
    raise RuntimeError("cannot load the phase-local media renderer")
legacy = importlib.util.module_from_spec(_legacy_spec)
_legacy_spec.loader.exec_module(legacy)

DT = 0.01
ALIVE_THRESHOLD = 1e-6
FRAME_MS = 230
FIRST_MS = 900
LAST_MS = 1800
INK = (34, 34, 34)
SOFT = (85, 85, 85)
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
LABEL_ROLES = ("reference", "model")

REGIMES = {
    "r240_n1000": {
        "grid": 240,
        "grains": 1000,
        "cases": ("microstructure 1", "microstructure 2", "microstructure 3"),
        "final_step": 13500,
        "layout": "two rows by three columns",
        "cube_width": 140.0,
        "gif": "r240_n1000_reference_vs_pinn_phase.gif",
        "poster": "r240_n1000_reference_vs_pinn_phase_poster.png",
        "title": "240\u00b3, 1,000 grains: reference and PINN-Phase",
        "published_disagreement": (0.92, 0.82, 0.70),
        "published_alive": ((196, 195), (192, 192), (180, 181)),
    },
    "r416_n5000": {
        "grid": 416,
        "grains": 5000,
        "cases": ("microstructure 1",),
        "final_step": 5000,
        "layout": "reference and PINN-Phase side by side",
        "cube_width": 245.0,
        "gif": "r416_n5000_reference_vs_pinn_phase.gif",
        "poster": "r416_n5000_reference_vs_pinn_phase_poster.png",
        "title": "416\u00b3, 5,000 grains: reference and PINN-Phase",
        "published_disagreement": (0.35,),
        "published_alive": ((2889, 2891),),
    },
}


class SourceConflict(RuntimeError):
    """A released array disagrees with its record or published value."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def palette(count: int = 5000) -> np.ndarray:
    """Return identity-fixed RGB colours without reducing labels to 8 bits."""
    rows8 = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in legacy.QUALITATIVE_HEX]
    used = set(rows8)
    base_count = len(rows8)
    golden = (5**0.5 - 1) / 2
    for index in range(base_count, count):
        level = index - base_count
        saturation = (0.52, 0.66, 0.80, 0.94)[level % 4]
        value = (0.62, 0.74, 0.86, 0.98)[(level // 4) % 4]
        rgb_triplet = tuple(int(round(value * 255)) for value in colorsys.hsv_to_rgb(
            (level * golden) % 1.0, saturation, value))
        rgb_integer = (rgb_triplet[0] << 16) | (rgb_triplet[1] << 8) | rgb_triplet[2]
        while rgb_triplet in used:
            rgb_integer = (rgb_integer + 1) % (1 << 24)
            rgb_triplet = (rgb_integer >> 16, (rgb_integer >> 8) & 255, rgb_integer & 255)
        rows8.append(rgb_triplet)
        used.add(rgb_triplet)
    result = np.asarray(rows8[:count], dtype=np.float64) / 255.0
    if len(used) != count:
        raise SourceConflict("the identity palette contains duplicate colours")
    return result


PAL = palette()
PALETTE_SHA256 = hashlib.sha256(PAL.tobytes()).hexdigest()
legacy.PAL = PAL


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


def _label_path(case_dir: Path, side: str, step: int) -> Path:
    return case_dir / side / "labels" / f"step_{step:05d}.npz"


def _load_label(path: Path, step: int, grid: int, grains: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        recorded_step = int(archive["step"])
        labels = np.asarray(archive["labels"])
    if recorded_step != step or labels.shape != (grid, grid, grid):
        raise SourceConflict(f"{path}: step or shape differs from the expected value")
    if not np.issubdtype(labels.dtype, np.integer) or int(labels.min()) < 0 or int(labels.max()) >= grains:
        raise SourceConflict(f"{path}: labels are outside 0 to {grains - 1}")
    return labels


def _saved_steps(case_dir: Path) -> list[int]:
    sides = []
    for side in LABEL_ROLES:
        steps = [int(path.stem.removeprefix("step_")) for path in sorted((case_dir / side / "labels").glob("*.npz"))]
        sides.append(steps)
    if not sides[0] or sides[0] != sides[1]:
        raise SourceConflict(f"{case_dir}: reference and model saved steps differ")
    return sides[0]


def _load_volumes(case_dir: Path, side: str, grains: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(case_dir / side / "volumes.npz", allow_pickle=False) as archive:
        steps = np.asarray(archive["steps"], dtype=np.int64)
        volumes = np.asarray(archive["volumes"], dtype=np.float64)
    if volumes.shape != (len(steps), grains) or not np.array_equal(steps, np.arange(len(steps))):
        raise SourceConflict(f"{case_dir / side / 'volumes.npz'}: unexpected volume table")
    return steps, volumes


def _score_cases(score: dict, regime: str) -> dict[str, dict]:
    return {case["public_name"]: case for case in score[regime]["cases"]}


def _case_series(asset_root: Path, regime: str, case_index: int, score_case: dict) -> tuple[dict, list[dict]]:
    spec = REGIMES[regime]
    case_dir = asset_root / "population_scaling" / regime / f"microstructure_{case_index}"
    steps = _saved_steps(case_dir)
    if steps[-1] != spec["final_step"]:
        raise SourceConflict(f"{regime} microstructure {case_index}: horizon is step {steps[-1]}")
    _, reference_volumes = _load_volumes(case_dir, "reference", spec["grains"])
    _, model_volumes = _load_volumes(case_dir, "model", spec["grains"])
    extinction_steps = {}
    for side in LABEL_ROLES:
        record = json.loads((case_dir / side / "extinctions.json").read_text(encoding="utf-8"))
        if (record.get("alive_threshold") != ALIVE_THRESHOLD or record.get("grains") != spec["grains"]
                or record.get("steps") != spec["final_step"] or record.get("returns")):
            raise SourceConflict(f"{regime} microstructure {case_index}: {side} extinction metadata differs")
        extinction_steps[side] = [int(entry["step"]) for entry in record["extinctions"]]
    result = {"steps": steps, "t": [round(step * DT, 4) for step in steps], "differing_voxels": [],
              "disagreement_percent": [], "reference_alive": [], "model_alive": []}
    inputs = []
    for side in LABEL_ROLES:
        for fixed in ("volumes.npz", "extinctions.json"):
            path = case_dir / side / fixed
            inputs.append({"case": f"microstructure {case_index}", "role": f"{side}_{fixed.rsplit('.', 1)[0]}",
                           "path": path.relative_to(asset_root).as_posix(), "sha256": sha256_file(path)})
    expected_fraction = {int(step): float(value) for step, value in score_case["label_disagreement_fraction_by_step"].items()}
    voxels = spec["grid"] ** 3
    for step in steps:
        reference_path = _label_path(case_dir, "reference", step)
        model_path = _label_path(case_dir, "model", step)
        reference = _load_label(reference_path, step, spec["grid"], spec["grains"])
        model = _load_label(model_path, step, spec["grid"], spec["grains"])
        differing = int(np.count_nonzero(reference != model))
        fraction = differing / voxels
        if step not in expected_fraction or abs(fraction - expected_fraction[step]) > 0.5 / voxels:
            raise SourceConflict(f"{regime} microstructure {case_index}: disagreement at step {step} differs from score")
        result["differing_voxels"].append(differing)
        result["disagreement_percent"].append(round(100 * fraction, 6))
        reference_alive = int(np.count_nonzero(reference_volumes[step] > ALIVE_THRESHOLD))
        model_alive = int(np.count_nonzero(model_volumes[step] > ALIVE_THRESHOLD))
        extinction_alive = (spec["grains"] - sum(value <= step for value in extinction_steps["reference"]),
                            spec["grains"] - sum(value <= step for value in extinction_steps["model"]))
        if (reference_alive, model_alive) != extinction_alive:
            raise SourceConflict(f"{regime} microstructure {case_index}: alive count at step {step} differs from extinction records")
        result["reference_alive"].append(reference_alive)
        result["model_alive"].append(model_alive)
        for side, path in (("reference", reference_path), ("model", model_path)):
            inputs.append({"case": f"microstructure {case_index}", "role": f"{side}_labels_step_{step}",
                           "path": path.relative_to(asset_root).as_posix(), "sha256": sha256_file(path)})
    extinctions = score_case["extinctions"]
    expected_alive = (spec["grains"] - int(extinctions["reference"]),
                      spec["grains"] - int(extinctions["model"]))
    if (result["reference_alive"][-1], result["model_alive"][-1]) != expected_alive:
        raise SourceConflict(f"{regime} microstructure {case_index}: terminal alive counts differ from score")
    return result, inputs


def _check_published(regime: str, recomputed: dict[str, dict]) -> None:
    spec = REGIMES[regime]
    for index, name in enumerate(spec["cases"]):
        data = recomputed[name]
        if round(data["disagreement_percent"][-1], 2) != spec["published_disagreement"][index]:
            raise SourceConflict(f"{regime} {name}: terminal disagreement differs from the article")
        alive = (data["reference_alive"][-1], data["model_alive"][-1])
        if alive != spec["published_alive"][index]:
            raise SourceConflict(f"{regime} {name}: terminal alive counts differ from the article")


def _cube(labels: np.ndarray, width: float) -> Image.Image:
    return legacy._on_white(legacy.cube_image(labels, width))


def _frame(regime: str, asset_root: Path, step: int, recomputed: dict[str, dict]) -> Image.Image:
    spec = REGIMES[regime]
    cases = spec["cases"]
    cubes = []
    for case_index, name in enumerate(cases, start=1):
        case_dir = asset_root / "population_scaling" / regime / f"microstructure_{case_index}"
        cubes.append((_cube(_load_label(_label_path(case_dir, "reference", step), step, spec["grid"], spec["grains"]), spec["cube_width"]),
                      _cube(_load_label(_label_path(case_dir, "model", step), step, spec["grid"], spec["grains"]), spec["cube_width"])))
    cw, ch = cubes[0][0].size
    if len(cases) == 3:
        left, gap, top, row_gap = 116, 18, 118, 26
        width = left + 3 * cw + 2 * gap + 12
        counters_y = top + 2 * ch + row_gap + 9
        height = counters_y + 65
        positions = [((left + k * (cw + gap), top), (left + k * (cw + gap), top + ch + row_gap)) for k in range(3)]
    else:
        left, gap, top = 36, 26, 126
        width = left * 2 + 2 * cw + gap
        counters_y = top + ch + 12
        height = counters_y + 71
        positions = [((left, top), (left + cw + gap, top))]
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((14, 10), spec["title"], font=_font(18, True), fill=INK)
    draw.text((14, 40), f"paired saved labels, t = {step * DT:g}; no interpolation", font=_font(13), fill=SOFT)
    x0, x1, y = 105, width - 24, 82
    draw.line((x0, y, x1, y), fill=(205, 205, 205), width=4)
    marker = x0 + (x1 - x0) * step / spec["final_step"]
    draw.line((x0, y, marker, y), fill=(70, 70, 70), width=4)
    draw.ellipse((marker - 5, y - 5, marker + 5, y + 5), fill=(70, 70, 70))
    draw.text((x0, y + 8), "t = 0", font=_font(11), fill=SOFT, anchor="ma")
    draw.text((x1, y + 8), f"t = {spec['final_step'] * DT:g}", font=_font(11), fill=SOFT, anchor="ma")
    for index, (name, pair, locations) in enumerate(zip(cases, cubes, positions)):
        reference_xy, model_xy = locations
        image.paste(pair[0], reference_xy)
        image.paste(pair[1], model_xy)
        data = recomputed[name]
        at = data["steps"].index(step)
        if len(cases) == 3:
            centre = reference_xy[0] + cw // 2
            draw.text((centre, top - 7), name, font=_font(13, True), fill=INK, anchor="md")
            draw.text((centre, counters_y), f"alive: ref {data['reference_alive'][at]} | model {data['model_alive'][at]}",
                      font=_font(11), fill=INK, anchor="ma")
            draw.text((centre, counters_y + 22), f"disagreement {data['disagreement_percent'][at]:.2f}%",
                      font=_font(11), fill=INK, anchor="ma")
        else:
            draw.text((reference_xy[0] + cw // 2, top - 7), "Reference", font=_font(14, True), fill=INK, anchor="md")
            draw.text((model_xy[0] + cw // 2, top - 7), "PINN-Phase", font=_font(14, True), fill=INK, anchor="md")
            draw.text((width // 2, counters_y), f"grains alive: reference {data['reference_alive'][at]}  |  PINN-Phase {data['model_alive'][at]}",
                      font=_font(13), fill=INK, anchor="ma")
            draw.text((width // 2, counters_y + 25), f"label disagreement {data['disagreement_percent'][at]:.2f} %",
                      font=_font(13), fill=INK, anchor="ma")
    if len(cases) == 3:
        draw.text((14, top + ch // 2), "Reference", font=_font(13, True), fill=INK, anchor="lm")
        draw.text((14, top + ch + row_gap + ch // 2), "PINN-Phase", font=_font(13, True), fill=INK, anchor="lm")
    return image


def _save_gif(frames: list[Image.Image], path: Path) -> list[int]:
    delays = [FRAME_MS] * len(frames)
    delays[0], delays[-1] = FIRST_MS, LAST_MS
    quantized = [frame.quantize(colors=80, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for frame in frames]
    quantized[0].save(path, save_all=True, append_images=quantized[1:], duration=delays, loop=0,
                      optimize=True, disposal=1)
    return delays


def _pixel_sha256(image: Image.Image) -> str:
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def _environment() -> dict:
    regular = FONT_DIR / "DejaVuSans.ttf"
    bold = FONT_DIR / "DejaVuSans-Bold.ttf"
    return {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__,
            "pillow": PIL.__version__, "freetype": features.version("freetype2"),
            "font_file": regular.name, "font_sha256": sha256_file(regular), "bold_font_sha256": sha256_file(bold),
            "text_rasterizer": "Pillow ImageDraw with FreeType"}


def _release_assets(release_asset_dir: Path) -> list[dict]:
    listed = {entry["filename"]: entry for entry in json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))["assets"]}
    result = []
    for regime in REGIMES:
        filename = f"population_scaling_{regime}.zip"
        path = release_asset_dir / filename
        entry = listed[filename]
        identity = {"filename": filename, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if identity["bytes"] != entry["bytes"] or identity["sha256"] != entry["sha256"]:
            raise SourceConflict(f"{filename}: release asset identity differs from the benchmark manifest")
        result.append(identity)
    return result


def render(asset_root: Path, release_asset_dir: Path, output: Path) -> Path:
    score = json.loads(SCORE.read_text(encoding="utf-8"))
    release_assets = _release_assets(release_asset_dir)
    all_inputs, regime_records, written = [], {}, []
    output.mkdir(parents=True, exist_ok=True)
    for regime, spec in REGIMES.items():
        recomputed = {}
        for case_index, name in enumerate(spec["cases"], start=1):
            data, inputs = _case_series(asset_root, regime, case_index, _score_cases(score, regime)[name])
            recomputed[name] = data
            all_inputs.extend(inputs)
        _check_published(regime, recomputed)
        steps = recomputed[spec["cases"][0]]["steps"]
        frames = [_frame(regime, asset_root, step, recomputed) for step in steps]
        gif = output / spec["gif"]
        poster = output / spec["poster"]
        delays = _save_gif(frames, gif)
        frames[-1].save(poster, format="PNG", optimize=True)
        if gif.stat().st_size > 3 * 1024 * 1024:
            raise SourceConflict(f"{gif.name}: GIF is larger than 3 MiB")
        written.extend((gif, poster))
        if len(spec["cases"]) == 3:
            terminal_pairs = ", ".join(
                f"{recomputed[name]['reference_alive'][-1]}/{recomputed[name]['model_alive'][-1]}"
                for name in spec["cases"])
            terminal_disagreement = ", ".join(
                f"{recomputed[name]['disagreement_percent'][-1]:.2f}%" for name in spec["cases"])
            alt_text = (f"Animation of three 240-cubed, 1,000-grain microstructures in columns, with "
                        f"reference labels above paired PINN-Phase labels at all {len(frames)} saved times "
                        f"from t = 0 to t = 135. At t = 135, reference/PINN-Phase alive counts are "
                        f"{terminal_pairs}, and label disagreements are {terminal_disagreement}.")
        else:
            data = recomputed[spec["cases"][0]]
            alt_text = (f"Animation of one 416-cubed, 5,000-grain microstructure with reference and "
                        f"PINN-Phase labels side by side at all {len(frames)} saved times from t = 0 to t = 50. "
                        f"At t = 50, reference/PINN-Phase alive counts are {data['reference_alive'][-1]}/"
                        f"{data['model_alive'][-1]}, and label disagreement is "
                        f"{data['disagreement_percent'][-1]:.2f}%.")
        regime_records[regime] = {
            "layout": spec["layout"], "cases_shown": list(spec["cases"]), "steps": steps,
            "t": recomputed[spec["cases"][0]]["t"], "frames": len(frames), "size": list(frames[0].size),
            "frame_durations_ms": delays, "frame_pixel_sha256": [_pixel_sha256(frame) for frame in frames],
            "recomputed": recomputed, "alt_text": alt_text,
            "camera": {**legacy.camera_metadata(spec["grid"]), "cube_width_px": spec["cube_width"]},
        }
    common_camera = legacy.camera_metadata(1)
    common_camera.pop("grid")
    common_camera.pop("cube_width_px")
    common_camera.update({"palette": "identity-fixed 5,000-colour palette; uint16 labels are indexed without truncation",
                          "palette_sha256": PALETTE_SHA256, "palette_entries": len(PAL)})
    manifest = {
        "schema": "pinn-phase-population-scaling-media-v1", "license": "CC-BY-4.0",
        "renderer": "scripts/render_population_scaling_media.py", "renderer_sha256": sha256_file(Path(__file__)),
        "score_record": "benchmarks/population_scaling/expected_score.json", "score_record_sha256": sha256_file(SCORE),
        "asset_manifest": "benchmarks/population_scaling/manifest.json", "asset_manifest_sha256": sha256_file(ASSET_MANIFEST),
        "release_assets": release_assets, "inputs": all_inputs, "alive_rule": "volume > 1e-6 in the per-step volume table",
        "time_per_step": DT, "no_interpolation": True,
        "pairing": "every frame pairs reference and PINN-Phase labels saved at the same step",
        "camera": common_camera,
        "renderer_environment": _environment(), "regimes": regime_records,
        "outputs": {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in written},
    }
    (output / "asset_manifest.json").write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    for path in written:
        print(f"Rendered {path} ({path.stat().st_size:,} bytes)")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True,
                        help="directory containing population_scaling/r240_n1000 and r416_n5000")
    parser.add_argument("--release-asset-dir", type=Path, required=True, help="directory containing the two release ZIPs")
    parser.add_argument("--output", type=Path, default=Path("media/population_scaling"))
    args = parser.parse_args()
    render(args.asset_root, args.release_asset_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
