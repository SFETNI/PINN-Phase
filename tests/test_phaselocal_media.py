"""Guards for the phase-local transfer media (150-grain animation, 38-grain still).

The label and volume arrays behind the media are release assets, not part of the
repository. Without them these tests check the shipped manifests, outputs and
renderer; with them (``PINN_PHASE_ASSET_ROOT`` pointing at the directory that holds
``phaselocal_transfer/``) they also re-render one frame and compare its pixels.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import runpy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageSequence, features


ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "scripts/render_phaselocal_media.py"
N150 = ROOT / "media/phaselocal_n150"
N38 = ROOT / "media/phaselocal_n38"
N150_GIF = N150 / "phaselocal_n150_reference_vs_pinn_phase.gif"
N150_POSTER = N150 / "phaselocal_n150_reference_vs_pinn_phase_poster.png"
N38_STILL = N38 / "phaselocal_n38_reference_vs_pinn_phase.png"
CASES = ["microstructure 1", "microstructure 2", "microstructure 3"]
N150_STEPS = [0, 200, 400, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 13500]
PALETTE_SHA256 = "0447c04a4563e3f62286644b69a0e8783cafe03c00c8179100ff02522a81462e"
ASSET_FILES = ("reference_labels.npz", "model_labels.npz", "reference_volumes.npz", "model_volumes.npz")
HEX_DIGEST = re.compile(r"\b[0-9a-f]{64}\b")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _renderer() -> dict:
    return runpy.run_path(str(RENDERER))


@pytest.mark.parametrize(
    "folder, regime, outputs, steps",
    [
        (N150, "r128_n150", (N150_GIF, N150_POSTER), N150_STEPS),
        (N38, "r128_n38", (N38_STILL,), [6750]),
    ],
)
def test_manifest_binds_outputs_renderer_and_inputs(folder: Path, regime: str, outputs, steps) -> None:
    media = _json(folder / "asset_manifest.json")
    assert media["schema"] == "pinn-phase-phaselocal-media-v1" and media["regime"] == regime
    assert media["license"] == "CC-BY-4.0" and media["no_interpolation"] is True
    assert media["renderer_sha256"] == _sha256(RENDERER)
    assert media["cases_shown"] == CASES and set(media["recomputed"]) == set(CASES)
    assert media["steps"] == steps and media["frames"] == len(steps)
    assert media["camera"]["palette_sha256"] == PALETTE_SHA256
    assert set(media["outputs"]) == {p.name for p in outputs}
    for path in outputs:
        assert media["outputs"][path.name] == {"sha256": _sha256(path), "bytes": path.stat().st_size}
    assert {p.name for p in folder.iterdir() if p.is_file()} == {p.name for p in outputs} | {"asset_manifest.json"}
    expected_inputs = {(name, f"phaselocal_transfer/{regime}/microstructure_{k}/{filename}")
                       for k, name in enumerate(CASES, start=1) for filename in ASSET_FILES}
    assert {(item["case"], item["path"]) for item in media["inputs"]} == expected_inputs
    assert all(HEX_DIGEST.fullmatch(item["sha256"]) for item in media["inputs"])
    assert len(media["frame_pixel_sha256"]) == media["frames"]
    record = ROOT / "benchmarks/phaselocal_transfer/expected_score.json"
    if record.is_file():
        assert media["record_sha256"] == _sha256(record) and media["record_checks"]
    width, height = media["size"]
    assert 880 <= width <= 920 and height < 900


def test_animation_timing_size_and_paired_frames() -> None:
    media = _json(N150 / "asset_manifest.json")
    with Image.open(N150_GIF) as image:
        durations = [frame.info["duration"] for frame in ImageSequence.Iterator(image)]
        assert len(durations) == 13 and image.size == tuple(media["size"])
    assert durations[0] == 1400 and durations[-1] == 2600 and set(durations[1:-1]) == {700}
    assert media["frame_durations_ms"] == durations
    assert N150_GIF.stat().st_size < 2 * 1024 * 1024
    assert media["t"] == [0, 2, 4, 10, 20, 30, 40, 50, 60, 80, 100, 120, 135]
    with Image.open(N150_POSTER) as poster, Image.open(N38_STILL) as still:
        assert poster.size == tuple(media["size"]) and still.size[0] == media["size"][0]


def test_recomputed_values_match_the_article() -> None:
    n150 = _json(N150 / "asset_manifest.json")["recomputed"]
    n38 = _json(N38 / "asset_manifest.json")["recomputed"]
    assert [round(n150[c]["disagreement_percent"][-1], 2) for c in CASES] == [0.51, 0.70, 0.88]
    assert [n150[c]["reference_alive"][-1] for c in CASES] == [24, 28, 30]
    assert [n150[c]["model_alive"][-1] for c in CASES] == [24, 28, 30]
    # All 368 extinctions: 126 + 122 + 120 grains of 150 are extinct at t = 135 in both.
    assert sum(150 - n150[c]["model_alive"][-1] for c in CASES) == 368
    offsets = {(c, t) for c in CASES for t, r, m in zip(n150[c]["t"], n150[c]["reference_alive"],
                                                         n150[c]["model_alive"]) if r != m}
    assert offsets == {("microstructure 1", 50), ("microstructure 3", 40)}
    for c in CASES:
        assert n150[c]["steps"] == N150_STEPS and n150[c]["disagreement_percent"][0] == 0.0
        assert max(abs(r - m) for r, m in zip(n150[c]["reference_alive"], n150[c]["model_alive"])) <= 1
    assert [round(n38[c]["disagreement_percent"][-1], 3) for c in CASES] == [0.356, 0.334, 0.405]
    for c, survivors in zip(CASES, (35, 32, 35)):
        assert n38[c]["reference_alive"] == n38[c]["model_alive"]
        assert n38[c]["model_alive"][-1] == survivors and n38[c]["steps"][-1] == 6750


@pytest.mark.parametrize("path", [N150 / "asset_manifest.json", N38 / "asset_manifest.json", RENDERER,
                                  Path(__file__)])
def test_public_text_carries_no_seed_shaped_numbers_or_private_paths(path: Path) -> None:
    text = HEX_DIGEST.sub("", path.read_text(encoding="utf-8"))
    assert not re.search(r"\d{8,}", text), "8+ digit run outside a digest"
    assert "\u2014" not in text
    for forbidden in ("/" "home/", "C:" "\\Users", "128" "cubed", "128" "^3/"):
        assert forbidden not in text
    for media_file in list(N150.iterdir()) + list(N38.iterdir()):
        assert not re.search(r"128(?:\^3|cubed|x128)|\u00b3", media_file.name)


def test_renderer_draws_a_frame_on_a_tiny_grid_and_refuses_disagreeing_values() -> None:
    renderer = _renderer()
    camera = renderer["camera_metadata"](16)
    assert (camera["camera_elevation_deg"], camera["camera_azimuth_deg"]) == (24.0, 38.0)
    assert camera["grid"] == [16, 16, 16] and camera["palette_sha256"] == PALETTE_SHA256
    cube = renderer["cube_image"](np.zeros((16, 16, 16), dtype=np.uint8))
    assert cube.shape[2] == 4 and cube.shape[1] == 237
    shades = {round(float(v), 4) for v in np.unique(cube[..., 0][cube[..., 3] == 1.0])}
    faces = {round(float(np.float32(renderer["PAL"][0, 0] * s)), 4) for s in (0.84, 0.70, 1.00)}
    assert faces | {0.22} <= shades, "three shaded faces of grain 0 and dark cube edges"

    rng = np.random.default_rng(0)
    steps = [0, 6750]
    reference = rng.integers(0, 38, size=(2, 16, 16, 16), dtype=np.uint8)
    model = reference.copy()
    model[1, :2] = 0
    volumes = np.ones((2, 38))
    volumes[1, :3] = 0.0
    data = renderer["series"](reference, model, volumes, volumes, steps)
    assert data["disagreement_percent"][0] == 0.0 and data["disagreement_percent"][1] > 0.0
    assert data["reference_alive"] == [38, 35] and data["t"] == [0.0, 67.5]
    cases = [{"public_name": name, "reference": reference[1:], "model": model[1:],
              "series": {key: values[1:] for key, values in data.items()}} for name in CASES]
    image = renderer["frame"]("r128_n38", cases, 0)
    assert list(image.size) == _json(N38 / "asset_manifest.json")["size"]

    with pytest.raises(renderer["SourceConflict"]):
        renderer["check_against_published"]("r128_n38", {name: data for name in CASES})
    agreeing = {"r128_n38": {"cases": [
        {"public_name": name, "reference_extinction_count": 3, "model_extinction_count": 3,
         "alive": [{"step": 6750, "reference_alive_count": 35, "model_alive_count": 35}],
         "label_disagreement": [{"step": 6750, "disagreeing_voxels": data["differing_voxels"][1],
                                 "percent": data["disagreement_percent"][1]}]}
        for name in CASES]}}
    assert len(renderer["check_against_record"](agreeing, "r128_n38", {n: data for n in CASES})) == 18
    for field, entry, key, value in (("alive", 0, "model_alive_count", 36),
                                     ("label_disagreement", 0, "disagreeing_voxels", 1)):
        disagreeing = json.loads(json.dumps(agreeing))
        disagreeing["r128_n38"]["cases"][1][field][entry][key] = value
        with pytest.raises(renderer["SourceConflict"]):
            renderer["check_against_record"](disagreeing, "r128_n38", {n: data for n in CASES})


def _asset_root() -> Path:
    value = os.environ.get("PINN_PHASE_ASSET_ROOT")
    if not value:
        pytest.skip("PINN_PHASE_ASSET_ROOT is not set; the label and volume arrays are release assets")
    root = Path(value)
    if not (root / "phaselocal_transfer/r128_n38").is_dir():
        pytest.skip(f"no phaselocal_transfer/r128_n38 under {root}")
    return root


def test_release_assets_rerender_the_still_pixel_for_pixel() -> None:
    root = _asset_root()
    media = _json(N38 / "asset_manifest.json")
    for item in media["inputs"]:
        assert _sha256(root / item["path"]) == item["sha256"], item["path"]
    environment = media["renderer_environment"]
    import PIL

    if (PIL.__version__, features.version("freetype2")) != (environment["pillow"], environment["freetype"]):
        pytest.skip("text rasterization differs outside the recorded Pillow and FreeType versions")
    renderer = _renderer()
    spec = renderer["REGIMES"]["r128_n38"]
    cases = []
    for k, name in enumerate(CASES, start=1):
        loaded = renderer["load_case"](root / f"phaselocal_transfer/r128_n38/microstructure_{k}", spec, name)
        s = renderer["series"](loaded["reference"], loaded["model"], loaded["reference_volumes"],
                               loaded["model_volumes"], loaded["steps"])
        assert s == media["recomputed"][name]
        cases.append({"public_name": name, "reference": loaded["reference"][-1:], "model": loaded["model"][-1:],
                      "series": {key: values[-1:] for key, values in s.items()}})
    image = renderer["frame"]("r128_n38", cases, 0)
    assert renderer["pixel_sha256"](image) == media["frame_pixel_sha256"][0]
    with Image.open(N38_STILL) as still:
        assert still.convert("RGB").tobytes() == image.tobytes()
