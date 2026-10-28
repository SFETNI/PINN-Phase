#!/usr/bin/env python3
"""Build public population-scaling release archives from run return archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import numpy as np


ALIVE = 1.0e-6
METHODS = {"model", "reference"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member(name: str) -> bool:
    return name == "output/volumes.npz" or name == "output/timing.npz" or name.startswith(
        "output/labels/step_"
    ) and name.endswith(".npz")


def extract_selected(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as source:
        members = [member for member in source.getmembers() if safe_member(member.name)]
        names = {member.name for member in members}
        if "output/volumes.npz" not in names or not any(name.startswith("output/labels/") for name in names):
            raise ValueError(f"required arrays are absent from {archive}")
        for member in members:
            relative = Path(member.name).relative_to("output")
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            extracted = source.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read {member.name}")
            with extracted, destination.open("wb") as output:
                shutil.copyfileobj(extracted, output)


def extinction_record(volumes_path: Path) -> dict:
    with np.load(volumes_path, allow_pickle=False) as archive:
        volumes = archive["volumes"]
    events = []
    returns = []
    for grain in range(volumes.shape[1]):
        below = np.flatnonzero(volumes[1:, grain] <= ALIVE)
        if not below.size:
            continue
        step = int(below[0]) + 1
        events.append({"grain": grain, "step": step})
        if np.any(volumes[step:, grain] > ALIVE):
            returns.append(grain)
    return {
        "alive_threshold": ALIVE,
        "extinctions": events,
        "returns": returns,
        "steps": int(volumes.shape[0] - 1),
        "grains": int(volumes.shape[1]),
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def build_entry(entry: dict, scratch: Path, staging: Path) -> dict:
    regime = entry["regime"]
    case = entry["case"]
    method = entry["method"]
    if method not in METHODS or not regime.startswith("r") or not case.startswith("microstructure_"):
        raise ValueError("configuration contains an invalid public identifier")
    archive = Path(entry["archive"]).resolve()
    extracted = scratch / regime / case / method
    extract_selected(archive, extracted)
    write_json(extracted / "extinctions.json", extinction_record(extracted / "volumes.npz"))
    destination = staging / regime / case / method
    destination.mkdir(parents=True, exist_ok=False)
    for source in sorted(extracted.rglob("*")):
        if source.is_file():
            out = destination / source.relative_to(extracted)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, out)
    return {
        "regime": regime,
        "case": case,
        "method": method,
        "source_sha256": sha256(archive),
    }


def zip_regime(staging: Path, regime: str, output_root: Path) -> Path:
    output = output_root / f"population_scaling_{regime}.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source in sorted((staging / regime).rglob("*")):
            if source.is_file():
                relative = Path("population_scaling") / regime / source.relative_to(staging / regime)
                archive.write(source, relative.as_posix())
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.scratch.exists() or args.output_root.exists():
        parser.error("scratch and output roots must not already exist")
    staging = args.scratch / "staging"
    staging.mkdir(parents=True)
    args.output_root.mkdir(parents=True)
    inputs = [build_entry(entry, args.scratch / "extracted", staging) for entry in config["runs"]]
    assets = []
    for regime in sorted({entry["regime"] for entry in config["runs"]}):
        path = zip_regime(staging, regime, args.output_root)
        assets.append({"filename": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(args.scratch / "source_inputs.json", {"runs": inputs})
    upload = args.output_root / "population_scaling_upload"
    upload.mkdir()
    for asset in assets:
        shutil.copy2(args.output_root / asset["filename"], upload / asset["filename"])
    upload_sums = "".join(f"{asset['sha256']}  {asset['filename']}\n" for asset in assets)
    (upload / "SHA256SUMS").write_text(upload_sums, encoding="utf-8", newline="\n")
    root_files = [path for path in args.output_root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"]
    root_files.append(upload / "SHA256SUMS")
    sums = "".join(
        f"{sha256(path)}  {path.relative_to(args.output_root).as_posix()}\n" for path in sorted(set(root_files))
    )
    (args.output_root / "SHA256SUMS").write_text(sums, encoding="utf-8", newline="\n")
    print(f"wrote {len(assets)} population-scaling release archives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
