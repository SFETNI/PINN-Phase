"""Run one phase-channel evaluation from a distributed initial label array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_phase.evaluation.phase_channels import PhaseChannelState
from pinn_phase.io.public_weights import load_public_state_dict
from pinn_phase.models.phase_local import PhaseLocalRule
from pinn_phase.physics.obstacle_reference import load_initial_labels


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _model(weights: Path, lineage: Path, device: torch.device) -> PhaseLocalRule:
    model = PhaseLocalRule()
    model.load_state_dict(load_public_state_dict(weights, lineage), strict=True)
    return model.to(device).eval()


def run(args: argparse.Namespace) -> dict:
    if args.steps < 0 or args.save_every < 1:
        raise ValueError("steps must be nonnegative and save-every must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not args.deterministic_gpu:
            raise ValueError("a GPU rollout requires --deterministic-gpu")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    labels = load_initial_labels(args.initial_labels)
    phases = int(labels.max()) + 1
    radius = 3 if args.method == "learned" else 1
    state = PhaseChannelState.from_labels(labels, phases, radius, device=device)
    model = _model(args.weights, args.lineage, device) if args.method == "learned" else None

    saved_steps = [0]
    volumes = [state.volumes().cpu().numpy()]
    saved_labels = [state.physical_labels().cpu().numpy().astype(np.uint16, copy=False)]
    with torch.no_grad():
        for step in range(1, args.steps + 1):
            state.step(
                args.method,
                model=model,
                channels_per_batch=args.channels_per_batch,
                spatial_slab=args.spatial_slab,
            )
            volumes.append(state.volumes().cpu().numpy())
            if step % args.save_every == 0 or step == args.steps:
                saved_steps.append(step)
                saved_labels.append(
                    state.physical_labels().cpu().numpy().astype(np.uint16, copy=False)
                )

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "volumes.npz",
        steps=np.arange(args.steps + 1, dtype=np.int64),
        volumes=np.asarray(volumes, dtype=np.float64),
    )
    np.savez_compressed(
        args.output / "labels.npz",
        steps=np.asarray(saved_steps, dtype=np.int64),
        labels=np.asarray(saved_labels, dtype=np.uint16),
    )
    summary = {
        "schema": "pinn-phase-phase-channel-rollout-v1",
        "method": args.method,
        "initial_labels": args.initial_labels.name,
        "grid": list(labels.shape),
        "grains": phases,
        "steps": args.steps,
        "time": args.steps * 0.01,
        "channels_initial": state.events[0]["channels"],
        "channels_final": state.channels,
        "channel_reassignments": sum(event["kind"] == "move" for event in state.events),
        "minimum_same_channel_distance": state.minimum_same_channel_distance(),
        "labels_saved_at_steps": saved_steps,
    }
    _write_json(args.output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-labels", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("learned", "reference"), default="learned")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic-gpu", action="store_true")
    parser.add_argument("--channels-per-batch", type=int)
    parser.add_argument("--spatial-slab", type=int)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--weights", type=Path, default=ROOT / "checkpoints/phase_local_48_n8.weights.npz"
    )
    parser.add_argument(
        "--lineage", type=Path, default=ROOT / "checkpoints/phase_local_48_n8.lineage.json"
    )
    print(json.dumps(run(parser.parse_args()), allow_nan=False))


if __name__ == "__main__":
    main()
