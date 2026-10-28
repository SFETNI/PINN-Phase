"""Focused CPU checks for phase-channel assignment and evolution."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from pinn_phase.evaluation.phase_channels import (
    PhaseChannelState,
    channel_field_from_physical,
    initial_voronoi_phase_channels,
    physical_field_from_channels,
    support_graph,
    update_channel_ownership,
)
from pinn_phase.io.public_weights import load_public_state_dict
from pinn_phase.models.phase_local import PhaseLocalRule, phase_local_step
from pinn_phase.physics.obstacle_reference import obstacle_reference_step, one_hot_from_labels

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model() -> PhaseLocalRule:
    rule = PhaseLocalRule()
    rule.load_state_dict(
        load_public_state_dict(
            ROOT / "checkpoints/phase_local_48_n8.weights.npz",
            ROOT / "checkpoints/phase_local_48_n8.lineage.json",
        ),
        strict=True,
    )
    return rule.eval()


def _small_labels() -> torch.Tensor:
    labels = torch.ones((30, 8, 8), dtype=torch.int64)
    labels[:4] = 0
    labels[15:19] = 2
    return labels


def test_channel_round_trip_preserves_value_bits() -> None:
    physical = torch.zeros((1, 3, 12, 12, 12), dtype=torch.float64)
    physical[0, 0, 0, 0, 0] = np.nextafter(0.5, 1.0)
    physical[0, 1, 6, 6, 6] = 0.25
    physical[0, 2, 0, 0, 1] = 0.75
    fields, owner = channel_field_from_physical(physical, [0, 0, 1], 3)
    assert torch.equal(physical_field_from_channels(fields, owner, 3), physical)
    assert owner.dtype == torch.int16


def test_periodic_support_graph_and_channel_reassignment() -> None:
    supports = torch.zeros((2, 20, 20, 20), dtype=torch.bool)
    supports[0, 0, 2, 2] = True
    supports[1, 19, 2, 2] = True
    assert support_graph(supports, 1) == [{1}, {0}]
    physical = supports.to(torch.float64).unsqueeze(0)
    fields, owner = channel_field_from_physical(physical, [0, 0])
    state = PhaseChannelState(fields, owner, [0, 0], radius=1, buffer=2)
    with pytest.raises(ValueError, match="same-channel separation"):
        state.validate_separation()
    assert state.reassign_channels() == 1
    state.validate_separation(before_step=True)
    assert torch.equal(state.physical_field(), physical)


def test_ownership_split_extinction_and_ambiguity() -> None:
    owner = torch.full((1, 15, 15, 15), -1, dtype=torch.int16)
    owner[0, 5, 5, 5:8] = 0
    fields = torch.zeros((1, 1, 15, 15, 15), dtype=torch.float64)
    fields[0, 0, 5, 5, 5] = 0.5
    fields[0, 0, 5, 5, 7] = 0.5
    updated = update_channel_ownership(owner, fields, 1, 1)
    assert int((updated == 0).sum()) == 2
    assert int((update_channel_ownership(updated, torch.zeros_like(fields), 1, 1) >= 0).sum()) == 0

    owner[0, 7, 5, 5] = 1
    fields.zero_()
    fields[0, 0, 6, 5, 5] = 1.0
    with pytest.raises(ValueError, match="ambiguous or unattributed ownership"):
        update_channel_ownership(owner, fields, 1, 2)


@pytest.mark.parametrize("method", ["reference", "learned"])
def test_small_phase_channel_step_matches_dense_roundoff(
    method: str, model: PhaseLocalRule
) -> None:
    physical = one_hot_from_labels(_small_labels(), 3)
    radius = 3 if method == "learned" else 1
    state = PhaseChannelState.from_physical_field(physical, radius)
    dense_step = phase_local_step(model, physical) if method == "learned" else obstacle_reference_step(physical)
    with torch.no_grad():
        state.step(method, model=model if method == "learned" else None, spatial_slab=3)
    actual = state.physical_field()
    assert float((actual - dense_step).abs().max()) <= 1e-12
    assert torch.equal(state.physical_labels(), dense_step[0].argmax(dim=0))
    state.validate_separation()


def test_voronoi_initialization_has_every_grain_and_valid_graph() -> None:
    centers = np.array([[2.0, 3.0, 1.5], [9.0, 4.0, 7.0], [5.5, 10.0, 3.0]])
    state = initial_voronoi_phase_channels(centers, 12, radius=1)
    assert sorted(torch.unique(state.physical_labels()).tolist()) == [0, 1, 2]
    assert state.channels == state.grains + 2
    state.validate_separation(before_step=True)


def test_driver_runs_one_reference_evaluation(tmp_path: Path) -> None:
    labels = _small_labels().to(torch.uint8).numpy()
    initial = tmp_path / "initial_labels.npz"
    output = tmp_path / "output"
    np.savez_compressed(initial, labels=labels)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_phase_channel_rollout.py"),
            "--initial-labels",
            str(initial),
            "--steps",
            "2",
            "--method",
            "reference",
            "--output",
            str(output),
            "--save-every",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["schema"] == "pinn-phase-phase-channel-rollout-v1"
    assert summary["steps"] == 2 and summary["time"] == 0.02
    assert summary["initial_labels"] == initial.name
    assert set(path.name for path in output.iterdir()) == {"labels.npz", "summary.json", "volumes.npz"}
    with np.load(output / "volumes.npz", allow_pickle=False) as archive:
        assert archive["volumes"].shape == (3, 3)
        np.testing.assert_allclose(archive["volumes"].sum(axis=1), labels.size, rtol=0, atol=1e-10)
