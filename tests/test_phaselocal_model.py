"""The phase-local model: structure, admissibility, distributed weights, recorded replay.

Each property the phase-local realization rests on is checked against the code
rather than the prose: the parameter count does not depend on the number of phases,
appended empty phases stay exactly zero and leave the others bitwise unchanged, the
step commutes with a relabelling of the phases, every state is on the simplex, the
projection agrees with an independent NumPy implementation, an initial field is the
one-hot field of its labels, the distributed weights match their lineage, and a
short rollout reproduces the recorded output of the published 96^3 evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from pinn_phase.evaluation import phase_local as evaluation
from pinn_phase.io.public_weights import (
    as_torch_state_dict,
    load_public_weights,
    load_weight_lineage,
    model_state_fingerprint,
)
from pinn_phase.models.factory import ArchitectureFamily, architecture_family_from_config
from pinn_phase.models.phase_local import PhaseLocalRule, margin_repair, phase_local_step
from pinn_phase.physics.obstacle_reference import (
    BETA_O,
    DT_O,
    KAPPA_O,
    load_initial_labels,
    obstacle_reference_step,
    one_hot_from_labels,
    periodic_voronoi_labels,
    project_simplex_euclidean,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/models/phase_local_48_n8.yaml"
WEIGHTS = ROOT / "checkpoints/phase_local_48_n8.weights.npz"
LINEAGE = ROOT / "checkpoints/phase_local_48_n8.lineage.json"
REPLAY_FIXTURE = ROOT / "tests/fixtures/phaselocal_replay_r96_n16_m1.json"
# The 96^3 replay takes several minutes on a CPU, so it runs only when asked for.
RUN_SLOW = os.environ.get("PINN_PHASE_SLOW_TESTS") == "1"

# Fixed grain centers of a small periodic test microstructure, in voxel units.
CENTERS = [[2.0, 3.0, 1.5], [9.0, 4.0, 7.0], [5.5, 10.0, 3.0], [1.0, 8.0, 10.5], [10.0, 10.0, 10.0]]
SIDE = 12


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@pytest.fixture(scope="module")
def model() -> PhaseLocalRule:
    rule = PhaseLocalRule()
    state = load_public_weights(WEIGHTS, lineage=load_weight_lineage(LINEAGE))
    rule.load_state_dict(as_torch_state_dict(state), strict=True)
    return rule.eval()


@pytest.fixture(scope="module")
def field(model: PhaseLocalRule) -> torch.Tensor:
    """The test microstructure after a few learned steps, so interfaces are diffuse."""
    phi = one_hot_from_labels(periodic_voronoi_labels(CENTERS, SIDE), len(CENTERS))
    with torch.no_grad():
        for _ in range(20):
            phi = phase_local_step(model, phi)
    return phi


# ---------------------------------------------------------------- structure


@pytest.mark.parametrize("phases", [1, 2, 8, 16, 38, 150])
def test_parameter_count_is_2160_for_any_phase_count(phases: int) -> None:
    rule = PhaseLocalRule()
    assert sum(p.numel() for p in rule.parameters()) == 2160
    assert {name: tuple(p.shape) for name, p in rule.state_dict().items()} == {
        "c1.weight": (8, 1, 3, 3, 3),
        "c2.weight": (8, 8, 3, 3, 3),
        "c3.weight": (1, 8, 3, 3, 3),
    }
    labels = torch.arange(4 * 4 * 4).reshape(4, 4, 4) % phases
    with torch.no_grad():
        out = phase_local_step(rule, one_hot_from_labels(labels, phases))
    assert out.shape == (1, phases, 4, 4, 4)
    assert sum(p.numel() for p in rule.parameters()) == 2160


def test_zero_phase_padding_is_exact(model: PhaseLocalRule, field: torch.Tensor) -> None:
    n = field.shape[1]
    padded = torch.zeros(1, n + 7, *field.shape[2:], dtype=field.dtype)
    padded[:, :n] = field
    with torch.no_grad():
        for step in (lambda x: phase_local_step(model, x), obstacle_reference_step):
            base, out = step(field), step(padded)
            assert torch.equal(out[:, :n], base), "existing phases changed when empty ones were appended"
            assert torch.equal(out[:, n:], torch.zeros_like(out[:, n:])), "an empty phase gained mass"


def test_permutation_equivariance(model: PhaseLocalRule, field: torch.Tensor) -> None:
    perm = [3, 0, 4, 1, 2]
    with torch.no_grad():
        base = phase_local_step(model, field)
        permuted = phase_local_step(model, field[:, perm])[:, torch.argsort(torch.tensor(perm))]
    assert float((permuted - base).abs().max()) <= evaluation.PERMUTATION_TOLERANCE
    report = evaluation.structural_checks(model, field, perm, zero_phases=4)
    assert report["pass"], report
    assert report["learned_step_padding"]["existing_bitwise_equal"]
    assert report["reference_step_padding"]["appended_exactly_zero"]


def test_architecture_family_resolves_from_the_configuration() -> None:
    settings = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert architecture_family_from_config(settings) is ArchitectureFamily.PHASE_LOCAL
    assert settings["architecture_family"] == str(ArchitectureFamily.PHASE_LOCAL)
    assert settings["class_name"] == "PhaseLocalRule"
    physics = settings["physics"]
    assert (physics["kappa_o"], physics["beta_o"], physics["dx"], physics["dt_o"]) == (
        KAPPA_O, BETA_O, 1.0, DT_O)
    with pytest.raises(ValueError):
        architecture_family_from_config(
            {"model": {"architecture": "phase_local_rule", "arch_variant": "perm_equivariant_v1"}}
        )


# ---------------------------------------------------------------- admissibility


def test_simplex_feasibility_after_a_step(model: PhaseLocalRule, field: torch.Tensor) -> None:
    with torch.no_grad():
        outputs = [phase_local_step(model, field), obstacle_reference_step(field)]
    for out in outputs:
        assert torch.isfinite(out).all()
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
        assert evaluation.simplex_error(out) <= evaluation.SIMPLEX_TOLERANCE


def test_margin_repair_raises_the_phase_sum_where_phases_are_present(field: torch.Tensor) -> None:
    z = field * 0.5
    repaired = margin_repair(z, field)
    assert float(repaired.sum(dim=1).min()) >= 1.0 + DT_O * BETA_O - 1e-12
    assert torch.equal(repaired[field == 0], z[field == 0])


def _numpy_projection(v: np.ndarray) -> np.ndarray:
    """Euclidean simplex projection of one vector, written independently in NumPy."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    k = np.arange(1, len(v) + 1)
    rho = np.nonzero(u * k > css - 1.0)[0][-1]
    tau = (css[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - tau, 0.0)


def _bisection_projection(v: np.ndarray) -> np.ndarray:
    """The same projection from its definition: find tau with sum(max(v - tau, 0)) = 1."""
    low, high = v.min() - 1.0, v.max()
    for _ in range(200):
        mid = 0.5 * (low + high)
        if np.maximum(v - mid, 0.0).sum() > 1.0:
            low = mid
        else:
            high = mid
    return np.maximum(v - 0.5 * (low + high), 0.0)


@pytest.mark.parametrize("phases", [1, 2, 3, 8, 16, 38, 150])
def test_projection_matches_an_independent_numpy_implementation(phases: int) -> None:
    rng = np.random.default_rng(phases)
    z = rng.normal(0.3, 1.5, size=(2, phases, 3, 2, 2))
    out = project_simplex_euclidean(torch.from_numpy(z)).numpy()
    for b in range(2):
        for x in np.ndindex(3, 2, 2):
            vector = z[(b, slice(None)) + x]
            got = out[(b, slice(None)) + x]
            np.testing.assert_allclose(got, _numpy_projection(vector), rtol=0, atol=1e-14)
            np.testing.assert_allclose(got, _bisection_projection(vector), rtol=0, atol=1e-12)
    assert np.abs(out.sum(axis=1) - 1.0).max() <= 1e-12
    assert out.min() >= 0.0
    again = project_simplex_euclidean(torch.from_numpy(out)).numpy()
    np.testing.assert_allclose(again, out, rtol=0, atol=1e-15)


# ---------------------------------------------------------------- initial fields


def test_initial_field_equals_the_one_hot_field_of_its_labels() -> None:
    labels = periodic_voronoi_labels(CENTERS, SIDE)
    assert labels.shape == (SIDE, SIDE, SIDE)
    assert sorted(torch.unique(labels).tolist()) == list(range(len(CENTERS)))
    phi0 = one_hot_from_labels(labels.to(torch.uint8).numpy(), len(CENTERS))
    assert phi0.dtype == torch.float64 and phi0.shape == (1, len(CENTERS), SIDE, SIDE, SIDE)
    assert torch.equal(project_simplex_euclidean(phi0), phi0), "a one-hot field is a projection fixed point"
    assert torch.equal(phi0[0].argmax(dim=0), labels)
    np.testing.assert_array_equal(evaluation.argmax_labels(phi0), labels.numpy().astype(np.uint8))
    volumes = evaluation.grain_volumes(phi0)
    assert float(volumes.sum()) == SIDE**3
    with pytest.raises(ValueError):
        one_hot_from_labels(labels, 3)
    with pytest.raises(ValueError):
        one_hot_from_labels(labels.to(torch.float64))


def test_initial_label_reader_is_strict(tmp_path: Path) -> None:
    labels = periodic_voronoi_labels(CENTERS, SIDE).to(torch.uint8).numpy()
    good = tmp_path / "labels.npz"
    np.savez(good, labels=labels)
    np.testing.assert_array_equal(load_initial_labels(good), labels)
    with pytest.raises(ValueError):
        load_initial_labels(good, expected_sha256="0" * 64)
    wrong = tmp_path / "wrong.npz"
    np.savez(wrong, labels=labels.astype(np.int64))
    with pytest.raises(ValueError):
        load_initial_labels(wrong)


# ---------------------------------------------------------------- evaluation helpers


def test_rollout_records_volumes_labels_and_admissibility(model: PhaseLocalRule) -> None:
    labels = periodic_voronoi_labels(CENTERS, SIDE).to(torch.uint8).numpy()
    run = evaluation.rollout_from_labels(model, labels, 6, label_steps=(0, 6), field_steps=(6,))
    assert run.admissible and run.steps_completed == 6
    assert run.volumes.shape == (7, len(CENTERS)) and run.volumes.dtype == np.float64
    np.testing.assert_array_equal(run.labels[0], labels)
    np.testing.assert_array_equal(run.volumes[6], evaluation.grain_volumes(run.fields[6]).numpy())
    np.testing.assert_array_equal(run.labels[6], evaluation.argmax_labels(run.fields[6]))
    assert evaluation.alive_set(run.volumes[6]) == tuple(range(len(CENTERS)))


def test_cpu_rollout_never_touches_cuda(model: PhaseLocalRule, monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("a CPU rollout queried CUDA")

    for name in ("is_available", "device_count", "current_device", "synchronize", "init"):
        monkeypatch.setattr(torch.cuda, name, refuse)
    labels = periodic_voronoi_labels(CENTERS, SIDE).to(torch.uint8).numpy()
    run = evaluation.rollout_from_labels(model, labels, 2)
    assert run.admissible
    with pytest.raises(ValueError):
        evaluation.configure_device("cuda", deterministic_gpu=False)


def test_extinction_events_and_alive_sets_follow_the_volume_threshold() -> None:
    table = np.array([
        [5.0, 3.0, 2.0],
        [4.0, 1e-7, 2.0],
        [3.0, 0.0, 1e-6],
        [2.0, 0.0, 0.5],
        [1.0, 0.0, 0.0],
    ])
    events = evaluation.extinction_events(table)
    assert events == {1: {"step": 1, "returned": None},
                      2: {"step": 2, "returned": {"step": 3, "volume": 0.5}}}
    assert evaluation.alive_set(table[2]) == (0,)
    assert evaluation.max_normalized_volume_error([1.0, 2.0], [1.5, 2.0], 10.0) == 0.05


def test_field_error_and_persistence_ratio() -> None:
    a = one_hot_from_labels(torch.zeros(2, 2, 2, dtype=torch.int64), 2)
    b = one_hot_from_labels(torch.ones(2, 2, 2, dtype=torch.int64), 2)
    assert evaluation.total_variation_error(a, b) == 1.0
    half = 0.5 * (a + b)
    assert evaluation.total_variation_error(half, b) == 0.5
    assert evaluation.persistence_ratio(half, b, a) == 0.5


# ---------------------------------------------------------------- distributed weights


def test_public_weights_load_and_fingerprint_match_the_lineage(model: PhaseLocalRule) -> None:
    lineage = json.loads(LINEAGE.read_text(encoding="utf-8"))
    settings = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    state = load_public_weights(WEIGHTS, lineage=load_weight_lineage(LINEAGE))
    assert model_state_fingerprint(state) == lineage["model_state_fingerprint_sha256"]
    assert lineage["parameter_count"] == 2160 and lineage["tensor_count"] == 3
    assert lineage["tensor_key_order"] == ["c1.weight", "c2.weight", "c3.weight"]
    assert lineage["accepted_parent_checkpoint"]["sha256"] == settings["parent_checkpoint_sha256"]
    assert lineage["accepted_parent_checkpoint"]["distributed"] is False
    assert all(array.dtype == np.float32 for array in state.values())
    for name, tensor in model.state_dict().items():
        assert tensor.numpy().tobytes() == state[name].tobytes()


# ---------------------------------------------------------------- recorded replay


def test_replay_fixture_is_well_formed() -> None:
    fixture = json.loads(REPLAY_FIXTURE.read_text(encoding="utf-8"))
    assert fixture["regime"] == "r96_n16" and fixture["case"] == "microstructure 1"
    assert fixture["parent_checkpoint_sha256"] == yaml.safe_load(
        CONFIG.read_text(encoding="utf-8"))["parent_checkpoint_sha256"]
    for step in fixture["steps"]:
        volumes = fixture["model_volumes"][str(step)]
        assert len(volumes) == fixture["phases"]
        assert abs(sum(volumes) - 96**3) < 1e-6 * 96**3
        assert len(fixture["model_labels_sha256"][str(step)]) == 64


@pytest.mark.skipif(not RUN_SLOW, reason="slow (several minutes on CPU); set PINN_PHASE_SLOW_TESTS=1")
def test_short_rollout_reproduces_the_recorded_96_output(model: PhaseLocalRule) -> None:
    """Microstructure 1 of r96_n16: 200 steps against the recorded volumes and labels.

    The expected values are the recorded output of the published evaluation (labels
    and grain volumes every 100 steps); the initial labels ship with the benchmark.
    """
    fixture = json.loads(REPLAY_FIXTURE.read_text(encoding="utf-8"))
    labels_path = ROOT / fixture["initial_labels"]
    if not labels_path.is_file():
        pytest.skip(f"{fixture['initial_labels']} is not present in this tree")
    labels = load_initial_labels(labels_path)
    assert _array_sha256(labels) == fixture["initial_labels_value_sha256"]
    steps = [int(s) for s in fixture["steps"]]
    run = evaluation.rollout_from_labels(model, labels, max(steps), phases=fixture["phases"],
                                         label_steps=steps)
    assert run.admissible
    for step in steps:
        expected = np.asarray(fixture["model_volumes"][str(step)], dtype=np.float64)
        difference = float(np.abs(run.volumes[step] - expected).max())
        assert difference <= fixture["volume_tolerance"], (
            f"step {step}: grain volumes differ from the record by {difference}"
        )
        assert _array_sha256(run.labels[step]) == fixture["model_labels_sha256"][str(step)], (
            f"step {step}: argmax labels differ from the record"
        )
