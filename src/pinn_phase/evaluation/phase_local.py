"""Autonomous rollout and per-case measurements for the phase-local realization.

A rollout starts from a sharp initial field rebuilt from integer labels and applies the
learned step repeatedly; no reference state enters after step 0. What is recorded per
step is what the criteria are written in:

* grain volumes ``V_k(s)``, the integral of each diffuse phase field (not a label
  count), from which alive sets and extinction steps follow;
* admissibility: finiteness, nonnegativity and the voxelwise unit-sum error;
* optionally the argmax label arrays and the full fields at chosen steps.

The helpers below compute the field error ``E_TV``, its ratio to static persistence,
the maximum normalized grain-volume error, and the structural checks (phase
permutation and zero-phase padding).

Device handling. Everything runs on CPU by default and the CPU path never touches
CUDA. ``configure_device("cuda", deterministic_gpu=True)`` is the documented GPU
option: it disables TF32, disables cuDNN autotuning, sets
``CUBLAS_WORKSPACE_CONFIG=:4096:8`` and turns on PyTorch's deterministic algorithms.
Call it before the first CUDA operation of the process. GPU results agree with CPU
results to round-off in the single-precision increment, not bitwise.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import os
from typing import Any

import numpy as np
import torch
from torch import Tensor

from pinn_phase.models.phase_local import PhaseLocalRule, phase_local_step
from pinn_phase.physics.obstacle_reference import obstacle_reference_step, one_hot_from_labels

__all__ = [
    "ALIVE_THRESHOLD",
    "PERMUTATION_TOLERANCE",
    "SIMPLEX_TOLERANCE",
    "VOLUME_ERROR_LIMIT",
    "PhaseLocalRollout",
    "alive_set",
    "argmax_labels",
    "configure_device",
    "extinction_events",
    "grain_volumes",
    "max_normalized_volume_error",
    "persistence_ratio",
    "rollout_from_labels",
    "simplex_error",
    "structural_checks",
    "total_variation_error",
]

#: A phase is alive at a step when its volume exceeds this value.
ALIVE_THRESHOLD = 1.0e-6
#: Criterion (ii): maximum voxelwise unit-sum error of an admissible state.
SIMPLEX_TOLERANCE = 1.0e-12
#: Criterion (i): maximum difference after permuting phases and permuting back.
PERMUTATION_TOLERANCE = 1.0e-15
#: Criterion (v): limit of the maximum normalized grain-volume error.
VOLUME_ERROR_LIMIT = 0.05


def configure_device(device: str | torch.device = "cpu", *, deterministic_gpu: bool = False) -> torch.device:
    """Resolve the execution device; on a GPU, require and apply deterministic settings."""

    resolved = torch.device(device)
    if resolved.type == "cpu":
        return resolved
    if resolved.type != "cuda":
        raise ValueError(f"unsupported device: {resolved}")
    if not deterministic_gpu:
        raise ValueError("a GPU rollout requires deterministic_gpu=True")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return resolved


def grain_volumes(phi: Tensor) -> Tensor:
    """``V_k = sum_x phi_k(x)`` for the first batch member, shape ``(N,)``."""

    return phi[0].reshape(phi.shape[1], -1).sum(dim=1)


def simplex_error(phi: Tensor) -> float:
    """Maximum voxelwise ``|sum_k phi_k - 1|``."""

    return float((phi.sum(dim=1) - 1.0).abs().max())


def argmax_labels(phi: Tensor) -> np.ndarray:
    """``uint8`` argmax label of every voxel of the first batch member."""

    if phi.shape[1] > 256:
        raise ValueError("uint8 labels hold at most 256 phases")
    return phi[0].argmax(dim=0).to(torch.uint8).cpu().numpy()


@dataclass
class PhaseLocalRollout:
    """Result of :func:`rollout_from_labels`.

    ``volumes[s]`` holds the grain volumes after ``s`` learned steps (row 0 is the
    initial field). ``violations`` lists every step that was non-finite, negative or
    off the simplex by more than the tolerance; a non-finite state stops the rollout.
    """

    steps_requested: int
    steps_completed: int
    volumes: np.ndarray
    labels: dict[int, np.ndarray] = field(default_factory=dict)
    fields: dict[int, Tensor] = field(default_factory=dict)
    violations: list[dict[str, Any]] = field(default_factory=list)
    min_value: float = 0.0
    max_simplex_error: float = 0.0

    @property
    def admissible(self) -> bool:
        """Criterion (ii): every step completed, finite, nonnegative and on the simplex."""

        return self.steps_completed == self.steps_requested and not self.violations


def rollout_from_labels(
    model: PhaseLocalRule,
    labels: np.ndarray | Tensor,
    steps: int,
    *,
    phases: int | None = None,
    label_steps: Iterable[int] = (),
    field_steps: Iterable[int] = (),
    device: str | torch.device = "cpu",
    deterministic_gpu: bool = False,
    simplex_tolerance: float = SIMPLEX_TOLERANCE,
) -> PhaseLocalRollout:
    """Run ``steps`` autonomous learned steps from the one-hot field of ``labels``.

    The model is moved to ``device`` and put in evaluation mode. Volumes are recorded at
    every step; labels and fields only at the requested steps (step 0 allowed).
    """

    if steps < 0:
        raise ValueError("steps must be nonnegative")
    target = configure_device(device, deterministic_gpu=deterministic_gpu)
    model = model.to(target).eval()
    label_steps = {int(s) for s in label_steps}
    field_steps = {int(s) for s in field_steps}
    phi = one_hot_from_labels(labels, phases).to(target)

    volumes = [grain_volumes(phi).cpu().numpy()]
    result = PhaseLocalRollout(steps_requested=int(steps), steps_completed=0,
                               volumes=np.empty((0, phi.shape[1])))
    if 0 in label_steps:
        result.labels[0] = argmax_labels(phi)
    if 0 in field_steps:
        result.fields[0] = phi.cpu().clone()
    min_value, max_error = float(phi.min()), simplex_error(phi)

    with torch.no_grad():
        for s in range(1, int(steps) + 1):
            phi = phase_local_step(model, phi)
            if not bool(torch.isfinite(phi).all()):
                result.violations.append({"step": s, "kind": "non-finite"})
                break
            low, error = float(phi.min()), simplex_error(phi)
            min_value, max_error = min(min_value, low), max(max_error, error)
            if low < 0.0:
                result.violations.append({"step": s, "kind": "negative", "min": low})
            if error > simplex_tolerance:
                result.violations.append({"step": s, "kind": "simplex", "error": error})
            volumes.append(grain_volumes(phi).cpu().numpy())
            result.steps_completed = s
            if s in label_steps:
                result.labels[s] = argmax_labels(phi)
            if s in field_steps:
                result.fields[s] = phi.cpu().clone()

    result.volumes = np.stack(volumes).astype(np.float64, copy=False)
    result.min_value, result.max_simplex_error = min_value, max_error
    return result


def alive_set(volumes: Sequence[float] | np.ndarray, threshold: float = ALIVE_THRESHOLD) -> tuple[int, ...]:
    """Indices of the phases whose volume exceeds ``threshold``."""

    return tuple(int(k) for k, v in enumerate(np.asarray(volumes, dtype=np.float64)) if v > threshold)


def extinction_events(
    volumes: np.ndarray | Sequence[Sequence[float]],
    threshold: float = ALIVE_THRESHOLD,
) -> dict[int, dict[str, Any]]:
    """Extinction step and first return of every phase that goes extinct.

    ``volumes`` has shape ``(steps + 1, N)``. The extinction step is the first step
    ``s >= 1`` with ``V_k(s) <= threshold``; a later step with ``V_k > threshold`` is
    recorded as a return (the first one only). Phases that never go extinct are absent.
    """

    table = np.asarray(volumes, dtype=np.float64)
    events: dict[int, dict[str, Any]] = {}
    for k in range(table.shape[1]):
        series = table[:, k]
        step = None
        returned = None
        for s in range(1, len(series)):
            if step is None and series[s] <= threshold:
                step = s
            elif step is not None and series[s] > threshold:
                returned = {"step": s, "volume": float(series[s])}
                break
        if step is not None:
            events[k] = {"step": step, "returned": returned}
    return events


def total_variation_error(a: Tensor, b: Tensor) -> float:
    """``E_TV = sum_x sum_k |a_k - b_k| / (2 n_voxels)`` for single-member fields."""

    voxels = a[0, 0].numel()
    return float((a - b).abs().sum() / (2 * voxels))


def persistence_ratio(model_state: Tensor, reference_state: Tensor, initial_state: Tensor) -> float:
    """Persistence-relative field error ``E_TV(model, ref) / E_TV(initial, ref)``."""

    persistence = total_variation_error(initial_state, reference_state)
    if persistence == 0.0:
        raise ValueError("the reference has not moved from the initial field")
    return total_variation_error(model_state, reference_state) / persistence


def max_normalized_volume_error(
    model_volumes: Sequence[float] | np.ndarray,
    reference_volumes: Sequence[float] | np.ndarray,
    mean_initial_volume: float,
) -> float:
    """``max_k |V_k^model - V_k^ref| / (n^3 / N)``; ``n^3 / N`` is the mean initial volume."""

    model = np.asarray(model_volumes, dtype=np.float64)
    reference = np.asarray(reference_volumes, dtype=np.float64)
    if model.shape != reference.shape:
        raise ValueError("model and reference volumes differ in shape")
    return float(np.abs(model - reference).max() / float(mean_initial_volume))


def _admissibility(phi: Tensor) -> dict[str, Any]:
    error, low = simplex_error(phi), float(phi.min())
    finite = bool(torch.isfinite(phi).all())
    return {"simplex_error": error, "min": low, "finite": finite,
            "pass": bool(finite and low >= 0.0 and error <= SIMPLEX_TOLERANCE)}


def _padding(step, phi: Tensor, zero_phases: int) -> dict[str, Any]:
    base = step(phi)
    padded = torch.zeros(phi.shape[0], phi.shape[1] + zero_phases, *phi.shape[2:],
                         dtype=phi.dtype, device=phi.device)
    padded[:, : phi.shape[1]] = phi
    out = step(padded)
    n = phi.shape[1]
    existing = bool(torch.equal(out[:, :n], base))
    appended = bool(torch.equal(out[:, n:], torch.zeros_like(out[:, n:])))
    return {"existing_bitwise_equal": existing, "appended_exactly_zero": appended,
            "pass": existing and appended, "base": base, "output": out}


def structural_checks(
    model: PhaseLocalRule,
    phi: Tensor,
    permutation: Sequence[int] | Tensor,
    *,
    zero_phases: int = 10,
) -> dict[str, Any]:
    """Criterion (i) on one field: one step each of the checks below.

    1. permutation: step on ``phi[:, perm]``, permuted back, against the step on
       ``phi``; maximum absolute difference at most :data:`PERMUTATION_TOLERANCE`;
    2. learned step with ``zero_phases`` appended empty phases: existing phases
       bitwise equal, appended phases exactly zero;
    3. the same for one reference step;
    4. every output finite, nonnegative and on the simplex.
    """

    perm = torch.as_tensor(permutation, dtype=torch.int64, device=phi.device)
    if sorted(perm.tolist()) != list(range(phi.shape[1])):
        raise ValueError("permutation must reorder all phases")
    with torch.no_grad():
        base = phase_local_step(model, phi)
        permuted = phase_local_step(model, phi[:, perm])[:, torch.argsort(perm)]
        model_padding = _padding(lambda x: phase_local_step(model, x), phi, zero_phases)
        reference_padding = _padding(obstacle_reference_step, phi, zero_phases)
    model_padding.pop("base")
    difference = float((permuted - base).abs().max())
    outputs: Mapping[str, Tensor] = {
        "learned_step": base,
        "learned_step_permuted_back": permuted,
        "learned_step_padded": model_padding.pop("output"),
        "reference_step": reference_padding.pop("base"),
        "reference_step_padded": reference_padding.pop("output"),
    }
    admissibility = {name: _admissibility(value) for name, value in outputs.items()}
    permutation_row = {"max_abs_difference": difference,
                       "pass": bool(difference <= PERMUTATION_TOLERANCE)}
    passed = bool(permutation_row["pass"] and model_padding["pass"] and reference_padding["pass"]
                  and all(row["pass"] for row in admissibility.values()))
    return {"permutation": permutation_row, "learned_step_padding": model_padding,
            "reference_step_padding": reference_padding, "admissibility": admissibility,
            "pass": passed}
