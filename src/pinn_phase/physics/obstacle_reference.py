"""Projected obstacle reference map of the phase-local realization.

One reference step on a multiphase field ``phi`` of shape ``(batch, N, X, Y, Z)``
(phase axis second, double precision, periodic boundaries) is

    T_o(phi) = Pi_Sigma[ phi + dt_o (kappa_o Lap phi + beta_o phi) ],

where ``Lap`` is the periodic seven-point Laplacian applied to every phase field and
``Pi_Sigma`` is the exact Euclidean projection of the phase vector at every voxel onto
the probability simplex, with one threshold per voxel shared by all phases
(Condat, 2016). The constants are fixed and do not depend on the grid.

This is a different discrete operator from the double-well reference of the hybrid
realization, and :func:`project_simplex_euclidean` is a nearest-point projection,
unlike the clip-and-renormalize map :func:`pinn_phase.physics.explicit_mpf.project_simplex`.

Initial fields are sharp one-hot periodic Voronoi partitions. A one-hot field is a
fixed point of the projection, so an initial field is fully determined by its integer
labels; :func:`one_hot_from_labels` rebuilds it and :func:`load_initial_labels` reads
a distributed label array.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from pinn_phase.io.artifacts import load_npz_arrays, sha256_file

__all__ = [
    "BETA_O",
    "DT_O",
    "DX_O",
    "KAPPA_O",
    "PHASE_DIM",
    "PRESENT_THRESHOLD",
    "laplacian_periodic_7pt",
    "load_initial_labels",
    "obstacle_proposal",
    "obstacle_reference_step",
    "one_hot_from_labels",
    "periodic_voronoi_labels",
    "project_simplex_euclidean",
]

PHASE_DIM = 1

#: kappa_o = 64 / pi^2, written as the double-precision literal every recorded run
#: used. It is one unit in the last place above ``64 / math.pi ** 2`` evaluated in
#: double precision, so the literal is kept rather than the expression.
KAPPA_O = 6.4845557531096185
BETA_O = 1.0
DX_O = 1.0
DT_O = 0.01
#: A phase component is present at a voxel when it is above this value. The
#: projection produces exact zeros, so absence is a true zero.
PRESENT_THRESHOLD = 0.0


def _running_sum(x: Tensor, dim: int) -> Tensor:
    """Cumulative sum along ``dim``.

    On CPU this is ``torch.cumsum``. On any other device it is an explicit sequential
    sum in index order, which is bitwise equal to the CPU kernel and, unlike the CUDA
    cumulative-sum kernel, has a deterministic implementation.
    """

    if x.device.type == "cpu":
        return torch.cumsum(x, dim=dim)
    out = torch.empty_like(x)
    running = x.select(dim, 0).clone()
    out.select(dim, 0).copy_(running)
    for j in range(1, x.shape[dim]):
        running = running + x.select(dim, j)
        out.select(dim, j).copy_(running)
    return out


def project_simplex_euclidean(z: Tensor, *, phase_dim: int = PHASE_DIM) -> Tensor:
    """Exact Euclidean projection of every phase vector onto the probability simplex.

    ``[Pi(z)]_k = max(z_k - tau, 0)`` with one threshold ``tau`` per voxel such that the
    components sum to one (Condat, 2016). The threshold is found with the sort-based
    algorithm of Duchi, Shalev-Shwartz, Singer and Chandra (2008). ``rho`` is the last
    index whose condition holds, taken as an index maximum rather than ``argmax``,
    whose choice among tied maxima is not specified.
    """

    if z.dim() < 1:
        raise ValueError("z must have a phase axis")
    k = z.shape[phase_dim]
    ordered, _ = torch.sort(z, dim=phase_dim, descending=True)
    cumulative = _running_sum(ordered, phase_dim) - 1.0
    shape = [1] * z.dim()
    shape[phase_dim] = k
    index = torch.arange(1, k + 1, dtype=z.dtype, device=z.device).reshape(shape)
    positive = ordered - cumulative / index > 0
    order = torch.arange(k, dtype=torch.int64, device=z.device).reshape(shape)
    rho = (positive.to(torch.int64) * order).amax(dim=phase_dim, keepdim=True)
    theta = torch.gather(cumulative, phase_dim, rho) / (rho.to(z.dtype) + 1.0)
    return torch.clamp(z - theta, min=0.0)


def laplacian_periodic_7pt(phi: Tensor, dx: float = DX_O, *, phase_dim: int = PHASE_DIM) -> Tensor:
    """Periodic seven-point Laplacian on every spatial axis of ``phi``.

    The per-axis term is accumulated as ``out + (roll(+1) + roll(-1) - 2 phi)``; the
    parenthesis fixes the floating-point association of the recorded runs.
    """

    out = torch.zeros_like(phi)
    for axis in range(phi.dim()):
        if axis in (0, phase_dim):
            continue
        out = out + (torch.roll(phi, 1, dims=axis) + torch.roll(phi, -1, dims=axis) - 2.0 * phi)
    return out / (dx * dx)


def obstacle_proposal(
    phi: Tensor,
    *,
    kappa: float = KAPPA_O,
    beta: float = BETA_O,
    dx: float = DX_O,
    h: float = DT_O,
    phase_dim: int = PHASE_DIM,
) -> Tensor:
    """The unconstrained proposal ``phi + h (kappa Lap phi + beta phi)`` before projection."""

    return phi + h * (kappa * laplacian_periodic_7pt(phi, dx, phase_dim=phase_dim) + beta * phi)


def obstacle_reference_step(
    phi: Tensor,
    *,
    kappa: float = KAPPA_O,
    beta: float = BETA_O,
    dx: float = DX_O,
    h: float = DT_O,
    phase_dim: int = PHASE_DIM,
) -> Tensor:
    """One step of the reference map ``T_o``; the reference trajectory iterates it."""

    return project_simplex_euclidean(
        obstacle_proposal(phi, kappa=kappa, beta=beta, dx=dx, h=h, phase_dim=phase_dim),
        phase_dim=phase_dim,
    )


def periodic_voronoi_labels(centers: Tensor | np.ndarray, side: int) -> Tensor:
    """Label every voxel of a periodic ``side^3`` cube with its nearest center.

    ``centers`` has shape ``(N, 3)`` in voxel units. Distances use the minimum image on
    each axis, so the partition is compatible with the periodic Laplacian. Ties keep
    the lower center index. Returns ``int64`` labels of shape ``(side, side, side)``.
    """

    centers = torch.as_tensor(centers, dtype=torch.float64)
    if centers.dim() != 2 or centers.shape[1] != 3 or centers.shape[0] < 1:
        raise ValueError("centers must have shape (N, 3) with N >= 1")
    axis = torch.arange(side, dtype=torch.float64)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    best = None
    label = torch.zeros(side, side, side, dtype=torch.long)
    for k in range(centers.shape[0]):
        delta = grid - centers[k]
        delta = delta - side * torch.round(delta / side)
        distance = (delta * delta).sum(-1)
        if best is None:
            best = distance
        else:
            closer = distance < best
            best = torch.where(closer, distance, best)
            label = torch.where(closer, torch.full_like(label, k), label)
    return label


def one_hot_from_labels(labels: Tensor | np.ndarray, phases: int | None = None) -> Tensor:
    """Sharp initial field ``(1, N, X, Y, Z)`` in double precision from integer labels.

    ``phases`` defaults to ``labels.max() + 1``. The result is a fixed point of
    :func:`project_simplex_euclidean`, so it equals the projected one-hot field exactly.
    """

    labels = torch.as_tensor(np.array(labels)) if isinstance(labels, np.ndarray) else labels
    if labels.dim() != 3:
        raise ValueError("labels must be a three-dimensional array")
    if labels.dtype.is_floating_point or labels.dtype == torch.bool:
        raise ValueError("labels must be integers")
    labels = labels.to(torch.int64)
    if int(labels.min()) < 0:
        raise ValueError("labels must be nonnegative")
    count = int(labels.max()) + 1 if phases is None else int(phases)
    if int(labels.max()) >= count:
        raise ValueError("a label is not smaller than the number of phases")
    one_hot = torch.nn.functional.one_hot(labels, count)
    return one_hot.permute(3, 0, 1, 2).to(torch.float64).unsqueeze(0)


def load_initial_labels(path: str | Path, *, expected_sha256: str | None = None) -> np.ndarray:
    """Read a distributed initial label array (``labels``, ``uint8``, three-dimensional).

    The archive is opened with pickle disabled after its container is validated. When
    ``expected_sha256`` is given, other bytes are refused before anything is decoded.
    """

    digest = expected_sha256 if expected_sha256 is not None else sha256_file(path)
    arrays = load_npz_arrays(path, expected_sha256=digest, required_keys=frozenset({"labels"}))
    labels = arrays["labels"]
    if labels.dtype != np.uint8 or labels.ndim != 3:
        raise ValueError(
            f"initial labels must be a three-dimensional uint8 array, got {labels.dtype} "
            f"with shape {labels.shape}"
        )
    return labels
