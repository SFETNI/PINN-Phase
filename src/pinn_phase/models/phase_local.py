"""Phase-local realization: one shared single-channel rule applied to every phase.

The learned rule is

    G_theta(u) = W3 * tanh(W2 * tanh(W1 * u)),

three bias-free 3x3x3 convolutions with widths 1 -> 8 -> 8 -> 1 and periodic padding,
27 (8 + 64 + 8) = 2,160 parameters whatever the number of phases N. It has no hidden
state and no phase-index or coordinate input. One learned step on a field ``phi`` of
shape ``(batch, N, X, Y, Z)`` in double precision is

    z_k       = phi_k + G_theta(phi_k)
    z~_k      = z_k + phi_k max(0, 1 + dt_o beta_o - sum_j z_j)     (margin repair)
    U(phi)    = Pi_Sigma(z~)                                         (Euclidean projection)

The increment is computed in single precision and converted to double precision before
the addition, repair and projection. Phases interact only through the repair and the
projection. A phase that is zero everywhere has a zero increment (bias-free layers,
tanh(0) = 0), receives no repair and stays exactly zero, so appending empty phases
leaves the existing ones unchanged.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pinn_phase.physics.obstacle_reference import BETA_O, DT_O, project_simplex_euclidean

__all__ = [
    "PhaseLocalRule",
    "PeriodicConv3d",
    "margin_repair",
    "phase_local_step",
    "phase_sequential_sum",
]


class PeriodicConv3d(nn.Module):
    """Bias-free cubic 3D convolution, periodic padding or valid (unpadded) mode."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        weight = torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        bound = 1.0 / (in_channels * kernel_size**3) ** 0.5
        weight.uniform_(-bound, bound, generator=generator)
        self.weight = nn.Parameter(weight * scale)
        self.radius = kernel_size // 2

    def forward(self, x: Tensor, periodic: bool = True) -> Tensor:
        if periodic and self.radius:
            r = self.radius
            x = F.pad(x, (r, r, r, r, r, r), mode="circular")
        return F.conv3d(x, self.weight)


class PhaseLocalRule(nn.Module):
    """The shared local rule ``G_theta``; ``forward`` performs one full learned step.

    State-dict keys are ``c1.weight`` (8, 1, 3, 3, 3), ``c2.weight`` (8, 8, 3, 3, 3) and
    ``c3.weight`` (1, 8, 3, 3, 3). The constructor's random initialization is only a
    starting point for training; replay loads distributed weights.
    """

    def __init__(
        self,
        width: int = 8,
        kernel_size: int = 3,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.kernel_size = int(kernel_size)
        self.c1 = PeriodicConv3d(1, width, kernel_size, generator=generator)
        self.c2 = PeriodicConv3d(width, width, kernel_size, generator=generator)
        self.c3 = PeriodicConv3d(width, 1, kernel_size, scale=0.1, generator=generator)

    @property
    def receptive_radius(self) -> int:
        return 3 * (self.kernel_size // 2)

    def increment(self, u: Tensor, periodic: bool = True) -> Tensor:
        """``G_theta`` on single-phase fields ``u`` of shape ``(B, 1, X, Y, Z)``, float32."""

        x = torch.tanh(self.c1(u, periodic))
        x = torch.tanh(self.c2(x, periodic))
        return self.c3(x, periodic)

    def forward(self, phi: Tensor) -> Tensor:
        return phase_local_step(self, phi)


def phase_sequential_sum(x: Tensor, dim: int = 1) -> Tensor:
    """Sum over ``dim`` in index order, one addition per phase.

    Appended all-zero phases add ``+0.0`` and leave the sum bitwise unchanged, which a
    blocked reduction does not guarantee.
    """

    out = x.select(dim, 0).clone()
    for j in range(1, x.shape[dim]):
        out = out + x.select(dim, j)
    return out


def margin_repair(z: Tensor, phi: Tensor, *, h: float = DT_O, beta: float = BETA_O) -> Tensor:
    """``z + phi max(0, 1 + h beta - sum_j z_j)`` on the phase axis (axis 1).

    On a simplex-valued ``phi`` this raises the proposal's phase sum to at least
    ``1 + h beta``, so the projection threshold is positive, and adds mass only where
    ``phi_k > 0``.
    """

    total = phase_sequential_sum(z, 1).unsqueeze(1)
    return z + phi * torch.relu(1.0 + h * beta - total)


def phase_local_step(model: PhaseLocalRule, phi: Tensor) -> Tensor:
    """One learned step on a periodic field ``phi`` (``(B, N, X, Y, Z)``, float64)."""

    if phi.dim() != 5:
        raise ValueError("phi must have shape (batch, phases, X, Y, Z)")
    if phi.dtype != torch.float64:
        raise ValueError("the carried state is double precision")
    b, n = phi.shape[:2]
    u = phi.reshape(b * n, 1, *phi.shape[2:]).to(torch.float32)
    g = model.increment(u, periodic=True)
    z = phi + g.reshape(phi.shape).to(torch.float64)
    return project_simplex_euclidean(margin_repair(z, phi))
