"""Phase-channel execution for the phase-local and obstacle-reference maps.

Each physical grain has a stable integer identity and one current phase channel.
Grains may share a channel only while their positive supports are farther apart
than the separation required by the operator stencil. An owner array retains the
physical grain identity at every positive channel voxel. Before every step,
conflicting grains are moved to other channels. After the step, ownership is
extended by at most the stencil radius and the separation condition is checked.

The default buffer is twice the stencil radius. Thus same-channel supports are
strictly farther apart than four radii before a step and remain strictly farther
apart than two radii afterwards. This makes the channel-wise learned and reference
maps equivalent to their dense physical-grain forms up to floating-point summation
order in the shared repair and projection operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from pinn_phase.models.phase_local import PhaseLocalRule, margin_repair
from pinn_phase.physics.obstacle_reference import (
    obstacle_proposal,
    one_hot_from_labels,
    periodic_voronoi_labels,
    project_simplex_euclidean,
)

__all__ = [
    "PhaseChannelState",
    "assign_grains_to_channels",
    "channel_field_from_physical",
    "initial_voronoi_phase_channels",
    "physical_field_from_channels",
    "support_graph",
    "update_channel_ownership",
]


def _dilate(values: Tensor, radius: int) -> Tensor:
    """Periodic Chebyshev maximum filter on ``(channel, X, Y, Z)`` values."""

    if values.dim() != 4 or radius < 0:
        raise ValueError("values must have shape (channel, X, Y, Z) and radius must be nonnegative")
    if radius == 0:
        return values
    result = values.unsqueeze(0)
    for spatial_axis in range(3):
        reach = min(radius, result.shape[spatial_axis + 2] // 2)
        padding = [0, 0, 0, 0, 0, 0]
        padding[2 * (2 - spatial_axis)] = reach
        padding[2 * (2 - spatial_axis) + 1] = reach
        result = F.pad(result, tuple(padding), mode="circular")
        kernel = [1, 1, 1]
        kernel[spatial_axis] = 2 * reach + 1
        result = F.max_pool3d(result, kernel_size=tuple(kernel), stride=1)
    return result[0]


def support_graph(supports: Tensor | np.ndarray, separation: int) -> list[set[int]]:
    """Return the periodic support-proximity graph at Chebyshev distance ``separation``.

    ``supports`` has shape ``(grain, X, Y, Z)``. Two nonempty grains are adjacent
    exactly when their supports are no farther apart than ``separation`` voxels.
    """

    values = torch.as_tensor(supports, dtype=torch.bool)
    if values.dim() != 4 or separation < 0:
        raise ValueError("supports must have shape (grain, X, Y, Z) and separation must be nonnegative")
    count = values.shape[0]
    graph = [set() for _ in range(count)]
    for grain in range(count):
        if not bool(values[grain].any()):
            continue
        nearby = _dilate(values[grain : grain + 1].to(torch.float32), separation)[0] > 0
        for other in range(grain + 1, count):
            if bool(values[other].any()) and bool((nearby & values[other]).any()):
                graph[grain].add(other)
                graph[other].add(grain)
    return graph


def _color_graph(graph: list[set[int]], reserve: int) -> tuple[list[int], int]:
    if reserve < 0:
        raise ValueError("reserve must be nonnegative")
    assignment = [-1] * len(graph)
    for grain in sorted(range(len(graph)), key=lambda index: (-len(graph[index]), index)):
        used = {assignment[other] for other in graph[grain] if assignment[other] >= 0}
        assignment[grain] = next(channel for channel in range(len(graph) + 1) if channel not in used)
    return assignment, max(assignment, default=-1) + 1 + reserve


def assign_grains_to_channels(
    physical: Tensor,
    radius: int,
    *,
    buffer: int | None = None,
    reserve: int = 2,
) -> tuple[list[int], int]:
    """Assign grains by largest-degree-first coloring of the support graph."""

    if physical.dim() != 5 or physical.shape[0] != 1:
        raise ValueError("physical must have shape (1, grain, X, Y, Z)")
    if radius < 1:
        raise ValueError("radius must be positive")
    resolved_buffer = 2 * radius if buffer is None else int(buffer)
    if resolved_buffer < 2 * radius:
        raise ValueError("buffer must be at least twice the stencil radius")
    graph = support_graph(physical[0] > 0, 2 * radius + resolved_buffer)
    return _color_graph(graph, reserve)


def channel_field_from_physical(
    physical: Tensor,
    assignment: list[int],
    channels: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Copy a physical-grain field into phase channels and build its owner array."""

    if physical.dim() != 5 or physical.shape[0] != 1 or physical.dtype != torch.float64:
        raise ValueError("physical must be float64 with shape (1, grain, X, Y, Z)")
    if len(assignment) != physical.shape[1] or not assignment or min(assignment) < 0:
        raise ValueError("one nonnegative channel assignment is required per grain")
    if not bool(torch.isfinite(physical).all()) or bool((physical < 0).any()):
        raise ValueError("physical state must be finite and nonnegative")
    channel_count = max(assignment) + 1 if channels is None else int(channels)
    if channel_count <= max(assignment):
        raise ValueError("channel count does not cover the assignment")
    fields = physical.new_zeros((1, channel_count, *physical.shape[2:]))
    owner_dtype = torch.int16 if physical.shape[1] < 32768 else torch.int32
    owner = torch.full(
        (channel_count, *physical.shape[2:]), -1, dtype=owner_dtype, device=physical.device
    )
    for grain, channel in enumerate(assignment):
        active = physical[0, grain] > 0
        if bool(((owner[channel] >= 0) & active).any()):
            raise ValueError(f"same-voxel owner collision for grain {grain} in channel {channel}")
        fields[0, channel][active] = physical[0, grain][active]
        owner[channel][active] = grain
    return fields, owner


def physical_field_from_channels(fields: Tensor, owner: Tensor, grains: int) -> Tensor:
    """Restore physical grain order from channel values and owner identities."""

    if fields.dim() != 5 or fields.shape[0] != 1 or owner.shape != fields.shape[1:]:
        raise ValueError("field and owner shapes do not agree")
    if bool(((fields[0] > 0) != (owner >= 0)).any()):
        raise ValueError("field and owner positive supports do not agree")
    if bool((owner >= grains).any()) or bool((owner < -1).any()):
        raise ValueError("physical owner is outside the grain range")
    physical = fields.new_zeros((1, grains, *fields.shape[2:]))
    for channel in range(fields.shape[1]):
        for grain in torch.unique(owner[channel]).tolist():
            if grain >= 0:
                active = owner[channel] == grain
                physical[0, grain][active] = fields[0, channel][active]
    return physical


def _owner_extrema(owner: Tensor, radius: int) -> tuple[Tensor, Tensor]:
    identities = owner.to(torch.float32) + 1.0
    maximum = _dilate(identities, radius)
    minimum = -_dilate(torch.where(identities > 0, -identities, -1_000_000.0), radius)
    return minimum, maximum


def update_channel_ownership(old_owner: Tensor, new_fields: Tensor, radius: int, grains: int) -> Tensor:
    """Attribute every new positive voxel to its unique prior grain within ``radius``."""

    if new_fields.dim() != 5 or new_fields.shape[0] != 1 or old_owner.shape != new_fields.shape[1:]:
        raise ValueError("field and owner shapes do not agree")
    result = torch.full_like(old_owner, -1)
    for channel in range(old_owner.shape[0]):
        positive = new_fields[0, channel] > 0
        minimum, maximum = _owner_extrema(old_owner[channel : channel + 1], radius)
        minimum, maximum = minimum[0], maximum[0]
        invalid = positive & ((maximum == 0) | (minimum != maximum))
        if bool(invalid.any()):
            location = torch.nonzero(invalid, as_tuple=False)[0].tolist()
            owner_min = int(minimum[tuple(location)].item()) - 1
            owner_max = int(maximum[tuple(location)].item()) - 1
            raise ValueError(
                "ambiguous or unattributed ownership "
                f"channel={channel} voxel={tuple(location)} owner_range={owner_min},{owner_max}"
            )
        result[channel] = torch.where(positive, maximum.to(old_owner.dtype) - 1, -1)
    if bool((result >= grains).any()):
        raise ValueError("new owner is outside the grain range")
    return result


def _first_conflict(owner: Tensor, separation: int) -> tuple[int, int, tuple[int, int, int]] | None:
    for channel in range(owner.shape[0]):
        identities = owner[channel : channel + 1].to(torch.float32) + 1.0
        conflict = (owner[channel : channel + 1] >= 0) & (_dilate(identities, separation) > identities)
        if bool(conflict.any()):
            location = torch.nonzero(conflict[0], as_tuple=False)[0].tolist()
            return int(owner[channel][tuple(location)].item()), channel, tuple(location)
    return None


def _project_in_slabs(proposal: Tensor, slab: int | None) -> Tensor:
    if slab is None:
        return project_simplex_euclidean(proposal)
    if slab < 1:
        raise ValueError("slab must be positive")
    for start in range(0, proposal.shape[2], slab):
        stop = min(start + slab, proposal.shape[2])
        proposal[:, :, start:stop] = project_simplex_euclidean(proposal[:, :, start:stop])
    return proposal


def _learned_channel_step(
    model: PhaseLocalRule,
    fields: Tensor,
    channels_per_batch: int | None,
    slab: int | None,
) -> Tensor:
    batch, channels = fields.shape[:2]
    step = channels if channels_per_batch is None else int(channels_per_batch)
    if step < 1:
        raise ValueError("channels_per_batch must be positive")
    proposal = torch.empty_like(fields)
    for start in range(0, channels, step):
        stop = min(start + step, channels)
        values = fields[:, start:stop].reshape(batch * (stop - start), 1, *fields.shape[2:])
        increment = model.increment(values.to(torch.float32), periodic=True)
        proposal[:, start:stop] = fields[:, start:stop] + increment.reshape(
            batch, stop - start, *fields.shape[2:]
        ).to(torch.float64)
    if slab is None:
        return project_simplex_euclidean(margin_repair(proposal, fields))
    if slab < 1:
        raise ValueError("slab must be positive")
    for start in range(0, proposal.shape[2], slab):
        stop = min(start + slab, proposal.shape[2])
        proposal[:, :, start:stop] = project_simplex_euclidean(
            margin_repair(proposal[:, :, start:stop], fields[:, :, start:stop])
        )
    return proposal


def _reference_channel_step(fields: Tensor, channels_per_batch: int | None, slab: int | None) -> Tensor:
    channels = fields.shape[1]
    step = channels if channels_per_batch is None else int(channels_per_batch)
    if step < 1:
        raise ValueError("channels_per_batch must be positive")
    proposal = torch.empty_like(fields)
    for start in range(0, channels, step):
        stop = min(start + step, channels)
        proposal[:, start:stop] = obstacle_proposal(fields[:, start:stop])
    return _project_in_slabs(proposal, slab)


@dataclass
class PhaseChannelState:
    """A physical-grain evolution state represented by separated phase channels."""

    fields: Tensor
    owner: Tensor
    assignment: list[int]
    radius: int
    buffer: int
    events: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0

    @classmethod
    def from_physical_field(
        cls,
        physical: Tensor,
        radius: int,
        *,
        buffer: int | None = None,
        reserve: int = 2,
    ) -> "PhaseChannelState":
        resolved_buffer = 2 * radius if buffer is None else int(buffer)
        assignment, channels = assign_grains_to_channels(
            physical, radius, buffer=resolved_buffer, reserve=reserve
        )
        fields, owner = channel_field_from_physical(physical, assignment, channels)
        state = cls(fields, owner, assignment, radius, resolved_buffer)
        state.events.append(
            {
                "step": 0,
                "kind": "initial",
                "assignment": assignment.copy(),
                "channels": channels,
                "minimum_same_channel_distance": state.minimum_same_channel_distance(),
            }
        )
        state.validate_separation(before_step=True)
        return state

    @classmethod
    def from_labels(
        cls,
        labels: Tensor | np.ndarray,
        phases: int | None,
        radius: int,
        *,
        device: str | torch.device = "cpu",
        buffer: int | None = None,
        reserve: int = 2,
    ) -> "PhaseChannelState":
        physical = one_hot_from_labels(labels, phases).to(device)
        return cls.from_physical_field(physical, radius, buffer=buffer, reserve=reserve)

    @property
    def grains(self) -> int:
        return len(self.assignment)

    @property
    def channels(self) -> int:
        return int(self.fields.shape[1])

    def physical_field(self) -> Tensor:
        return physical_field_from_channels(self.fields, self.owner, self.grains)

    def volumes(self) -> Tensor:
        """Diffuse physical-grain volumes, accumulated channel by channel."""

        volumes = self.fields.new_zeros(self.grains)
        deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            if deterministic and self.fields.device.type == "cuda":
                torch.use_deterministic_algorithms(False)
            for channel in range(self.channels):
                identities = self.owner[channel].reshape(-1)
                values = self.fields[0, channel].reshape(-1)
                present = identities >= 0
                volumes += torch.bincount(
                    identities[present].to(torch.int64),
                    weights=values[present],
                    minlength=self.grains,
                )
        finally:
            if deterministic and self.fields.device.type == "cuda":
                torch.use_deterministic_algorithms(True)
        return volumes

    def physical_labels(self) -> Tensor:
        """Return physical-grain labels with dense argmax tie ordering."""

        maximum = self.fields[0].amax(dim=0, keepdim=True)
        identities = torch.where(
            (self.fields[0] == maximum) & (self.owner >= 0), self.owner, self.grains
        )
        labels = identities.amin(dim=0)
        if bool((labels == self.grains).any()):
            raise ValueError("a label voxel has no positive physical owner")
        return labels

    def minimum_same_channel_distance(self) -> int | None:
        """Exact minimum periodic Chebyshev distance between same-channel grains."""

        upper = max(int(size // 2) for size in self.owner.shape[1:])
        low = 0
        high = min(upper, 2 * self.radius + 1)
        while _first_conflict(self.owner, high) is None:
            if high == upper:
                return None
            low = high
            high = min(upper, 2 * high)
        while low < high:
            middle = (low + high) // 2
            if _first_conflict(self.owner, middle) is None:
                low = middle + 1
            else:
                high = middle
        return low

    def reassign_channels(self) -> int:
        """Move grains until every current same-channel pair satisfies the buffer."""

        separation = 2 * self.radius + self.buffer
        moves = 0
        while True:
            found = _first_conflict(self.owner, separation)
            if found is None:
                break
            grain, source, location = found
            if self.assignment[grain] != source:
                raise ValueError(
                    f"assignment and owner disagree for grain {grain} in channel {source}"
                )
            grain_mask = self.owner[source] == grain
            nearby = _dilate(grain_mask.unsqueeze(0).to(torch.float32), separation)[0] > 0
            target = next(
                (
                    channel
                    for channel in range(self.channels)
                    if channel != source and not bool((nearby & (self.owner[channel] >= 0)).any())
                ),
                None,
            )
            if target is None:
                target = self.channels
                self.fields = torch.cat((self.fields, torch.zeros_like(self.fields[:, :1])), dim=1)
                self.owner = torch.cat((self.owner, torch.full_like(self.owner[:1], -1)), dim=0)
                self.events.append(
                    {"step": self.steps, "kind": "allocation", "channel": target, "channels": self.channels}
                )
            active = self.owner[source] == grain
            if bool((self.owner[target][active] >= 0).any()):
                raise ValueError(f"channel reassignment collision for grain {grain}")
            self.fields[0, target][active] = self.fields[0, source][active]
            self.fields[0, source][active] = 0
            self.owner[target][active] = grain
            self.owner[source][active] = -1
            self.assignment[grain] = target
            moves += 1
            self.events.append(
                {
                    "step": self.steps,
                    "kind": "move",
                    "grain": grain,
                    "source": source,
                    "target": target,
                    "conflict_voxel": list(location),
                }
            )
            if moves > self.grains:
                raise ValueError(f"channel reassignment did not converge at step {self.steps}")
        if moves:
            self.events.append(
                {
                    "step": self.steps,
                    "kind": "graph",
                    "moves": moves,
                    "channels": self.channels,
                    "minimum_same_channel_distance": self.minimum_same_channel_distance(),
                }
            )
        return moves

    def validate_separation(self, *, before_step: bool = False) -> None:
        """Raise if same-channel supports violate the required separation."""

        separation = 2 * self.radius + self.buffer if before_step else 2 * self.radius
        found = _first_conflict(self.owner, separation)
        if found is not None:
            grain, channel, location = found
            raise ValueError(
                "same-channel separation failure "
                f"grain={grain} channel={channel} voxel={location}"
            )

    def step(
        self,
        method: Literal["learned", "reference"],
        *,
        model: PhaseLocalRule | None = None,
        channels_per_batch: int | None = None,
        spatial_slab: int | None = None,
    ) -> "PhaseChannelState":
        """Advance one learned or reference step and retain physical identities."""

        self.reassign_channels()
        self.validate_separation(before_step=True)
        if method == "learned":
            if model is None:
                raise ValueError("the learned method requires a phase-local model")
            new_fields = _learned_channel_step(model, self.fields, channels_per_batch, spatial_slab)
        elif method == "reference":
            new_fields = _reference_channel_step(self.fields, channels_per_batch, spatial_slab)
        else:
            raise ValueError("method must be 'learned' or 'reference'")
        new_owner = update_channel_ownership(self.owner, new_fields, self.radius, self.grains)
        self.fields, self.owner = new_fields, new_owner
        self.steps += 1
        self.validate_separation()
        return self


def initial_voronoi_phase_channels(
    centers: Tensor | np.ndarray,
    side: int,
    radius: int,
    *,
    device: str | torch.device = "cpu",
    buffer: int | None = None,
    reserve: int = 2,
) -> PhaseChannelState:
    """Build a sharp periodic Voronoi microstructure and its phase-channel state."""

    labels = periodic_voronoi_labels(centers, side)
    return PhaseChannelState.from_labels(
        labels,
        int(torch.as_tensor(centers).shape[0]),
        radius,
        device=device,
        buffer=buffer,
        reserve=reserve,
    )
