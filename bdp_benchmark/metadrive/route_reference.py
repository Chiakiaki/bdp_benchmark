"""Route-connected MetaDrive lane references for Frenet candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from bdp_benchmark.common.frenet import ReferencePath
from bdp_benchmark.common.features import wrap_angle


@dataclass(frozen=True)
class RoutePosition:
    lane_index: int
    lane: Any
    start_longitudinal: float
    start_lateral: float
    start_heading: float
    lateral_normal_sign: float


@dataclass(frozen=True)
class RouteReferenceResult:
    reference: ReferencePath
    start_lane: Any
    start_longitudinal: float
    start_lateral: float
    start_heading: float
    lateral_normal_sign: float


def _lane_edge(lane: Any) -> tuple[Any, Any]:
    index = getattr(lane, "index", None)
    if not isinstance(index, tuple) or len(index) < 3:
        raise ValueError(f"MetaDrive route lane must have a three-part index, got {index!r}")
    return index[0], index[1]


def _lane_number(lane: Any) -> int:
    index = getattr(lane, "index", None)
    try:
        return int(index[-1])
    except (TypeError, ValueError, IndexError):
        return 0


def _successor_score(previous: Any, candidate: Any) -> tuple[float, float, int]:
    gap = float(np.linalg.norm(np.asarray(previous.end, dtype=np.float64) - np.asarray(candidate.start, dtype=np.float64)))
    previous_heading = float(previous.heading_theta_at(float(previous.length)))
    candidate_heading = float(candidate.heading_theta_at(0.0))
    heading_gap = abs(float(wrap_angle(candidate_heading - previous_heading)))
    lane_gap = abs(_lane_number(candidate) - _lane_number(previous))
    return gap, heading_gap, lane_gap


def _select_connected_successor(previous: Any, candidates: Sequence[Any]) -> Any:
    connected = [candidate for candidate in candidates if previous.is_previous_lane_of(candidate)]
    if not connected:
        previous_index = getattr(previous, "index", None)
        candidate_indices = [getattr(candidate, "index", None) for candidate in candidates]
        raise ValueError(
            "MetaDrive assigned route has no geometrically connected successor for "
            f"lane {previous_index!r}; candidates={candidate_indices!r}"
        )
    return min(connected, key=lambda candidate: _successor_score(previous, candidate))


def resolve_route_lanes(
    navigation: Any,
    current_lane: Any,
    *,
    planning_xy: np.ndarray | None = None,
) -> tuple[Any, ...]:
    """Resolve a connected lane sequence from the current lane through the assigned route."""

    checkpoints = tuple(getattr(navigation, "checkpoints", ()) or ())
    if len(checkpoints) < 2:
        raise ValueError("MetaDrive navigation route must contain at least two checkpoints")
    current_edge = _lane_edge(current_lane)
    route_edges = tuple(zip(checkpoints[:-1], checkpoints[1:]))
    hint = int(getattr(navigation, "_target_checkpoints_index", (0,))[0])
    search_order = list(range(max(0, hint - 1), len(route_edges))) + list(range(0, max(0, hint - 1)))
    edge_index = next((index for index in search_order if route_edges[index] == current_edge), None)
    if edge_index is None:
        edge_index = int(np.clip(hint, 0, len(route_edges) - 1))
        start_node, end_node = route_edges[edge_index]
        route_candidates = tuple(navigation.map.road_network.graph[start_node][end_node])
        if not route_candidates:
            raise ValueError(f"MetaDrive assigned route edge {(start_node, end_node)!r} has no lanes")
        if planning_xy is None:
            current_number = _lane_number(current_lane)
            current_lane = min(route_candidates, key=lambda lane: abs(_lane_number(lane) - current_number))
        else:
            position = np.asarray(planning_xy, dtype=np.float64)
            current_lane = min(route_candidates, key=lambda lane: float(lane.distance(position)))

    graph = navigation.map.road_network.graph
    lanes = [current_lane]
    previous = current_lane
    for start_node, end_node in route_edges[edge_index + 1 :]:
        candidates = tuple(graph[start_node][end_node])
        if not candidates:
            raise ValueError(f"MetaDrive assigned route edge {(start_node, end_node)!r} has no lanes")
        previous = _select_connected_successor(previous, candidates)
        lanes.append(previous)
    return tuple(lanes)


def _closest_route_lane(lanes: Sequence[Any], planning_xy: np.ndarray) -> tuple[int, Any, float, float]:
    candidates = []
    for index, lane in enumerate(lanes):
        longitudinal, lateral = lane.local_coordinates(planning_xy)
        candidates.append((float(lane.distance(planning_xy)), index, lane, float(longitudinal), float(lateral)))
    _distance, index, lane, longitudinal, lateral = min(candidates, key=lambda value: (value[0], value[1]))
    longitudinal = float(np.clip(longitudinal, 0.0, float(lane.length)))
    return index, lane, longitudinal, lateral


def project_route_position(lanes: Sequence[Any], planning_xy: np.ndarray) -> RoutePosition:
    """Project a planning position onto the closest bounded lane in a route sequence."""

    route_lanes = tuple(lanes)
    if not route_lanes:
        raise ValueError("At least one route lane is required")
    position = np.asarray(planning_xy, dtype=np.float64)
    if position.shape != (2,):
        raise ValueError(f"planning_xy must have shape (2,), got {position.shape}")
    lane_index, lane, longitudinal, lateral = _closest_route_lane(route_lanes, position)
    center = np.asarray(lane.position(longitudinal, 0.0), dtype=np.float64)
    heading = float(lane.heading_theta_at(longitudinal))
    standard_left = np.asarray([-np.sin(heading), np.cos(heading)], dtype=np.float64)
    lane_lateral = np.asarray(lane.position(longitudinal, 1.0), dtype=np.float64) - center
    normal_sign = 1.0 if float(np.dot(standard_left, lane_lateral)) >= 0.0 else -1.0
    return RoutePosition(
        lane_index=lane_index,
        lane=lane,
        start_longitudinal=longitudinal,
        start_lateral=lateral,
        start_heading=heading,
        lateral_normal_sign=normal_sign,
    )


def _append_distinct(points: list[np.ndarray], point: np.ndarray) -> None:
    value = np.asarray(point, dtype=np.float64)
    if not points or float(np.linalg.norm(value - points[-1])) > 1.0e-9:
        points.append(value)


def build_route_reference(
    lanes: Sequence[Any],
    *,
    planning_xy: np.ndarray,
    required_length: float,
    point_count: int,
    route_position: RoutePosition | None = None,
) -> RouteReferenceResult:
    """Build a centerline reference from the closest route lane through its successors."""

    route_lanes = tuple(lanes)
    if not route_lanes:
        raise ValueError("At least one route lane is required")
    if required_length <= 0.0:
        raise ValueError("required_length must be positive")
    if point_count < 2:
        raise ValueError("point_count must be at least two")
    projected = route_position or project_route_position(route_lanes, planning_xy)
    lane_index = projected.lane_index
    start_lane = projected.lane
    start_longitudinal = projected.start_longitudinal
    start_lateral = projected.start_lateral
    start_heading = projected.start_heading
    lateral_normal_sign = projected.lateral_normal_sign

    spacing = min(float(required_length) / float(point_count - 1), 1.0)
    spacing = max(spacing, 1.0e-3)
    remaining = float(required_length)
    points: list[np.ndarray] = []
    local_start = start_longitudinal
    for lane in route_lanes[lane_index:]:
        _append_distinct(points, np.asarray(lane.position(local_start, 0.0), dtype=np.float64))
        available = max(float(lane.length) - local_start, 0.0)
        consumed = min(available, remaining)
        if consumed > 1.0e-9:
            intervals = max(1, int(np.ceil(consumed / spacing)))
            for longitudinal in np.linspace(local_start, local_start + consumed, intervals + 1)[1:]:
                _append_distinct(points, np.asarray(lane.position(float(longitudinal), 0.0), dtype=np.float64))
            remaining -= consumed
        if remaining <= 1.0e-9:
            break
        local_start = 0.0

    if len(points) < 2:
        tangent = np.asarray([np.cos(start_heading), np.sin(start_heading)], dtype=np.float64)
        points.append(points[0] + tangent * 1.0e-6)

    reference = ReferencePath.from_xy(np.asarray(points), lateral_normal_sign=lateral_normal_sign)
    return RouteReferenceResult(
        reference=reference,
        start_lane=start_lane,
        start_longitudinal=start_longitudinal,
        start_lateral=start_lateral,
        start_heading=start_heading,
        lateral_normal_sign=lateral_normal_sign,
    )
