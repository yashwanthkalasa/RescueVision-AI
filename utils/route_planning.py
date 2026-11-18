"""Approximate route ordering for detected rescue targets.

This is a straight-line planning estimate between approximate GPS points,
not a road-network route or a guarantee of safe physical access.
"""

from __future__ import annotations

from math import radians, sin, cos, sqrt, atan2
from typing import Any


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    radius = 6_371_000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    h = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * radius * atan2(sqrt(h), sqrt(max(0.0, 1 - h)))


def _route_length(start: tuple[float, float], ordered: list[dict[str, Any]]) -> float:
    total = 0.0
    current = start
    for item in ordered:
        point = (float(item["latitude"]), float(item["longitude"]))
        total += haversine_m(current, point)
        current = point
    return total


def plan_visit_order(
    start: tuple[float, float],
    targets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    """Nearest-neighbour route followed by 2-opt improvement."""
    remaining = [dict(t) for t in targets]
    if not remaining:
        return [], 0.0

    route: list[dict[str, Any]] = []
    current = start

    while remaining:
        nearest_index = min(
            range(len(remaining)),
            key=lambda i: haversine_m(
                current,
                (float(remaining[i]["latitude"]), float(remaining[i]["longitude"])),
            ),
        )
        chosen = remaining.pop(nearest_index)
        route.append(chosen)
        current = (float(chosen["latitude"]), float(chosen["longitude"]))

    # 2-opt route-order improvement, keeping the start point fixed.
    improved = True
    while improved and len(route) >= 3:
        improved = False
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                before = start if i == 0 else (
                    float(route[i - 1]["latitude"]),
                    float(route[i - 1]["longitude"]),
                )
                a = (float(route[i]["latitude"]), float(route[i]["longitude"]))
                b = (float(route[j]["latitude"]), float(route[j]["longitude"]))
                after = (
                    (float(route[j + 1]["latitude"]), float(route[j + 1]["longitude"]))
                    if j + 1 < len(route)
                    else None
                )
                old = haversine_m(before, a)
                new = haversine_m(before, b)
                if after is not None:
                    old += haversine_m(b, after)
                    new += haversine_m(a, after)
                if new + 0.01 < old:
                    route[i:j + 1] = reversed(route[i:j + 1])
                    improved = True

    return route, _route_length(start, route)
