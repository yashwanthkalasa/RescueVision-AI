"""ByteTrack-facing candidate identity layer."""

from __future__ import annotations

from typing import Any

from .detection import overlap_over_smaller
from .triage import priority_score

MIN_OBSERVATIONS = 3


def _center_distance_is_close(box_a: list[float], box_b: list[float]) -> bool:
    ax = (box_a[0] + box_a[2]) / 2
    ay = (box_a[1] + box_a[3]) / 2
    bx = (box_b[0] + box_b[2]) / 2
    by = (box_b[1] + box_b[3]) / 2
    scale = max((box_a[2] - box_a[0]) + (box_a[3] - box_a[1]), 1.0)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= scale * 0.75


class CandidateManager:
    def __init__(self) -> None:
        self.tracks: dict[str, dict[str, Any]] = {}
        self.candidates: dict[str, dict[str, Any]] = {}
        self.track_to_candidate: dict[str, str] = {}
        self.next_id = 1

    def _candidate_for_box(self, bbox: list[float]) -> str | None:
        for candidate_id, candidate in self.candidates.items():
            previous = candidate.get("last_bbox")
            if previous and (overlap_over_smaller(bbox, previous) >= 0.50 or _center_distance_is_close(bbox, previous)):
                return candidate_id
        return None

    def update(self, detection: dict[str, Any], latitude: float, longitude: float, crop: Any):
        source = detection["source"]
        tracker_id = detection.get("track_id")
        track_key = f"{source}:{tracker_id}" if tracker_id is not None else f"{source}:box"
        bbox = detection["bbox"]
        track = self.tracks.setdefault(track_key, {
            "observations": 0, "max_confidence": 0.0, "latitude": latitude,
            "longitude": longitude, "last_bbox": bbox, "best_crop": None,
            "source": source, "track_id": tracker_id,
        })
        track["observations"] += 1
        if detection["confidence"] >= track["max_confidence"] and crop is not None:
            track["best_crop"] = crop
        track["max_confidence"] = max(track["max_confidence"], detection["confidence"])
        track.update(latitude=latitude, longitude=longitude, last_bbox=bbox)
        if track["observations"] < MIN_OBSERVATIONS:
            return None, False

        candidate_id = self.track_to_candidate.get(track_key)
        created = False
        if candidate_id is None:
            candidate_id = self._candidate_for_box(bbox)
            if candidate_id is None:
                candidate_id = f"S{self.next_id:03d}"
                self.next_id += 1
                created = True
            self.track_to_candidate[track_key] = candidate_id

        score, priority = priority_score(track["max_confidence"], track["observations"])
        candidate = self.candidates.setdefault(candidate_id, {
            "track_id": tracker_id, "source": source, "latitude": latitude,
            "longitude": longitude, "max_confidence": 0.0, "observations": 0,
            "priority": priority, "triage_score": score, "evidence": None,
            "last_bbox": bbox, "latest": bbox,
        })
        candidate.update(
            latitude=latitude, longitude=longitude, last_bbox=bbox,
            latest=bbox,
            observations=max(candidate["observations"], track["observations"]),
            max_confidence=max(candidate["max_confidence"], track["max_confidence"]),
        )
        if track["best_crop"] is not None:
            candidate["evidence"] = track["best_crop"]
        candidate["triage_score"], candidate["priority"] = priority_score(candidate["max_confidence"], candidate["observations"])
        return candidate_id, created
