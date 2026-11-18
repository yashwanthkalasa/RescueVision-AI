"""Person detection with full-frame tracking and tiled small-person inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None

CUSTOM_CONFIDENCE = 0.20
FALLBACK_CONFIDENCE = 0.18
TILED_CONFIDENCE = 0.12
VEHICLE_CONFIDENCE = 0.25
DUPLICATE_OVERLAP = 0.60
VEHICLE_OVERLAP = 0.50
VEHICLE_CLASSES = (2, 3, 5, 7)

TILE_SIZE = 768
TILE_OVERLAP = 0.25


def box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def overlap_over_smaller(box_a: list[float], box_b: list[float]) -> float:
    smaller = min(box_area(box_a), box_area(box_b))
    return intersection(box_a, box_b) / smaller if smaller else 0.0


def overlap_over_candidate(candidate: list[float], other: list[float]) -> float:
    area = box_area(candidate)
    return intersection(candidate, other) / area if area else 0.0


def _result_detections(result: Any, source: str) -> list[dict[str, Any]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    track_ids_obj = result.boxes.id
    track_ids = (
        track_ids_obj.int().cpu().tolist()
        if track_ids_obj is not None
        else [None] * len(boxes)
    )
    return [
        {
            "bbox": box.astype(float).tolist(),
            "confidence": float(confidence),
            "track_id": track_id,
            "source": source,
        }
        for box, confidence, track_id in zip(boxes, confidences, track_ids)
    ]


def resolve_primary_model(root: Path) -> Path | None:
    candidates = [
        root / "models" / "best.pt",
        root / "runs" / "detect" / "runs" / "disaster_survivor" / "final_sar_yolo11s" / "weights" / "best.pt",
        root / "runs" / "detect" / "runs" / "disaster_survivor" / "combined_test-2" / "weights" / "best.pt",
    ]
    return next((path for path in candidates if path.exists()), None)


def _tile_starts(length: int, tile_size: int) -> list[int]:
    if length <= tile_size:
        return [0]
    step = max(1, int(tile_size * (1.0 - TILE_OVERLAP)))
    starts = list(range(0, max(1, length - tile_size + 1), step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


class DetectorSuite:
    """Run SAR, COCO fallback, vehicle filtering and supplemental tiled inference."""

    def __init__(self, root: Path, device: str | int = "cpu") -> None:
        self.root = root
        self.device = device
        self.primary_path = resolve_primary_model(root)
        self.primary = None
        self.fallback = None
        self.vehicle = None
        self.load_error: str | None = None

        if YOLO is None:
            self.load_error = "Ultralytics is not installed. Install requirements.txt."
            return

        if self.primary_path:
            try:
                self.primary = YOLO(str(self.primary_path))
            except Exception as exc:
                self.load_error = f"Custom SAR model could not load: {exc}"

        try:
            self.fallback = YOLO(str(root / "yolo11s.pt"))
            self.vehicle = YOLO(str(root / "yolo11n.pt"))
        except Exception as exc:
            self.load_error = f"Fallback model could not load: {exc}"

    @property
    def custom_available(self) -> bool:
        return self.primary is not None

    def _tiled_person_detections(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Run prediction on overlapping crops to make small people larger to the model."""
        height, width = frame.shape[:2]
        if self.primary is None and self.fallback is None:
            return []

        tile_w = min(TILE_SIZE, width)
        tile_h = min(TILE_SIZE, height)
        xs = _tile_starts(width, tile_w)
        ys = _tile_starts(height, tile_h)

        # A small image is already processed at full resolution; avoid duplicate work.
        if len(xs) == 1 and len(ys) == 1:
            return []

        tiled: list[dict[str, Any]] = []
        tile_number = 0

        for y0 in ys:
            for x0 in xs:
                tile_number += 1
                crop = frame[y0:min(y0 + tile_h, height), x0:min(x0 + tile_w, width)]
                if crop.size == 0:
                    continue

                models = []
                if self.primary is not None:
                    models.append((self.primary, "SAR-TILED", None))
                if self.fallback is not None:
                    models.append((self.fallback, "FALLBACK-TILED", [0]))

                for model, source, classes in models:
                    kwargs = {
                        "conf": TILED_CONFIDENCE,
                        "imgsz": 960,
                        "device": self.device,
                        "verbose": False,
                    }
                    if classes is not None:
                        kwargs["classes"] = classes
                    result = model.predict(crop, **kwargs)[0]
                    for item in _result_detections(result, source):
                        box = item["bbox"]
                        item["bbox"] = [
                            box[0] + x0,
                            box[1] + y0,
                            box[2] + x0,
                            box[3] + y0,
                        ]
                        # Tiled predictions are not persistent tracker IDs.
                        item["track_id"] = None
                        item["tile_id"] = tile_number
                        tiled.append(item)

        return tiled

    def process(self, frame: np.ndarray) -> tuple[list[dict[str, Any]], int]:
        vehicle_boxes: list[list[float]] = []
        rejected = 0

        if self.vehicle is not None:
            result = self.vehicle.predict(
                frame,
                classes=list(VEHICLE_CLASSES),
                conf=VEHICLE_CONFIDENCE,
                imgsz=640,
                device=self.device,
                verbose=False,
            )[0]
            vehicle_boxes = [
                item["bbox"] for item in _result_detections(result, "VEHICLE")
            ]

        detections: list[dict[str, Any]] = []

        # Full-frame tracking remains in place for video candidate tracking.
        if self.primary is not None:
            result = self.primary.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=CUSTOM_CONFIDENCE,
                imgsz=960,
                device=self.device,
                verbose=False,
            )[0]
            detections.extend(_result_detections(result, "SAR"))

        accepted: list[dict[str, Any]] = []
        for detection in detections:
            if any(
                overlap_over_candidate(detection["bbox"], vehicle) >= VEHICLE_OVERLAP
                for vehicle in vehicle_boxes
            ):
                rejected += 1
            else:
                accepted.append(detection)

        if self.fallback is not None:
            result = self.fallback.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=FALLBACK_CONFIDENCE,
                imgsz=960,
                device=self.device,
                verbose=False,
            )[0]
            for detection in _result_detections(result, "FALLBACK"):
                if any(
                    overlap_over_candidate(detection["bbox"], vehicle) >= VEHICLE_OVERLAP
                    for vehicle in vehicle_boxes
                ):
                    rejected += 1
                elif not any(
                    overlap_over_smaller(detection["bbox"], item["bbox"])
                    >= DUPLICATE_OVERLAP
                    for item in accepted
                ):
                    accepted.append(detection)

        # Supplemental tiled detections are added only when not duplicates.
        for detection in self._tiled_person_detections(frame):
            if any(
                overlap_over_candidate(detection["bbox"], vehicle) >= VEHICLE_OVERLAP
                for vehicle in vehicle_boxes
            ):
                rejected += 1
                continue

            duplicate = any(
                overlap_over_smaller(detection["bbox"], item["bbox"])
                >= DUPLICATE_OVERLAP
                for item in accepted
            )
            if not duplicate:
                accepted.append(detection)

        return accepted, rejected
