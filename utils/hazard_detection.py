
from pathlib import Path
from typing import Any

from ultralytics import YOLO


class HazardDetector:
    """Detect fire and smoke in disaster imagery."""

    def __init__(
        self,
        model_path: Path,
        device: str | int = "cpu",
        confidence: float = 0.25,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.confidence = confidence

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Hazard model not found: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))
        self.names = self.model.names

    def process(self, frame) -> list[dict[str, Any]]:
        """Return detected fire and smoke bounding boxes."""

        result = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=640,
            device=self.device,
            verbose=False,
        )[0]

        detections = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            detections.append({
                "class_name": str(self.names[class_id]).lower(),
                "bbox": box.xyxy[0].cpu().numpy().astype(float).tolist(),
                "confidence": confidence,
            })

        return detections
