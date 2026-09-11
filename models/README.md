# Models

Place custom disaster-specific YOLO weights at `models/best.pt`.

The app also searches existing local training output for a compatible `best.pt`. If no custom model is available, it remains usable with the local `yolo11s.pt` COCO person detector and labels the run as fallback mode.
