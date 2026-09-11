from ultralytics import YOLO

model = YOLO("yolov8s.pt")

results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    save=True,
    conf=0.25
)

print("DONE!")