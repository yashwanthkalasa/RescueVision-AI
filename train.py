from ultralytics import YOLO
import torch


def main():

    # Check GPU
    if torch.cuda.is_available():
        device = 0
        print("GPU detected!")
        print("Using:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("CUDA GPU not detected.")
        print("Using CPU.")

    # Load pretrained YOLOv8 Small
    print("\nLoading YOLOv8s...")
    model = YOLO("yolov8s.pt")

    # Train
    print("\nStarting training...")

    model.train(
        data="archive/search-and-rescue/data.yaml",

        # Only testing first
        epochs=3,

        # Good starting size for RTX 3050
        imgsz=640,

        # Safe batch size
        batch=4,

        # GPU 0
        device=device,

        # IMPORTANT FOR WINDOWS
        workers=0,

        patience=10,

        project="runs/disaster_survivor",

        name="sard_yolov8s_test",

        save=True
    )

    print("\nTRAINING FINISHED!")


if __name__ == "__main__":
    main()