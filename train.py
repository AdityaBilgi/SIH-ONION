import os
from ultralytics import YOLO

def train_onion_model(data_yaml_path="data.yaml"):
    """
    Trains a YOLOv8 nano model on the provided dataset.
    We use the nano model (yolov8n.pt) because it's fast and perfect for mobile deployment.
    """
    print("Starting YOLOv8 training for Onion Quality Assessment...")

    # Load a pre-trained YOLOv8 nano model
    model = YOLO("yolov8n.pt") 

    if not os.path.exists(data_yaml_path):
        print(f"\n❌ Error: '{data_yaml_path}' not found!")
        print("Please export your dataset from Roboflow in YOLOv8 format and place the 'data.yaml' file and 'train/val' folders in this directory.")
        return

    # Train the model
    # epochs=25 is a good starting point for a small dataset. We can increase it later.
    # imgsz=640 is standard for YOLOv8.
    results = model.train(
        data=data_yaml_path,
        epochs=25,
        imgsz=640,
        project="onion_grading",
        name="model_v1"
    )

    print("\n✅ Training complete!")
    print("The best model weights are saved at: onion_grading/model_v1/weights/best.pt")
    print("We will use this 'best.pt' file in the backend API to grade onions.")

if __name__ == "__main__":
    train_onion_model()
