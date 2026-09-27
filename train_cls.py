from ultralytics import YOLO

def train_classification_model():
    print("Starting YOLOv8 Image Classification Training...")
    
    # Load a pre-trained YOLOv8 nano classification model
    model = YOLO("yolov8n-cls.pt")
    
    # Train the model on our dataset
    results = model.train(
        data="dataset",      # path to the dataset folder containing train/ and val/
        epochs=15,           # small number for quick testing
        imgsz=224,           # smaller image size for classification
        project="onion_grading",
        name="cls_model_v5"
    )
    
    print("\n✅ Training complete!")
    print("The best classification model weights are saved at: onion_grading/cls_model_v1/weights/best.pt")

if __name__ == "__main__":
    train_classification_model()
