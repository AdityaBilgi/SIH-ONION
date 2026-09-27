import sys
from ultralytics import YOLO

def predict_onion_quality(image_path):
    print(f"Testing image: {image_path}")
    
    # Load our freshly trained model
    model_path = "runs/classify/onion_grading/cls_model_v1/weights/best.pt"
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run inference
    results = model(image_path)
    
    # Extract prediction
    result = results[0]
    top_class_id = result.probs.top1
    top_class_name = result.names[top_class_id]
    confidence = result.probs.top1conf.item() * 100
    
    print("\n" + "="*30)
    print("🧅 PREDICTION RESULTS 🧅")
    print("="*30)
    print(f"Quality Assessment: {top_class_name.upper()}")
    print(f"Confidence Level:   {confidence:.2f}%")
    print("="*30 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide an image path to test.")
        print("Usage: python test_model.py <path_to_image>")
    else:
        predict_onion_quality(sys.argv[1])
