from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os

app = FastAPI(title="Onion Quality API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
MODEL_PATH = "best.pt"
try:
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.post("/analyze")
async def analyze_onion(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}

    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        results = model(temp_file)
        result = results[0]
        
        top_class_id = result.probs.top1
        top_class_name = result.names[top_class_id]
        confidence = float(result.probs.top1conf.item() * 100)
        
        os.remove(temp_file)
        
        return {
            "status": "success",
            "prediction": top_class_name,
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"message": "Onion Quality Assessment API is running!"}
