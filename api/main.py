from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Paths relative to the Train_Poultry_Model folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "poultry_disease_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "label_encoder.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "model", "feature_names.pkl")

app = FastAPI(title="FlockGuard AI — ML API", version="1.0.0")

# Allow the web app to call this API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your web app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(ENCODER_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)

ALL_SYMPTOMS = [
    "Twisted Neck", "Difficulty Breathing", "Coughing", "Sneezing",
    "Reduced Egg Production", "Reduced Feeding", "Nasal Discharge",
    "Swollen Head", "Watery Eyes", "Lethargy", "Diarrhea", "Ruffled Feathers"
]

FEED_MAP = {"High": 2, "Medium": 1, "Low": 0}
ACTIVITY_MAP = {"Active": 2, "Moderate": 1, "Weak": 0}


class PredictRequest(BaseModel):
    temperature: float
    humidity: float
    feed_intake: str = "Medium"
    water_consumption: str = "Medium"
    activity_level: str = "Active"
    symptoms: list[str] = []


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    risk_level: str
    all_probabilities: dict[str, float]
    model_source: str = "ml_random_forest"


@app.get("/")
def read_root():
    return {
        "message": "FlockGuard AI ML API is running",
        "model": "RandomForestClassifier",
        "classes": list(le.classes_),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        # Build feature vector
        row = {
            "temperature": float(req.temperature),
            "humidity": float(req.humidity),
            "feed_intake_num": FEED_MAP.get(req.feed_intake, 1),
            "water_consumption_num": FEED_MAP.get(req.water_consumption, 1),
            "activity_level_num": ACTIVITY_MAP.get(req.activity_level, 1),
        }
        for symptom in ALL_SYMPTOMS:
            row[f"symptom_{symptom.replace(' ', '_')}"] = 1 if symptom in req.symptoms else 0

        feature_order = [
            "temperature", "humidity", "feed_intake_num",
            "water_consumption_num", "activity_level_num"
        ] + [f"symptom_{s.replace(' ', '_')}" for s in ALL_SYMPTOMS]

        features = np.array([row[f] for f in feature_order], dtype=np.float32).reshape(1, -1)
        features_s = scaler.transform(features)

        pred_idx = clf.predict(features_s)[0]
        proba = clf.predict_proba(features_s)[0]
        label = le.inverse_transform([pred_idx])[0]
        confidence = round(proba[pred_idx] * 100, 1)

        risk = "High" if confidence > 70 else "Medium" if confidence > 40 else "Low"

        all_probs = dict(zip(le.classes_, [round(p * 100, 1) for p in proba]))

        return PredictResponse(
            prediction=label,
            confidence=confidence,
            risk_level=risk,
            all_probabilities=all_probs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True, "classes": list(le.classes_)}


@app.get("/model-info")
def model_info():
    return {
        "algorithm": "RandomForestClassifier",
        "estimators": clf.n_estimators,
        "max_depth": clf.max_depth,
        "classes": list(le.classes_),
        "features": feature_names,
        "feature_count": len(feature_names),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
