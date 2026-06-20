"""
Interactive Prediction Script
Load the trained model and predict disease from user input or batch CSV.
"""

import joblib
import numpy as np
import pandas as pd
import os

# Paths
MODEL_PATH = "model/poultry_disease_model.pkl"
SCALER_PATH = "model/scaler.pkl"
ENCODER_PATH = "model/label_encoder.pkl"
FEATURE_NAMES_PATH = "model/feature_names.pkl"

# Symptom list (must match training)
ALL_SYMPTOMS = [
    "Twisted Neck", "Difficulty Breathing", "Coughing", "Sneezing",
    "Reduced Egg Production", "Reduced Feeding", "Nasal Discharge",
    "Swollen Head", "Watery Eyes", "Lethargy", "Diarrhea", "Ruffled Feathers"
]

FEED_MAP = {"High": 2, "Medium": 1, "Low": 0}
ACTIVITY_MAP = {"Active": 2, "Moderate": 1, "Weak": 0}


def load_model():
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(ENCODER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return clf, scaler, le, feature_names


def encode_record(temperature, humidity, feed_intake, water_consumption, activity_level, symptoms_list):
    """Convert a raw record to the numeric feature vector."""
    row = {
        "temperature": float(temperature),
        "humidity": float(humidity),
        "feed_intake_num": FEED_MAP.get(feed_intake, 1),
        "water_consumption_num": FEED_MAP.get(water_consumption, 1),
        "activity_level_num": ACTIVITY_MAP.get(activity_level, 1),
    }
    for symptom in ALL_SYMPTOMS:
        row[f"symptom_{symptom.replace(' ', '_')}"] = 1 if symptom in symptoms_list else 0

    feature_order = [
        "temperature", "humidity", "feed_intake_num",
        "water_consumption_num", "activity_level_num"
    ] + [f"symptom_{s.replace(' ', '_')}" for s in ALL_SYMPTOMS]

    return np.array([row[f] for f in feature_order], dtype=np.float32).reshape(1, -1)


def predict(clf, scaler, le, features):
    features_s = scaler.transform(features)
    pred_idx = clf.predict(features_s)[0]
    proba = clf.predict_proba(features_s)[0]
    label = le.inverse_transform([pred_idx])[0]
    confidence = round(proba[pred_idx] * 100, 1)
    return label, confidence, dict(zip(le.classes_, [round(p * 100, 1) for p in proba]))


def interactive_predict():
    print("=" * 55)
    print("   🐔 FLOCKGUARD AI — Poultry Disease Predictor")
    print("=" * 55)

    clf, scaler, le, feature_names = load_model()
    print("✅ Model loaded successfully!\n")

    print("Enter the poultry health record details:")
    temp = float(input("  Body Temperature (°C) [e.g. 42.5]: ").strip() or "42.0")
    hum = float(input("  Humidity (%) [e.g. 65]: ").strip() or "60")
    feed = input("  Feed Intake (High/Medium/Low) [Medium]: ").strip() or "Medium"
    water = input("  Water Consumption (High/Medium/Low) [Medium]: ").strip() or "Medium"
    activity = input("  Activity Level (Active/Moderate/Weak) [Active]: ").strip() or "Active"
    symptoms = input("  Symptoms (comma-separated) [e.g. Coughing, Sneezing]: ").strip()
    symptoms_list = [s.strip() for s in symptoms.split(",")] if symptoms else []

    features = encode_record(temp, hum, feed, water, activity, symptoms_list)
    label, confidence, probs = predict(clf, scaler, le, features)

    risk = "High" if confidence > 70 else "Medium" if confidence > 40 else "Low"

    print("\n" + "=" * 55)
    print("   📊 PREDICTION RESULT")
    print("=" * 55)
    print(f"   🏷️  Disease:     {label}")
    print(f"   📈 Confidence:  {confidence}%")
    print(f"   ⚠️  Risk Level:  {risk}")
    print("\n   📋 All Probabilities:")
    for cls, prob in probs.items():
        bar = "█" * int(prob / 3)
        print(f"      {cls:<20} {prob:>6.1f}%  {bar}")
    print("=" * 55)


def batch_predict(csv_path: str, output_path: str = "predictions.csv"):
    """Predict on a batch CSV file."""
    print(f"📂 Loading batch data from {csv_path}...")
    df = pd.read_csv(csv_path)
    clf, scaler, le, _ = load_model()

    results = []
    for _, row in df.iterrows():
        symptoms_list = [s.strip() for s in str(row.get("symptoms", "")).split("|") if s.strip() and s.strip() != "None"]
        features = encode_record(
            row.get("temperature", 41.0),
            row.get("humidity", 60),
            row.get("feed_intake", "Medium"),
            row.get("water_consumption", "Medium"),
            row.get("activity_level", "Active"),
            symptoms_list
        )
        label, confidence, probs = predict(clf, scaler, le, features)
        results.append({
            **row.to_dict(),
            "predicted_disease": label,
            "confidence": confidence,
            **{f"prob_{k}": v for k, v in probs.items()}
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to: {output_path}")
    print(out_df[["predicted_disease", "confidence"]].head(10))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        csv_file = sys.argv[2] if len(sys.argv) > 2 else "data/test_batch.csv"
        batch_predict(csv_file)
    else:
        interactive_predict()
