"""
Poultry Disease Model Trainer
Trains a Random Forest classifier on the synthetic poultry health dataset.
Saves the model, scaler, and label encoder for later inference.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Paths
DATA_PATH = "data/poultry_health_dataset.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "poultry_disease_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


def preprocess(df: pd.DataFrame):
    """Preprocess raw data into numeric features and labels."""
    df = df.copy()

    # Encode categorical columns
    feed_map = {"High": 2, "Medium": 1, "Low": 0}
    activity_map = {"Active": 2, "Moderate": 1, "Weak": 0}

    df["feed_intake_num"] = df["feed_intake"].map(feed_map)
    df["water_consumption_num"] = df["water_consumption"].map(feed_map)
    df["activity_level_num"] = df["activity_level"].map(activity_map)

    # One-hot encode symptoms
    all_symptoms = [
        "Twisted Neck", "Difficulty Breathing", "Coughing", "Sneezing",
        "Reduced Egg Production", "Reduced Feeding", "Nasal Discharge",
        "Swollen Head", "Watery Eyes", "Lethargy", "Diarrhea", "Ruffled Feathers"
    ]

    for symptom in all_symptoms:
        df[f"symptom_{symptom.replace(' ', '_')}"] = df["symptoms"].fillna("None").astype(str).apply(
            lambda x: 1 if symptom in x.split("|") else 0
        )

    # Feature columns
    feature_cols = [
        "temperature", "humidity", "feed_intake_num",
        "water_consumption_num", "activity_level_num"
    ] + [f"symptom_{s.replace(' ', '_')}" for s in all_symptoms]

    X = df[feature_cols].values.astype(np.float32)
    y = df["disease"].values

    return X, y, feature_cols


def main():
    print("📂 Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Loaded {len(df)} records")

    print("🔧 Preprocessing...")
    X, y, feature_names = preprocess(df)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train Random Forest
    print("🌲 Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    clf.fit(X_train_s, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Test Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Feature importance
    print("\n🔍 Top 10 Feature Importances:")
    importances = sorted(zip(feature_names, clf.feature_importances_), key=lambda x: x[1], reverse=True)
    for name, score in importances[:10]:
        print(f"   {name}: {score:.4f}")

    # Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, ENCODER_PATH)
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

    print(f"\n💾 Model saved to: {MODEL_PATH}")
    print(f"💾 Scaler saved to: {SCALER_PATH}")
    print(f"💾 Label encoder saved to: {ENCODER_PATH}")
    print("\n🎉 Training complete! Run evaluate_model.py for charts.")


if __name__ == "__main__":
    main()
