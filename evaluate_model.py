"""
Model Evaluation & Visualization
Generates confusion matrix, feature importance chart, and classification metrics.
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

# Paths
DATA_PATH = "data/poultry_health_dataset.csv"
MODEL_PATH = "model/poultry_disease_model.pkl"
SCALER_PATH = "model/scaler.pkl"
ENCODER_PATH = "model/label_encoder.pkl"
OUTPUT_DIR = "model"


def preprocess(df: pd.DataFrame):
    """Same preprocessing as train_model.py"""
    df = df.copy()
    feed_map = {"High": 2, "Medium": 1, "Low": 0}
    activity_map = {"Active": 2, "Moderate": 1, "Weak": 0}
    df["feed_intake_num"] = df["feed_intake"].map(feed_map)
    df["water_consumption_num"] = df["water_consumption"].map(feed_map)
    df["activity_level_num"] = df["activity_level"].map(activity_map)

    all_symptoms = [
        "Twisted Neck", "Difficulty Breathing", "Coughing", "Sneezing",
        "Reduced Egg Production", "Reduced Feeding", "Nasal Discharge",
        "Swollen Head", "Watery Eyes", "Lethargy", "Diarrhea", "Ruffled Feathers"
    ]
    for symptom in all_symptoms:
        df[f"symptom_{symptom.replace(' ', '_')}"] = df["symptoms"].fillna("None").astype(str).apply(
            lambda x: 1 if symptom in x.split("|") else 0
        )

    feature_cols = [
        "temperature", "humidity", "feed_intake_num",
        "water_consumption_num", "activity_level_num"
    ] + [f"symptom_{s.replace(' ', '_')}" for s in all_symptoms]

    return df[feature_cols].values.astype(np.float32)


def main():
    print("📂 Loading data and model...")
    df = pd.read_csv(DATA_PATH)
    X = preprocess(df)
    y = df["disease"].values

    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(ENCODER_PATH)

    X_s = scaler.transform(X)
    y_enc = le.transform(y)

    y_pred = clf.predict(X_s)
    acc = accuracy_score(y_enc, y_pred)
    print(f"✅ Overall Accuracy: {acc:.4f} ({acc*100:.1f}%)")

    # Cross-validation
    print("\n🔁 5-Fold Cross-Validation:")
    cv_scores = cross_val_score(clf, X_s, y_enc, cv=5, n_jobs=-1)
    print(f"   Scores: {cv_scores}")
    print(f"   Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Classification report
    print("\n📋 Detailed Report:")
    print(classification_report(y_enc, y_pred, target_names=le.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_enc, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix — Poultry Disease Prediction")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"\n📊 Confusion matrix saved to: {cm_path}")
    plt.close()

    # Feature Importance
    feature_names = joblib.load(os.path.join(OUTPUT_DIR, "feature_names.pkl"))
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:15]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[sorted_idx], y=[feature_names[i] for i in sorted_idx], palette="viridis")
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    print(f"📊 Feature importance chart saved to: {fi_path}")
    plt.close()

    print("\n🎉 Evaluation complete!")


if __name__ == "__main__":
    main()
