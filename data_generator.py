"""
Poultry Health Dataset Generator
Generates synthetic but realistic training data for poultry disease prediction.
Based on veterinary symptom-weights for Newcastle Disease and Avian Influenza.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# Configuration
N_SAMPLES = 5000
OUTPUT_PATH = "data/poultry_health_dataset.csv"

# Symptom lists
ALL_SYMPTOMS = [
    "Twisted Neck",
    "Difficulty Breathing",
    "Coughing",
    "Sneezing",
    "Reduced Egg Production",
    "Reduced Feeding",
    "Nasal Discharge",
    "Swollen Head",
    "Watery Eyes",
    "Lethargy",
    "Diarrhea",
    "Ruffled Feathers",
]

# Disease-specific symptom weights (higher = more likely for that disease)
NEWCASTLE_WEIGHTS = {
    "Twisted Neck": 0.80,
    "Difficulty Breathing": 0.60,
    "Coughing": 0.55,
    "Sneezing": 0.50,
    "Reduced Egg Production": 0.45,
    "Reduced Feeding": 0.40,
    "Nasal Discharge": 0.35,
    "Swollen Head": 0.20,
    "Watery Eyes": 0.25,
    "Lethargy": 0.50,
    "Diarrhea": 0.30,
    "Ruffled Feathers": 0.35,
}

AVIAN_WEIGHTS = {
    "Twisted Neck": 0.10,
    "Difficulty Breathing": 0.70,
    "Coughing": 0.55,
    "Sneezing": 0.50,
    "Reduced Egg Production": 0.40,
    "Reduced Feeding": 0.45,
    "Nasal Discharge": 0.60,
    "Swollen Head": 0.75,
    "Watery Eyes": 0.50,
    "Lethargy": 0.55,
    "Diarrhea": 0.20,
    "Ruffled Feathers": 0.40,
}

HEALTHY_WEIGHTS = {
    "Twisted Neck": 0.02,
    "Difficulty Breathing": 0.03,
    "Coughing": 0.03,
    "Sneezing": 0.05,
    "Reduced Egg Production": 0.05,
    "Reduced Feeding": 0.05,
    "Nasal Discharge": 0.02,
    "Swollen Head": 0.01,
    "Watery Eyes": 0.02,
    "Lethargy": 0.04,
    "Diarrhea": 0.03,
    "Ruffled Feathers": 0.03,
}


def generate_sample(label: str):
    """Generate one synthetic poultry health record."""
    weights = NEWCASTLE_WEIGHTS if label == "Newcastle Disease" else AVIAN_WEIGHTS if label == "Avian Influenza" else HEALTHY_WEIGHTS

    # Number of symptoms (diseased birds have more symptoms)
    if label == "Healthy":
        n_symptoms = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
    else:
        n_symptoms = np.random.randint(3, 8)

    # Pick symptoms based on disease weights
    symptoms = []
    symptom_probs = [weights[s] for s in ALL_SYMPTOMS]
    symptom_probs = np.array(symptom_probs) / sum(symptom_probs)
    chosen = np.random.choice(ALL_SYMPTOMS, size=min(n_symptoms, len(ALL_SYMPTOMS)), replace=False, p=symptom_probs)
    symptoms = list(chosen)

    # Body temperature
    if label == "Healthy":
        temperature = np.random.normal(41.2, 0.4)
    elif label == "Newcastle Disease":
        temperature = np.random.normal(42.5, 0.6)
    else:  # Avian Influenza
        temperature = np.random.normal(42.0, 0.7)
    temperature = round(np.clip(temperature, 39.0, 44.0), 1)

    # Humidity
    if label == "Healthy":
        humidity = np.random.normal(60, 8)
    else:
        humidity = np.random.normal(75, 12)  # stressed birds often in poor humidity
    humidity = round(np.clip(humidity, 20, 100), 1)

    # Feed intake
    if label == "Healthy":
        feed_intake = np.random.choice(["High", "Medium", "Low"], p=[0.5, 0.4, 0.1])
    else:
        feed_intake = np.random.choice(["High", "Medium", "Low"], p=[0.05, 0.25, 0.7])

    # Water consumption
    if label == "Healthy":
        water_consumption = np.random.choice(["High", "Medium", "Low"], p=[0.6, 0.3, 0.1])
    else:
        water_consumption = np.random.choice(["High", "Medium", "Low"], p=[0.1, 0.3, 0.6])

    # Activity level
    if label == "Healthy":
        activity_level = np.random.choice(["Active", "Moderate", "Weak"], p=[0.6, 0.3, 0.1])
    else:
        activity_level = np.random.choice(["Active", "Moderate", "Weak"], p=[0.05, 0.20, 0.75])

    return {
        "temperature": temperature,
        "humidity": humidity,
        "feed_intake": feed_intake,
        "water_consumption": water_consumption,
        "activity_level": activity_level,
        "symptoms": "|".join(symptoms) if symptoms else "None",
        "disease": label,
    }


def generate_dataset(n_samples: int = N_SAMPLES):
    """Generate the full labeled dataset."""
    # Class distribution
    labels = np.random.choice(
        ["Healthy", "Newcastle Disease", "Avian Influenza"],
        size=n_samples,
        p=[0.40, 0.30, 0.30]
    )

    records = [generate_sample(label) for label in labels]
    df = pd.DataFrame(records)
    return df


def main():
    print(f"🐔 Generating {N_SAMPLES} synthetic poultry health records...")
    df = generate_dataset(N_SAMPLES)

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Dataset saved to: {OUTPUT_PATH}")
    print(f"📊 Shape: {df.shape}")
    print(f"🏷️  Class distribution:")
    print(df["disease"].value_counts())
    print("\n🔍 First 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()
