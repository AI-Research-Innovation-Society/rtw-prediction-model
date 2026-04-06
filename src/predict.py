import pandas as pd
import joblib
import argparse

print("🚀 Starting the prediction robot...")

def load_model():
    """Load the trained model from the specified file."""
    model_path = "model.pkl"
    print(f"Loading model from '{model_path}'...")
    model = joblib.load(model_path)
    print("Model successfully loaded.")
    return model

def read_claims(file_path):
    print(f"Reading the claim papers from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"✅ Read {len(df)} new claims")
    return df

def make_predictions(model, df):
    print("Asking the magic box for predictions...")

    predictions = model.predict(df)

    df["prediction"] = predictions
    print("✅ Predictions done!")
    return df

if __name__ == "__main__":
    print("Starting Prediction...")
