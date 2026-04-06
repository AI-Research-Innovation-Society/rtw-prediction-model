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

def main():
    parser = argparse.ArgumentParser(description="Run RTW predictions on claims CSV data.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--output", default="predictions.csv", help="Path to the output CSV file")
    args = parser.parse_args()

    model = load_model()
    claims_df = read_claims(args.input)
    predicted_df = make_predictions(model, claims_df)
    predicted_df.to_csv(args.output, index=False)
    print(f"Saved predictions to '{args.output}'")


if __name__ == "__main__":
    main()








