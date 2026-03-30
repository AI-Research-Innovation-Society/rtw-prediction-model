import pandas as pd 
import joblib
import argparse

print("Starting Prediction...")

def load_model():
    "Load the trained model from the specified file.""
    model_path = "model.pkl"
    print(f"Load models from '{model_path}'...")
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    return model