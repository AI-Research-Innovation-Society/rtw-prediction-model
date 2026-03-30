import pandas as pd 
import joblib
import argparse

print("Starting Prediction...")

def load_model():
    "Load the trained model from the specified file.""
    model_path = "model.pkl"