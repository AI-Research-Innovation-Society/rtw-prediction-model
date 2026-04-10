import sys
from pathlib import Path

# Ensure project root is importable for tests
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from src.preprocess import (
    load_data,
    clean_data,
    encode_features,
    get_features_and_target,
    split_data,
)


def test_clean_data():
    df = pd.DataFrame({
        "Age": [25, None, 40],
        "Name": ["Alice", None, "Charlie"],
        "Start Date": ["01/02/2020", None, "15/03/2019"],
    })

    df_clean = clean_data(df)

    # No missing values should remain
    assert df_clean.isnull().sum().sum() == 0

    # Date column converted
    assert pd.api.types.is_datetime64_any_dtype(df_clean["Start Date"]) 


def test_encode_features():
    df = pd.DataFrame({
        "Gender": ["M", "F", None],
        "RTW Category": [" RTW 13 Weeks ", "RTW 26 Weeks", "Not RTW 26 Weeks"],
    })

    df = clean_data(df)
    df_enc = encode_features(df)

    # Target exists and is numeric
    assert "RTW Category" in df_enc.columns
    assert pd.api.types.is_integer_dtype(df_enc["RTW Category"].dropna())

    # Values must be in the expected set
    assert set(df_enc["RTW Category"].unique()) <= {0, 1, 2}


def test_get_features_and_target():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "RTW Category": [0, 1, 2],
    })

    X, y = get_features_and_target(df)

    assert "RTW Category" not in X.columns
    assert y.shape[0] == X.shape[0]


def test_split_data():
    X = pd.DataFrame({"feat": range(100)})
    # Create a balanced y with three classes
    y = pd.Series([0, 1, 2] * 33 + [0])

    X_train, X_test, y_train, y_test = split_data(X, y)

    # Check sizes
    assert len(X_test) == 20
    assert len(X_train) == 80

    # Stratify should keep class proportions similar
    prop_full = y.value_counts(normalize=True).sort_index()
    prop_train = y_train.value_counts(normalize=True).sort_index()
    prop_test = y_test.value_counts(normalize=True).sort_index()

    # proportions approximately equal (allow small floating differences)
    assert np.allclose(prop_full.values, prop_train.values, atol=0.05)
    assert np.allclose(prop_full.values, prop_test.values, atol=0.05)