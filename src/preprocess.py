# src/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    # ---------------------------------------------------------------------
    # load_data
    #
    # Purpose:
    # - Read a CSV file from disk into a pandas DataFrame.
    #
    # Inputs:
    # - filepath: path-like string to the CSV file.
    #
    # Outputs:
    # - pandas.DataFrame containing the CSV contents.
    #
    # Notes / edge-cases:
    # - Any IO or parsing error will propagate from pandas (FileNotFoundError,
    #   pandas.errors.ParserError, etc.). Callers may want to catch these.
    # ---------------------------------------------------------------------
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # ---------------------------------------------------------------------
    # clean_data
    #
    # Purpose:
    # - Normalize and impute a DataFrame so downstream modeling code can
    #   safely consume it without NaNs in common columns.
    #
    # Inputs:
    # - df: pandas.DataFrame to be cleaned (function operates on a copy).
    #
    # Outputs:
    # - Cleaned pandas.DataFrame with duplicates removed, date columns parsed,
    #   categorical columns imputed by mode (or 'Unknown'), numeric columns
    #   imputed by median, and datetime columns imputed by mode or a fallback
    #   epoch timestamp.
    #
    # Behavior and edge-cases:
    # - Columns with 'date' (case-insensitive) are parsed with dayfirst=True.
    # - If a categorical column has no non-null values, it is filled with
    #   the string 'Unknown'. If a mode exists it is used.
    # - Numeric columns are filled with their median (NaN-safe).
    # - Datetime columns are filled with the mode date when available; if the
    #   entire column is null a fallback of 1970-01-01 is used.
    # - The function returns a copy and does not mutate the input DataFrame.
    # ---------------------------------------------------------------------
    """Clean the DataFrame in-place-safe manner and return a cleaned copy.

    Steps:
    - drop duplicates
    - convert any column with 'date' in the name to datetime (dayfirst=True)
    - fill categorical missing values with mode (fallback 'Unknown')
    - fill numeric missing values with median
    """
    df = df.copy()

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Convert date-like columns to datetime
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    # Identify categorical and numeric columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # Identify datetime columns (after conversion)
    datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

    # Fill categorical columns: mode if available, otherwise 'Unknown'
    for col in cat_cols:
        series = df[col]
        if series.isnull().all():
            df[col] = series.fillna("Unknown")
            continue
        mode_vals = series.mode(dropna=True)
        if not mode_vals.empty:
            fill_val = mode_vals.iloc[0]
        else:
            fill_val = "Unknown"
        df[col] = series.fillna(fill_val)

    # Fill numeric columns with median
    for col in num_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    # Fill datetime columns: use mode if available, else a fallback (1970-01-01)
    for col in datetime_cols:
        series = df[col]
        if series.isnull().all():
            df[col] = series.fillna(pd.Timestamp("1970-01-01"))
            continue
        mode_vals = series.mode(dropna=True)
        if not mode_vals.empty:
            fill_val = mode_vals.iloc[0]
        else:
            # fallback to minimum non-null date or epoch
            non_null = series.dropna()
            fill_val = non_null.min() if not non_null.empty else pd.Timestamp("1970-01-01")
        df[col] = series.fillna(fill_val)

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    # ---------------------------------------------------------------------
    # encode_features
    #
    # Purpose:
    # - Convert human-readable categorical variables into numeric encodings
    #   suitable for ML models. Specifically the target 'RTW Category' is
    #   mapped to integers and remaining object/category columns are
    #   one-hot-encoded.
    #
    # Inputs:
    # - df: pandas.DataFrame containing at least the 'RTW Category' column
    #   (optional for one-hot encoding of other categorical fields).
    #
    # Outputs:
    # - pandas.DataFrame with 'RTW Category' mapped to {0,1,2} and other
    #   categorical columns replaced by dummy columns via pd.get_dummies.
    #
    # Behavior and edge-cases:
    # - Leading/trailing whitespace on the RTW Category values is stripped
    #   before mapping.
    # - Unknown or unmapped target values will become NaN after mapping.
    # - One-hot encoding uses drop_first=True to avoid perfect multicollinearity.
    # ---------------------------------------------------------------------
    """Encode target and categorical features.

    - Strip whitespace from 'RTW Category' and map to integers
    - One-hot encode remaining categorical columns (drop_first=True)
    """
    df = df.copy()

    # Map target
    mapping = {
        "RTW 13 Weeks": 0,
        "RTW 26 Weeks": 1,
        "Not RTW 26 Weeks": 2,
    }

    if "RTW Category" in df.columns:
        # Ensure strings and strip whitespace before mapping
        df["RTW Category"] = df["RTW Category"].astype(str).str.strip()
        df["RTW Category"] = df["RTW Category"].map(mapping)

    # One-hot encode remaining categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Ensure we don't one-hot encode the target if it remains object-like
    if "RTW Category" in cat_cols:
        cat_cols.remove("RTW Category")

    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


def get_features_and_target(df: pd.DataFrame):
    # ---------------------------------------------------------------------
    # get_features_and_target
    #
    # Purpose:
    # - Split a DataFrame into feature matrix X and target vector y using the
    #   column 'RTW Category' as the target.
    #
    # Inputs:
    # - df: pandas.DataFrame that must contain the 'RTW Category' column.
    #
    # Outputs:
    # - X: DataFrame with the target column dropped.
    # - y: Series containing the target values.
    #
    # Behavior and edge-cases:
    # - Raises KeyError if 'RTW Category' is missing.
    # - Returns y as-is (caller may want to verify dtype/NA handling).
    # ---------------------------------------------------------------------
    """Return (X, y) where y is the 'RTW Category' column."""
    if "RTW Category" not in df.columns:
        raise KeyError("RTW Category column is required in the dataframe")

    X = df.drop(columns=["RTW Category"])
    y = df["RTW Category"]
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series):
    # ---------------------------------------------------------------------
    # split_data
    #
    # Purpose:
    # - Split feature matrix X and target y into train/test partitions while
    #   preserving class distribution via stratification.
    #
    # Inputs:
    # - X: pandas.DataFrame of features
    # - y: pandas.Series or array-like of target labels
    #
    # Outputs:
    # - (X_train, X_test, y_train, y_test) as returned by sklearn.model_selection
    #
    # Behavior and edge-cases:
    # - Uses test_size=0.2, random_state=42 for reproducibility, and stratify=y.
    # - If y has too few samples per class for stratification, sklearn will raise
    #   a ValueError; callers should ensure class counts are sufficient.
    # ---------------------------------------------------------------------
    """Split features and target into train/test with stratification."""
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)