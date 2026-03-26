"""Data loading, cleaning, and preprocessing utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    CLEANED_PARQUET,
    COLS_TO_DROP_LEAKAGE,
    RANDOM_STATE,
    RAW_CSV,
    TARGET_COL,
    TEST_PARQUET,
    TEST_SIZE,
    TRAIN_PARQUET,
)


def load_raw_data(path: Path | str = RAW_CSV) -> pd.DataFrame:
    """Load raw dataset from CSV."""
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def remove_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are not available at order creation time."""
    cols_to_drop = [col for col in COLS_TO_DROP_LEAKAGE if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} leakage columns")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values based on column type."""
    for col in df.columns:
        missing_pct = df[col].isnull().mean()
        if missing_pct > 0.5:
            df = df.drop(columns=[col])
            print(f"Dropped {col} (>{50}% missing)")
            continue

        if df[col].dtype == "object":
            try:
                numeric_vals = pd.to_numeric(df[col], errors="coerce")
                if numeric_vals.notna().mean() > 0.5:
                    df[col] = numeric_vals.fillna(numeric_vals.median())
                else:
                    df[col] = df[col].fillna("Unknown")
            except (ValueError, TypeError):
                df[col] = df[col].fillna("Unknown")
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("Unknown")
    return df


def encode_categoricals(
    df: pd.DataFrame, label_encoders: dict | None = None, cat_cols: list | None = None
) -> tuple[pd.DataFrame, dict]:
    """Encode categorical columns using LabelEncoder."""
    if label_encoders is None:
        label_encoders = {}

    if cat_cols is None:
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in cat_cols:
        if col not in df.columns:
            continue
        if col not in label_encoders:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    return df, label_encoders


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    df = df.copy()
    df = remove_leakage_columns(df)
    df = handle_missing_values(df)

    date_cols = ["sale_date", "lead_Дата создания сделки"]
    for col in date_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    if "lead_id" in df.columns:
        df["lead_id"] = df["lead_id"].astype("category").cat.codes
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")
    print(
        f"Target distribution - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}"
    )

    return X_train, X_test, y_train, y_test


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    train_path: Path = TRAIN_PARQUET,
    test_path: Path = TEST_PARQUET,
) -> None:
    """Save processed train and test sets."""
    train_path.parent.mkdir(parents=True, exist_ok=True)

    train_df = pd.concat([X_train, y_train.rename("target")], axis=1)
    test_df = pd.concat([X_test, y_test.rename("target")], axis=1)

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"Saved train to {train_path}")
    print(f"Saved test to {test_path}")


def load_processed_data(
    train_path: Path = TRAIN_PARQUET,
    test_path: Path = TEST_PARQUET,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load processed train and test sets."""
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    return X_train, X_test, y_train, y_test


def get_feature_info(df: pd.DataFrame) -> dict:
    """Get information about features."""
    info = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "categorical": df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist(),
        "numeric": df.select_dtypes(include=["number"]).columns.tolist(),
        "missing": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
    }
    return info
