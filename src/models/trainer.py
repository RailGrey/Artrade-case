"""Model training utilities."""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_BASELINE, MODEL_LGBM, N_FOLDS, RANDOM_STATE


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scaler_path: Path | None = None,
) -> tuple[LogisticRegression, StandardScaler, dict[str, float]]:
    """Train baseline Logistic Regression model."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    model.fit(X_scaled, y_train)

    y_pred_proba = model.predict_proba(X_scaled)[:, 1]

    metrics = {
        "train_auc": roc_auc_score(y_train, y_pred_proba),
        "train_f1": f1_score(y_train, model.predict(X_scaled)),
        "train_precision": precision_score(y_train, model.predict(X_scaled)),
        "train_recall": recall_score(y_train, model.predict(X_scaled)),
    }

    if scaler_path:
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_BASELINE.with_name("scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(MODEL_BASELINE, "wb") as f:
            pickle.dump(model, f)

    return model, scaler, metrics


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = N_FOLDS,
) -> tuple[np.ndarray, dict[str, float]]:
    """Perform stratified k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    y_pred_proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]

    metrics = {
        "cv_auc": roc_auc_score(y, y_pred_proba),
        "cv_f1": f1_score(y, (y_pred_proba > 0.5).astype(int)),
        "cv_precision": precision_score(y, (y_pred_proba > 0.5).astype(int)),
        "cv_recall": recall_score(y, (y_pred_proba > 0.5).astype(int)),
    }

    return y_pred_proba, metrics


def train_lgb_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    params: dict | None = None,
) -> tuple[Any, dict[str, Any], dict[str, float]]:
    """Train LightGBM model."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM not installed. Falling back to RandomForest.")
        return train_random_forest(X_train, y_train, X_val, y_val, params)

    if params is None:
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": RANDOM_STATE,
            "n_estimators": 300,
            "class_weight": "balanced",
        }

    model = lgb.LGBMClassifier(**params)

    if X_val is not None and y_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
    else:
        model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_train)[:, 1]

    metrics = {
        "train_auc": roc_auc_score(y_train, y_pred_proba),
        "train_f1": f1_score(y_train, model.predict(X_train)),
        "train_precision": precision_score(y_train, model.predict(X_train)),
        "train_recall": recall_score(y_train, model.predict(X_train)),
    }

    return model, params, metrics


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    params: dict | None = None,
) -> tuple[Any, dict[str, Any], dict[str, float]]:
    """Train Random Forest model as fallback."""
    from sklearn.ensemble import RandomForestClassifier

    if params is None:
        params = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
            "n_jobs": -1,
        }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_train)[:, 1]

    metrics = {
        "train_auc": roc_auc_score(y_train, y_pred_proba),
        "train_f1": f1_score(y_train, model.predict(X_train)),
        "train_precision": precision_score(y_train, model.predict(X_train)),
        "train_recall": recall_score(y_train, model.predict(X_train)),
    }

    return model, params, metrics


def save_model(model: Any, path: Path = MODEL_LGBM) -> None:
    """Save trained model to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def load_model(path: Path = MODEL_LGBM) -> Any:
    """Load trained model from file."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def get_feature_importance(
    model: Any, feature_names: list[str], top_n: int = 20
) -> pd.DataFrame:
    """Get feature importance from trained model."""
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)

    return importance_df.head(top_n)
