"""Model evaluation utilities with visualizations."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Calculate classification metrics."""
    metrics = {
        "accuracy": (y_true == y_pred).mean(),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }

    if y_pred_proba is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_pred_proba)

    return metrics


def plot_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    save_path: Path | None = None,
    normalize: bool = False,
) -> None:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2%" if normalize else "d",
        cmap="Blues",
        xticklabels=["Not Buyout", "Buyout"],
        yticklabels=["Not Buyout", "Buyout"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")

    plt.show()
    plt.close()


def plot_roc_curve(
    y_true: pd.Series | np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Path | None = None,
) -> float:
    """Plot ROC curve and return AUC score."""
    auc_score = roc_auc_score(y_true, y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_score:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved ROC curve to {save_path}")

    plt.show()
    plt.close()

    return auc_score


def plot_precision_recall_curve(
    y_true: pd.Series | np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    baseline = y_true.mean()

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label="Model", linewidth=2)
    plt.axhline(
        y=baseline, color="k", linestyle="--", label=f"Baseline ({baseline:.3f})"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved PR curve to {save_path}")

    plt.show()
    plt.close()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    save_path: Path | None = None,
    top_n: int = 20,
) -> None:
    """Plot feature importance."""
    if importance_df.empty:
        print("No feature importance available")
        return

    top_df = importance_df.head(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=top_df,
        x="importance",
        y="feature",
        palette="viridis",
        orient="h",
    )
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importance")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved feature importance to {save_path}")

    plt.show()
    plt.close()


def plot_threshold_analysis(
    y_true: pd.Series | np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Analyze metrics at different thresholds."""
    thresholds = np.arange(0.1, 1.0, 0.05)
    results = []

    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        results.append(
            {
                "threshold": thresh,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }
        )

    results_df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    plt.plot(results_df["threshold"], results_df["precision"], label="Precision")
    plt.plot(results_df["threshold"], results_df["recall"], label="Recall")
    plt.plot(results_df["threshold"], results_df["f1"], label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Metrics at Different Thresholds")
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved threshold analysis to {save_path}")

    plt.show()
    plt.close()

    return results_df


def analyze_errors(
    X: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    """Analyze prediction errors."""
    results_df = X.copy()
    results_df["y_true"] = y_true.values
    results_df["y_pred"] = y_pred
    results_df["y_pred_proba"] = y_pred_proba
    results_df["error"] = results_df["y_true"] != results_df["y_pred"]

    false_positives = (
        results_df[(results_df["y_true"] == 0) & (results_df["y_pred"] == 1)]
        .sort_values("y_pred_proba", ascending=False)
        .head(top_n)
    )

    false_negatives = (
        results_df[(results_df["y_true"] == 1) & (results_df["y_pred"] == 0)]
        .sort_values("y_pred_proba", ascending=True)
        .head(top_n)
    )

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "all_predictions": results_df,
    }


def print_classification_summary(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None = None,
) -> None:
    """Print comprehensive classification summary."""
    print("=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)

    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)

    for name, value in metrics.items():
        print(f"{name.upper():20s}: {value:.4f}")

    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=["Not Buyout", "Buyout"]))

    print("=" * 60)
