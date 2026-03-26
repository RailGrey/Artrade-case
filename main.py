"""Main entry point for the ML pipeline."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    MODEL_LGBM,
    RAW_CSV,
    TARGET_COL,
    TEST_PARQUET,
    TRAIN_PARQUET,
)
from src.data.loader import (
    encode_categoricals,
    load_processed_data,
    load_raw_data,
    preprocess_data,
    save_processed_data,
    split_data,
)
from src.evaluation.analyze import (
    analyze_errors,
    calculate_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    plot_threshold_analysis,
    print_classification_summary,
)
from src.features.engineer import add_all_features
from src.models.trainer import (
    cross_validate_model,
    get_feature_importance,
    save_model,
    train_baseline,
    train_lgb_model,
)


def run_preprocessing() -> None:
    """Run data preprocessing pipeline."""
    print("=" * 60)
    print("STEP 1: Data Preprocessing")
    print("=" * 60)

    df = load_raw_data(RAW_CSV)
    df = preprocess_data(df)
    df = add_all_features(df)

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"Encoding {len(cat_cols)} categorical columns...")
    df, _ = encode_categoricals(df)

    X_train, X_test, y_train, y_test = split_data(df)
    save_processed_data(X_train, X_test, y_train, y_test)

    print(f"\nProcessed data saved to {TRAIN_PARQUET} and {TEST_PARQUET}")
    print(f"Features created: {X_train.shape[1]}")


def run_training() -> None:
    """Run model training pipeline."""
    print("=" * 60)
    print("STEP 2: Model Training")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_processed_data()

    print("\n--- Baseline: Logistic Regression ---")
    baseline_model, scaler, baseline_metrics = train_baseline(X_train, y_train)
    for name, value in baseline_metrics.items():
        print(f"  {name}: {value:.4f}")

    cv_pred, cv_metrics = cross_validate_model(baseline_model, X_train, y_train)
    print("\n  Cross-validation metrics:")
    for name, value in cv_metrics.items():
        print(f"    {name}: {value:.4f}")

    print("\n--- Main Model: LightGBM ---")
    lgb_model, params, lgb_metrics = train_lgb_model(X_train, y_train, X_test, y_test)
    for name, value in lgb_metrics.items():
        print(f"  {name}: {value:.4f}")

    cv_pred, cv_metrics = cross_validate_model(lgb_model, X_train, y_train)
    print("\n  Cross-validation metrics:")
    for name, value in cv_metrics.items():
        print(f"    {name}: {value:.4f}")

    save_model(lgb_model, MODEL_LGBM)

    print("\n--- Feature Importance ---")
    feature_names = X_train.columns.tolist()
    importance_df = get_feature_importance(lgb_model, feature_names)
    print(importance_df.head(10).to_string(index=False))


def run_evaluation() -> None:
    """Run model evaluation."""
    print("=" * 60)
    print("STEP 3: Model Evaluation")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_processed_data()

    lgb_model, params, _ = train_lgb_model(X_train, y_train, X_test, y_test)

    y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
    y_pred = lgb_model.predict(X_test)

    print_classification_summary(y_test, y_pred, y_pred_proba)

    plot_roc_curve(y_test, y_pred_proba)
    plot_confusion_matrix(y_test, y_pred)
    plot_threshold_analysis(y_test, y_pred_proba)

    importance_df = get_feature_importance(lgb_model, X_train.columns.tolist())
    plot_feature_importance(importance_df)

    errors = analyze_errors(X_test, y_test, y_pred, y_pred_proba)
    print("\n--- Top False Positives (predicted buyout, actually not) ---")
    print(errors["false_positives"][["y_pred_proba", "y_true", "y_pred"]].head())

    print("\n--- Top False Negatives (predicted no buyout, actually bought) ---")
    print(errors["false_negatives"][["y_pred_proba", "y_true", "y_pred"]].head())


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="ML Pipeline for Order Buyout Prediction"
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["preprocess", "train", "evaluate", "all"],
        default="all",
        help="Which step to run",
    )

    args = parser.parse_args()

    if args.step in ("preprocess", "all"):
        run_preprocessing()

    if args.step in ("train", "all"):
        run_training()

    if args.step in ("evaluate", "all"):
        run_evaluation()

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
