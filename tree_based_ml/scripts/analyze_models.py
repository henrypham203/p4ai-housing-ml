"""Tree-based feature importance, training/validation curves, and error analysis.

This script intentionally does NOT run hyperparameter tuning and does NOT create a
new train/test split. It reuses the saved fixed train/test row indices and the CV-selected Gradient Boosting hyperparameters for both datasets.

Run from the project root:
    python scripts/analyze_models.py

Outputs are written under results/analysis/ and figures/analysis/.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS = ROOT / "results"
SPLITS = RESULTS / "splits"
ANALYSIS_DIR = RESULTS / "analysis"
FIGURES = ROOT / "figures" / "analysis"

RANDOM_STATE = 36
N_FOLDS = 5
TOP_N_FEATURES = 20
TOP_N_ERROR_CASES = 20

DATASETS = {
    "Housing": {
        "filename": "Housing.csv",
        "train_indices": "housing_train_indices.csv",
        "test_indices": "housing_test_indices.csv",
    },
    "KingCounty": {
        "filename": "kc_house_data.csv",
        "train_indices": "kingcounty_train_indices.csv",
        "test_indices": "kingcounty_test_indices.csv",
    },
}

PARAMS_FILE = ROOT / "configs" / "selected_model_params.json"
SELECTED_RESULTS_FILE = RESULTS / "tables" / "selected_model_test_metrics.csv"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            "Keep data/, results/, and results/splits/ folders. "
            "Do not create a new split or rerun tuning as a workaround."
        )


def read_row_indices(path: Path, n_rows: int) -> np.ndarray:
    require_file(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Split file is empty: {path}")

    preferred = [
        "row_index_after_dedup", "row_index", "index", "indices",
        "original_index", "idx", "row",
    ]
    lower = {str(col).strip().lower(): col for col in frame.columns}
    selected_col = next((lower[name] for name in preferred if name in lower), None)
    if selected_col is None:
        numeric_cols = [
            col for col in frame.columns
            if pd.to_numeric(frame[col], errors="coerce").notna().all()
        ]
        if not numeric_cols:
            raise ValueError(f"No numeric index column found in {path}")
        selected_col = numeric_cols[0]

    values = pd.to_numeric(frame[selected_col], errors="raise").to_numpy()
    if (
        len(values) == 0
        or not np.isfinite(values).all()
        or not np.equal(values, np.floor(values)).all()
    ):
        raise ValueError(f"Invalid row indices in {path}")

    indices = values.astype(int)
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"Duplicate row indices in {path}")
    if indices.min() < 0 or indices.max() >= n_rows:
        raise ValueError(f"Index outside dataset range in {path}")
    return indices


def make_dense_onehot() -> OneHotEncoder:
    # Support both newer and older scikit-learn keyword names.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def prepare_data(raw: pd.DataFrame, filename: str):
    """Mirror the fixed train/test split / tuned Gradient Boosting preparation, preserving row positions."""
    # fixed train/test split generated indices after exact duplicate removal and index reset.
    data = raw.copy().drop_duplicates().reset_index(drop=True)
    if "price" not in data.columns:
        raise ValueError(f"Target 'price' is missing from {filename}")

    y = pd.to_numeric(data["price"], errors="coerce")
    if y.isna().any():
        raise ValueError(
            f"{filename} contains missing/non-numeric target values. "
            "Rows cannot be dropped because saved fixed train/test split indices depend on row order."
        )

    X = data.drop(columns=["price"]).copy()
    if filename.lower() == "kc_house_data.csv":
        if "id" in X.columns:
            X = X.drop(columns=["id"])
        if "date" in X.columns:
            dates = pd.to_datetime(
                X["date"], format="%Y%m%dT%H%M%S", errors="coerce"
            )
            if dates.isna().any():
                raise ValueError(
                    "Some King County sale dates could not be parsed with the "
                    "expected date format."
                )
            X["sale_year"] = dates.dt.year
            X["sale_month"] = dates.dt.month
            X = X.drop(columns=["date"])
        if "zipcode" in X.columns:
            X["zipcode"] = X["zipcode"].astype("string")

    all_missing = [col for col in X.columns if X[col].isna().all()]
    if all_missing:
        warnings.warn(f"Dropping all-missing features: {all_missing}")
        X = X.drop(columns=all_missing)

    return X, y


def build_pipeline(X_train: pd.DataFrame) -> Pipeline:
    numeric_columns = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()
    categorical_columns = [
        col for col in X_train.columns if col not in numeric_columns
    ]

    transformers = []
    if numeric_columns:
        transformers.append((
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_columns,
        ))
    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_dense_onehot()),
        ])
        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ))
    if not transformers:
        raise ValueError("No usable feature columns remain.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    estimator = GradientBoostingRegressor(random_state=RANDOM_STATE)
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator),
    ])


def make_feature_importance(
    fitted_pipeline: Pipeline,
    original_feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preprocessor = fitted_pipeline.named_steps["preprocess"]
    model = fitted_pipeline.named_steps["model"]
    transformed_names = preprocessor.get_feature_names_out().tolist()
    importances = np.asarray(model.feature_importances_, dtype=float)

    source_features: list[str] = []
    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer == "drop" or columns is None:
            continue
        columns = list(columns)
        if transformer_name == "numeric":
            source_features.extend(columns)
        elif transformer_name == "categorical":
            encoder = transformer.named_steps["onehot"]
            for column, categories in zip(columns, encoder.categories_):
                source_features.extend([column] * len(categories))
        else:
            # This pipeline currently has only numeric and categorical blocks.
            raise ValueError(f"Unexpected transformer block: {transformer_name}")

    if not (len(transformed_names) == len(source_features) == len(importances)):
        raise ValueError(
            "Feature-name/importance count mismatch. Check the fitted preprocessing "
            "pipeline before interpreting feature importance."
        )

    transformed = pd.DataFrame({
        "transformed_feature": transformed_names,
        "source_feature": source_features,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    aggregated = (
        transformed.groupby("source_feature", as_index=False)
        .agg(
            importance=("importance", "sum"),
            encoded_feature_count=("transformed_feature", "count"),
        )
        .sort_values("importance", ascending=False)
    )
    total = float(aggregated["importance"].sum())
    aggregated["importance_share_percent"] = (
        aggregated["importance"] / total * 100.0 if total > 0 else 0.0
    )
    aggregated.insert(0, "rank", range(1, len(aggregated) + 1))
    return transformed, aggregated


def save_feature_importance_plot(
    importance: pd.DataFrame, dataset: str, output_path: Path
) -> None:
    plot_data = importance.head(TOP_N_FEATURES).sort_values("importance")
    height = max(5.0, 0.32 * len(plot_data) + 1.3)
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(plot_data["source_feature"], plot_data["importance"])
    ax.set_title(f"{dataset}: Gradient Boosting Feature Importance")
    ax.set_xlabel("Aggregated impurity-based importance")
    ax.set_ylabel("Original feature")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_error_analysis(
    dataset: str,
    test_indices: np.ndarray,
    y_test: pd.Series,
    predictions: np.ndarray,
):
    actual = np.asarray(y_test, dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    residual = actual - predicted  # positive = underprediction
    denominator = np.maximum(np.abs(actual), np.finfo(float).eps)

    errors = pd.DataFrame({
        "row_index": test_indices,
        "actual_price": actual,
        "predicted_price": predicted,
        "residual_actual_minus_predicted": residual,
        "absolute_error": np.abs(residual),
        "absolute_percentage_error": np.abs(residual) / denominator * 100.0,
    })

    # Test-set bins are used only for post-hoc error diagnosis, never selection.
    bins = pd.qcut(errors["actual_price"], q=5, duplicates="drop")
    # Keep the ordered categorical intervals so groups/plots stay low-to-high.
    errors["actual_price_quantile_bin"] = bins
    grouped = errors.groupby(
        "actual_price_quantile_bin", sort=True, observed=True
    ).agg(
        count=("actual_price", "size"),
        actual_price_min=("actual_price", "min"),
        actual_price_max=("actual_price", "max"),
        actual_price_mean=("actual_price", "mean"),
        MAE=("absolute_error", "mean"),
        RMSE=("residual_actual_minus_predicted", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        mean_residual_actual_minus_predicted=("residual_actual_minus_predicted", "mean"),
        median_absolute_error=("absolute_error", "median"),
    ).reset_index()
    grouped.insert(0, "dataset", dataset)
    errors.insert(0, "dataset", dataset)
    top_errors = errors.nlargest(TOP_N_ERROR_CASES, "absolute_error").copy()
    return errors, grouped, top_errors


def save_prediction_and_residual_plots(
    dataset: str, errors: pd.DataFrame, figures_dir: Path
) -> None:
    actual = errors["actual_price"].to_numpy()
    predicted = errors["predicted_price"].to_numpy()
    residual = errors["residual_actual_minus_predicted"].to_numpy()

    # Actual versus predicted.
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(actual, predicted, alpha=0.65)
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_title(f"{dataset}: Actual vs Predicted Price")
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{dataset.lower()}_actual_vs_predicted.png", dpi=180)
    plt.close(fig)

    # Residuals versus predictions.
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(predicted, residual, alpha=0.65)
    ax.axhline(0, linestyle="--")
    ax.set_title(f"{dataset}: Residuals vs Predicted Price")
    ax.set_xlabel("Predicted price")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{dataset.lower()}_residuals_vs_predicted.png", dpi=180)
    plt.close(fig)

    # Residual distribution.
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.hist(residual, bins=30, edgecolor="black", alpha=0.8)
    ax.axvline(0, linestyle="--")
    ax.set_title(f"{dataset}: Residual Distribution")
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{dataset.lower()}_residual_distribution.png", dpi=180)
    plt.close(fig)

    # Error by actual-price quantile bin.
    grouped = errors.groupby(
        "actual_price_quantile_bin", sort=True, observed=True
    )["absolute_error"].mean()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(range(len(grouped)), grouped.to_numpy())
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped.index.tolist(), rotation=25, ha="right")
    ax.set_title(f"{dataset}: MAE by Actual-Price Quantile")
    ax.set_xlabel("Actual-price quantile bin")
    ax.set_ylabel("Mean absolute error")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{dataset.lower()}_error_by_price_bin.png", dpi=180)
    plt.close(fig)


def fit_diagnostic_model_and_curve(
    dataset: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    figures_dir: Path,
):
    """Fit a diagnostic model on training-only data and plot boosting-stage RMSE.

    For King County, a deterministic 8,000-row sample is used to keep this
    interpretability step practical. This diagnostic model is NOT the model
    used for the final test predictions; those are read from the verified tuned Gradient Boosting
    output generated by the full training partition.
    """
    max_diagnostic_rows = 8000 if dataset == "KingCounty" else len(X_train)
    candidate_positions = np.arange(len(X_train))
    if len(candidate_positions) > max_diagnostic_rows:
        rng = np.random.RandomState(RANDOM_STATE)
        diagnostic_positions = np.sort(
            rng.choice(candidate_positions, size=max_diagnostic_rows, replace=False)
        )
    else:
        diagnostic_positions = candidate_positions

    X_diagnostic = X_train.iloc[diagnostic_positions]
    y_diagnostic = y_train.iloc[diagnostic_positions]
    curve_train_pos, curve_valid_pos = train_test_split(
        np.arange(len(X_diagnostic)), test_size=0.20, random_state=RANDOM_STATE
    )
    X_curve_train = X_diagnostic.iloc[curve_train_pos]
    X_curve_valid = X_diagnostic.iloc[curve_valid_pos]
    y_curve_train = y_diagnostic.iloc[curve_train_pos]
    y_curve_valid = y_diagnostic.iloc[curve_valid_pos]

    diagnostic_pipeline = build_pipeline(X_curve_train).set_params(**{f"model__{key}": value for key, value in params.items()})
    fit_start = time.perf_counter()
    diagnostic_pipeline.fit(X_curve_train, y_curve_train)
    diagnostic_fit_seconds = time.perf_counter() - fit_start

    preprocessor = diagnostic_pipeline.named_steps["preprocess"]
    model = diagnostic_pipeline.named_steps["model"]
    train_matrix = preprocessor.transform(X_curve_train)
    valid_matrix = preprocessor.transform(X_curve_valid)

    rows = []
    train_predictions = model.staged_predict(train_matrix)
    valid_predictions = model.staged_predict(valid_matrix)
    for iteration, (train_pred, valid_pred) in enumerate(
        zip(train_predictions, valid_predictions), start=1
    ):
        rows.append({
            "dataset": dataset,
            "n_estimators_so_far": iteration,
            "training_RMSE": float(np.sqrt(mean_squared_error(y_curve_train, train_pred))),
            "validation_RMSE": float(np.sqrt(mean_squared_error(y_curve_valid, valid_pred))),
            "diagnostic_sample_rows": len(X_diagnostic),
            "curve_train_rows": len(curve_train_pos),
            "curve_validation_rows": len(curve_valid_pos),
            "validation_strategy": "20% holdout from a training-only diagnostic sample",
            "validation_random_state": RANDOM_STATE,
        })

    curve = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(
        curve["n_estimators_so_far"], curve["training_RMSE"],
        label="Training RMSE",
    )
    ax.plot(
        curve["n_estimators_so_far"], curve["validation_RMSE"],
        label="Validation RMSE (training-only holdout)",
    )
    ax.set_title(f"{dataset}: Training / Validation Curve — Gradient Boosting")
    ax.set_xlabel("Number of boosting stages (estimators)")
    ax.set_ylabel("RMSE (lower is better)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{dataset.lower()}_training_validation_curve.png", dpi=180)
    plt.close(fig)
    return curve, diagnostic_pipeline, diagnostic_fit_seconds, len(X_diagnostic), len(X_curve_train), len(X_curve_valid)


def validate_saved_protocol(train_indices, test_indices, n_rows, dataset):
    train_set = set(train_indices.tolist())
    test_set = set(test_indices.tolist())
    if train_set.intersection(test_set):
        raise ValueError(f"{dataset}: train/test split contains overlap.")
    if len(train_indices) + len(test_indices) != n_rows:
        raise ValueError(f"{dataset}: split count does not match row count.")
    if train_set.union(test_set) != set(range(n_rows)):
        raise ValueError(f"{dataset}: split does not cover every row exactly once.")


def write_notes(
    metrics: pd.DataFrame,
    figures_dir: Path,
    notes_path: Path,
    feature_importance: pd.DataFrame,
    error_bins: pd.DataFrame,
    curves: pd.DataFrame,
) -> None:
    lines = [
        "# model analysis — Tree-based Feature Importance and Error Analysis",
        "",
        "## Methodology",
        "- The saved fixed train/test membership is unchanged; the only additional diagnostic validation holdout is drawn from training rows and does not touch the test set.",
        "- The saved split is 80/20 with random_state=42.",
        "- Gradient Boosting hyperparameters are loaded from training-only CV-based model selection.",
        "- Final test predictions are reused from the tuned Gradient Boosting run and checked against the saved fixed test row indices and actual target values.",
        "- A separate diagnostic Gradient Boosting model is fitted only on a deterministic sample of fixed train/test split training rows to compute impurity-based feature importance and the boosting-stage curve; it is not used for final test evaluation.",
        "- The King County diagnostic model uses 8,000 sampled training rows (then an internal 80/20 train/validation split) to keep the interpretability step computationally practical.",
        "- Training/validation curves use a 20% internal holdout from the diagnostic training sample (random_state=36); they are diagnostics, not model selection.",
        "- The held-out test partition is used only for final metrics and post-hoc error diagnosis.",
        "- Numeric missing values are median-imputed; categorical values use most-frequent imputation and one-hot encoding inside the pipeline.",
        "- King County preprocessing drops id, converts date to sale_year/sale_month, and treats zipcode as categorical.",
        "",
        "## Feature-importance interpretation",
        "- Importance is the Gradient Boosting impurity-based feature_importances_ value.",
        "- One-hot encoded category importances are summed to the original categorical feature.",
        "- This importance is model-specific and is not causal evidence; impurity-based importance can favor continuous or high-cardinality features.",
        "",
        "## Results",
        "",
        "| Dataset | CV RMSE | Test MAE | Test RMSE | Test R² | Diagnostic fit (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['CV_RMSE']:.3f} | {row['MAE']:.3f} | "
            f"{row['RMSE']:.3f} | {row['R2']:.4f} | {row['diagnostic_model_fit_seconds']:.3f} |"
        )
    lines.extend(["", "## Descriptive findings"])
    for dataset in metrics["dataset"].tolist():
        ds_features = feature_importance[feature_importance["dataset"] == dataset].head(5)
        feature_text = ", ".join(
            f"{row.source_feature} ({row.importance_share_percent:.1f}%)"
            for row in ds_features.itertuples(index=False)
        )
        lines.append(f"- **{dataset} — leading features:** {feature_text}.")

        ds_bins = error_bins[error_bins["dataset"] == dataset]
        if not ds_bins.empty:
            worst = ds_bins.sort_values("MAE", ascending=False).iloc[0]
            lines.append(
                f"- **{dataset} — largest test-set MAE segment:** actual-price bin "
                f"`{worst['actual_price_quantile_bin']}` (n={int(worst['count'])}), "
                f"MAE={worst['MAE']:.3f}, RMSE={worst['RMSE']:.3f}, "
                f"mean residual={worst['mean_residual_actual_minus_predicted']:.3f}."
            )

        ds_curve = curves[curves["dataset"] == dataset]
        if not ds_curve.empty:
            best_stage = ds_curve.loc[ds_curve["validation_RMSE"].idxmin()]
            last_stage = ds_curve.sort_values("n_estimators_so_far").iloc[-1]
            lines.append(
                f"- **{dataset} — training/validation curve:** lowest diagnostic validation "
                f"RMSE={best_stage['validation_RMSE']:.3f} at stage "
                f"{int(best_stage['n_estimators_so_far'])}; at the final stage "
                f"{int(last_stage['n_estimators_so_far'])}, training RMSE="
                f"{last_stage['training_RMSE']:.3f} and validation RMSE="
                f"{last_stage['validation_RMSE']:.3f}. This is descriptive and does not alter the CV-selected model."
            )

    lines.extend([
        "",
        "## Error-analysis conventions",
        "- Residual is defined as actual_price minus predicted_price; positive residual indicates underprediction.",
        "- Test-set price quantiles are used only to describe where errors occur, not to tune or select the model.",
        "- MAE and RMSE are reported separately for each dataset because the datasets represent different housing markets and price scales.",
        "",
        "## Generated figures",
    ])
    for path in sorted(figures_dir.glob("*.png")):
        lines.append(f"- `{path.relative_to(ROOT)}`")
    lines.extend([
        "",
        "## Limitations to state in the report",
        "- The feature importance values are associative model diagnostics, not causal effects.",
        "- Training/validation curves are computed across Gradient Boosting boosting stages on an internal training-only sample; hyperparameters remain selected by the original 5-fold CV.",
        "- Diagnostic-model fit time is measured on the stated sample and must not be reported as the full final model training time.",
        "- Test error analysis is descriptive and must not be used to revise model parameters after seeing the test results.",
    ])
    notes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    # This is a dedicated output directory; remove stale plots from earlier reruns.
    for stale_plot in FIGURES.glob("*.png"):
        stale_plot.unlink()
    require_file(PARAMS_FILE)
    require_file(SELECTED_RESULTS_FILE)

    best_params = json.loads(PARAMS_FILE.read_text(encoding="utf-8"))
    selected_results = pd.read_csv(SELECTED_RESULTS_FILE)

    all_transformed_importance = []
    all_original_importance = []
    all_errors = []
    all_error_bins = []
    all_top_errors = []
    all_training_validation_curves = []
    all_metrics = []
    audit_lines = [
        "MODEL ANALYSIS VALIDATION REPORT",
        "No hyperparameter tuning was run. The saved fixed train/test membership was unchanged; the diagnostic validation holdout was drawn only from training rows.",
        "Training/validation curves use an internal 20% holdout from a deterministic sample of the training partition.",
    ]

    for dataset, config in DATASETS.items():
        print("\n" + "=" * 76)
        print(f"MODEL ANALYSIS — {dataset}")

        data_path = DATA_DIR / config["filename"]
        require_file(data_path)
        raw = pd.read_csv(data_path)
        original_rows = len(raw)
        duplicate_count = int(raw.duplicated().sum())
        X, y = prepare_data(raw, config["filename"])

        train_indices = read_row_indices(
            SPLITS / config["train_indices"], len(X)
        )
        test_indices = read_row_indices(
            SPLITS / config["test_indices"], len(X)
        )
        validate_saved_protocol(train_indices, test_indices, len(X), dataset)

        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        print(
            f"Rows: {len(X)} (raw {original_rows}, duplicates removed {duplicate_count}) | "
            f"train={len(X_train)} | test={len(X_test)}"
        )

        params = best_params.get(dataset)
        if not params:
            raise ValueError(f"No saved CV-based model selection best parameters for {dataset}.")

        selected_row = selected_results[
            selected_results["dataset"].astype(str).str.lower() == dataset.lower()
        ]
        if selected_row.empty:
            raise ValueError(f"{dataset} is missing from CV-based model selection selected-model results.")
        selected_row = selected_row.iloc[0]
        if str(selected_row.get("model", "")).strip().lower() != "gradient boosting":
            raise ValueError(f"{dataset}: CV-based model selection selected model is not Gradient Boosting.")
        cv_rmse = float(selected_row["CV_RMSE"])

        # Reuse tuned Gradient Boosting predictions from the CV-selected model trained on the full
        # training partition. Verify row membership and actual prices.
        prediction_path = RESULTS / "predictions" / "tuned" / f"{dataset.lower()}_gradient_boosting.csv"
        require_file(prediction_path)
        saved_predictions = pd.read_csv(prediction_path)
        required_prediction_columns = {"row_index", "actual_price", "predicted_price"}
        if not required_prediction_columns.issubset(saved_predictions.columns):
            raise ValueError(f"{prediction_path.name} is missing required prediction columns.")
        if saved_predictions["row_index"].duplicated().any():
            raise ValueError(f"{prediction_path.name} contains duplicate row indices.")
        saved_predictions = saved_predictions.set_index("row_index").reindex(test_indices)
        if saved_predictions["predicted_price"].isna().any():
            raise ValueError(f"{prediction_path.name} does not cover the fixed train/test split test rows.")
        if not np.allclose(saved_predictions["actual_price"].to_numpy(), y_test.to_numpy()):
            raise ValueError(f"{prediction_path.name}: saved actual prices do not match fixed train/test split test rows.")
        predictions = saved_predictions["predicted_price"].to_numpy(dtype=float)

        test_mae = float(mean_absolute_error(y_test, predictions))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        test_r2 = float(r2_score(y_test, predictions))
        comparisons = {
            "MAE": (test_mae, float(selected_row["MAE"])),
            "RMSE": (test_rmse, float(selected_row["RMSE"])),
            "R2": (test_r2, float(selected_row["R2"])),
        }
        for metric_name, (new_value, saved_value) in comparisons.items():
            if not math.isclose(new_value, saved_value, rel_tol=1e-5, abs_tol=1e-5):
                audit_lines.append(
                    f"WARNING {dataset}: saved tuned Gradient Boosting predictions yield {metric_name}={new_value:.10f} "
                    f"but CV-based model selection summary records {saved_value:.10f}."
                )

        print("Fitting diagnostic model on training-only sample for feature importance and curve...", flush=True)
        curve_df, diagnostic_model, diagnostic_fit_seconds, diagnostic_rows, curve_train_rows, curve_valid_rows = fit_diagnostic_model_and_curve(
            dataset, params, X_train, y_train, FIGURES
        )
        all_training_validation_curves.append(curve_df)

        transformed_importance, original_importance = make_feature_importance(
            diagnostic_model, X_train.columns.tolist()
        )
        transformed_importance.insert(0, "dataset", dataset)
        original_importance.insert(0, "dataset", dataset)
        all_transformed_importance.append(transformed_importance)
        all_original_importance.append(original_importance)
        save_feature_importance_plot(
            original_importance,
            dataset,
            FIGURES / f"{dataset.lower()}_feature_importance_top{TOP_N_FEATURES}.png",
        )

        errors, error_bins, top_errors = make_error_analysis(
            dataset, test_indices, y_test, predictions
        )
        all_errors.append(errors)
        all_error_bins.append(error_bins)
        all_top_errors.append(top_errors)
        save_prediction_and_residual_plots(dataset, errors, FIGURES)

        metric_row = {
            "dataset": dataset,
            "model": "Gradient Boosting",
            "stage": "Tuned (selected by training-only CV RMSE)",
            "train_rows": len(train_indices),
            "test_rows": len(test_indices),
            "CV_folds": N_FOLDS,
            "CV_RMSE": cv_rmse,
            "MAE": test_mae,
            "RMSE": test_rmse,
            "R2": test_r2,
            "diagnostic_sample_rows": diagnostic_rows,
            "diagnostic_curve_train_rows": curve_train_rows,
            "diagnostic_curve_validation_rows": curve_valid_rows,
            "diagnostic_model_fit_seconds": diagnostic_fit_seconds,
            "best_params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
        }
        all_metrics.append(metric_row)

        audit_lines.extend([
            f"PASS {dataset}: saved train/test split has no overlap and covers all {len(X)} rows.",
            f"PASS {dataset}: training rows={len(train_indices)}, test rows={len(test_indices)}.",
            f"PASS {dataset}: tuned Gradient Boosting saved predictions match the fixed test row indices and actual prices.",
            f"PASS {dataset}: diagnostic Gradient Boosting model uses only fixed train/test split training rows.",
            f"PASS {dataset}: training/validation curve uses an internal holdout from diagnostic training rows.",
            f"INFO {dataset}: diagnostic sample={diagnostic_rows}, curve train={curve_train_rows}, curve validation={curve_valid_rows}.",
            f"INFO {dataset}: diagnostic model fit time={diagnostic_fit_seconds:.4f} seconds (not full final-model training time).",
            f"INFO {dataset}: test MAE={test_mae:.6f}, RMSE={test_rmse:.6f}, R2={test_r2:.8f}.",
        ])
        print(
            f"Test MAE={test_mae:,.3f} | RMSE={test_rmse:,.3f} | "
            f"R2={test_r2:.5f} | diagnostic fit={diagnostic_fit_seconds:.2f}s on {diagnostic_rows} training rows"
        )
        print("Feature importance, error analysis, and training/validation curve completed.")

    outputs = {
        "analysis_feature_importance_transformed.csv": pd.concat(
            all_transformed_importance, ignore_index=True
        ),
        "analysis_feature_importance.csv": pd.concat(
            all_original_importance, ignore_index=True
        ),
        "analysis_test_predictions_with_errors.csv": pd.concat(
            all_errors, ignore_index=True
        ),
        "analysis_error_analysis_by_price_bin.csv": pd.concat(
            all_error_bins, ignore_index=True
        ),
        "analysis_error_cases_top20.csv": pd.concat(
            all_top_errors, ignore_index=True
        ),
        "analysis_training_validation_curve.csv": pd.concat(
            all_training_validation_curves, ignore_index=True
        ),
        "analysis_selected_model_metrics.csv": pd.DataFrame(all_metrics),
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(ANALYSIS_DIR / filename, index=False)

    notes_path = ANALYSIS_DIR / "analysis_notes.md"
    write_notes(
        pd.DataFrame(all_metrics), FIGURES, notes_path,
        outputs["analysis_feature_importance.csv"],
        outputs["analysis_error_analysis_by_price_bin.csv"],
        outputs["analysis_training_validation_curve.csv"],
    )
    report_path = ANALYSIS_DIR / "analysis_validation_report.txt"
    report_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 76)
    print("MODEL ANALYSIS COMPLETED")
    print("No tuning rerun; no new split; test set was used only for final metrics and descriptive error analysis.")
    print(pd.DataFrame(all_metrics)[[
        "dataset", "CV_RMSE", "MAE", "RMSE", "R2", "diagnostic_model_fit_seconds"
    ]].to_string(index=False))
    print("\nSaved CSV files:")
    for filename in outputs:
        print(f"  results/analysis/{filename}")
    print("  results/analysis/analysis_notes.md")
    print("  results/analysis/analysis_validation_report.txt")
    print(f"Figures: {FIGURES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
