from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from tree_based_ml.common import DATASETS, RESULTS_DIR, CV_DIR, PREDICTION_DIR, load_split, build_pipeline, regression_metrics, MODEL_NAMES

SEARCH_SPACES = {
    "Decision Tree": {
        "model__max_depth": [None, 5, 8, 10, 15, 20, 30],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8, 12],
        "model__max_features": [None, "sqrt", "log2"],
    },
    "Random Forest": {
        "model__n_estimators": [100, 150, 200],
        "model__max_depth": [None, 10, 20, 30, 40],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": [1.0, "sqrt", "log2"],
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 150, 200, 300],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
        "model__max_depth": [2, 3, 4],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 6],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__max_features": [None, 1.0, "sqrt", "log2"],
    },
}
SEARCH_SETTINGS = {
    "Decision Tree": {"n_iter": 20, "cv_seed": 42, "search_seed": 36, "search_n_jobs": -1},
    "Random Forest": {"n_iter": 10, "cv_seed": 36, "search_seed": 36, "search_n_jobs": 1},
    "Gradient Boosting": {"n_iter": 10, "cv_seed": 36, "search_seed": 36, "search_n_jobs": -1},
}


def run_one(dataset_name, model_name):
    raw, X, y, train_idx, test_idx = load_split(dataset_name)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    settings = SEARCH_SETTINGS[model_name]
    cv = KFold(n_splits=5, shuffle=True, random_state=settings["cv_seed"])
    pipeline = build_pipeline(model_name, X_train)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=SEARCH_SPACES[model_name],
        n_iter=settings["n_iter"],
        scoring="neg_root_mean_squared_error",
        cv=cv,
        refit=True,
        random_state=settings["search_seed"],
        n_jobs=settings["search_n_jobs"],
        verbose=1,
        return_train_score=True,
        error_score="raise",
    )
    print(f"\n{dataset_name} — {model_name}: {settings['n_iter']} candidates × 5 folds")
    start = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)
    metrics = regression_metrics(y_test, predictions)
    key = DATASETS[dataset_name]["key"]
    model_key = model_name.lower().replace(" ", "_")

    cv_frame = pd.DataFrame(search.cv_results_)
    cv_frame["mean_cv_RMSE"] = -cv_frame["mean_test_score"]
    cv_frame["std_cv_RMSE"] = cv_frame["std_test_score"]
    cv_frame["mean_train_RMSE"] = -cv_frame["mean_train_score"]
    CV_DIR.mkdir(parents=True, exist_ok=True)
    cv_frame.sort_values("rank_test_score").to_csv(CV_DIR / f"{key}_{model_key}.csv", index=False)

    pred_dir = PREDICTION_DIR / "tuned"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "row_index": test_idx, "actual_price": y_test.to_numpy(),
        "predicted_price": predictions,
        "residual_actual_minus_predicted": y_test.to_numpy() - predictions,
        "absolute_error": np.abs(y_test.to_numpy() - predictions),
    }).to_csv(pred_dir / f"{key}_{model_key}.csv", index=False)

    row = {
        "dataset": dataset_name, "model": model_name, "stage": "Tuned",
        **metrics, "CV_RMSE": float(-search.best_score_),
        "search_elapsed_seconds_including_refit": float(elapsed),
        "refit_seconds_reported_by_search": float(getattr(search, "refit_time_", np.nan)),
        "candidate_count": settings["n_iter"], "cv_folds": 5,
        "cv_fold_random_state": settings["cv_seed"],
        "search_random_state": settings["search_seed"],
        "best_params": search.best_params_,
    }
    print(f"Best CV RMSE={row['CV_RMSE']:,.3f} | Test MAE={metrics['MAE']:,.3f} | Test RMSE={metrics['RMSE']:,.3f} | R2={metrics['R2']:.4f}")
    print("Best parameters:", search.best_params_)
    return row


def main():
    rows = []
    for dataset_name in DATASETS:
        for model_name in MODEL_NAMES:
            rows.append(run_one(dataset_name, model_name))
    summary = pd.DataFrame([{k:v for k,v in row.items() if k != "best_params"} | {"best_params_json": json.dumps(row["best_params"], sort_keys=True)} for row in rows])
    out = RESULTS_DIR / "tables" / "tuned_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    # Store estimator-level parameter names (without the Pipeline's "model__" prefix)
    # so the JSON can be reused directly by build_pipeline/analyze_models.py.
    params = {
        model: {
            row["dataset"]: {key.replace("model__", "", 1): value for key, value in row["best_params"].items()}
            for row in rows if row["model"] == model
        }
        for model in MODEL_NAMES
    }
    (PROJECT_ROOT / "configs" / "best_hyperparameters_reproduced.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )
    selected_gb = params["Gradient Boosting"]
    (PROJECT_ROOT / "configs" / "selected_model_params.json").write_text(
        json.dumps(selected_gb, indent=2), encoding="utf-8"
    )
    print(f"\nSaved tuned metrics: {out.relative_to(PROJECT_ROOT)}")
    print("Saved best configs: configs/best_hyperparameters_reproduced.json")

if __name__ == "__main__":
    main()
