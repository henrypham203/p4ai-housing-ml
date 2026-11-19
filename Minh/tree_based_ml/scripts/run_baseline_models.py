from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import json
import time
import numpy as np
import pandas as pd
from tree_based_ml.common import DATASETS, RESULTS_DIR, PREDICTION_DIR, load_split, build_pipeline, regression_metrics, MODEL_NAMES


def run_one(dataset_name, model_name):
    raw, X, y, train_idx, test_idx = load_split(dataset_name)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    pipeline = build_pipeline(model_name, X_train)
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    predictions = pipeline.predict(X_test)
    metrics = regression_metrics(y_test, predictions)
    key = DATASETS[dataset_name]["key"]
    model_key = model_name.lower().replace(" ", "_")
    out = PREDICTION_DIR / "baseline" / f"{key}_{model_key}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "row_index": test_idx, "actual_price": y_test.to_numpy(),
        "predicted_price": predictions,
        "residual_actual_minus_predicted": y_test.to_numpy() - predictions,
        "absolute_error": np.abs(y_test.to_numpy() - predictions),
    }).to_csv(out, index=False)
    row = {"dataset": dataset_name, "model": model_name, "stage": "Baseline",
           **metrics, "fit_time_seconds": fit_seconds, "train_rows": len(train_idx), "test_rows": len(test_idx)}
    print(f"{dataset_name:12s} | {model_name:18s} | MAE={metrics['MAE']:,.3f} | RMSE={metrics['RMSE']:,.3f} | R2={metrics['R2']:.4f} | fit={fit_seconds:.4f}s")
    return row


def main():
    rows = []
    for dataset_name in DATASETS:
        for model_name in MODEL_NAMES:
            rows.append(run_one(dataset_name, model_name))
    out = RESULTS_DIR / "tables" / "baseline_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved baseline metrics: {out.relative_to(RESULTS_DIR.parent)}")

if __name__ == "__main__":
    main()
