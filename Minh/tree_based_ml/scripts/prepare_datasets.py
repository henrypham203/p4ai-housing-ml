from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tree_based_ml.common import DATASETS, DATA_DIR, RESULTS_DIR, SPLIT_DIR, FIGURE_DIR, prepare_features


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (FIGURE_DIR / "data_distributions").mkdir(parents=True, exist_ok=True)
    summary_rows, feature_rows = [], []
    for dataset_name, spec in DATASETS.items():
        path = DATA_DIR / spec["filename"]
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. See data/README.md")
        raw = pd.read_csv(path)
        duplicate_count = int(raw.duplicated().sum())
        if duplicate_count:
            raise ValueError("The current fixed split files assume no duplicate rows. Inspect/reconcile data before regenerating splits.")
        X, y = prepare_features(raw, dataset_name)
        train_idx, test_idx = train_test_split(np.arange(len(raw)), test_size=0.20, random_state=42)
        key = spec["key"]
        pd.DataFrame({"row_index": np.concatenate([train_idx, test_idx]),
                      "split": ["train"] * len(train_idx) + ["test"] * len(test_idx)}).to_csv(SPLIT_DIR / f"{key}_split_indices.csv", index=False)
        pd.DataFrame({"row_index": train_idx}).to_csv(SPLIT_DIR / f"{key}_train_indices.csv", index=False)
        pd.DataFrame({"row_index": test_idx}).to_csv(SPLIT_DIR / f"{key}_test_indices.csv", index=False)
        numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical = [c for c in X.columns if c not in numeric]
        q1, q3 = float(y.quantile(.25)), float(y.quantile(.75))
        iqr = q3 - q1
        summary_rows.append({
            "dataset": dataset_name, "rows": len(raw), "columns_including_target": raw.shape[1],
            "duplicate_rows": duplicate_count, "missing_cells": int(raw.isna().sum().sum()),
            "features_before_encoding": X.shape[1], "numeric_features": len(numeric),
            "categorical_features": len(categorical), "train_rows": len(train_idx),
            "test_rows": len(test_idx), "split_random_state": 42,
            "price_min": float(y.min()), "price_median": float(y.median()),
            "price_mean": float(y.mean()), "price_max": float(y.max()),
            "iqr_outliers_flagged_not_removed": int(((y < q1 - 1.5*iqr) | (y > q3 + 1.5*iqr)).sum())
        })
        for col in X.columns:
            feature_rows.append({
                "dataset": dataset_name, "feature": col, "dtype": str(X[col].dtype),
                "role": "numeric" if col in numeric else "categorical",
                "missing_count": int(X[col].isna().sum()),
                "unique_values": int(X[col].nunique(dropna=True)),
            })
        fig, ax = plt.subplots(figsize=(8,5))
        ax.hist(y, bins=40)
        ax.set(title=f"{dataset_name}: house price distribution", xlabel="Price", ylabel="Observations")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "data_distributions" / f"{key}_price_distribution.png", dpi=160)
        plt.close(fig)
        print(f"{dataset_name}: train={len(train_idx)}, test={len(test_idx)}, features={X.shape[1]}")
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "tables" / "dataset_summary.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(RESULTS_DIR / "tables" / "feature_documentation.csv", index=False)
    metadata = {
        "test_fraction": 0.20, "split_random_state": 42,
        "target": "price", "cv_folds": 5,
        "model_random_state": 36, "search_random_state": 36,
        "selection_metric": "RMSE on training-only 5-fold CV",
        "test_set_use": "final evaluation only",
        "outliers": "flagged with 1.5xIQR; not removed",
    }
    (SPLIT_DIR / "split_protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Data preparation complete. Fixed split indices and dataset documentation saved.")

if __name__ == "__main__":
    main()
