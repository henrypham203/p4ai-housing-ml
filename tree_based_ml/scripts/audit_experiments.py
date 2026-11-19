from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import ast
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tree_based_ml.common import DATASETS, RESULTS_DIR, load_split, build_pipeline, MODEL_NAMES


def load_best_params():
    reproduced_path = PROJECT_ROOT / "configs" / "best_hyperparameters_reproduced.json"
    if reproduced_path.exists():
        payload = json.loads(reproduced_path.read_text(encoding="utf-8"))
        return payload

    path = PROJECT_ROOT / "configs" / "best_hyperparameters.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        model: {
            dataset: details["best_params"]
            for dataset, details in info["best_by_dataset"].items()
        }
        for model, info in payload["algorithms"].items()
    }


def normalize_params(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return {str(k).replace("model__", ""): v for k, v in dict(value).items()}


def main():
    parser = argparse.ArgumentParser(description="Audit CV/predictions; optionally remeasure final-fit time.")
    parser.add_argument(
        "--measure-fit-time", action="store_true",
        help="Fit each tuned pipeline 3 times on the complete training partition to refresh comparable timing. This may take several minutes.",
    )
    args = parser.parse_args()

    tables = RESULTS_DIR / "tables"
    cv_dir = RESULTS_DIR / "cv_results"
    predictions_dir = RESULTS_DIR / "predictions" / "tuned"
    audit_dir = RESULTS_DIR / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    tuned = pd.read_csv(tables / "tuned_metrics.csv")
    best_params = load_best_params()
    cv_rows, pred_rows, report = [], [], []
    fit_rows = []
    report.append("EXPERIMENT AUDIT REPORT")
    report.append("No hyperparameter search was run by this audit.")
    report.append("No estimator was fitted on test data.")
    report.append("Final-fit timing is refreshed only when --measure-fit-time is explicitly supplied.")

    for dataset_name, spec in DATASETS.items():
        raw, X, y, train_idx, test_idx = load_split(dataset_name)
        expected_train, expected_test = train_test_split(
            np.arange(len(raw)), test_size=0.20, random_state=42
        )
        split_match = np.array_equal(train_idx, expected_train) and np.array_equal(test_idx, expected_test)
        if not split_match:
            raise ValueError(f"Saved split differs from 80/20 random_state=42 for {dataset_name}")
        report.append(
            f"PASS {dataset_name}: fixed split matches train_test_split(0.20, random_state=42); "
            f"train={len(train_idx)}, test={len(test_idx)}."
        )
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

        for model_name in MODEL_NAMES:
            model_key = model_name.lower().replace(" ", "_")
            cv_file = cv_dir / f"{spec['key']}_{model_key}.csv"
            cv = pd.read_csv(cv_file)
            if "mean_cv_RMSE" in cv.columns:
                cv_rmse = pd.to_numeric(cv["mean_cv_RMSE"], errors="raise")
            else:
                cv_rmse = -pd.to_numeric(cv["mean_test_score"], errors="raise")
            min_cv = float(cv_rmse.min())
            tolerance = max(1e-6, abs(min_cv) * 1e-10)
            tied = cv_rmse.sub(min_cv).abs() <= tolerance
            candidates = [
                normalize_params(ast.literal_eval(p) if isinstance(p, str) else p)
                for p in cv.loc[tied, "params"]
            ]
            expected = best_params[model_name][dataset_name]
            params_match = normalize_params(expected) in candidates
            summary = tuned[(tuned.dataset == dataset_name) & (tuned.model == model_name)].iloc[0]
            score_match = abs(float(summary.CV_RMSE) - min_cv) <= tolerance
            fold_cols = [c for c in cv.columns if c.startswith("split") and c.endswith("_test_score")]
            if len(fold_cols) != 5 or not params_match or not score_match:
                raise ValueError(f"CV audit failed for {dataset_name}/{model_name}")
            cv_rows.append({
                "dataset": dataset_name, "model": model_name,
                "candidate_count": len(cv), "fold_count": len(fold_cols),
                "best_cv_rmse": min_cv, "summary_cv_rmse": float(summary.CV_RMSE),
                "best_params_found_among_tied_best_candidates": params_match,
                "status": "PASS",
            })
            pred_file = predictions_dir / f"{spec['key']}_{model_key}.csv"
            pred = pd.read_csv(pred_file)
            row_match = np.array_equal(pred.row_index.to_numpy(dtype=int), test_idx)
            actual_match = np.allclose(
                pred.actual_price.to_numpy(dtype=float),
                y.iloc[test_idx].to_numpy(dtype=float), rtol=0, atol=1e-8,
            )
            if not row_match or not actual_match:
                raise ValueError(f"Prediction alignment failed for {dataset_name}/{model_name}")
            pred_rows.append({
                "dataset": dataset_name, "model": model_name,
                "prediction_rows": len(pred), "row_index_order_match": row_match,
                "actual_price_match": actual_match, "status": "PASS",
            })
            report.append(
                f"PASS {dataset_name}/{model_name}: best CV score and parameters verified; "
                f"{len(cv)} candidates, 5 folds."
            )
            report.append(
                f"PASS {dataset_name}/{model_name}: saved test predictions match the fixed test rows and actual prices."
            )

            if args.measure_fit_time:
                durations = []
                for _ in range(3):
                    pipeline = build_pipeline(model_name, X_train, expected)
                    start = time.perf_counter()
                    pipeline.fit(X_train, y_train)
                    durations.append(time.perf_counter() - start)
                fit_rows.append({
                    "dataset": dataset_name, "model": model_name,
                    "training_rows": len(train_idx), "repeats": 3,
                    "final_fit_time_median_seconds": float(np.median(durations)),
                    "final_fit_time_mean_seconds": float(np.mean(durations)),
                    "best_params_json": json.dumps(expected, sort_keys=True),
                    "timing_scope": "preprocessing + estimator fit on complete training partition; excludes CV/search and test prediction",
                })
                report.append(
                    f"INFO {dataset_name}/{model_name}: median final fit={np.median(durations):.4f}s across 3 fits."
                )

    pd.DataFrame(cv_rows).to_csv(audit_dir / "cross_validation_audit.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(audit_dir / "prediction_alignment_audit.csv", index=False)
    if args.measure_fit_time:
        pd.DataFrame(fit_rows).to_csv(audit_dir / "comparable_final_fit_times.csv", index=False)
        report.append("Final-fit timings were refreshed using three repeated fits per tuned model.")
    else:
        timing_file = audit_dir / "comparable_final_fit_times.csv"
        if timing_file.exists():
            report.append("Existing comparable final-fit timing results were retained; no additional fitting was performed.")
    (audit_dir / "validation_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print("\nSaved audit tables under results/audit/")
    if args.measure_fit_time:
        print(pd.DataFrame(fit_rows)[["dataset", "model", "final_fit_time_median_seconds"]].to_string(index=False))


if __name__ == "__main__":
    main()
