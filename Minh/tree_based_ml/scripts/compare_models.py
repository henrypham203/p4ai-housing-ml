from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tree_based_ml.common import RESULTS_DIR, FIGURE_DIR


def main():
    tables = RESULTS_DIR / "tables"
    baseline = pd.read_csv(tables / "baseline_metrics.csv")
    tuned = pd.read_csv(tables / "tuned_metrics.csv")
    baseline["CV_RMSE"] = float("nan")
    baseline["fit_time_scope"] = "single baseline fit; historical timing"
    tuned["fit_time_scope"] = "search elapsed includes CV and final refit; use separate audit table for comparable final-fit time"
    cols = ["dataset","model","stage","MAE","RMSE","R2","CV_RMSE"]
    combined = pd.concat([baseline[cols], tuned[cols]], ignore_index=True)
    combined.to_csv(tables / "model_performance_summary.csv", index=False)
    ranking = tuned.sort_values(["dataset","CV_RMSE"]).copy()
    ranking["CV_rank"] = ranking.groupby("dataset")["CV_RMSE"].rank(method="first").astype(int)
    ranking["selected_by_CV"] = ranking["CV_rank"].eq(1)
    ranking[["dataset","model","CV_RMSE","MAE","RMSE","R2","CV_rank","selected_by_CV"]].to_csv(tables / "cv_model_ranking.csv", index=False)
    selected = ranking[ranking["selected_by_CV"]].copy()
    selected[["dataset","model","CV_RMSE","MAE","RMSE","R2","CV_rank"]].to_csv(tables / "selected_model_test_metrics.csv", index=False)
    FIGURE_DIR.joinpath("model_comparison").mkdir(parents=True, exist_ok=True)
    for dataset in combined["dataset"].unique():
        frame = combined[combined["dataset"] == dataset]
        for metric in ["MAE","RMSE","R2"]:
            pivot = frame.pivot(index="model", columns="stage", values=metric)
            ax = pivot.plot(kind="bar", figsize=(8,5), rot=0)
            ax.set_title(f"{dataset}: {metric} by model and training stage")
            ax.set_ylabel(metric)
            ax.set_xlabel("Model")
            ax.legend(title="Stage")
            ax.figure.tight_layout()
            ax.figure.savefig(FIGURE_DIR / "model_comparison" / f"{dataset.lower()}_{metric.lower()}_comparison.png", dpi=170)
            plt.close(ax.figure)
    print("Model comparison complete.")
    print("Selection is based only on training 5-fold CV RMSE; test metrics are reported after selection.")
    print(selected[["dataset","model","CV_RMSE","MAE","RMSE","R2"]].to_string(index=False))

if __name__ == "__main__":
    main()
