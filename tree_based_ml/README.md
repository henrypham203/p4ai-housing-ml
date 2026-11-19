# Tree-Based Classical Machine Learning — House Price Prediction

This folder contains the complete Tree-based contribution to the house-price prediction project. Files are organized by **what they do**, not by the order or date on which they were created.

## Algorithms and datasets

Algorithms: Decision Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor.

Datasets:
- Housing Prices Dataset (`Housing.csv`)
- King County House Sales Prediction (`kc_house_data.csv`)

Download instructions and source links are in `data/README.md`. The raw CSVs are not included in this GitHub handoff.

## Start here

1. `configs/experiment_protocol.json` — fixed split, random seeds, CV policy, and evaluation rules.
2. `configs/baseline_model_configurations.json` — initial model configurations.
3. `configs/hyperparameter_search_spaces.json` — search spaces and search budgets for each algorithm.
4. `configs/best_hyperparameters.json` — best configurations of all 6 tuned models, CV RMSE, test metrics, and audited fit times.
5. `results/tables/model_performance_summary.csv` — all 12 baseline/tuned experiments.
6. `results/tables/cv_model_ranking.csv` — valid tuned-model ranking based on training-only CV RMSE.
7. `report/Tree_Based_Classical_ML_Report.docx` — report contribution.
8. `docs/REPORT_REVIEW_NOTES.md` — review notes and experimental caveats.
9. `docs/GITHUB_PUSH_GUIDE.md` — how to add this folder to the group repository.

## Code map — one file per task

| File | Responsibility |
|---|---|
| `scripts/prepare_datasets.py` | Inspect datasets, engineer the King County fields, create the fixed 80/20 split, and export dataset/feature documentation. |
| `scripts/run_baseline_models.py` | Run baseline Decision Tree, Random Forest, and Gradient Boosting on both datasets in one process. |
| `scripts/tune_hyperparameters.py` | Run 5-fold RandomizedSearchCV for all three algorithms on both datasets; export best params, candidate CV tables, and tuned test metrics. |
| `scripts/compare_models.py` | Combine baseline/tuned metrics, rank tuned models by CV RMSE, select the best model without looking at test scores, and create comparison charts. |
| `scripts/analyze_models.py` | Feature importance, actual-vs-predicted, residuals, price-segment error analysis, and diagnostic learning curves for the CV-selected Gradient Boosting models. |
| `scripts/audit_experiments.py` | Verify the fixed split, CV score/parameter consistency, test-prediction alignment, and comparable final-fit times. No tuning is run by this script. |
| `src/tree_based_ml/common.py` | Shared data preparation, split loading, preprocessing pipelines, model constructors, and metrics used by the scripts. |

**Baseline runs are combined:** there is one script for all three baseline algorithms.  
**Hyperparameter tuning is combined:** there is one script for all three tuning workflows.  
Algorithm-specific differences (search spaces, CV fold seed, and estimator settings) are stored in the central script/configuration rather than scattered across separate files.

## Folder map

```text
tree_based_ml/
├── README.md
├── requirements.txt
├── data/                         # put the two source CSVs here
├── configs/                      # protocol, baseline params, search spaces, best params
├── scripts/                      # one script per project task
├── src/tree_based_ml/             # shared reusable code
├── results/
│   ├── splits/                   # fixed train/test row indices
│   ├── tables/                   # metric and dataset summary tables
│   ├── cv_results/                # full candidate-level CV records for six tuned runs
│   ├── predictions/               # baseline and tuned test predictions
│   ├── analysis/                  # feature importance and error-analysis tables
│   └── audit/                     # validation and fit-time audit outputs
├── figures/
│   ├── data_distributions/
│   ├── model_comparison/
│   └── analysis/
├── report/
└── docs/
```

## Setup on Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Put the two CSV datasets into `data/` and run commands from the `tree_based_ml` folder.

## Run order

The submitted `results/`, `figures/`, and report already contain the recorded experiments. You do **not** need to rerun the pipeline just to review or submit the results.

For a full reproduction from the original CSV files, use the following order:

```powershell
python scripts/prepare_datasets.py
python scripts/run_baseline_models.py
python scripts/tune_hyperparameters.py
python scripts/compare_models.py
python scripts/analyze_models.py
python scripts/audit_experiments.py
```

`prepare_datasets.py` creates or refreshes the saved split indices. For reproducing the reported experiments, use the original CSV files in their original row order. Do not sort or edit the raw rows after saving the indices.

`tune_hyperparameters.py` performs all six searches (three algorithms × two datasets); it can take several minutes, especially for King County Random Forest. Do not run it merely to inspect the saved best configurations. The analysis and default audit scripts do not rerun hyperparameter search. `audit_experiments.py --measure-fit-time` is optional and refits each tuned model three times to refresh comparable training-time measurements.

## Experimental policy

- Train/test split: 80/20, split random state 42.
- Model random state: 36 where supported.
- Hyperparameter selection: 5-fold CV on training only, scoring by RMSE.
- Test set: final evaluation only; never use for tuning or model selection.
- Metrics: MAE, RMSE, and R² on original price scale.
- Outliers flagged by the 1.5×IQR rule are retained.

The CV fold shuffling seed differs by algorithm in the current recorded experiments: Decision Tree uses 42; Random Forest and Gradient Boosting use 36. The exact setting is documented in `configs/hyperparameter_search_spaces.json`.

## Results interpretation

The selected model is the tuned model with the lowest training-only CV RMSE within each dataset. In the current experiment, Gradient Boosting is selected for both datasets. The Random Forest model has a slightly lower test MAE than Gradient Boosting on King County, but model selection remains based on CV RMSE, not test metrics.

Do not use any post-hoc ranking by test RMSE as a model-selection rule. Use `results/tables/cv_model_ranking.csv` and `results/tables/selected_model_test_metrics.csv`.

## GitHub policy

Commit scripts, configs, summarized result tables, CV records, prediction outputs, figures, docs, and the report. Do not commit `.venv/`, Python caches, IDE settings, or raw Kaggle CSVs unless the group has checked and agreed to the dataset redistribution terms.
