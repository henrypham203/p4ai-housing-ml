# File guide

## Recommended review order

1. `README.md` — project structure, execution instructions, and experimental rules.
2. `configs/experiment_protocol.json` — fixed split, random seeds, CV policy, and evaluation rules.
3. `configs/baseline_model_configurations.json` — baseline estimator settings.
4. `configs/hyperparameter_search_spaces.json` — search spaces, candidate budgets, and CV settings.
5. `configs/best_hyperparameters.json` — best settings for all six tuned models, CV RMSE, test metrics, and comparable fit times.
6. `results/tables/model_performance_summary.csv` — baseline/tuned comparison for all 12 experiments.
7. `results/tables/cv_model_ranking.csv` — valid tuned-model ranking based on training-only CV RMSE.
8. `results/cv_results/` — candidate-level CV outcomes for each dataset/model pair.
9. `results/audit/validation_report.txt` — split, CV, and prediction checks.
10. `report/Tree_Based_Classical_ML_Report.docx` — report contribution.

## Which script does what?

- `scripts/prepare_datasets.py`: inspect raw data, prepare features, and create fixed split indices.
- `scripts/run_baseline_models.py`: run all three baseline algorithms on both datasets.
- `scripts/tune_hyperparameters.py`: tune all three algorithms on both datasets using 5-fold CV.
- `scripts/compare_models.py`: build unified performance tables and rank tuned models by CV RMSE.
- `scripts/analyze_models.py`: generate feature importance, residual/error analysis, and diagnostic learning curves.
- `scripts/audit_experiments.py`: verify split, CV records, and prediction alignment; no tuning is run.

## Reproduction commands

Run from this folder after installing `requirements.txt` and placing the source CSV files in `data/`:

```powershell
python scripts/prepare_datasets.py
python scripts/run_baseline_models.py
python scripts/tune_hyperparameters.py
python scripts/compare_models.py
python scripts/analyze_models.py
python scripts/audit_experiments.py
```

For a quick validation only, run `python scripts/audit_experiments.py`. To refresh final-fit timings, explicitly add `--measure-fit-time`; this performs 18 additional model fits (six tuned models × three repetitions).
