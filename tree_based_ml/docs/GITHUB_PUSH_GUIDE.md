# GitHub integration guide

## Where to place this contribution

Copy the complete `tree_based_ml/` folder into the root of the group's shared repository. Keep it as a self-contained folder; do not merge its scripts into the repository root.

A typical group repository can look like this:

```text
GROUP_PROJECT/
├── linear_ml/
├── deep_learning/
└── tree_based_ml/
```

## Files to commit

Commit the scripts, shared source module, configs, saved result tables, CV records, prediction CSVs, figures, documentation, and the Word report contained in this folder.

Do not commit `.venv/`, `__pycache__/`, raw Kaggle CSV files, or editor-specific files. The `.gitignore` in this folder already excludes common environment/cache files and `data/*.csv`.

## Suggested Git commands

Run these commands from the group's repository root after copying the folder:

```powershell
git status
git switch -c feature/tree-based-ml
git add tree_based_ml
git commit -m "Add tree-based classical ML experiments and report"
git push -u origin feature/tree-based-ml
```

Then open a Pull Request for the group lead to review. If the branch name already exists, switch to it or choose a different branch name rather than creating it again.

## What the group lead should review first

1. `tree_based_ml/README.md`
2. `tree_based_ml/configs/best_hyperparameters.json`
3. `tree_based_ml/results/tables/model_performance_summary.csv`
4. `tree_based_ml/results/tables/cv_model_ranking.csv`
5. `tree_based_ml/results/audit/validation_report.txt`
6. `tree_based_ml/report/Tree_Based_Classical_ML_Report.docx`
