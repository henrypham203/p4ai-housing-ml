# Tree-based experiment protocol

## Datasets and target
- Housing Prices Dataset: `Housing.csv`
- King County House Sales Prediction: `kc_house_data.csv`
- Target: `price`

## Data preparation
- Preserve source row order.
- Verify duplicate/missing records.
- For King County, drop `id`, derive `sale_year` and `sale_month` from `date`, and treat `zipcode` as categorical.
- Flag 1.5×IQR price outliers; do not automatically remove them.
- Keep imputation and one-hot encoding inside sklearn pipelines.

## Evaluation protocol
- Train/test: 80/20, random_state=42; fixed row indices are saved under `results/splits/`.
- Hyperparameter search: 5-fold CV using training rows only.
- Selection metric: RMSE.
- Test set: final evaluation only.
- Model random_state: 36 where supported.
- Metrics: MAE, RMSE, R² on the original price scale.

## Recorded CV fold seeds
- Decision Tree: KFold shuffle seed 42.
- Random Forest: KFold shuffle seed 36.
- Gradient Boosting: KFold shuffle seed 36.
- RandomizedSearchCV sampler seed: 36 for all three.

## Search budget
- Decision Tree: 20 sampled candidates × 5 folds per dataset.
- Random Forest: 10 sampled candidates × 5 folds per dataset.
- Gradient Boosting: 10 sampled candidates × 5 folds per dataset.
