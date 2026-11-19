# model analysis — Tree-based Feature Importance and Error Analysis

## Methodology
- The saved fixed train/test membership is unchanged; the only additional diagnostic validation holdout is drawn from training rows and does not touch the test set.
- The saved split is 80/20 with random_state=42.
- Gradient Boosting hyperparameters are loaded from training-only CV-based model selection.
- Final test predictions are reused from the tuned Gradient Boosting run and checked against the saved fixed test row indices and actual target values.
- A separate diagnostic Gradient Boosting model is fitted only on a deterministic sample of fixed train/test split training rows to compute impurity-based feature importance and the boosting-stage curve; it is not used for final test evaluation.
- The King County diagnostic model uses 8,000 sampled training rows (then an internal 80/20 train/validation split) to keep the interpretability step computationally practical.
- Training/validation curves use a 20% internal holdout from the diagnostic training sample (random_state=36); they are diagnostics, not model selection.
- The held-out test partition is used only for final metrics and post-hoc error diagnosis.
- Numeric missing values are median-imputed; categorical values use most-frequent imputation and one-hot encoding inside the pipeline.
- King County preprocessing drops id, converts date to sale_year/sale_month, and treats zipcode as categorical.

## Feature-importance interpretation
- Importance is the Gradient Boosting impurity-based feature_importances_ value.
- One-hot encoded category importances are summed to the original categorical feature.
- This importance is model-specific and is not causal evidence; impurity-based importance can favor continuous or high-cardinality features.

## Results

| Dataset | CV RMSE | Test MAE | Test RMSE | Test R² | Diagnostic fit (s) |
|---|---:|---:|---:|---:|---:|
| Housing | 1047919.012 | 967592.865 | 1325441.217 | 0.6524 | 0.098 |
| KingCounty | 119859.842 | 73701.121 | 141604.478 | 0.8674 | 3.080 |

## Descriptive findings
- **Housing — leading features:** area (46.2%), bathrooms (13.6%), stories (6.9%), airconditioning (6.8%), parking (5.3%).
- **Housing — largest test-set MAE segment:** actual-price bin `(6650000.0, 13300000.0]` (n=19), MAE=1796623.024, RMSE=2294502.403, mean residual=1466282.419.
- **Housing — training/validation curve:** lowest diagnostic validation RMSE=1101543.035 at stage 140; at the final stage 150, training RMSE=390198.358 and validation RMSE=1107094.422. This is descriptive and does not alter the CV-selected model.
- **KingCounty — leading features:** sqft_living (33.8%), grade (22.0%), lat (14.4%), long (6.1%), zipcode (5.8%).
- **KingCounty — largest test-set MAE segment:** actual-price bin `(711360.0, 5570000.0]` (n=865), MAE=160910.159, RMSE=280228.763, mean residual=43556.345.
- **KingCounty — training/validation curve:** lowest diagnostic validation RMSE=137796.282 at stage 150; at the final stage 150, training RMSE=83402.636 and validation RMSE=137796.282. This is descriptive and does not alter the CV-selected model.

## Error-analysis conventions
- Residual is defined as actual_price minus predicted_price; positive residual indicates underprediction.
- Test-set price quantiles are used only to describe where errors occur, not to tune or select the model.
- MAE and RMSE are reported separately for each dataset because the datasets represent different housing markets and price scales.

## Generated figures
- `figures/analysis/housing_actual_vs_predicted.png`
- `figures/analysis/housing_error_by_price_bin.png`
- `figures/analysis/housing_feature_importance_top20.png`
- `figures/analysis/housing_residual_distribution.png`
- `figures/analysis/housing_residuals_vs_predicted.png`
- `figures/analysis/housing_training_validation_curve.png`
- `figures/analysis/kingcounty_actual_vs_predicted.png`
- `figures/analysis/kingcounty_error_by_price_bin.png`
- `figures/analysis/kingcounty_feature_importance_top20.png`
- `figures/analysis/kingcounty_residual_distribution.png`
- `figures/analysis/kingcounty_residuals_vs_predicted.png`
- `figures/analysis/kingcounty_training_validation_curve.png`

## Limitations to state in the report
- The feature importance values are associative model diagnostics, not causal effects.
- Training/validation curves are computed across Gradient Boosting boosting stages on an internal training-only sample; hyperparameters remain selected by the original 5-fold CV.
- Diagnostic-model fit time is measured on the stated sample and must not be reported as the full final model training time.
- Test error analysis is descriptive and must not be used to revise model parameters after seeing the test results.
