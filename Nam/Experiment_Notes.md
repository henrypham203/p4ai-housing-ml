# Linear regression experiments

[Linear_Regression.ipynb](./Linear_Regression.ipynb) contains the executed basic pipeline on Housing and King County. The target is the original sale price. Model selection uses mean RMSE over five training folds (CV seed 36).

## Final settings

These are the settings in Table 2 of the submitted ML draft.

| Model | Housing | King County |
|---|---|---|
| OLS | No regularization parameter | No regularization parameter |
| Ridge | `alpha=18.205642` | `alpha=0.457305` |
| Lasso | `alpha=3162.277660`, `selection="cyclic"` | `alpha=0.1`, `selection="random"` |
| ElasticNet | `alpha=0.056234`, `l1_ratio=0.10` | `alpha=0.02`, `l1_ratio=0.98` |

## Why the King County Lasso output differs from Table 2

The uploaded notebook contains the earlier run. At `alpha=0.1`, cyclic updates failed the cold-start convergence check. The notebook then fitted a fallback at `alpha=316.227766`. Its saved Lasso test score and exported parameters belong to that fallback.

Section 2.3 of the submitted ML draft records a later check at `alpha=0.1` using random coordinate updates, `warm_start=False`, `max_iter=300000` and `tol=1e-4`. All five fold fits and the full-training fit passed the convergence check. Mean CV RMSE was **159,554.895**, with a sample SD of **5,908.184**. The supplementary cell is absent from the uploaded notebook. The draft reports only training and validation results for this final Lasso setting.

## Selected models and holdout results

| Dataset | Selected by CV | Test MAE | Test RMSE | Test R² |
|---|---|---:|---:|---:|
| Housing | ElasticNet | 981,196.95 | 1,342,666.90 | 0.6433 |
| King County | Ridge | 98,284.67 | 170,328.04 | 0.8081 |

For King County Ridge, Table 4 uses the audit CV RMSE of **159,554.621**. The tuning step recorded **159,548.81**. The draft leaves this small difference unresolved. The team's full-pipeline Ridge results are reported separately.