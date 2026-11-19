# Report review notes — Tree-based contribution

## Scope
The report describes only the Tree-based part of the Classical ML contribution. It should be integrated with the Linear ML and Deep Learning results by the group lead; it does not claim to be the final whole-group comparison.

## Experimental details to preserve
1. The fixed data split is 80/20 with `random_state=42`. The Decision Tree tuning code reconstructs this deterministic split, while the Random Forest and Gradient Boosting workflows read the saved row indices. The audit confirmed the same train/test membership and order.
2. All tuning uses 5-fold CV on the training partition. Decision Tree uses shuffled KFold seed 42; Random Forest and Gradient Boosting use shuffled KFold seed 36. All search samplers use seed 36. Do not claim identical fold assignments across algorithms.
3. Select models using the lowest CV RMSE on training data. Test MAE/RMSE/R² are final evaluation only. Do not choose models by test RMSE.
4. Gradient Boosting was selected for both datasets by the CV criterion. On King County, Random Forest has a slightly lower test MAE (73,418.790 vs 73,701.121), while Gradient Boosting has lower test RMSE and higher R².
5. Feature importance and learning curves are diagnostic. The King County diagnostic curve uses a deterministic 8,000-row sample of the training partition. It is not the 5-fold CV curve and was not used to tune or select the final model.
6. The comparable final-fit timing is the median of three fits on the full training partition; it excludes cross-validation, search, and test prediction.

## Naming and selection
Files are named by function, not by the order/date on which they were created. The legacy table that ranked models by test RMSE is excluded because test data must not be used for model selection.
