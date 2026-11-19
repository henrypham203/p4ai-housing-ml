"""Shared data, preprocessing, split, model, and metric utilities."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SPLIT_DIR = RESULTS_DIR / "splits"
TABLE_DIR = RESULTS_DIR / "tables"
CV_DIR = RESULTS_DIR / "cv_results"
PREDICTION_DIR = RESULTS_DIR / "predictions"
FIGURE_DIR = PROJECT_ROOT / "figures"
DATASETS = {
    "Housing": {"filename": "Housing.csv", "key": "housing"},
    "KingCounty": {"filename": "kc_house_data.csv", "key": "kingcounty"},
}
MODEL_NAMES = ["Decision Tree", "Random Forest", "Gradient Boosting"]


def prepare_features(raw: pd.DataFrame, dataset_name: str):
    """Prepare features without changing row order or dropping observations."""
    if "price" not in raw.columns:
        raise ValueError(f"Target 'price' not found in {dataset_name} dataset")
    if raw.duplicated().any():
        raise ValueError(f"Unexpected duplicate rows in {dataset_name}; fixed split indices assume original row order")
    X = raw.drop(columns=["price"]).copy()
    y = pd.to_numeric(raw["price"], errors="coerce")
    if y.isna().any():
        raise ValueError(f"Missing or non-numeric target values in {dataset_name}")
    if dataset_name == "KingCounty":
        if "id" in X.columns:
            X = X.drop(columns=["id"])
        if "date" in X.columns:
            date_values = pd.to_datetime(X["date"], format="%Y%m%dT%H%M%S", errors="coerce")
            X["sale_year"] = date_values.dt.year
            X["sale_month"] = date_values.dt.month
            X = X.drop(columns=["date"])
        if "zipcode" in X.columns:
            X["zipcode"] = X["zipcode"].astype("string")
    empty = [col for col in X.columns if X[col].isna().all()]
    if empty:
        X = X.drop(columns=empty)
    return X, y


def read_indices(path: Path, n_rows: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing fixed split file: {path}. Run scripts/prepare_datasets.py first.")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Split file is empty: {path}")
    preferred = ["row_index", "row_index_after_dedup", "index", "indices", "idx", "row"]
    lower = {str(c).strip().lower(): c for c in frame.columns}
    column = next((lower[c] for c in preferred if c in lower), None)
    if column is None:
        numeric = [c for c in frame.columns if pd.to_numeric(frame[c], errors="coerce").notna().all()]
        if not numeric:
            raise ValueError(f"No numeric index column in {path}")
        column = numeric[0]
    values = pd.to_numeric(frame[column], errors="raise").to_numpy()
    if len(values) == 0 or not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"Invalid row indices in {path}")
    indices = values.astype(int)
    if len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= n_rows:
        raise ValueError(f"Duplicate or out-of-range row indices in {path}")
    return indices


def load_split(dataset_name: str):
    spec = DATASETS[dataset_name]
    data_path = DATA_DIR / spec["filename"]
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. See data/README.md")
    raw = pd.read_csv(data_path)
    X, y = prepare_features(raw, dataset_name)
    key = spec["key"]
    train_idx = read_indices(SPLIT_DIR / f"{key}_train_indices.csv", len(raw))
    test_idx = read_indices(SPLIT_DIR / f"{key}_test_indices.csv", len(raw))
    if set(train_idx) & set(test_idx):
        raise ValueError(f"Train/test overlap in {dataset_name}")
    if len(train_idx) + len(test_idx) != len(raw):
        raise ValueError(f"Train/test indices do not cover {dataset_name}")
    return raw, X, y, train_idx, test_idx


def make_onehot(dense: bool = False):
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=not dense)


def build_preprocessor(X_train: pd.DataFrame, dense_onehot: bool = False):
    numeric = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X_train.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot(dense=dense_onehot)),
        ])
        transformers.append(("categorical", categorical_pipe, categorical))
    if not transformers:
        raise ValueError("No usable input features remain")
    return ColumnTransformer(transformers, remainder="drop")


def build_estimator(model_name: str, params: dict | None = None):
    params = dict(params or {})
    if model_name == "Decision Tree":
        return DecisionTreeRegressor(random_state=36, **params)
    if model_name == "Random Forest":
        return RandomForestRegressor(random_state=36, n_jobs=-1, **params)
    if model_name == "Gradient Boosting":
        baseline = {
            "loss": "squared_error", "learning_rate": 0.1,
            "n_estimators": 100, "subsample": 1.0,
            "criterion": "friedman_mse", "max_depth": 3,
        }
        baseline.update(params)
        return GradientBoostingRegressor(random_state=36, **baseline)
    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(model_name: str, X_train: pd.DataFrame, params: dict | None = None):
    dense = model_name == "Gradient Boosting"
    preprocessor = build_preprocessor(X_train, dense_onehot=dense)
    estimator = build_estimator(model_name, params)
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def regression_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }
