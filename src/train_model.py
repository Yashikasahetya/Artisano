
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

try:
    from src.data_generator import DATA_PATH, generate_dataset
except ModuleNotFoundError:
    from data_generator import DATA_PATH, generate_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "price_model.pkl"

FEATURES = ["craft", "material_cost", "labor_hours", "market_tier"]
TARGET = "selling_price"


def build_model() -> TransformedTargetRegressor:
    """One-hot encode craft -> XGBoost; the model learns log(price) and converts back."""
    prep = ColumnTransformer(
        [("craft", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["craft"])],
        remainder="passthrough",
    )
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    pipe = Pipeline([("prep", prep), ("xgb", xgb)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def main() -> None:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded {len(df):,} rows from {DATA_PATH}")
    else:
        print("data/artisan_market_data.csv not found - generating data on the fly.")
        df = generate_dataset()

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model()
    cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred)),
        "MAPE_%": float(np.mean(np.abs((y_test.values - pred) / y_test.values)) * 100),
        "CV_R2_mean": float(cv_r2.mean()),
    }

    
    ratio = y_test.values / pred
    band = (float(np.quantile(ratio, 0.10)), float(np.quantile(ratio, 0.90)))

    inner = model.regressor_
    names = inner.named_steps["prep"].get_feature_names_out()
    importance = dict(zip(names, inner.named_steps["xgb"].feature_importances_.astype(float)))

    artefact = {
        "model": model,
        "features": FEATURES,
        "metrics": metrics,
        "price_band": band,
        "feature_importance": importance,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artefact, f)

    print("\n=== Test-set performance ===")
    for k, v in metrics.items():
        print(f"{k:>11}: {v:,.4f}")
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
