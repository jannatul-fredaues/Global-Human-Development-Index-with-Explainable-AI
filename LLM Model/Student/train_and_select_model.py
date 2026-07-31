"""
train_and_select_model.py
Runs after update_dataset.py each week, on the freshly-updated data.

Trains several candidate regression models to predict Readiness_Score_0to100,
evaluates each on a held-out test split, and saves whichever one performs
best as best_model.joblib - the single file the website loads for
predictions. Since this reruns on the same 7-day schedule as the dataset
update, the "best" model is never fixed: it's re-selected from scratch every
week against that week's data, so a different candidate can win next time.

Outputs:
  best_model.joblib       - a full sklearn Pipeline (preprocessing + model),
                             ready to call .predict(dataframe) directly
  model_leaderboard.json  - every candidate's metrics + which one was chosen
"""

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_PATH = "student_dataset.xlsx"
MODEL_PATH = "best_model.joblib"
LEADERBOARD_PATH = "model_leaderboard.json"

TARGET = "Readiness_Score_0to100"
NUMERIC_FEATURES = [
    "Country_HDI", "Country_Life_Expectancy",
    "Health_Quality_Score_1to5", "Stress_Level_1to5",
    "Monthly_Earnings_Savings_USD",
]
CATEGORICAL_FEATURES = [
    "Country", "Education_Level", "Field_of_Study", "Language_Proficiency",
]
# University is deliberately excluded: ~5,900 unique values is too
# high-cardinality to one-hot encode cheaply and risks overfitting to
# individual universities rather than learning general patterns.

CANDIDATES = {
    "linear_regression": LinearRegression(),
    "ridge": Ridge(alpha=1.0),
    "knn": KNeighborsRegressor(n_neighbors=15),
    "random_forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingRegressor(random_state=42),
}


def build_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def main():
    xl = pd.ExcelFile(DATASET_PATH)
    df = xl.parse(xl.sheet_names[0])
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X, y = df[feature_cols], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []
    best_name, best_rmse = None, np.inf

    for name, model in CANDIDATES.items():
        pipe = build_pipeline(clone(model))
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        results.append({"model": name, "rmse": round(rmse, 3), "mae": round(mae, 3), "r2": round(r2, 4)})
        print(f"[info] {name}: RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.4f}")
        if rmse < best_rmse:
            best_rmse, best_name = rmse, name

    # Refit the winning model on the FULL dataset (not just the 80% split)
    # before shipping it, so the deployed model uses all available data.
    final_pipeline = build_pipeline(clone(CANDIDATES[best_name]))
    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, MODEL_PATH)

    leaderboard = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows_used": len(df),
        "target": TARGET,
        "features": feature_cols,
        "results": sorted(results, key=lambda r: r["rmse"]),
        "selected_model": best_name,
        "selected_test_rmse": round(best_rmse, 3),
    }
    with open(LEADERBOARD_PATH, "w") as f:
        json.dump(leaderboard, f, indent=2)

    print(f"[done] best model this run: {best_name} (test RMSE={best_rmse:.3f}) -> saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
