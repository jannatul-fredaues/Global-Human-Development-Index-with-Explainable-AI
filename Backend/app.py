"""
app.py
Flask backend for the Global HDI x Explainable AI platform.

Serves three things to the frontend:
  GET  /api/health     - which model is currently live and when it was trained
  GET  /api/countries   - per-country HDI, life expectancy, and dataset stats
  POST /api/predict     - predicted readiness score for a student profile,
                           plus a plain-language explanation of *why*

The model file (best_model.joblib) is replaced every 7 days by the separate
weekly GitHub Actions pipeline (update_dataset.py -> train_and_select_model.py).
This app just loads whatever the latest winning model is - it doesn't care
which algorithm that turns out to be, since the whole pipeline (encoding
included) is bundled in the saved sklearn Pipeline.

Explanation method: for each feature, the query row is duplicated across a
sample of real peer values for that one feature (everything else held fixed
at the query's actual values), and the model is re-run on each variant. The
feature's "contribution" is how much the prediction shifts, on average, from
using the query's actual value versus a typical peer's value for that
feature alone. This is a straightforward occlusion/sensitivity method - it
avoids the SHAP library's masker, which does not handle mixed string and
numeric columns cleanly.
"""

import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

DATASET_PATH = "student_dataset.xlsx"
MODEL_PATH = "best_model.joblib"
LEADERBOARD_PATH = "model_leaderboard.json"
BACKGROUND_SAMPLE_SIZE = 300  # rows drawn from as "typical peer" values per feature
PER_FEATURE_SAMPLES = 30      # how many peer values to average over, per feature, per request

app = Flask(__name__)
CORS(app)

# --- Load once at startup, not per-request ---
xl = pd.ExcelFile(DATASET_PATH)
DF = xl.parse(xl.sheet_names[0])
if "Unnamed: 0" in DF.columns:
    DF = DF.drop(columns=["Unnamed: 0"])

MODEL = joblib.load(MODEL_PATH)

with open(LEADERBOARD_PATH) as f:
    LEADERBOARD = json.load(f)

FEATURES = LEADERBOARD["features"]
BACKGROUND = DF[FEATURES].sample(n=min(BACKGROUND_SAMPLE_SIZE, len(DF)), random_state=42).reset_index(drop=True)


def explain_prediction(row: pd.DataFrame, base_pred: float) -> dict:
    rng = np.random.default_rng(0)
    n = min(PER_FEATURE_SAMPLES, len(BACKGROUND))
    peer_idx = rng.choice(len(BACKGROUND), size=n, replace=False)
    peers = BACKGROUND.iloc[peer_idx].reset_index(drop=True)

    contributions = []
    for feature in FEATURES:
        variants = pd.concat([row] * n, ignore_index=True)
        variants[feature] = peers[feature].values
        alt_preds = MODEL.predict(variants)
        contributions.append({
            "feature": feature,
            "contribution": round(base_pred - float(np.mean(alt_preds)), 3),
        })
    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    return {
        "method": "occlusion (prediction shift vs. typical peer values)",
        "feature_contributions": contributions,
    }


@app.get("/api/health")
def health():
    return jsonify(
        status="ok",
        model_used=LEADERBOARD["selected_model"],
        model_test_rmse=LEADERBOARD["selected_test_rmse"],
        trained_at=LEADERBOARD["trained_at"],
        rows_used=LEADERBOARD["rows_used"],
    )


@app.get("/api/countries")
def countries():
    agg = (
        DF.groupby("Country")
        .agg(
            hdi=("Country_HDI", "first"),
            life_expectancy=("Country_Life_Expectancy", "first"),
            avg_readiness_score=("Readiness_Score_0to100", "mean"),
            student_count=("Readiness_Score_0to100", "size"),
        )
        .reset_index()
        .round(3)
    )
    return jsonify(agg.to_dict(orient="records"))


@app.post("/api/predict")
def predict():
    payload = request.get_json(force=True, silent=True) or {}
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        return jsonify(error=f"missing required fields: {missing}"), 400

    row = pd.DataFrame([{f: payload[f] for f in FEATURES}])

    try:
        pred = float(MODEL.predict(row)[0])
        explanation = explain_prediction(row, pred)
    except Exception as e:
        return jsonify(error=f"prediction failed: {e}"), 400

    return jsonify(
        predicted_readiness_score=round(pred, 2),
        model_used=LEADERBOARD["selected_model"],
        explanation=explanation,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)

