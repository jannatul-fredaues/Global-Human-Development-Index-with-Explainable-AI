"""
predict_example.py
Shows how your website's backend loads best_model.joblib and uses it.
best_model.joblib is a full sklearn Pipeline (preprocessing + model already
bundled together), so you never need to know which model won this week -
you just call .predict() and it works, whatever got selected.

Run standalone to sanity-check, or copy the load_model()/predict_readiness()
pattern into your Flask app.
"""

import json
import joblib
import pandas as pd

MODEL_PATH = "best_model.joblib"
LEADERBOARD_PATH = "model_leaderboard.json"


def load_model():
    return joblib.load(MODEL_PATH)


def current_model_info() -> dict:
    """Optional: show which model is live and how it scored, e.g. for an
    'about this prediction' note on the website."""
    with open(LEADERBOARD_PATH) as f:
        board = json.load(f)
    return {
        "model": board["selected_model"],
        "test_rmse": board["selected_test_rmse"],
        "trained_at": board["trained_at"],
    }


def predict_readiness(model, student: dict) -> float:
    """student must contain: Country_HDI, Country_Life_Expectancy,
    Health_Quality_Score_1to5, Stress_Level_1to5,
    Monthly_Earnings_Savings_USD, Country, Education_Level, Field_of_Study,
    Language_Proficiency"""
    row = pd.DataFrame([student])
    return float(model.predict(row)[0])


if __name__ == "__main__":
    model = load_model()
    print("Live model this week:", current_model_info())

    example_student = {
        "Country_HDI": 0.785,
        "Country_Life_Expectancy": 74.2,
        "Health_Quality_Score_1to5": 3,
        "Stress_Level_1to5": 2,
        "Monthly_Earnings_Savings_USD": 1500.0,
        "Country": "Bangladesh",
        "Education_Level": "Master's",
        "Field_of_Study": "Data Science & Artificial Intelligence",
        "Language_Proficiency": "Advanced (C1-C2)",
    }
    print("Predicted readiness score:", predict_readiness(model, example_student))
