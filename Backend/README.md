---
title: HDI XAI Backend
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Global HDI x Explainable AI - Backend

Flask API backing the Global HDI x Explainable AI platform's frontend.

## Endpoints

- `GET /api/health` - which model is currently live, its test RMSE, and when
  it was last trained.
- `GET /api/countries` - HDI, life expectancy, average readiness score, and
  student count per country.
- `POST /api/predict` - predicted readiness score for a student profile,
  plus a per-feature explanation of why. Body must include:
  ```json
  {
    "Country_HDI": 0.785,
    "Country_Life_Expectancy": 74.2,
    "Health_Quality_Score_1to5": 3,
    "Stress_Level_1to5": 2,
    "Monthly_Earnings_Savings_USD": 1500.0,
    "Country": "Bangladesh",
    "Education_Level": "Master's",
    "Field_of_Study": "Data Science & Artificial Intelligence",
    "Language_Proficiency": "Advanced (C1-C2)"
  }
  ```

## Keeping it current

`student_dataset.xlsx`, `best_model.joblib`, and `model_leaderboard.json`
are produced weekly by the GitHub Actions pipeline in the
`hdi-dataset-updater-student` repo. Copy the three files here (or wire up
an automated push from that repo's workflow) after each weekly run so the
API always serves the current model.

## Running locally

```bash
pip install -r requirements.txt
python app.py
# then: curl http://localhost:7860/api/health
```
