"""
update_dataset.py
Weekly refresh pipeline for the Global HDI x Explainable AI student dataset.

Two independent steps, kept deliberately separate so real facts stay real:

1. REAL DATA REFRESH
   Pulls the latest published Country_HDI and Country_Life_Expectancy from
   Our World in Data (sourced from UNDP and UN World Population Prospects)
   and overwrites those two columns for every existing row, matched by
   country name. HDI is only released ~once a year, so most weekly runs
   will simply find no change - that's expected.

2. SYNTHETIC EXPANSION (no paid API required)
   Generates new plausible student records by bootstrap-resampling from
   the existing 55k rows: each new row starts as a real observed
   combination of country/university/education/field/language/scores, then
   gets light random jitter on the numeric columns so it isn't an exact
   duplicate. This preserves the real joint correlations in the data
   without needing any LLM call, API key, or network access for this step.
"""

import numpy as np
import requests
import pandas as pd
from datetime import datetime, timezone

DATASET_PATH = "student_dataset.xlsx"
HDI_URL = "https://ourworldindata.org/grapher/human-development-index.csv?v=1&csvType=full&useColumnShortNames=false"
LIFE_EXP_URL = "https://ourworldindata.org/grapher/life-expectancy.csv?v=1&csvType=full&useColumnShortNames=false"
NEW_ROWS_PER_RUN = 200
REQUEST_HEADERS = {"User-Agent": "hdi-xai-dataset-bot/1.0"}


def refresh_country_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Overwrite Country_HDI / Country_Life_Expectancy with the latest
    published values per country. Returns the updated dataframe."""
    hdi = pd.read_csv(HDI_URL, storage_options=REQUEST_HEADERS)
    life = pd.read_csv(LIFE_EXP_URL, storage_options=REQUEST_HEADERS)

    # OWID grapher CSVs are long-format: Entity, Code, Year, <value column>
    hdi_value_col = [c for c in hdi.columns if c not in ("Entity", "Code", "Year")][0]
    life_value_col = [c for c in life.columns if c not in ("Entity", "Code", "Year")][0]

    latest_hdi = hdi.sort_values("Year").groupby("Entity").tail(1).set_index("Entity")[hdi_value_col]
    latest_life = life.sort_values("Year").groupby("Entity").tail(1).set_index("Entity")[life_value_col]

    updated, missing = 0, set()
    for country in df["Country"].unique():
        if country in latest_hdi.index:
            df.loc[df["Country"] == country, "Country_HDI"] = round(float(latest_hdi[country]), 3)
            updated += 1
        else:
            missing.add(country)
        if country in latest_life.index:
            df.loc[df["Country"] == country, "Country_Life_Expectancy"] = round(float(latest_life[country]), 2)

    if missing:
        print(f"[warn] {len(missing)} countries had no exact match in the source data, left unchanged: {sorted(missing)[:10]}")
    print(f"[info] refreshed HDI/life expectancy for {updated} countries")
    return df


def generate_synthetic_rows(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create n new rows by bootstrap-resampling real existing rows and
    lightly jittering the numeric columns. No API key or network call
    needed - every categorical combination sampled is one that genuinely
    occurred in the data, so joint correlations (e.g. field of study by
    country, or HDI by country) stay intact."""
    base = df.sample(n=n, replace=True, random_state=rng.integers(0, 2**32 - 1)).reset_index(drop=True).copy()

    # Jitter numeric columns a little so new rows aren't exact duplicates,
    # then clip back into each column's valid range.
    base["Health_Quality_Score_1to5"] = np.clip(
        base["Health_Quality_Score_1to5"] + rng.integers(-1, 2, size=n), 1, 5)
    base["Stress_Level_1to5"] = np.clip(
        base["Stress_Level_1to5"] + rng.integers(-1, 2, size=n), 1, 5)
    base["Monthly_Earnings_Savings_USD"] = (
        base["Monthly_Earnings_Savings_USD"] * rng.normal(1.0, 0.08, size=n)
    ).round(2).clip(lower=0)
    base["Readiness_Score_0to100"] = np.clip(
        base["Readiness_Score_0to100"] + rng.integers(-5, 6, size=n), 0, 100)

    print(f"[info] generated {len(base)} new synthetic rows via bootstrap resampling")
    return base


def main():
    xl = pd.ExcelFile(DATASET_PATH)
    sheet_name = xl.sheet_names[0]
    df = xl.parse(sheet_name)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = refresh_country_stats(df)
    rng = np.random.default_rng()
    new_rows = generate_synthetic_rows(df, NEW_ROWS_PER_RUN, rng)
    df = pd.concat([df, new_rows], ignore_index=True)

    with pd.ExcelWriter(DATASET_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    with open("update_log.txt", "a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} - rows: {len(df)} (+{len(new_rows)} new)\n")

    print(f"[done] dataset now has {len(df)} rows")


if __name__ == "__main__":
    main()
