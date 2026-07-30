# Weekly dataset updater

Keeps `student_dataset.xlsx` current on a 7-day schedule via GitHub Actions.
No paid API or API key required.

## What it does each run
1. **Real data refresh** — pulls the latest published HDI and life expectancy
   per country from [Our World in Data](https://ourworldindata.org) (sourced
   from UNDP / UN World Population Prospects) and overwrites those columns.
2. **Synthetic expansion** — adds ~200 new rows by bootstrap-resampling real
   existing rows (a genuinely-observed country/university/education/field/
   language combination each time) and lightly jittering the numeric scores
   so they're not exact duplicates. This keeps the real joint correlations
   in the data intact without needing any LLM call or network access for
   that step.

## Setup
1. Push this folder (including `student_dataset.xlsx`) to a GitHub repo.
2. That's it — no secrets or API keys to configure.
3. The workflow in `.github/workflows/weekly-update.yml` runs automatically
   every Monday at 03:00 UTC. You can also trigger it manually from the
   **Actions** tab → *Weekly dataset update* → *Run workflow*.

## Running locally to test
```bash
pip install -r requirements.txt
python update_dataset.py
```

## Notes
- HDI is only published roughly once a year, so most weekly runs will find
  no change there — that's expected, not a bug.
- `NEW_ROWS_PER_RUN` in `update_dataset.py` controls how many synthetic rows
  get added each run; adjust to control dataset growth rate.
- `update_log.txt` accumulates a one-line history of each run.
- If you'd rather generate new rows with an LLM instead of resampling, a
  free Anthropic trial ($5 in credits, no card needed, at console.anthropic.com)
  would cover this workload for a while — but resampling costs nothing and
  needs no key at all, so it's the more sustainable default.
