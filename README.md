# Flight Delay & Event Impact Analysis (2024)

This project explores 2024 US flight delays, event-day impacts (sports schedules), and baseline ML models.

## Datasets (place in `rawdata/`)
- `flight_data_2024.csv` (full) or `flight_data_2024_sample.csv` (quick tests)
- `combined_schedules_with_airports.csv` (sports/venue events mapped to primary airports)

## Setup
- Python 3 with `pandas`, `scikit-learn`, `numpy`, `seaborn`, `matplotlib`.
- `lightgbm` for the full modeling pipeline: `python3 -m pip install lightgbm` (requires build tools).
- Install essentials:  
  `python3 -m pip install pandas scikit-learn numpy seaborn matplotlib`

## Tests
- Install pytest: `python3 -m pip install pytest`
- Run: `python3 -m pytest`

## Runner (interactive)
`python3 runner.py` — choose between analysis, plots, route classifier, baseline model, or full pipeline.

## Core scripts

### Analysis: delays, routes, events, weather
`python3 analysis_script --chunk-size 250000 --top 10 --min-airport-flights 1000 --event-top 30`
- Outputs: delay rates by month/weekday/hour, carrier/airport counts, routes, avg delay minutes for top origins/dests, largest airports vs overall, event-day table, weather-driven delays.
- Flags: `--chunk-size`, `--top`, `--min-airport-flights`, `--event-top`.

### Plot generation (PNGs)
`python3 plot_generation.py --chunk-size 250000 --top 10`
- Saves plots to `plots/`: delay-rate by hour/month, weather top origins, event delays, top-5 origins heatmap, carrier delay rates.
- Flags: `--sample`, `--nrows`, `--chunk-size`, `--top`.

### Route/event delay classifier
`python3 route_event_prediction.py --model logreg --top-routes 15 --min-route-flights 50`
- Features: origin, dest, carrier, month, day_of_week, dep_hour, distance, event flag/name.
- Models: logistic regression (default) or random forest (`--model rf`).
- Shows metrics, route risk summary, event comparison. Use `--sample` / `--nrows` for quick runs.

### Baseline delay model
`python3 baseline_model.py --sample --nrows 50000 --model logreg`
- Random Forest variant: `--model rf`; full run: `--model logreg --test-size 0.2`.
- Predicts arr_delay ≥ 15 with One-Hot + StandardScaler; metrics: Accuracy, Precision, Recall, F1, Confusion Matrix.

### Modeling pipeline (heavy)
`python3 modeling_pipeline.py --sample --nrows 50000 --diagnostics`
- Chunked loading from `rawdata/`, feature engineering, models: Logistic Regression, Random Forest, LightGBM (classification); Linear Regression, Gradient Boosting, LightGBM (regression); MiniBatchKMeans clustering.
- Flags: `--sample`, `--nrows`, `--chunk-size`, `--diagnostics`, `--ev-costs C_TP C_FP C_FN C_TN`.
- Saves ROC/learning/complexity curves if matplotlib is available; requires `lightgbm`.

## Notes
- Delayed flight: `arr_delay >= 15`.
- Cancelled/diverted flights and rows without `arr_delay` are dropped.
- Event mapping comes from `combined_schedules_with_airports.csv`; missing airports default to `none`.
