# Flight Delay & Event Impact Analysis (2024)

This project explores 2024 US flight delays, event-day impacts (sports schedules), weather effects, and baseline ML models.

## Datasets (place in `rawdata/`)
- `flight_data_2024.csv` (full) or `flight_data_2024_sample.csv` (for quick tests)
- `combined_schedules_with_airports.csv` (sports/venue events mapped to primary airports)
- Optional: `GlobalWeatherRepository.csv` (only if you want to run the weather loader)

## Setup
- Python 3 with `pandas`, `scikit-learn`, `numpy`, `seaborn`, `matplotlib`.
- `lightgbm` for the full modeling pipeline (`python3 -m pip install lightgbm`)—requires a C/C++ build toolchain.
- Optional: `xgboost` (only used by the legacy `Felipe` script).
- Install deps (example):  
  `python3 -m pip install pandas scikit-learn numpy seaborn matplotlib lightgbm`

## Core scripts

### End-to-end modeling pipeline (classification, regression, clustering)
`python3 modeling_pipeline.py --sample --nrows 50000 --diagnostics`
- Uses memory-optimized, chunked loading from `CSV Files/flight_data_2024.csv` (or `flight_data_2024_sample.csv` when `--sample` is set).
- Models: Logistic Regression, Random Forest, LightGBM classifier; Linear Regression, Gradient Boosting Regressor, LightGBM regressor; MiniBatchKMeans clustering.
- Flags: `--sample`, `--nrows`, `--chunk-size`, `--diagnostics` (plots/learning curves), `--ev-costs C_TP C_FP C_FN C_TN` for expected-value thresholds.
- Outputs metrics to stdout; saves ROC/learning/complexity curves when matplotlib is available.

### Analysis: delays, routes, events, weather
`python3 analysis_script --chunk-size 250000 --top 10 --min-airport-flights 1000 --event-top 30`
- Outputs: delay rates by month/weekday/hour, carrier/airport counts, routes, average delay minutes for top origins/dests, largest airports vs overall, event-day table, weather-driven delays (using `weather_delay`).
- Flags: `--chunk-size` (streaming size), `--top` (rows shown), `--min-airport-flights` (display threshold), `--event-top` (rows in event table).

### Plot generation (PNG charts)
`python3 plot_generation.py --chunk-size 250000 --top 10`
- Saves plots to `plots/`: delay-rate by hour/month, weather top origins, event delays, top-5 origins heatmap, carrier delay rates.
- Flags: `--sample`, `--nrows`, `--chunk-size`, `--top` (for bar charts).

### Route/event delay classifier
`python3 route_event_prediction.py --model logreg --top-routes 15 --min-route-flights 50`
- Features: origin, dest, carrier, month, day_of_week, dep_hour, distance, event flag/name.
- Models: logistic regression (default) or random forest (`--model rf`).
- Shows metrics, route risk summary, event comparison. Use `--sample` / `--nrows` for quick runs.

### Baseline delay model
`python3 baseline_model.py --sample --nrows 50000 --model logreg`
- Random Forest variant: `--model rf`; full run: `--model logreg --test-size 0.2`.
- Predicts arr_delay ≥ 15 with One-Hot + StandardScaler; metrics: Accuracy, Precision, Recall, F1, Confusion Matrix.

## Notes
- Delayed flight definition: `arr_delay >= 15`.
- Cancelled/diverted flights and rows without `arr_delay` are dropped.
- Event mapping comes from `combined_schedules_with_airports.csv`; missing airports default to `none`.
