# Flight Delay & Event Impact Analysis (2024)

This project explores 2024 US flight delays, event-day impacts (sports schedules), weather effects, and baseline ML models for delay prediction.

## Datasets (place in `rawdata/`)
- `flight_data_2024.csv` (full) or `flight_data_2024_sample.csv` (quick tests)
- `combined_schedules_with_airports.csv` (sports/venue events mapped to primary airports)
- Optional: `GlobalWeatherRepository.csv` (for the weather loader script)

## Core scripts

### Analysis: delay breakdowns, routes, events, weather
`python3 analysis_script --chunk-size 250000 --top 10 --min-airport-flights 1000 --event-top 30`
- `--chunk-size`: streaming size (lower if RAM is tight; higher if plenty available).
- `--top`: rows shown in most summary tables.
- `--min-airport-flights`: display threshold; calculations still use all data.
- `--event-top`: how many event rows to show (from `combined_schedules_with_airports.csv`).
Outputs: delay rates by month/weekday/hour, carrier/airport counts, routes, average delay minutes for top origins/dests, largest airports vs overall, event-day table, plus weather-driven delays (using `weather_delay`).

### Route/event delay classifier
`python3 route_event_prediction.py --model logreg --top-routes 15 --min-route-flights 50`
- Features: origin, dest, carrier, month, day_of_week, dep_hour, distance, event flag/name.
- Models: logistic regression (default) or random forest (`--model rf`).
- Shows metrics, route risk summary, and event comparison. Use `--sample` / `--nrows` for quick runs.

### Baseline delay model
`python3 baseline_model.py --sample --nrows 50000 --model logreg`
- Random Forest variant: `--model rf`
- Full run: `python3 baseline_model.py --model logreg --test-size 0.2`
- Predicts arr_delay ≥ 15 with One-Hot + StandardScaler; metrics: Accuracy, Precision, Recall, F1, Confusion Matrix.

### Weather data loader (Kaggle CSV)
`python3 weather_data_script`  
Prints head/columns/shape/missing stats for `rawdata/GlobalWeatherRepository.csv`.

### Legacy end-to-end modeling (Felipe)
`python3 Felipe`  
Runs classification/regression/clustering on the flight dataset (uses `rawdata/flight_data_2024.csv`, falls back to sample).

## Setup
- Python 3 with `pandas`, `scikit-learn`, `numpy`, `seaborn`, `matplotlib`. Optional: `xgboost` for Felipe.
- Place CSVs in `rawdata/` as listed above.

## Notes
- Delayed flight definition: `arr_delay >= 15`.
- Cancelled/diverted flights and rows without `arr_delay` are dropped in analyses/models.
- Event mapping comes from `combined_schedules_with_airports.csv`; missing airports default to `none`.
