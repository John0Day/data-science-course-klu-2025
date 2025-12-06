# Flight Delay & Event Impact Analysis (2024)

This project explores 2024 US flight delays, event-day impacts (sports schedules), weather effects, and baseline ML models.

## Datasets (place in `rawdata/`)
- `flight_data_2024.csv` (full) or `flight_data_2024_sample.csv` (for quick tests)
- `combined_schedules_with_airports.csv` (sports/venue events mapped to primary airports)
- Optional: `GlobalWeatherRepository.csv` (for the weather loader)

## Core scripts

### Analysis: delays, routes, events, weather
`python3 analysis_script --chunk-size 250000 --top 10 --min-airport-flights 1000 --event-top 30`
- Outputs: delay rates by month/weekday/hour, carrier/airport counts, routes, average delay minutes for top origins/dests, largest airports vs overall, event-day table, weather-driven delays (using `weather_delay`).
- Flags: `--chunk-size` (streaming size), `--top` (rows shown), `--min-airport-flights` (display threshold), `--event-top` (rows in event table).

### Route/event delay classifier
`python3 route_event_prediction.py --model logreg --top-routes 15 --min-route-flights 50`
- Features: origin, dest, carrier, month, day_of_week, dep_hour, distance, event flag/name.
- Models: logistic regression (default) or random forest (`--model rf`).
- Shows metrics, route risk summary, event comparison. Use `--sample` / `--nrows` for quick runs.

### Baseline delay model
`python3 baseline_model.py --sample --nrows 50000 --model logreg`
- Random Forest variant: `--model rf`; full run: `--model logreg --test-size 0.2`.
- Predicts arr_delay ≥ 15 with One-Hot + StandardScaler; metrics: Accuracy, Precision, Recall, F1, Confusion Matrix.

### Weather data loader (optional)
`python3 weather_data_script`  
Prints head/columns/shape/missing stats for `rawdata/GlobalWeatherRepository.csv`.

### Legacy end-to-end script
`python3 Felipe`  
Older all-in-one modeling (classification/regression/clustering) on the flight dataset.

## Setup
- Python 3 with `pandas`, `scikit-learn`, `numpy`, `seaborn`, `matplotlib`. Optional: `xgboost` for `Felipe`.
- Place the CSVs in `rawdata/` as listed above.

## Notes
- Delayed flight definition: `arr_delay >= 15`.
- Cancelled/diverted flights and rows without `arr_delay` are dropped.
- Event mapping comes from `combined_schedules_with_airports.csv`; missing airports default to `none`.
