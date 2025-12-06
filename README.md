# data-science-course-klu-2025
# a simple Data-Science project for the course Data-Science @ KLU
#
# Flight delay analysis

- Full dataset: `python3 test_script --chunk-size 250000 --top 10 --min-airport-flights 1000 --event-top 30`
  - `--chunk-size`: Streaming chunk size; lower if RAM is tight, higher if plenty available.
  - `--top`: How many rows to display in most summary tables.
  - `--min-airport-flights`: Minimum flights for airport-level rate tables (display only; calculations use all data).
  - `--event-top`: How many event rows to show (events from `rawdata/combined_schedules_with_airports.csv`).
- Outputs include: delay rates (month, weekday, hour), carrier/airport counts, route stats, average delay minutes for top origins/destinations, top routes, largest airports vs overall, event-day delay table.

# Baseline delay model

- Quick test on sample: `python3 baseline_model.py --sample --nrows 50000 --model logreg`
- Random Forest on a subset: `python3 baseline_model.py --nrows 200000 --model rf`
- Full run: `python3 baseline_model.py --model logreg --test-size 0.2`

Notes:
- `baseline_model.py` uses One-Hot + StandardScaler to predict delays (arr_delay ≥ 15); metrics: Accuracy, Precision, Recall, F1, Confusion Matrix.
- Dependencies: `pandas`, `scikit-learn`.
