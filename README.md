# Flight Delay & Event Impact Analysis (2024)

This project explores 2024 US flight delays, event-day impacts (sports schedules), and baseline ML models.

## Setup 
1. Download the Dataset from Kaggle, under this link: https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024.
2. Place the big dataset called `flight_data_2024.csv` in the `rawdata/` folder.
3. Install all the needed dependencies: `python3 -m pip install pandas scikit-learn numpy seaborn matplotlib lightgbm`
4. Run the command `python3 runner.py` to open the choices menu or run commands straight from the ##Core scirpt list further down.

## Core script

### Analysis: delays, routes, events, weather
`python3 analysis_script --chunk-size 250000 --top 10 --min-airport-flights 1000 --event-top 30`
- Outputs: delay rates by month/weekday/hour, carrier/airport counts, routes, avg delay minutes for top origins/dests, largest airports vs overall, event-day table, weather-driven delays.
- Flags: `--sample`, `--nrows`, `--chunk-size`, `--top`, `--min-airport-flights`, `--event-top`.
`python3 analysis_script --sample --nrows 50000 --chunk-size 250000 --top 10 --min-airport-flights 1000 --event-top 30`

### Plot generation (PNGs)
`python3 plot_generation.py --chunk-size 250000 --top 10`
- Saves plots to `plots/`: delay-rate by hour/month, weather top origins, event delays, top-5 origins heatmap, carrier delay rates.
- Flags: `--sample`, `--nrows`, `--chunk-size`, `--top`.

### Modeling pipeline (heavy)
`python3 modeling_pipeline.py --sample --nrows 50000 --diagnostics`
- Chunked loading from `rawdata/`, feature engineering, models: Logistic Regression, Random Forest, LightGBM (classification); Linear Regression, Gradient Boosting, LightGBM (regression); MiniBatchKMeans clustering.
- Flags: `--sample`, `--nrows`, `--chunk-size`, `--diagnostics`, `--plots`, `--ev-costs C_TP C_FP C_FN C_TN`.
- Use `--plots` to save confusion matrices, profit curve, feature importance, regression, and clustering PNGs; `--diagnostics` adds ROC/learning/complexity curves (matplotlib required). Requires `lightgbm`.
- Use `--sample` for the smaller dataset; omit it to run on the full CSV.
`python3 modeling_pipeline.py --sample --nrows 50000 --plots --diagnostics`

## Notes
- Delayed flight: `arr_delay >= 15`.
- Cancelled/diverted flights and rows without `arr_delay` are dropped.
- Event mapping comes from `combined_schedules_with_airports.csv`; missing airports default to `none`.

