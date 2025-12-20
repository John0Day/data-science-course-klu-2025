import pandas as pd

from plot_generation import aggregate
from event_utils import load_event_lookup


def test_aggregate_shapes_with_sample_events(tmp_path):
    # Minimal synthetic data to verify aggregate returns expected keys and non-negative rates
    # Create tiny sample CSV
    data = pd.DataFrame(
        {
            "fl_date": ["2024-01-01", "2024-01-01", "2024-02-02"],
            "origin": ["AAA", "BBB", "AAA"],
            "dest": ["BBB", "AAA", "CCC"],
            "op_unique_carrier": ["XX", "YY", "XX"],
            "crs_dep_time": [800, 1500, 930],
            "arr_delay": [20, -5, 30],
            "cancelled": [0, 0, 0],
            "diverted": [0, 0, 0],
            "month": [1, 1, 2],
            "day_of_week": [1, 1, 5],
            "distance": [500, 600, 700],
            "weather_delay": [5, 0, 10],
        }
    )
    sample_path = tmp_path / "flight_data_2024_sample.csv"
    data.to_csv(sample_path, index=False)

    # Minimal events CSV
    events_path = tmp_path / "combined_schedules_with_airports.csv"
    events_path.write_text(
        "League,Team Name,Stadium Name,Stadium Lat,Stadium Lon,Primary Airport,Airport Lat,Airport Lon,Primary Airport Distance,Date,Home Team,Away Team\n"
        "NFL,Team A,Stadium A,0,0,AAA,0,0,1.0,Mon,2024-01-01,Team A,Team B\n"
    )

    events_lookup = load_event_lookup(events_path)

    summary = aggregate(
        use_sample=True,
        nrows=None,
        chunk_size=10,
        events_lookup=events_lookup,
    )

    # Basic shape checks
    assert "month_rates" in summary and not summary["month_rates"].empty
    assert "hour_rates" in summary
    assert "weekday_rates" in summary
    assert "weather_origin_counts" in summary
    assert "event_total" in summary

    # Rates should be non-negative
    assert (summary["month_rates"] >= 0).all()
    assert (summary["hour_rates"].fillna(0) >= 0).all()
