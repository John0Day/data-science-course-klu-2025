#!/usr/bin/env python3
"""
Analyze flight delays for 2024 data.

The script cleans the raw CSV in chunks (to handle the full dataset size) and
summarizes which flights were delayed and what likely caused the delays. It also
breaks down delays by carrier, airport, route, month, weekday, and departure
hour, plus aggregates total delay minutes per cause.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from pandas import DataFrame, Series

from event_utils import load_event_lookup

DATA_PATH = Path("rawdata/flight_data_2024.csv")
SAMPLE_PATH = Path("rawdata/flight_data_2024_sample.csv")
EVENTS_PATH = Path("rawdata/combined_schedules_with_airports.csv")

# FAA convention: arrival delay of 15 minutes or more counts as a delayed flight.
ARRIVAL_DELAY_THRESHOLD = 15
DEFAULT_CHUNK_SIZE = 250_000
DEFAULT_MIN_AIRPORT_FLIGHTS = 1000

# Basic dictionaries for readability in outputs.
CARRIER_NAMES = {
    "AA": "American Airlines",
    "AS": "Alaska Airlines",
    "B6": "JetBlue Airways",
    "DL": "Delta Air Lines",
    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines",
    "NK": "Spirit Airlines",
    "OO": "SkyWest Airlines",
    "UA": "United Airlines",
    "WN": "Southwest Airlines",
    "YX": "Republic Airways",
    "MQ": "Envoy Air",
    "9E": "Endeavor Air",
    "YV": "Mesa Airlines",
    "QX": "Horizon Air",
}

DELAY_REASON_NAMES = {
    "carrier_delay": "Carrier (Airline)",
    "weather_delay": "Weather",
    "nas_delay": "NAS (ATC/System)",
    "security_delay": "Security",
    "late_aircraft_delay": "Late Aircraft",
    "unknown": "Unknown/Other",
}

# Lightweight airport name map for readability in top tables (not exhaustive).
AIRPORT_NAMES = {
    "ATL": "Atlanta",
    "DFW": "Dallas/Fort Worth",
    "DEN": "Denver",
    "ORD": "Chicago O'Hare",
    "LAX": "Los Angeles",
    "CLT": "Charlotte",
    "MCO": "Orlando",
    "PHX": "Phoenix",
    "LAS": "Las Vegas",
    "SEA": "Seattle",
    "SFO": "San Francisco",
    "IAH": "Houston Intercontinental",
    "EWR": "Newark",
    "BOS": "Boston",
    "MSP": "Minneapolis",
    "FLL": "Fort Lauderdale",
    "MDW": "Chicago Midway",
    "DTW": "Detroit",
    "SLC": "Salt Lake City",
    "IAD": "Dulles",
    "BWI": "Baltimore",
    "LGA": "LaGuardia",
    "JFK": "JFK",
    "DCA": "Reagan National",
}

# Columns needed for the delay analysis.
USECOLS = [
    "year",
    "month",
    "day_of_month",
    "day_of_week",
    "fl_date",
    "op_unique_carrier",
    "op_carrier_fl_num",
    "origin",
    "dest",
    "crs_dep_time",
    "dep_time",
    "dep_delay",
    "taxi_out",
    "wheels_off",
    "wheels_on",
    "taxi_in",
    "crs_arr_time",
    "arr_time",
    "arr_delay",
    "cancelled",
    "cancellation_code",
    "diverted",
    "crs_elapsed_time",
    "actual_elapsed_time",
    "air_time",
    "distance",
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
]

# Memory-conscious dtype mapping based on the data dictionary.
DTYPES = {
    "year": "int16",
    "month": "int8",
    "day_of_month": "int8",
    "day_of_week": "int8",
    "op_unique_carrier": "category",
    "op_carrier_fl_num": "float32",
    "origin": "category",
    "dest": "category",
    "crs_dep_time": "Int32",
    "dep_time": "float32",
    "dep_delay": "float32",
    "taxi_out": "float32",
    "wheels_off": "float32",
    "wheels_on": "float32",
    "taxi_in": "float32",
    "crs_arr_time": "Int32",
    "arr_time": "float32",
    "arr_delay": "float32",
    "cancelled": "int8",
    "cancellation_code": "category",
    "diverted": "int8",
    "crs_elapsed_time": "float32",
    "actual_elapsed_time": "float32",
    "air_time": "float32",
    "distance": "float32",
    "carrier_delay": "float32",
    "weather_delay": "float32",
    "nas_delay": "float32",
    "security_delay": "float32",
    "late_aircraft_delay": "float32",
}

DELAY_REASON_COLS = [
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
]


def read_in_chunks(
    use_sample: bool = False,
    nrows: Optional[int] = None,
    chunk_size: Optional[int] = DEFAULT_CHUNK_SIZE,
) -> Iterable[DataFrame]:
    """
    Yield pandas DataFrames in manageable chunks so we can process the full dataset.
    """
    path = SAMPLE_PATH if use_sample else DATA_PATH
    reader = pd.read_csv(
        path,
        usecols=USECOLS,
        dtype=DTYPES,
        parse_dates=["fl_date"],
        na_values=["", "NA", "NaN"],
        chunksize=chunk_size,
        nrows=nrows,
        low_memory=False,
    )
    if isinstance(reader, DataFrame):
        yield reader
    else:
        yield from reader


def clean_chunk(df: DataFrame) -> DataFrame:
    """
    Drop cancelled/diverted flights, coerce numeric columns, and tag delay reasons.
    """
    cleaned = df.copy()
    cleaned = cleaned[(cleaned["cancelled"] == 0) & (cleaned["diverted"] == 0)]

    numeric_cols = [
        "dep_delay",
        "arr_delay",
        "taxi_out",
        "wheels_off",
        "wheels_on",
        "taxi_in",
        "arr_time",
        "crs_arr_time",
        "crs_dep_time",
        "actual_elapsed_time",
        "air_time",
        "distance",
        "crs_elapsed_time",
        *DELAY_REASON_COLS,
    ]
    cleaned[numeric_cols] = cleaned[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    cleaned["dep_hour"] = pd.to_numeric(
        cleaned["crs_dep_time"] // 100, errors="coerce"
    ).astype("Int16")

    cleaned = cleaned[cleaned["arr_delay"].notna()]
    cleaned["is_delayed_arr"] = cleaned["arr_delay"] >= ARRIVAL_DELAY_THRESHOLD

    reason_flags = cleaned[DELAY_REASON_COLS].gt(0)
    cleaned["reason"] = "unknown"
    for col in DELAY_REASON_COLS:
        mask = reason_flags[col] & (cleaned["reason"] == "unknown")
        cleaned.loc[mask, "reason"] = col

    return cleaned


def summarize_delays(chunks: Iterable[DataFrame], events_lookup: Optional[dict] = None) -> dict:
    total_rows = 0
    processed_rows = 0
    delayed_rows = 0
    arr_delay_sum_all = 0.0
    arr_delay_count_all = 0
    arr_delay_sum_delayed = 0.0
    max_arr_delay = None
    weather_delayed = 0
    weather_delay_minutes_total = 0.0

    reason_counts = Series(dtype="int64")
    carrier_counts = Series(dtype="int64")
    origin_counts = Series(dtype="int64")  # delayed flights per origin
    dest_counts = Series(dtype="int64")  # delayed flights per destination
    origin_total = Series(dtype="int64")  # all flights per origin
    dest_total = Series(dtype="int64")  # all flights per destination
    origin_delay_minutes = Series(dtype="float64")
    dest_delay_minutes = Series(dtype="float64")
    weather_origin_counts = Series(dtype="int64")
    weather_dest_counts = Series(dtype="int64")
    weather_origin_minutes = Series(dtype="float64")
    weather_dest_minutes = Series(dtype="float64")
    route_counts = Series(dtype="int64")  # delayed flights per route
    route_total = Series(dtype="int64")  # all flights per route
    route_delay_minutes = Series(dtype="float64")
    month_total = Series(dtype="int64")
    month_delayed = Series(dtype="int64")
    weekday_total = Series(dtype="int64")
    weekday_delayed = Series(dtype="int64")
    hour_total = Series(dtype="int64")
    hour_delayed = Series(dtype="int64")
    empty_origin_hour_index = pd.MultiIndex.from_arrays(
        [[], []], names=["origin", "dep_hour"]
    )
    origin_hour_total = Series(dtype="int64", index=empty_origin_hour_index)
    origin_hour_delayed = Series(dtype="int64", index=empty_origin_hour_index)
    reason_delay_minutes = Series(dtype="float64")
    event_total = Series(dtype="int64")
    event_delayed = Series(dtype="int64")
    event_delay_minutes = Series(dtype="float64")

    empty_event_origin_index = pd.MultiIndex.from_arrays(
        [[], []], names=["event_name", "origin"]
    )
    event_origin_total = Series(dtype="int64", index=empty_event_origin_index)
    event_origin_delayed = Series(dtype="int64", index=empty_event_origin_index)
    event_origin_delay_minutes = Series(
        dtype="float64", index=empty_event_origin_index
    )
    sample_delays = []

    for chunk in chunks:
        total_rows += len(chunk)
        cleaned = clean_chunk(chunk)
        processed_rows += len(cleaned)

        if cleaned.empty:
            continue

        arr_delay_sum_all += cleaned["arr_delay"].sum(skipna=True)
        arr_delay_count_all += cleaned["arr_delay"].notna().sum()

        delayed_mask = cleaned["is_delayed_arr"]
        delayed_rows += delayed_mask.sum()
        arr_delay_sum_delayed += cleaned.loc[delayed_mask, "arr_delay"].sum(
            skipna=True
        )
        if not cleaned["arr_delay"].empty:
            chunk_max = cleaned["arr_delay"].max()
            if max_arr_delay is None or chunk_max > max_arr_delay:
                max_arr_delay = chunk_max

        reason_counts = reason_counts.add(
            cleaned.loc[delayed_mask, "reason"].value_counts(), fill_value=0
        )
        carrier_counts = carrier_counts.add(
            cleaned.loc[delayed_mask]
            .groupby("op_unique_carrier", observed=True)
            .size(),
            fill_value=0,
        )
        origin_counts = origin_counts.add(
            cleaned.loc[delayed_mask].groupby("origin", observed=True).size(),
            fill_value=0,
        )
        dest_counts = dest_counts.add(
            cleaned.loc[delayed_mask].groupby("dest", observed=True).size(),
            fill_value=0,
        )
        origin_total = origin_total.add(
            cleaned.groupby("origin", observed=True).size(), fill_value=0
        )
        dest_total = dest_total.add(
            cleaned.groupby("dest", observed=True).size(), fill_value=0
        )
        origin_delay_minutes = origin_delay_minutes.add(
            cleaned.loc[delayed_mask]
            .groupby("origin", observed=True)["arr_delay"]
            .sum(),
            fill_value=0,
        )
        dest_delay_minutes = dest_delay_minutes.add(
            cleaned.loc[delayed_mask]
            .groupby("dest", observed=True)["arr_delay"]
            .sum(),
            fill_value=0,
        )
        weather_mask = cleaned["weather_delay"].fillna(0) > 0
        weather_delayed += weather_mask.sum()
        weather_delay_minutes_total += cleaned.loc[weather_mask, "weather_delay"].sum(
            skipna=True
        )
        weather_origin_counts = weather_origin_counts.add(
            cleaned.loc[weather_mask].groupby("origin", observed=True).size(),
            fill_value=0,
        )
        weather_dest_counts = weather_dest_counts.add(
            cleaned.loc[weather_mask].groupby("dest", observed=True).size(),
            fill_value=0,
        )
        weather_origin_minutes = weather_origin_minutes.add(
            cleaned.loc[weather_mask]
            .groupby("origin", observed=True)["weather_delay"]
            .sum(),
            fill_value=0,
        )
        weather_dest_minutes = weather_dest_minutes.add(
            cleaned.loc[weather_mask]
            .groupby("dest", observed=True)["weather_delay"]
            .sum(),
            fill_value=0,
        )
        all_routes = cleaned["origin"].astype("string").str.cat(
            cleaned["dest"].astype("string"), sep="-"
        )
        route_total = route_total.add(all_routes.value_counts(), fill_value=0)

        delayed_routes = cleaned.loc[delayed_mask, "origin"].astype("string").str.cat(
            cleaned.loc[delayed_mask, "dest"].astype("string"), sep="-"
        )
        route_counts = route_counts.add(delayed_routes.value_counts(), fill_value=0)

        route_delay_minutes = route_delay_minutes.add(
            cleaned.loc[delayed_mask]
            .groupby(delayed_routes, observed=True)["arr_delay"]
            .sum(),
            fill_value=0,
        )
        month_total = month_total.add(cleaned["month"].value_counts(), fill_value=0)
        month_delayed = month_delayed.add(
            cleaned.loc[delayed_mask, "month"].value_counts(), fill_value=0
        )
        weekday_total = weekday_total.add(
            cleaned["day_of_week"].value_counts(), fill_value=0
        )
        weekday_delayed = weekday_delayed.add(
            cleaned.loc[delayed_mask, "day_of_week"].value_counts(), fill_value=0
        )
        hour_total = hour_total.add(
            cleaned["dep_hour"].value_counts(), fill_value=0
        )
        hour_delayed = hour_delayed.add(
            cleaned.loc[delayed_mask, "dep_hour"].value_counts(), fill_value=0
        )
        origin_hour_total = origin_hour_total.add(
            cleaned.groupby(["origin", "dep_hour"], observed=True).size(), fill_value=0
        )
        origin_hour_delayed = origin_hour_delayed.add(
            cleaned.loc[delayed_mask]
            .groupby(["origin", "dep_hour"], observed=True)
            .size(),
            fill_value=0,
        )
        reason_delay_minutes = reason_delay_minutes.add(
            cleaned.loc[delayed_mask].groupby("reason", observed=True)["arr_delay"].sum(),
            fill_value=0,
        )
        if events_lookup:
            cleaned["event_name"] = "none"
            origin_keys = pd.Series(
                list(zip(cleaned["fl_date"].dt.date, cleaned["origin"])),
                index=cleaned.index,
            )
            cleaned["event_name"] = origin_keys.map(events_lookup).fillna("none")

            no_event_mask = cleaned["event_name"] == "none"
            if no_event_mask.any():
                dest_keys = pd.Series(
                    list(
                        zip(
                            cleaned.loc[no_event_mask, "fl_date"].dt.date,
                            cleaned.loc[no_event_mask, "dest"],
                        )
                    ),
                    index=cleaned.index[no_event_mask],
                )
                cleaned.loc[no_event_mask, "event_name"] = (
                    dest_keys.map(events_lookup).fillna("none")
                )

            event_total = event_total.add(
                cleaned["event_name"].value_counts(), fill_value=0
            )
            event_delayed = event_delayed.add(
                cleaned.loc[delayed_mask, "event_name"].value_counts(), fill_value=0
            )
            event_delay_minutes = event_delay_minutes.add(
                cleaned.loc[delayed_mask]
                .groupby("event_name", observed=True)["arr_delay"]
                .sum(),
                fill_value=0,
            )
            # per-origin breakdown on event days
            origin_total_chunk = (
                cleaned.groupby(["event_name", "origin"], observed=True).size()
            )
            origin_total_chunk.index = origin_total_chunk.index.set_names(
                ["event_name", "origin"]
            )
            event_origin_total = event_origin_total.add(
                origin_total_chunk, fill_value=0
            )

            origin_delayed_chunk = (
                cleaned.loc[delayed_mask]
                .groupby(["event_name", "origin"], observed=True)
                .size()
            )
            origin_delayed_chunk.index = origin_delayed_chunk.index.set_names(
                ["event_name", "origin"]
            )
            event_origin_delayed = event_origin_delayed.add(
                origin_delayed_chunk, fill_value=0
            )

            origin_delay_minutes_chunk = (
                cleaned.loc[delayed_mask]
                .groupby(["event_name", "origin"], observed=True)["arr_delay"]
                .sum()
            )
            origin_delay_minutes_chunk.index = (
                origin_delay_minutes_chunk.index.set_names(["event_name", "origin"])
            )
            event_origin_delay_minutes = event_origin_delay_minutes.add(
                origin_delay_minutes_chunk, fill_value=0
            )

        if len(sample_delays) < 10:
            sample_rows = cleaned.loc[
                delayed_mask,
                [
                    "fl_date",
                    "op_unique_carrier",
                    "op_carrier_fl_num",
                    "origin",
                    "dest",
                    "arr_delay",
                    "reason",
                ],
            ].head(10 - len(sample_delays))
            sample_delays.extend(sample_rows.to_dict(orient="records"))

    return {
        "total_rows": total_rows,
        "processed_rows": processed_rows,
        "delayed_rows": delayed_rows,
        "delay_rate_pct": (delayed_rows / processed_rows * 100) if processed_rows else 0,
        "weather_delayed": weather_delayed,
        "weather_delay_minutes_total": weather_delay_minutes_total,
        "arr_delay_mean_all": (arr_delay_sum_all / arr_delay_count_all)
        if arr_delay_count_all
        else 0,
        "arr_delay_mean_delayed": (arr_delay_sum_delayed / delayed_rows)
        if delayed_rows
        else 0,
        "max_arr_delay": max_arr_delay,
        "reason_counts": reason_counts.sort_values(ascending=False).astype(int),
        "reason_delay_minutes": reason_delay_minutes.sort_values(
            ascending=False
        ),
        "carrier_counts": carrier_counts.sort_values(ascending=False).astype(int),
        "origin_counts": origin_counts.sort_values(ascending=False).astype(int),
        "dest_counts": dest_counts.sort_values(ascending=False).astype(int),
        "origin_total": origin_total.sort_values(ascending=False).astype(int),
        "dest_total": dest_total.sort_values(ascending=False).astype(int),
        "origin_delay_minutes": origin_delay_minutes.sort_values(
            ascending=False
        ),
        "dest_delay_minutes": dest_delay_minutes.sort_values(ascending=False),
        "weather_origin_counts": weather_origin_counts.sort_values(ascending=False).astype(int)
        if not weather_origin_counts.empty
        else Series(dtype="int64"),
        "weather_dest_counts": weather_dest_counts.sort_values(ascending=False).astype(int)
        if not weather_dest_counts.empty
        else Series(dtype="int64"),
        "weather_origin_minutes": weather_origin_minutes.sort_values(ascending=False)
        if not weather_origin_minutes.empty
        else Series(dtype="float64"),
        "weather_dest_minutes": weather_dest_minutes.sort_values(ascending=False)
        if not weather_dest_minutes.empty
        else Series(dtype="float64"),
        "route_counts": route_counts.sort_values(ascending=False).astype(int),
        "route_total": route_total.sort_values(ascending=False).astype(int),
        "route_delay_minutes": route_delay_minutes.sort_values(ascending=False),
        "month_rates": (month_delayed / month_total * 100).sort_index(),
        "weekday_rates": (weekday_delayed / weekday_total * 100).sort_index(),
        "hour_rates": (hour_delayed / hour_total * 100).sort_index(),
        "origin_hour_total": origin_hour_total.sort_values(ascending=False).astype(int),
        "origin_hour_delayed": origin_hour_delayed.sort_values(ascending=False).astype(int),
        "event_total": event_total.sort_values(ascending=False).astype(int)
        if not event_total.empty
        else Series(dtype="int64"),
        "event_delayed": event_delayed.sort_values(ascending=False).astype(int)
        if not event_delayed.empty
        else Series(dtype="int64"),
        "event_delay_minutes": event_delay_minutes.sort_values(
            ascending=False
        )
        if not event_delay_minutes.empty
        else Series(dtype="float64"),
        "event_origin_total": event_origin_total.sort_values(ascending=False).astype(int)
        if not event_origin_total.empty
        else Series(dtype="int64"),
        "event_origin_delayed": event_origin_delayed.sort_values(ascending=False).astype(int)
        if not event_origin_delayed.empty
        else Series(dtype="int64"),
        "event_origin_delay_minutes": event_origin_delay_minutes.sort_values(
            ascending=False
        )
        if not event_origin_delay_minutes.empty
        else Series(dtype="float64"),
        "sample_delays": sample_delays,
    }


def print_summary(
    summary: dict,
    top_n: int = 10,
    min_airport_flights: int = DEFAULT_MIN_AIRPORT_FLIGHTS,
    event_top: int = 20,
) -> None:
    print(f"Rows in file: {summary['total_rows']:,}")
    print(f"Rows after cleaning (no cancellations/diversions): {summary['processed_rows']:,}")
    print(
        f"Delayed flights (arr_delay >= {ARRIVAL_DELAY_THRESHOLD}): {summary['delayed_rows']:,} "
        f"({summary['delay_rate_pct']:.2f}% of clean flights)"
    )
    print(
        f"Mean arr_delay (all flights with data): {summary['arr_delay_mean_all']:.2f} minutes; "
        f"mean among delayed flights: {summary['arr_delay_mean_delayed']:.2f} minutes; "
        f"max observed arrival delay: {summary['max_arr_delay']}"
    )
    if summary["weather_delayed"]:
        weather_share = summary["weather_delayed"] / summary["delayed_rows"] * 100 if summary["delayed_rows"] else 0
        print(
            f"Weather-attributed delays (weather_delay > 0): {summary['weather_delayed']:,} flights "
            f"({weather_share:.2f}% of delayed), total weather minutes: {summary['weather_delay_minutes_total']:.0f}"
        )

    print("\nDelay reasons (flights with positive minutes in the column):")
    if summary["reason_counts"].empty:
        print("  No delays detected.")
    else:
        reason_named = summary["reason_counts"].rename(index=DELAY_REASON_NAMES)
        print(reason_named.to_string())
        print("\nTotal arrival delay minutes attributed to each reason:")
        print(summary["reason_delay_minutes"].rename(index=DELAY_REASON_NAMES).to_string())

    print(f"\nTop {top_n} carriers by delayed flights:")
    carrier_named = summary["carrier_counts"].rename(index=CARRIER_NAMES)
    print(carrier_named.head(top_n).to_string())

    print(f"\nTop {top_n} origins by delayed flights:")
    print(summary["origin_counts"].head(top_n).rename(index=AIRPORT_NAMES).to_string())

    print(f"\nTop {top_n} destinations by delayed flights:")
    print(summary["dest_counts"].head(top_n).rename(index=AIRPORT_NAMES).to_string())

    if not summary["weather_origin_counts"].empty:
        print(f"\nWeather-driven delays: top {top_n} origins (weather_delay > 0):")
        weather_orig_df = pd.DataFrame(
            {
                "weather_delayed": summary["weather_origin_counts"],
                "weather_minutes": summary["weather_origin_minutes"],
                "flights_total": summary["origin_total"],
            }
        ).fillna(0)
        weather_orig_df["weather_rate_pct"] = (
            weather_orig_df["weather_delayed"] / weather_orig_df["flights_total"] * 100
        )
        print(
            weather_orig_df.sort_values("weather_delayed", ascending=False)
            .head(top_n)
            .rename(index=AIRPORT_NAMES)
            .to_string(
                columns=[
                    "flights_total",
                    "weather_delayed",
                    "weather_rate_pct",
                    "weather_minutes",
                ]
            )
        )

    if not summary["weather_dest_counts"].empty:
        print(f"\nWeather-driven delays: top {top_n} destinations (weather_delay > 0):")
        weather_dest_df = pd.DataFrame(
            {
                "weather_delayed": summary["weather_dest_counts"],
                "weather_minutes": summary["weather_dest_minutes"],
                "flights_total": summary["dest_total"],
            }
        ).fillna(0)
        weather_dest_df["weather_rate_pct"] = (
            weather_dest_df["weather_delayed"] / weather_dest_df["flights_total"] * 100
        )
        print(
            weather_dest_df.sort_values("weather_delayed", ascending=False)
            .head(top_n)
            .rename(index=AIRPORT_NAMES)
            .to_string(
                columns=[
                    "flights_total",
                    "weather_delayed",
                    "weather_rate_pct",
                    "weather_minutes",
                ]
            )
        )

    # Airport delay stats (rate and total delay minutes) with a minimum flight threshold
    origin_stats = pd.DataFrame(
        {
            "flights": summary["origin_total"],
            "delayed": summary["origin_counts"],
            "delay_minutes": summary["origin_delay_minutes"],
        }
    )
    origin_stats["delay_rate_pct"] = (
        origin_stats["delayed"] / origin_stats["flights"] * 100
    )
    origin_stats = origin_stats[
        origin_stats["flights"] >= min_airport_flights
    ].sort_values("delay_rate_pct", ascending=False)

    dest_stats = pd.DataFrame(
        {
            "flights": summary["dest_total"],
            "delayed": summary["dest_counts"],
            "delay_minutes": summary["dest_delay_minutes"],
        }
    )
    dest_stats["delay_rate_pct"] = (
        dest_stats["delayed"] / dest_stats["flights"] * 100
    )
    dest_stats = dest_stats[
        dest_stats["flights"] >= min_airport_flights
    ].sort_values("delay_rate_pct", ascending=False)

    print(
        f"\nTop {top_n} origin airports by delay rate (min {min_airport_flights} flights):"
    )
    print(
        origin_stats.head(top_n)[
            ["flights", "delayed", "delay_rate_pct", "delay_minutes"]
        ]
        .rename(index=AIRPORT_NAMES)
        .to_string()
    )

    print(
        f"\nTop {top_n} destination airports by delay rate (min {min_airport_flights} flights):"
    )
    print(
        dest_stats.head(top_n)[
            ["flights", "delayed", "delay_rate_pct", "delay_minutes"]
        ]
        .rename(index=AIRPORT_NAMES)
        .to_string()
    )

    print(f"\nTop {top_n} routes by delayed flights:")
    print(summary["route_counts"].head(top_n).to_string())

    print("\nDelay rate by month (%): (1=Jan ... 12=Dec)")
    print(summary["month_rates"].sort_index().to_string())

    dow_names = {
        1: "Mon",
        2: "Tue",
        3: "Wed",
        4: "Thu",
        5: "Fri",
        6: "Sat",
        7: "Sun",
    }
    weekday_rates_named = summary["weekday_rates"].rename(index=dow_names)
    print("\nDelay rate by day_of_week (%):")
    print(weekday_rates_named.to_string())

    print("\nDelay rate by scheduled departure hour (%):")
    print(summary["hour_rates"].to_string())

    if not summary["event_total"].empty:
        print("\nDelay rate by notable event day (%), including 'none' for non-event days:")
        event_df = pd.DataFrame(
            {
                "flights": summary["event_total"],
                "delayed": summary["event_delayed"],
                "delay_rate_pct": summary["event_delayed"] / summary["event_total"] * 100,
                "delay_minutes": summary["event_delay_minutes"],
            }
        ).fillna(0)
        # Show a longer list of events; keep 'none' at top if present, then sort by delayed flights desc
        if "none" in event_df.index:
            event_df_sorted = pd.concat(
                [
                    event_df.loc[["none"]],
                    event_df.drop(index="none").sort_values("delayed", ascending=False),
                ]
            )
        else:
            event_df_sorted = event_df.sort_values("delayed", ascending=False)
        print(event_df_sorted.head(event_top).to_string())

    # Average delay minutes for top origins/destinations by delayed flights
    origin_avg_df = pd.DataFrame(
        {
            "flights": summary["origin_total"],
            "delayed": summary["origin_counts"],
            "delay_minutes": summary["origin_delay_minutes"],
        }
    ).fillna(0)
    origin_avg_df["delay_rate_pct"] = origin_avg_df["delayed"] / origin_avg_df["flights"] * 100
    origin_avg_df["avg_delay_minutes_delayed"] = origin_avg_df["delay_minutes"] / origin_avg_df["delayed"].replace(0, pd.NA)
    origin_avg_df = origin_avg_df.sort_values("delayed", ascending=False)
    print(f"\nAverage delay minutes for top {top_n} origin airports (by delayed flights):")
    print(
        origin_avg_df.head(top_n)[
            ["flights", "delayed", "delay_rate_pct", "avg_delay_minutes_delayed"]
        ].rename(index=AIRPORT_NAMES).to_string()
    )

    dest_avg_df = pd.DataFrame(
        {
            "flights": summary["dest_total"],
            "delayed": summary["dest_counts"],
            "delay_minutes": summary["dest_delay_minutes"],
        }
    ).fillna(0)
    dest_avg_df["delay_rate_pct"] = dest_avg_df["delayed"] / dest_avg_df["flights"] * 100
    dest_avg_df["avg_delay_minutes_delayed"] = dest_avg_df["delay_minutes"] / dest_avg_df["delayed"].replace(0, pd.NA)
    dest_avg_df = dest_avg_df.sort_values("delayed", ascending=False)
    print(f"\nAverage delay minutes for top {top_n} destination airports (by delayed flights):")
    print(
        dest_avg_df.head(top_n)[
            ["flights", "delayed", "delay_rate_pct", "avg_delay_minutes_delayed"]
        ].rename(index=AIRPORT_NAMES).to_string()
    )

    # Top routes with delayed counts and rate
    route_df = pd.DataFrame(
        {
            "delayed": summary["route_counts"],
            "flights": summary["route_total"],
            "delay_minutes": summary["route_delay_minutes"],
        }
    ).fillna(0)
    route_df["delay_rate_pct"] = route_df["delayed"] / route_df["flights"] * 100
    route_df["avg_delay_minutes_delayed"] = route_df["delay_minutes"] / route_df["delayed"].replace(0, pd.NA)
    route_df = route_df.sort_values("delayed", ascending=False)
    print(f"\nTop {top_n} routes by delayed flights:")
    print(
        route_df.head(top_n)[
            ["flights", "delayed", "delay_rate_pct", "avg_delay_minutes_delayed"]
        ].to_string()
    )

    # Top 5 most delayed airports: show delay rate by hour
    top5_origins = origin_avg_df.head(5).index.tolist()
    if top5_origins:
        rate_by_hour_rows = []
        for origin in top5_origins:
            for hour in range(0, 24):
                flights = summary["origin_hour_total"].get((origin, hour), 0)
                delayed = summary["origin_hour_delayed"].get((origin, hour), 0)
                rate = (delayed / flights * 100) if flights else None
                rate_by_hour_rows.append(
                    {
                        "origin": origin,
                        "dep_hour": hour,
                        "flights": flights,
                        "delayed": delayed,
                        "delay_rate_pct": rate,
                    }
                )
        rate_by_hour_df = pd.DataFrame(rate_by_hour_rows)
        rate_by_hour_df["origin_name"] = rate_by_hour_df["origin"].map(AIRPORT_NAMES).fillna(rate_by_hour_df["origin"])
        pivot_hour = (
            rate_by_hour_df.pivot_table(
                index="origin_name",
                columns="dep_hour",
                values="delay_rate_pct",
                aggfunc="first",
            )
            .reindex(columns=range(0, 24))
            .sort_index(axis=1)
        )
        print("\nTop 5 most delayed origins: delay rate by hour (%) [dep_hour 0-23]:")
        print(pivot_hour.round(1).to_string(na_rep=""))

    # Comparison: top 5 largest airports by traffic volume (origins)
    largest_origins = summary["origin_total"].sort_values(ascending=False).head(5)
    overall_rate = summary["delay_rate_pct"]
    largest_df = pd.DataFrame(
        {
            "flights": largest_origins,
            "delayed": summary["origin_counts"].reindex(largest_origins.index).fillna(0),
        }
    )
    largest_df["delay_rate_pct"] = largest_df["delayed"] / largest_df["flights"] * 100
    largest_df["delta_vs_overall"] = largest_df["delay_rate_pct"] - overall_rate
    print("\nTop 5 largest origin airports by traffic volume and their delay rates:")
    print(
        largest_df.rename(index=AIRPORT_NAMES)[
            ["flights", "delayed", "delay_rate_pct", "delta_vs_overall"]
        ].to_string()
    )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean flight data and summarize delay reasons."
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use the smaller sample file for a quick run.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Only read the first N rows (for smoke testing).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size for streaming the CSV. Lower this if memory is tight.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many carriers/origins/destinations to show in the summary.",
    )
    parser.add_argument(
        "--min-airport-flights",
        type=int,
        default=DEFAULT_MIN_AIRPORT_FLIGHTS,
        help="Minimum flights for an airport to appear in rate summaries.",
    )
    parser.add_argument(
        "--event-top",
        type=int,
        default=20,
        help="How many event rows to display in the event delay table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events_lookup = load_event_lookup(EVENTS_PATH)
    chunks = read_in_chunks(
        use_sample=args.sample,
        nrows=args.nrows,
        chunk_size=args.chunk_size,
    )
    summary = summarize_delays(chunks, events_lookup=events_lookup)
    print_summary(
        summary,
        top_n=args.top,
        min_airport_flights=args.min_airport_flights,
        event_top=args.event_top,
    )


if __name__ == "__main__":
    main()
