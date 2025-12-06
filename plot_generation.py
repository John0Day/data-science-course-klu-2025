#!/usr/bin/env python3
"""
Generate illustrative plots (PNG) for 2024 flight delay data.

Plots:
- Delay rate by scheduled departure hour (line)
- Delay rate by month (line/bar)
- Weather-driven delays: top origins by weather_delay counts (bar)
- Event-day delays: top events by delayed flights (bar)
- Top 5 delayed origins: delay rate by hour (heatmap)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from event_utils import load_event_lookup

sns.set_theme(style="whitegrid")

DATA_PATH = Path(__file__).resolve().parent / "rawdata" / "flight_data_2024.csv"
SAMPLE_PATH = Path(__file__).resolve().parent / "rawdata" / "flight_data_2024_sample.csv"
EVENTS_PATH = Path(__file__).resolve().parent / "rawdata" / "combined_schedules_with_airports.csv"

ARRIVAL_DELAY_THRESHOLD = 15
DEFAULT_CHUNK_SIZE = 250_000

PLOT_DIR = Path("plots")

# Lightweight airport name map for readability (extend as needed)
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

CARRIER_NAMES = {
    "AA": "American",
    "AS": "Alaska",
    "B6": "JetBlue",
    "DL": "Delta",
    "F9": "Frontier",
    "G4": "Allegiant",
    "HA": "Hawaiian",
    "NK": "Spirit",
    "OO": "SkyWest",
    "UA": "United",
    "WN": "Southwest",
    "YX": "Republic",
    "MQ": "Envoy",
    "9E": "Endeavor",
    "YV": "Mesa",
    "QX": "Horizon",
}


def read_in_chunks(
    use_sample: bool,
    nrows: Optional[int],
    chunk_size: int,
) -> Iterable[pd.DataFrame]:
    path = SAMPLE_PATH if use_sample else DATA_PATH
    usecols = [
        "fl_date",
        "origin",
        "dest",
        "op_unique_carrier",
        "crs_dep_time",
        "arr_delay",
        "cancelled",
        "diverted",
        "month",
        "day_of_week",
        "distance",
        "weather_delay",
    ]
    reader = pd.read_csv(
        path,
        usecols=usecols,
        parse_dates=["fl_date"],
        chunksize=chunk_size,
        nrows=nrows,
        low_memory=False,
    )
    if isinstance(reader, pd.DataFrame):
        yield reader
    else:
        yield from reader


def aggregate(use_sample: bool, nrows: Optional[int], chunk_size: int, events_lookup: dict):
    month_total = pd.Series(dtype="int64")
    month_delayed = pd.Series(dtype="int64")
    hour_total = pd.Series(dtype="int64")
    hour_delayed = pd.Series(dtype="int64")
    weekday_total = pd.Series(dtype="int64")
    weekday_delayed = pd.Series(dtype="int64")

    origin_delayed = pd.Series(dtype="int64")
    empty_origin_hour_index = pd.MultiIndex.from_arrays(
        [[], []], names=["origin", "dep_hour"]
    )
    origin_hour_total = pd.Series(dtype="int64", index=empty_origin_hour_index)
    origin_hour_delayed = pd.Series(dtype="int64", index=empty_origin_hour_index)

    carrier_total = pd.Series(dtype="int64")
    carrier_delayed = pd.Series(dtype="int64")

    weather_origin_counts = pd.Series(dtype="int64")
    weather_origin_minutes = pd.Series(dtype="float64")

    event_total = pd.Series(dtype="int64")
    event_delayed = pd.Series(dtype="int64")

    for chunk in read_in_chunks(use_sample, nrows, chunk_size):
        # basic cleaning
        chunk = chunk[(chunk["cancelled"] == 0) & (chunk["diverted"] == 0)]
        chunk["arr_delay"] = pd.to_numeric(chunk["arr_delay"], errors="coerce")
        chunk = chunk[chunk["arr_delay"].notna()]
        delayed_mask = chunk["arr_delay"] >= ARRIVAL_DELAY_THRESHOLD

        chunk["dep_hour"] = pd.to_numeric(chunk["crs_dep_time"] // 100, errors="coerce").fillna(-1).astype(int)

        # aggregates
        month_total = month_total.add(chunk["month"].value_counts(), fill_value=0)
        month_delayed = month_delayed.add(chunk.loc[delayed_mask, "month"].value_counts(), fill_value=0)

        weekday_total = weekday_total.add(chunk["day_of_week"].value_counts(), fill_value=0)
        weekday_delayed = weekday_delayed.add(chunk.loc[delayed_mask, "day_of_week"].value_counts(), fill_value=0)

        hour_total = hour_total.add(chunk["dep_hour"].value_counts(), fill_value=0)
        hour_delayed = hour_delayed.add(chunk.loc[delayed_mask, "dep_hour"].value_counts(), fill_value=0)

        origin_delayed = origin_delayed.add(
            chunk.loc[delayed_mask].groupby("origin").size(), fill_value=0
        )

        origin_hour_total = origin_hour_total.add(
            chunk.groupby(["origin", "dep_hour"]).size(), fill_value=0
        )
        origin_hour_delayed = origin_hour_delayed.add(
            chunk.loc[delayed_mask].groupby(["origin", "dep_hour"]).size(),
            fill_value=0,
        )

        carrier_total = carrier_total.add(
            chunk["op_unique_carrier"].value_counts(), fill_value=0
        )
        carrier_delayed = carrier_delayed.add(
            chunk.loc[delayed_mask, "op_unique_carrier"].value_counts(), fill_value=0
        )

        # weather
        weather_mask = pd.to_numeric(chunk["weather_delay"], errors="coerce").fillna(0) > 0
        weather_origin_counts = weather_origin_counts.add(
            chunk.loc[weather_mask].groupby("origin").size(), fill_value=0
        )
        weather_origin_minutes = weather_origin_minutes.add(
            chunk.loc[weather_mask].groupby("origin")["weather_delay"].sum(),
            fill_value=0,
        )

        # events
        if events_lookup:
            chunk["event_name"] = "none"
            origin_keys = pd.Series(
                list(zip(chunk["fl_date"].dt.date, chunk["origin"])), index=chunk.index
            )
            chunk["event_name"] = origin_keys.map(events_lookup).fillna("none")
            no_event_mask = chunk["event_name"] == "none"
            if no_event_mask.any():
                dest_keys = pd.Series(
                    list(zip(chunk.loc[no_event_mask, "fl_date"].dt.date, chunk.loc[no_event_mask, "dest"])),
                    index=chunk.index[no_event_mask],
                )
                chunk.loc[no_event_mask, "event_name"] = dest_keys.map(events_lookup).fillna("none")

            event_total = event_total.add(chunk["event_name"].value_counts(), fill_value=0)
            event_delayed = event_delayed.add(
                chunk.loc[delayed_mask, "event_name"].value_counts(), fill_value=0
            )

    return {
        "month_rates": (month_delayed / month_total * 100).sort_index(),
        "month_total": month_total.sort_index(),
        "month_delayed": month_delayed.sort_index(),
        "hour_rates": (hour_delayed / hour_total * 100).sort_index(),
        "weekday_rates": (weekday_delayed / weekday_total * 100).sort_index(),
        "origin_delayed": origin_delayed.sort_values(ascending=False),
        "origin_hour_total": origin_hour_total,
        "origin_hour_delayed": origin_hour_delayed,
        "carrier_total": carrier_total.sort_values(ascending=False),
        "carrier_delayed": carrier_delayed.sort_values(ascending=False),
        "weather_origin_counts": weather_origin_counts.sort_values(ascending=False),
        "weather_origin_minutes": weather_origin_minutes.sort_values(ascending=False),
        "event_total": event_total.sort_values(ascending=False),
        "event_delayed": event_delayed.sort_values(ascending=False),
    }


def plot_delay_by_hour(hour_rates: pd.Series) -> None:
    plt.figure()
    sns.lineplot(x=hour_rates.index, y=hour_rates.values, marker="o")
    plt.xlabel("Scheduled departure hour")
    plt.ylabel("Delay rate (%)")
    plt.title("Delay rate by hour")
    plt.tight_layout()
    (PLOT_DIR / "delay_rate_by_hour.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_DIR / "delay_rate_by_hour.png", dpi=200)
    plt.close()


def plot_delay_by_month(month_rates: pd.Series, month_total: pd.Series, month_delayed: pd.Series) -> None:
    plt.figure()
    df = pd.DataFrame(
        {
            "month": month_rates.index,
            "rate": month_rates.values,
            "flights": month_total.reindex(month_rates.index),
            "delayed": month_delayed.reindex(month_rates.index),
        }
    )
    ax = sns.barplot(x="month", y="rate", data=df, color="#4c72b0")
    # annotate only min/max bars for readability; place separately
    if not df.empty:
        max_idx = df["rate"].idxmax()
        min_idx = df["rate"].idxmin()
        for bar, (idx, row) in zip(ax.patches, df.iterrows()):
            if idx not in (max_idx, min_idx):
                continue
            x = bar.get_x() + bar.get_width()
            y = bar.get_height()
            if idx == max_idx:
                dx, dy = 0.5, 1.0  # farther right and higher to avoid overlap (e.g., month 10)
                ha, va = "left", "center"
            else:  # min
                dx, dy = 0.0, 2.5  # higher above the bar to avoid touching neighbor bars
                ha, va = "center", "bottom"
                x = bar.get_x() + bar.get_width() / 2
            ax.annotate(
                f"{int(row['flights']):,}\n{int(row['delayed']):,} delayed",
                xy=(bar.get_x() + bar.get_width() / 2, y),
                xytext=(x + dx, y + dy),
                textcoords="data",
                ha=ha,
                va=va,
                fontsize=8,
                fontweight="normal",
                color="#222",
                arrowprops=dict(arrowstyle="->", color="#444", lw=0.8),
            )
    plt.xlabel("Month (1-12)")
    plt.ylabel("Delay rate (%)")
    plt.title("Delay rate by month")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "delay_rate_by_month.png", dpi=200)
    plt.close()


def plot_weather_origins(weather_counts: pd.Series, weather_minutes: pd.Series, top_n: int) -> None:
    if weather_counts.empty:
        return
    df = pd.DataFrame({"weather_delayed": weather_counts, "weather_minutes": weather_minutes}).fillna(0)
    df = df.sort_values("weather_delayed", ascending=False).head(top_n)
    df = df.rename(index=AIRPORT_NAMES)
    plt.figure(figsize=(8, 4))
    sns.barplot(y=df.index, x=df["weather_delayed"], orient="h", color="#dd8452")
    plt.xlabel("Weather-driven delayed flights")
    plt.ylabel("Origin")
    plt.title(f"Top {top_n} origins by weather delays")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "weather_top_origins.png", dpi=200)
    plt.close()


def plot_events(event_total: pd.Series, event_delayed: pd.Series, top_n: int) -> None:
    if event_total.empty:
        return
    df = pd.DataFrame({"flights": event_total, "delayed": event_delayed}).fillna(0)
    if "none" in df.index:
        df = df.drop(index="none")
    df = df[df["flights"] > 0]
    df["delay_rate_pct"] = df["delayed"] / df["flights"] * 100
    # Keep high-traffic events first (by delayed count), but show delay rate as the metric.
    df = df.sort_values("delayed", ascending=False).head(top_n)
    height = max(5, 0.4 * len(df))
    plt.figure(figsize=(9, height))
    sns.barplot(y=df.index, x=df["delay_rate_pct"], orient="h", color="#55a868")
    plt.xlabel("Delay rate (%)")
    plt.ylabel("Event name")
    plt.title(f"Top {top_n} events by delay rate")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=9)
    plt.subplots_adjust(left=0.42, right=0.98, top=0.9, bottom=0.15)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "event_delays.png", dpi=200)
    plt.close()


def plot_top5_origin_hour_heatmap(origin_delayed: pd.Series, origin_hour_total: pd.Series, origin_hour_delayed: pd.Series) -> None:
    if origin_delayed.empty:
        return
    top5 = origin_delayed.head(5).index.tolist()
    rows = []
    for origin in top5:
        for hour in range(24):
            flights = origin_hour_total.get((origin, hour), 0)
            delayed = origin_hour_delayed.get((origin, hour), 0)
            rate = (delayed / flights * 100) if flights else None
            rows.append({"origin": origin, "dep_hour": hour, "delay_rate": rate})
    df = pd.DataFrame(rows)
    df["origin_name"] = df["origin"].map(AIRPORT_NAMES).fillna(df["origin"])
    pivot = df.pivot(index="origin_name", columns="dep_hour", values="delay_rate").reindex(columns=range(24))
    plt.figure(figsize=(10, 4))
    sns.heatmap(pivot, cmap="YlOrRd", cbar_kws={"label": "Delay rate (%)"}, linewidths=0.2, linecolor="white")
    plt.xlabel("Dep hour")
    plt.ylabel("Origin")
    plt.title("Top 5 delayed origins: delay rate by hour")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "top5_origins_by_hour_heatmap.png", dpi=200)
    plt.close()


def plot_carriers(carrier_total: pd.Series, carrier_delayed: pd.Series, top_n: int) -> None:
    if carrier_total.empty:
        return
    df = pd.DataFrame({"flights": carrier_total, "delayed": carrier_delayed}).fillna(0)
    df = df[df["flights"] > 0]
    df["delay_rate_pct"] = df["delayed"] / df["flights"] * 100
    df = df.sort_values("delay_rate_pct", ascending=False)
    df = df.rename(index=CARRIER_NAMES)

    worst = df.head(top_n)
    total_flights = int(carrier_total.sum())
    total_delayed = int(carrier_delayed.sum())
    height = max(5, 0.4 * len(worst))
    plt.figure(figsize=(9, height))
    sns.barplot(y=worst.index, x=worst["delay_rate_pct"], orient="h", color="#c44e52")
    plt.xlabel("Delay rate (%)")
    plt.ylabel("Carrier")
    plt.title(f"Worst {top_n} carriers by delay rate")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=9)
    plt.figtext(
        0.5,
        0.92,
        f"Total flights analyzed: {total_flights:,} | Total delayed: {total_delayed:,}",
        ha="center",
        fontsize=9,
    )
    plt.subplots_adjust(left=0.35, right=0.98, top=0.88, bottom=0.15)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "carrier_delay_rates.png", dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots for flight delay analysis.")
    parser.add_argument("--sample", action="store_true", help="Use sample file instead of full dataset.")
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows for a quick run.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size for streaming CSV.")
    parser.add_argument("--top", type=int, default=10, help="Top N for bars.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events_lookup = load_event_lookup(EVENTS_PATH)
    summary = aggregate(
        use_sample=args.sample,
        nrows=args.nrows,
        chunk_size=args.chunk_size,
        events_lookup=events_lookup,
    )
    PLOT_DIR.mkdir(exist_ok=True)

    plot_delay_by_hour(summary["hour_rates"])
    plot_delay_by_month(summary["month_rates"], summary["month_total"], summary["month_delayed"])
    plot_weather_origins(summary["weather_origin_counts"], summary["weather_origin_minutes"], top_n=args.top)
    plot_events(summary["event_total"], summary["event_delayed"], top_n=args.top)
    plot_top5_origin_hour_heatmap(
        summary["origin_delayed"],
        summary["origin_hour_total"],
        summary["origin_hour_delayed"],
    )
    plot_carriers(summary["carrier_total"], summary["carrier_delayed"], top_n=args.top)
    print(f"Plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
