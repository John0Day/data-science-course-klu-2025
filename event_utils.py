#!/usr/bin/env python3
"""
Utility helpers for loading notable 2024 events with primary airports.
Based on rawdata/combined_schedules_with_airports.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# Manual overrides for questionable airport assignments in the schedules file.
AIRPORT_FIXES = {
    "PBG": "YUL",  # Montreal should map to YUL instead of Plattsburgh
    "IAG": "YYZ",  # Toronto should map to YYZ instead of Niagara Falls
}

def load_event_lookup(path: Path) -> Dict[Tuple, str]:
    """
    Load combined schedules CSV and return a mapping:
    (date, primary_airport) -> event label.
    If the file is missing, return an empty dict.
    """
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if "Date" not in df.columns or "Primary Airport" not in df.columns:
        return {}

    # Normalize date like "Sat,2024-01-06" -> date
    df["date"] = (
        pd.to_datetime(df["Date"].astype(str).str.split(",", n=1).str[-1], errors="coerce")
        .dt.date
    )
    df = df[df["date"].notna()]

    df["airport"] = (
        df["Primary Airport"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace(AIRPORT_FIXES)
    )

    def _event_label(row: pd.Series) -> str:
        league = row.get("League", "")
        home = row.get("Team Name", "")
        away = row.get("Away Team", "")
        stadium = row.get("Stadium Name", "")
        airport = row.get("airport", "")
        return f"{league}: {away} @ {home} ({stadium} / {airport})"

    df["event_name"] = df.apply(_event_label, axis=1)

    return dict(zip(zip(df["date"], df["airport"]), df["event_name"]))
