#!/usr/bin/env python3
"""
Utility helpers for loading notable event days (holidays, major events) in 2024.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def load_events_map(path: Path) -> Dict:
    """
    Load notable events CSV and return mapping date -> event name.
    If the file is missing, return an empty dict.
    """
    if not path.exists():
        return {}
    events_df = pd.read_csv(path, parse_dates=["date"])
    events_df["date"] = events_df["date"].dt.date
    return dict(zip(events_df["date"], events_df["name"]))
