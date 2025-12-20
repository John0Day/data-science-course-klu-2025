import csv
from datetime import date
from pathlib import Path

from event_utils import load_event_lookup


def test_load_event_lookup_applies_airport_fixes_and_date_label(tmp_path: Path) -> None:
    """
    Ensure Primary Airport codes are fixed (PBG->YUL, IAG->YYZ) and dates are parsed.
    """
    path = tmp_path / "combined_schedules_with_airports.csv"
    rows = [
        [
            "NFL",
            "Team A",
            "Stadium A",
            "40.0",
            "-70.0",
            "PBG",  # should map to YUL
            "40.0",
            "-70.0",
            "1.0",
            "Sun,2024-02-11",
            "Team A",
            "Team B",
        ],
        [
            "NHL",
            "Team C",
            "Stadium C",
            "41.0",
            "-71.0",
            "IAG",  # should map to YYZ
            "41.0",
            "-71.0",
            "2.0",
            "Mon,2024-03-04",
            "Team C",
            "Team D",
        ],
    ]

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "League",
                "Team Name",
                "Stadium Name",
                "Stadium Lat",
                "Stadium Lon",
                "Primary Airport",
                "Airport Lat",
                "Airport Lon",
                "Primary Airport Distance",
                "Date",
                "Home Team",
                "Away Team",
            ]
        )
        writer.writerows(rows)

    lookup = load_event_lookup(path)

    assert lookup[(date(2024, 2, 11), "YUL")].startswith("2024-02-11")
    assert lookup[(date(2024, 3, 4), "YYZ")].startswith("2024-03-04")
