from pathlib import Path

import pytest


DATA_DIR = Path("rawdata")
FULL_FILE = DATA_DIR / "flight_data_2024.csv"
SAMPLE_FILE = DATA_DIR / "flight_data_2024_sample.csv"
EVENTS_FILE = DATA_DIR / "combined_schedules_with_airports.csv"


def test_required_data_directory_exists() -> None:
    assert DATA_DIR.exists(), "Expected a 'rawdata/' directory next to the repo root."


def test_flight_data_file_present() -> None:
    """
    Ensure the Kaggle flight dataset is placed in rawdata/.
    Accept either the full file or the bundled sample.
    """
    if FULL_FILE.exists() or SAMPLE_FILE.exists():
        assert True
    else:  # pragma: no cover - only triggered when user forgot to download
        pytest.fail(
            "Missing flight data. Download the 2024 dataset from Kaggle and place it at "
            "'rawdata/flight_data_2024.csv' (or use the sample as 'rawdata/flight_data_2024_sample.csv')."
        )


def test_events_file_present() -> None:
    """
    Verify the events mapping CSV is available where scripts expect it.
    """
    if not EVENTS_FILE.exists():  # pragma: no cover - warns during setup
        pytest.fail(
            "Missing events CSV. Place 'combined_schedules_with_airports.csv' in 'rawdata/'. "
            "See README for the expected file."
        )
