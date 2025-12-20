#!/usr/bin/env python3
"""
Simple runner to orchestrate analysis, plots, and models for the 2024 flight delay project.

Prompts the user for an action and executes the corresponding script/command.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def prompt_choice() -> str:
    print("What would you like to run?")
    print("1) Analysis (console tables)")
    print("2) Plot generation (PNG files)")
    print("3) Route/event delay classifier")
    print("4) Baseline delay model")
    print("5) Modeling pipeline (LightGBM, diagnostics) - heavy")
    choice = input("Enter number (1-5): ").strip()
    return choice


def run_cmd(cmd: list[str]) -> None:
    print(f"\nRunning: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def main() -> None:
    choice = prompt_choice()
    if choice == "1":
        # analysis_script with default params
        run_cmd(["python3", str(ROOT / "analysis_script")])
    elif choice == "2":
        # plot_generation with defaults
        run_cmd(["python3", str(ROOT / "plot_generation.py")])
    elif choice == "3":
        # route_event_prediction with defaults
        run_cmd(["python3", str(ROOT / "route_event_prediction.py")])
    elif choice == "4":
        # baseline_model with defaults
        run_cmd(["python3", str(ROOT / "baseline_model.py")])
    elif choice == "5":
        # modeling_pipeline (heavy); adjust args as needed
        run_cmd(["python3", str(ROOT / "modeling_pipeline.py")])
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
