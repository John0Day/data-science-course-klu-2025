#!/usr/bin/env python3
"""
Simple runner to orchestrate analysis, plots, and models for the 2024 flight delay project.

Prompts the user for an action and executes the corresponding script/command.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent

def prompt_choice() -> str:
    print("What would you like to run?")
    print("1) Analysis (console tables)")
    print("2) Plot generation (PNG files)")
    print("3) Modeling pipeline (LightGBM, diagnostics) - heavy")
    print("0) Exit")
    choice = input("Enter number (0-3): ").strip()
    return choice


def run_cmd(cmd: list[str]) -> None:
    print(f"\nRunning: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}.")


def main() -> None:
    install_cmd = "python3 -m pip install pandas scikit-learn numpy seaborn matplotlib lightgbm"
    print("Before using the menu, install the required dependencies:")
    print(f"\033[36m{install_cmd}\033[0m\n")
    while True:
        choice = prompt_choice()
        if choice == "1":
            # analysis_script with default params
            run_cmd(["python3", str(ROOT / "analysis_script.py")])
        elif choice == "2":
            # plot_generation with defaults
            run_cmd(["python3", str(ROOT / "plot_generation.py")])
        elif choice == "3":
            # modeling_pipeline (heavy); adjust args as needed
            run_cmd(["python3", str(ROOT / "modeling_pipeline.py")])
        elif choice == "0":
            print("Exiting runner.")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
