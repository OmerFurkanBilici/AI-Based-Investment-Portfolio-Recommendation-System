"""
main.py
-------
Runs all modules in sequence:
  1. Data download
  2. Financial analysis
  3. K-Means clustering
  4. Random Forest prediction

To run:
  python main.py
"""

import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

STEPS = [
    ("1_data_collection.py",  "Data Download (Yahoo Finance)"),
    ("2_analysis.py",         "Financial Analysis"),
    ("3_kmeans_clustering.py","K-Means Clustering"),
    ("4_random_forest.py",    "Random Forest Prediction"),
]


def run_step(script: str, title: str):
    print("\n" + "=" * 60)
    print(f"  STEP: {title}")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed. Stopping.")
        sys.exit(1)


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("#   AI-Based Investment Portfolio Recommendation System   #")
    print("#" * 60)

    for script, title in STEPS:
        run_step(script, title)

    print("\n" + "#" * 60)
    print("#  All steps completed successfully!")
    print("#  Outputs saved in: ./data/")
    print("#" * 60 + "\n")
