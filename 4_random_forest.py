"""
4_random_forest.py
------------------
Predicts monthly price direction for each stock using Random Forest:
  1  → Price will go UP next month
  0  → Price will go DOWN next month

Features used:
  - 1, 3, 6-month moving average ratios
  - 1-month rolling volatility
  - RSI (14-day Relative Strength Index)
  - Previous monthly return

Output: console report + data/rf_results.csv + data/rf_feature_importance.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(price: pd.Series, name: str) -> pd.DataFrame:
    df = pd.DataFrame({"Close": price})

    # Moving average ratios (price / MA)
    df["MA_21"]  = df["Close"].rolling(21).mean()
    df["MA_63"]  = df["Close"].rolling(63).mean()
    df["MA_126"] = df["Close"].rolling(126).mean()
    df["Ratio_21"]  = df["Close"] / df["MA_21"]
    df["Ratio_63"]  = df["Close"] / df["MA_63"]
    df["Ratio_126"] = df["Close"] / df["MA_126"]

    # 21-day rolling volatility
    df["Volatility_21"] = df["Close"].pct_change().rolling(21).std()

    # RSI
    df["RSI_14"] = rsi(df["Close"], 14)

    # Previous monthly return (~21 trading days)
    df["Prev_Return"] = df["Close"].pct_change(21)

    # Target: next month direction (1=UP, 0=DOWN)
    df["Target"] = (df["Close"].shift(-21) > df["Close"]).astype(int)

    df.dropna(inplace=True)
    return df


FEATURE_COLS = [
    "Ratio_21", "Ratio_63", "Ratio_126",
    "Volatility_21", "RSI_14", "Prev_Return",
]


def train_and_evaluate(prices: pd.DataFrame):
    results = {}

    for stock in prices.columns:
        price = prices[stock].dropna()
        if len(price) < 300:          # skip if not enough data
            continue

        df = build_features(price, stock)
        X  = df[FEATURE_COLS]
        y  = df["Target"]

        # Time-series split: first 80% train, last 20% test
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42,
            class_weight="balanced",
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        acc       = accuracy_score(y_test, y_pred)
        last_pred = rf.predict(X.iloc[[-1]])[0]
        last_prob = rf.predict_proba(X.iloc[[-1]])[0]

        results[stock] = {
            "Accuracy (%)":   round(acc * 100, 1),
            "Next Month":     "UP ↑" if last_pred == 1 else "DOWN ↓",
            "Confidence (%)": round(max(last_prob) * 100, 1),
            "_model":         rf,
            "_feature_names": FEATURE_COLS,
        }

        print(f"\n  [{stock}]  Accuracy: {acc*100:.1f}%  |  Prediction: {'UP ↑' if last_pred==1 else 'DOWN ↓'}  (confidence: {max(last_prob)*100:.1f}%)")
        print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"], zero_division=0))

    return results


def plot_feature_importance(results: dict):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (stock, info) in zip(axes, results.items()):
        importances = info["_model"].feature_importances_
        feat_names  = info["_feature_names"]
        sorted_idx  = np.argsort(importances)

        ax.barh(
            [feat_names[i] for i in sorted_idx],
            importances[sorted_idx],
            color="#3498db",
        )
        ax.set_title(f"{stock}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Importance")

    plt.suptitle("Random Forest - Feature Importance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("data/rf_feature_importance.png", dpi=150)
    plt.show()
    print("  Chart saved: data/rf_feature_importance.png")


if __name__ == "__main__":
    print("=" * 55)
    print("  Random Forest - Monthly Direction Prediction")
    print("=" * 55)

    prices = pd.read_csv("data/all_prices.csv", index_col=0, parse_dates=True)
    prices.columns = prices.columns.str.strip()

    results = train_and_evaluate(prices)

    print("\n--- Prediction Summary ---")
    summary = pd.DataFrame(
        {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()}
    ).T
    print(summary.to_string())
    summary.to_csv("data/rf_results.csv")
    print("\nResults saved: data/rf_results.csv")

    print("\nGenerating feature importance chart...")
    plot_feature_importance(results)

    print("\nRandom Forest complete.")
