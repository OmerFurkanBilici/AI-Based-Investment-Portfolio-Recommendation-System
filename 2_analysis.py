"""
2_analysis.py
-------------
Calculates core financial metrics for each stock:
  - Annual Average Return
  - Annual Volatility
  - Sharpe Ratio
  - Max Drawdown (largest peak-to-trough decline)
  - Total Return (%)

Results are printed to the console and saved to data/metrics.csv.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

RISK_FREE_RATE = 0.02   # Annual risk-free rate (2% - based on German government bond)
TRADING_DAYS   = 252    # Approximate trading days per year


def load_prices(path: str = "data/all_prices.csv") -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # Strip any whitespace from column headers
    df.columns = df.columns.str.strip()
    return df.dropna(how="all")


def compute_metrics(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate financial metrics for each stock."""
    daily_returns = prices.pct_change().dropna()
    metrics = {}

    for stock in prices.columns:
        r = daily_returns[stock].dropna()
        p = prices[stock].dropna()

        # Annualized return and volatility
        ann_return = r.mean() * TRADING_DAYS
        ann_vol    = r.std()  * np.sqrt(TRADING_DAYS)

        # Sharpe Ratio
        sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol != 0 else 0

        # Max Drawdown
        cumulative   = (1 + r).cumprod()
        rolling_max  = cumulative.cummax()
        drawdown     = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Total return over the full period
        total_return = (p.iloc[-1] / p.iloc[0] - 1) * 100

        metrics[stock] = {
            "Ann. Return (%)":     round(ann_return   * 100, 2),
            "Ann. Volatility (%)": round(ann_vol      * 100, 2),
            "Sharpe Ratio":        round(sharpe,            2),
            "Max Drawdown (%)":    round(max_drawdown * 100, 2),
            "Total Return (%)":    round(total_return,       2),
        }

    return pd.DataFrame(metrics).T


def plot_prices(prices: pd.DataFrame):
    """Normalized price chart (base = 100 at start date)."""
    normalized = prices / prices.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in normalized.columns:
        ax.plot(normalized.index, normalized[col], label=col, linewidth=1.5)
    ax.set_title("DAX Stocks - Normalized Price Performance (Base = 100)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/price_performance.png", dpi=150)
    plt.show()
    print("  Chart saved: data/price_performance.png")


def plot_metrics(metrics: pd.DataFrame):
    """Annual return and volatility comparison chart."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.Set2.colors

    # Return
    axes[0].barh(metrics.index, metrics["Ann. Return (%)"], color=colors[:len(metrics)])
    axes[0].set_title("Annual Average Return (%)")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("%")

    # Volatility
    axes[1].barh(metrics.index, metrics["Ann. Volatility (%)"], color=colors[:len(metrics)])
    axes[1].set_title("Annual Volatility (%)")
    axes[1].set_xlabel("%")

    plt.suptitle("Financial Metrics Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("data/metrics_comparison.png", dpi=150)
    plt.show()
    print("  Chart saved: data/metrics_comparison.png")


if __name__ == "__main__":
    print("=" * 55)
    print("  Financial Analysis")
    print("=" * 55)

    prices  = load_prices()
    metrics = compute_metrics(prices)

    print("\n--- Metrics Table ---")
    print(metrics.to_string())

    metrics.to_csv("data/metrics.csv")
    print("\nMetrics saved: data/metrics.csv")

    print("\nGenerating charts...")
    plot_prices(prices)
    plot_metrics(metrics)

    print("\nAnalysis complete.")
