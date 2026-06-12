# AI-Based Investment Portfolio Recommendation System

## Project Summary
A system that retrieves DAX stock data from Yahoo Finance and analyzes it using artificial intelligence algorithms to provide investment recommendations.

## Technologies Used
- **Data Source:** Yahoo Finance (`yfinance`) — free, legal, open data
- **AI Algorithms:**
  - K-Means Clustering (risk grouping)
  - Random Forest Classifier (price direction prediction)

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
# Run all steps in sequence:
python main.py

# Or run step by step:
python 1_data_collection.py
python 2_analysis.py
python 3_kmeans_clustering.py
python 4_random_forest.py
```

## File Structure

```
investment_portfolio/
│
├── main.py                    ← Runs all steps
├── 1_data_collection.py       ← Data download
├── 2_analysis.py              ← Financial metrics
├── 3_kmeans_clustering.py     ← Risk clustering (AI)
├── 4_random_forest.py         ← Direction prediction (AI)
├── requirements.txt
│
└── data/
    ├── all_prices.csv             ← All stock prices
    ├── *.csv                      ← Individual stock CSVs
    ├── metrics.csv                ← Financial metrics
    ├── clustered.csv              ← Risk groups
    ├── rf_results.csv             ← Random Forest predictions
    ├── price_performance.png
    ├── metrics_comparison.png
    ├── clusters_plot.png
    └── rf_feature_importance.png
```

## Stocks Analyzed
| Company        | Ticker  |
|----------------|---------|
| SAP            | SAP.DE  |
| Siemens        | SIE.DE  |
| Allianz        | ALV.DE  |
| Deutsche Bank  | DBK.DE  |
| BMW            | BMW.DE  |
| Volkswagen     | VOW3.DE |
| BASF           | BAS.DE  |
| Bayer          | BAYN.DE |

## Calculated Metrics
- **Annual Average Return (%)**
- **Annual Volatility (%)**
- **Sharpe Ratio** (risk-free rate = 2%)
- **Max Drawdown** (largest peak-to-trough decline)
- **Total Return (%)**

## AI Modules

### K-Means Clustering
Groups stocks into 3 clusters based on volatility and return:
- 🟢 **Low Risk**
- 🟡 **Medium Risk**
- 🔴 **High Risk**

### Random Forest Classifier
Uses technical indicators (RSI, moving average ratios, volatility) to predict the price direction for the next month:
- ↑ **UP**
- ↓ **DOWN**
