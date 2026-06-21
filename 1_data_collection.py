"""
1_data_collection.py
--------------------
Downloads DAX stock data legally from Yahoo Finance.
Source: Yahoo Finance (yfinance) - free and publicly available.
"""
 
import yfinance as yf
import pandas as pd
import os
 
# Set working directory to the location of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))
 
# -------------------------------------------------------------------
# DAX companies to analyze (Yahoo Finance ticker symbols)
# -------------------------------------------------------------------
TICKERS = {
    "SAP":            "SAP.DE",
    "Siemens":        "SIE.DE",
    "Allianz":        "ALV.DE",
    "Deutsche Bank":  "DBK.DE",
    "BMW":            "BMW.DE",
    "Volkswagen":     "VOW3.DE",
    "BASF":           "BAS.DE",
    "Bayer":          "BAYN.DE",
}
 
START_DATE = "2020-01-01"   # Last ~5 years
END_DATE   = "2025-01-01"
 
 
def download_data(tickers: dict, start: str, end: str) -> dict:
    """Download daily closing prices for each stock and save as CSV."""
    os.makedirs("data", exist_ok=True)
    all_close = {}
 
    for name, ticker in tickers.items():
        print(f"  Downloading: {name} ({ticker}) ...")
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
 
        if df.empty:
            print(f"  [WARNING] No data retrieved for {name}, skipping.")
            continue
 
        # --- FIX ---
        # Newer yfinance versions return columns as a MultiIndex,
        # e.g. ('Close', 'SAP.DE') instead of just 'Close'.
        # Flatten them so we can reliably grab the 'Close' column.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
 
        # Keep only the closing price (as a Series)
        close = df["Close"]
 
        # Save individual stock CSV
        csv_path = f"data/{name.replace(' ', '_')}.csv"
        close.to_frame(name=name).to_csv(csv_path)
        print(f"  Saved: {csv_path}  ({len(close)} rows)")
 
        all_close[name] = close
 
    # Save all stocks into a single combined CSV
    combined = pd.DataFrame(all_close)
    combined.to_csv("data/all_prices.csv")
    print(f"\nAll closing prices saved: data/all_prices.csv ({combined.shape[0]} days, {combined.shape[1]} stocks)")
    return combined
 
 
if __name__ == "__main__":
    print("=" * 55)
    print("  Data Download Started")
    print(f"  Source : Yahoo Finance (yfinance) - open data")
    print(f"  Period : {START_DATE}  ->  {END_DATE}")
    print("=" * 55)
    prices = download_data(TICKERS, START_DATE, END_DATE)
    print("\nData download complete.")
