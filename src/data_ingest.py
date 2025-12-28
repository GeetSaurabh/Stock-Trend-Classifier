from pathlib import Path
import pandas as pd
import yfinance as yf

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]
START = "2010-01-01"
END = "2025-01-01"

def fetch_one(ticker: str) -> pd.DataFrame:
    """
    Why: keep data download logic isolated and testable.
    """
    df = yf.download(ticker, start=START, end=END, auto_adjust=False, progress=False)
    df = df.reset_index()
    df["Ticker"] = ticker
    return df

def main():
    """
    Why: create a repeatable 'make raw data' step.
    If you retrain later, you can reproduce the same pipeline.
    """
    Path("data").mkdir(exist_ok=True)

    all_dfs = [fetch_one(t) for t in TICKERS]
    data = pd.concat(all_dfs, ignore_index=True)

    out_path = Path("data/raw_stock_data.csv")
    data.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(data):,} rows")

if __name__ == "__main__":
    main()
