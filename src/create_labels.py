import numpy as np
import pandas as pd
from pathlib import Path

THRESHOLD = 0.005  # 0.5% move

def main():
    """
    Why labeling matters:
    ML needs a clear target. We define "trend" in a way that's stable and explainable:
    - Up if next day's close is > +0.5%
    - Down if < -0.5%
    - Neutral otherwise
    """
    in_path = Path("data/raw_stock_data.csv")
    if not in_path.exists():
        raise FileNotFoundError("Run src/data_ingest.py first to create data/raw_stock_data.csv")

    df = pd.read_csv(in_path)

    # Next day's close per ticker
    df["Next_Close"] = df.groupby("Ticker")["Close"].shift(-1)
    df["Return_1d"] = (df["Next_Close"] - df["Close"]) / df["Close"]

    # 0=Down, 1=Neutral, 2=Up
    df["Label"] = np.where(
        df["Return_1d"] > THRESHOLD, 2,
        np.where(df["Return_1d"] < -THRESHOLD, 0, 1)
    )

    df = df.dropna().copy()

    out_path = Path("data/labeled_stock_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(df):,} rows")
    print("Label distribution:\n", df["Label"].value_counts(normalize=True).round(3))

if __name__ == "__main__":
    main()
