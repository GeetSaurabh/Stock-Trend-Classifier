import pandas as pd
from pathlib import Path
import ta

def main():
    in_path = Path("data/labeled_stock_data.csv")
    if not in_path.exists():
        raise FileNotFoundError("Run create_labels.py first")

    df = pd.read_csv(in_path)

    # Price-based features
    df["Return_1"] = df.groupby("Ticker")["Close"].pct_change()
    df["Return_5"] = df.groupby("Ticker")["Close"].pct_change(5)

    # Moving averages
    df["MA_10"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(10).mean())
    df["MA_20"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20).mean())

    # Volatility
    df["Volatility_10"] = df.groupby("Ticker")["Return_1"].transform(lambda x: x.rolling(10).std())

    # Momentum indicators
    df["RSI"] = df.groupby("Ticker")["Close"].transform(lambda x: ta.momentum.rsi(x, window=14))
    df["MACD"] = df.groupby("Ticker")["Close"].transform(lambda x: ta.trend.macd(x))

    df = df.dropna()

    out_path = Path("data/featured_stock_data.csv")
    df.to_csv(out_path, index=False)

    print(f"Saved {out_path} with {len(df):,} rows")

if __name__ == "__main__":
    main()
