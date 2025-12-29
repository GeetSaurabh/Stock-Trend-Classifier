import pandas as pd
import joblib
from pathlib import Path

FEATURES = ["Return_1","Return_5","MA_10","MA_20","Volatility_10","RSI","MACD"]

model = joblib.load("models/balanced_model.joblib")

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

Path("reports").mkdir(exist_ok=True)
out = Path("reports/feature_importance.csv")
importance.to_csv(out, index=False)

print("\nFeature importance:")
print(importance)
print(f"\nSaved to {out}")
