from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

MODEL_PATH = "models/balanced_model.joblib"

# Load once at startup (fast + standard practice)
model = joblib.load(MODEL_PATH)

FEATURES = ["Return_1", "Return_5", "MA_10", "MA_20", "Volatility_10", "RSI", "MACD"]
LABEL_MAP = {0: "Down", 1: "Neutral", 2: "Up"}

app = FastAPI(title="Stock Trend Classifier", version="1.0")

class FeaturesIn(BaseModel):
    Return_1: float
    Return_5: float
    MA_10: float
    MA_20: float
    Volatility_10: float
    RSI: float
    MACD: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(x: FeaturesIn):
    # Convert request into a single-row DataFrame in the correct feature order
    df = pd.DataFrame([[getattr(x, f) for f in FEATURES]], columns=FEATURES)

    pred = int(model.predict(df)[0])
    proba = model.predict_proba(df)[0].tolist()

    return {
        "prediction_id": pred,
        "prediction_label": LABEL_MAP[pred],
        "probabilities": {
            "Down": proba[0],
            "Neutral": proba[1],
            "Up": proba[2],
        }
    }
