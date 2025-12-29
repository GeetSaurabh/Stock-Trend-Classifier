# Stock Trend Classification System (End-to-End ML)

End-to-end ML project that predicts next-day stock trend (**Down / Neutral / Up**) using historical market data, technical indicators, and a trained classifier. Includes a FastAPI inference service for real-time predictions.

## Problem
Given today’s market signals, predict tomorrow’s trend:
- **Up (2)** if next-day return > +0.5%
- **Down (0)** if next-day return < −0.5%
- **Neutral (1)** otherwise

## Tech Stack
Python, pandas, scikit-learn, yfinance, ta, FastAPI, Uvicorn, joblib

## Project Structure
- `src/` data ingestion, labeling, feature engineering, training
- `api/` FastAPI model serving
- `models/` saved model artifacts (generated)
- `data/` generated datasets (ignored by git)

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
