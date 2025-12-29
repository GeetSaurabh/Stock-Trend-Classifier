import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def main():
    df = pd.read_csv("data/featured_stock_data.csv")

    features = [
        "Return_1", "Return_5",
        "MA_10", "MA_20",
        "Volatility_10",
        "RSI", "MACD"
    ]

    X = df[features]
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\nBalanced Model Report:")
    print(classification_report(y_test, preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/balanced_model.joblib")
    print("\nSaved models/balanced_model.joblib")

if __name__ == "__main__":
    main()
