from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('models/churn_model.pkl')

@app.post("/predict")
async def predict(data: dict):
    # Build a single-row DataFrame from the incoming JSON
    try:
        df = pd.DataFrame([data])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input format: {e}")

    # Ensure the DataFrame has exactly the columns the model was trained on
    # (fill missing cols with 0, drop extras)
    trained_cols = model.feature_names_in_
    for c in trained_cols:
        if c not in df.columns:
            df[c] = 0
    df = df[trained_cols]

    # Compute churn probability
    prob = float(model.predict_proba(df)[:, 1][0])

    # Translate into a human‑readable recommendation
    if prob > 0.8:
        rec = "Offer 20% discount"
    elif prob > 0.5:
        rec = "Offer 10% discount"
    else:
        rec = "Send thank-you note"

    return {"churn_prob": prob, "recommendation": rec}
