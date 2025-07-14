from fastapi import FastAPI, Request
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("eligibility_model.joblib")

@app.post("/predict")
async def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    score = prediction[0]
    return {"score": float(score)}
