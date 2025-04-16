
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Modeli yükle
model = joblib.load("models/ms_model.pkl")

@app.get("/")
def home():
    return {"status": "GOALMASTER with AI is alive!"}

@app.get("/predict")
def predict(home_id: int, away_id: int):
    # Dummy veri (normalde buraya gerçek istatistik verileri gelir)
    features = np.array([[home_id % 5, away_id % 5, (home_id - away_id) % 3]])
    prediction = model.predict(features)[0]
    probability = max(model.predict_proba(features)[0])

    return {
        "home_id": home_id,
        "away_id": away_id,
        "prediction": prediction,
        "confidence": round(float(probability), 2)
    }
