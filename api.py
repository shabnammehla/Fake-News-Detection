from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load ML assets
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

app = FastAPI(title="Fake News Detection Backend")

class NewsInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "Backend is running"}

@app.post("/predict")
def predict(news: NewsInput):
    text = news.text

    transformed = vectorizer.transform([text])
    prediction = model.predict(transformed)[0]
    proba = model.predict_proba(transformed)[0]

    label = "Real" if prediction == 1 else "Fake"
    confidence = max(proba)

    return {
        "prediction": label,
        "confidence": round(float(confidence), 3)
    }