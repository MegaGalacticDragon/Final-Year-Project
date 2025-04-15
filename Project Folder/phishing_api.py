from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import re
import string

# Load trained model and vectorizer
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = FastAPI()

# Input format
class EmailInput(BaseModel):
    text: str

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r'\d+', '', text)
    return text

# Endpoint to check phishing
@app.post("/predict")
def predict_phishing(input_data: EmailInput):
    cleaned_text = clean_text(input_data.text)
    vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized)[0]
    result = "Phishing" if prediction == 1 else "Legitimate"
    return {"prediction": result}
