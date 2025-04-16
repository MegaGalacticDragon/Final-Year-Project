from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import re
import string
import os
import urllib.request

def download_from_drive(file_id, filename):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    urllib.request.urlretrieve(url, filename)

# Download model if not already present
if not os.path.exists("phishing_model.pkl"):
    print("Downloading phishing_model.pkl from Google Drive...")
    download_from_drive("1dUZlaBLfk4UfpYzv6H6Y1MXtet_bXqLO", "phishing_model.pkl")

if not os.path.exists("tfidf_vectorizer.pkl"):
    print("Downloading tfidf_vectorizer.pkl from Google Drive...")
    download_from_drive("1l0KYxXZhif4v4lTr0cbHBXRsz1YD3K3U", "tfidf_vectorizer.pkl")

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
