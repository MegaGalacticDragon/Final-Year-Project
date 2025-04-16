from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import re
import string
import os
import gdown

# Google Drive file IDs
model_file_id = "1dUZlaBLfk4UfpYzv6H6Y1MXtet_bXqLO"
vectorizer_file_id = "1l0KYxXZhif4v4lTr0cbHBXRsz1YD3K3U"

# Output paths
model_path = "phishing_model.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"

# Full download links
model_url = f"https://drive.google.com/uc?id={model_file_id}"
vectorizer_url = f"https://drive.google.com/uc?id={vectorizer_file_id}"

# Download if missing
if not os.path.exists(model_path):
    print("Downloading phishing_model.pkl from Google Drive...")
    gdown.download(model_url, model_path, quiet=False)

if not os.path.exists(vectorizer_path):
    print("Downloading tfidf_vectorizer.pkl from Google Drive...")
    gdown.download(vectorizer_url, vectorizer_path, quiet=False)

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
