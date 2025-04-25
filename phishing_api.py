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

# Load model
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

app = FastAPI()

class EmailInput(BaseModel):
    text: str

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r'\d+', '', text)
    return text

@app.post("/predict")
def predict_phishing(input_data: EmailInput):
    cleaned_text = clean_text(input_data.text)

    # Handle empty/meaningless text
    if not cleaned_text.strip():
        return {"prediction": "Invalid or empty input"}
    if re.fullmatch(r"[\d\W]+", cleaned_text):
        return {"prediction": "Unable to classify meaningless content"}

    vectorized = vectorizer.transform([cleaned_text])
    proba = model.predict_proba(vectorized)[0][1]  # probability of phishing

    if proba >= 0.65:
        result = "Phishing"
    elif proba <= 0.35:
        result = "Legitimate"
    else:
        result = "Uncertain"

    return {"prediction": result, "confidence": f"{proba:.2f}"}