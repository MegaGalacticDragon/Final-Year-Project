import pandas as pd
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

# Load dataset
df = pd.read_csv("C:/Users/sifat/Python/phishing_email.csv")  # Ensure dataset is large enough

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r'\d+', '', text)
    return text

df['text_combined'] = df['text_combined'].astype(str).apply(clean_text)

# Balance dataset
phishing = df[df.label == 1]
legit = df[df.label == 0]

if len(phishing) > len(legit):
    legit = resample(legit, replace=True, n_samples=len(phishing), random_state=42)
else:
    phishing = resample(phishing, replace=True, n_samples=len(legit), random_state=42)

df_balanced = pd.concat([phishing, legit])

# Split data
X_train, X_test, y_train, y_test = train_test_split(df_balanced['text_combined'], df_balanced['label'], test_size=0.2, random_state=42)

# Faster TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# Train Random Forest
model = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "phishing_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")