import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "fake_news_model.pkl")
VECT_PATH = os.path.join(BASE_DIR, "..", "models", "tfidf_vectorizer.pkl")

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECT_PATH)


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def predict_news(text):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)
    return "REAL NEWS" if prediction[0] == 1 else "FAKE NEWS"


if __name__ == "__main__":
    text = "Breaking news: Scientists discover a new species of bird in the Amazon rainforest."
    result = predict_news(text)
    print("Prediction:", result)
