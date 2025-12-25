import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

model = joblib.load("../models/fake_news_model.pkl")
tfidf = joblib.load("../models/tfidf_vectorizer.pkl")

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
