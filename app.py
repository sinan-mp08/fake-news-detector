import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)

# ---------- LOAD MODEL ----------
import joblib

model = joblib.load("notebooks/fake_news_model.pkl")
vectorizer = joblib.load("notebooks/tfidf_vectorizer.pkl")


# ---------- CLEAN TEXT ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()

    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# ---------- UI ----------
st.set_page_config(page_title="Fake News Detector")

st.title("📰 Fake News Detector")
st.write("Check whether a news article is **Fake or Real**")



user_input = st.text_area("Paste news text here")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news text")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])

        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]
        confidence = max(proba) * 100

        threshold = 70  # percent

    if confidence < threshold:
        st.warning(
        f"⚠️ Low confidence prediction ({confidence:.2f}%). "
        "Result may be unreliable."
    )
    elif prediction == 1:
        st.success(f"✅ REAL NEWS ({confidence:.2f}% confidence)")
    else:
        st.error(f"❌ FAKE NEWS ({confidence:.2f}% confidence)")


        st.caption(
            "⚠️ This model detects writing patterns, not factual correctness. "
            "Short or context-less statements may be misclassified."
        )

