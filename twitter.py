import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# CONFIG
# -------------------------------
DATA_PATH = "twitter_training_10k_ml_ready.csv"
MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# -------------------------------
# TRAIN MODEL (ONLY IF NOT EXIST)
# -------------------------------
@st.cache_resource
def train_or_load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        accuracy = None
    else:
        df = pd.read_csv(DATA_PATH)

        X = df["text"].astype(str)   # ❌ No cleaning
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_vec, y_train)

        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)

        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)

    return model, vectorizer, accuracy

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="centered")

st.title("💬 Sentiment Analysis Dashboard")
st.write("NLP-based sentiment analysis using TF-IDF and Logistic Regression")

model, vectorizer, accuracy = train_or_load_model()

if accuracy:
    st.success(f"Model trained successfully — Accuracy: **{accuracy:.2%}**")

# -------------------------------
# USER INPUT
# -------------------------------
user_text = st.text_area("Enter text for sentiment analysis:", height=150)

if st.button("Analyze Sentiment"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        vec = vectorizer.transform([user_text])   # ❌ No cleaning
        prediction = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]

        st.subheader("🔍 Prediction Result")
        st.success(f"Sentiment: **{prediction}**")

        prob_df = pd.DataFrame({
            "Sentiment": model.classes_,
            "Probability": probs
        })

        fig, ax = plt.subplots()
        ax.bar(prob_df["Sentiment"], prob_df["Probability"])
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")

        st.pyplot(fig)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("NLP Project • Streamlit • TF-IDF • Logistic Regression")
