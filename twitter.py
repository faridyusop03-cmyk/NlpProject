import streamlit as st
import pandas as pd
import re
import string
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =====================================
# PATH CONFIG
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "twitter_training_10k_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

# =====================================
# TEXT CLEANING
# =====================================
stopwords = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)

    tokens = [
        word for word in text.split()
        if word not in stopwords and len(word) > 2
    ]

    return " ".join(tokens)

# =====================================
# TRAIN OR LOAD MODEL
# =====================================
@st.cache_resource
def train_or_load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        accuracy = None
        df = pd.read_csv(DATA_PATH)
    else:
        if not os.path.exists(DATA_PATH):
            st.error("Dataset file not found.")
            st.stop()

        df = pd.read_csv(DATA_PATH)

        # ❌ REMOVE IRRELEVANT
        df = df[df["label"] != "Irrelevant"]

        X = df["text"].astype(str).apply(clean_text)
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1
        )

        model.fit(X_train_vec, y_train)

        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)

        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)

    return model, vectorizer, accuracy, df

# =====================================
# STREAMLIT UI
# =====================================
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="centered")

st.title("💬 Sentiment Analysis Dashboard")
st.write("Sentiment classification using NLP and Machine Learning")

model, vectorizer, accuracy, df = train_or_load_model()

# =====================================
# ACCURACY DISPLAY
# =====================================
if accuracy:
    st.success(f"🎯 Model Accuracy: **{accuracy:.2%}**")

# =====================================
# PIE CHART (DATA DISTRIBUTION)
# =====================================
st.subheader("📊 Sentiment Distribution (Dataset)")

sentiment_counts = df["label"].value_counts()

fig1, ax1 = plt.subplots()
ax1.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
ax1.axis("equal")

st.pyplot(fig1)

# =====================================
# USER INPUT
# =====================================
st.subheader("📝 Sentiment Prediction")

user_text = st.text_area("Enter text:", height=150)

if st.button("Analyze Sentiment"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_text)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]

        st.success(f"Predicted Sentiment: **{prediction}**")

        prob_df = pd.DataFrame({
            "Sentiment": model.classes_,
            "Probability": probs
        })

        fig2, ax2 = plt.subplots()
        ax2.bar(prob_df["Sentiment"], prob_df["Probability"])
        ax2.set_ylabel("Probability")
        ax2.set_title("Prediction Confidence")
        st.pyplot(fig2)

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.caption("NLP Project • Streamlit • TF-IDF • Logistic Regression")
