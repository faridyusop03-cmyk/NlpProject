import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import re
import string

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# CONFIG
# -------------------------------
DATA_PATH = "twitter_training_10k_ml_ready.csv"
MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# -------------------------------
# TEXT CLEANING
# -------------------------------
stopwords = {
    "i","me","my","we","our","you","your","he","she","it","they","them",
    "is","am","are","was","were","be","been","being",
    "a","an","the","and","or","but","if","because","as","until","while",
    "of","at","by","for","with","about","against","between","into","through",
    "to","from","up","down","in","out","on","off","over","under",
    "again","further","then","once","here","there","when","where","why","how",
    "all","any","both","each","few","more","most","other","some","such",
    "no","nor","not","only","own","same","so","than","too","very"
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = " ".join(w for w in text.split() if w not in stopwords)
    return text.strip()

# -------------------------------
# TRAIN / LOAD MODEL
# -------------------------------
@st.cache_resource
def train_or_load_model():
    df = pd.read_csv(DATA_PATH)

    X = df["text"].astype(str).apply(clean_text)
    y = df["label"]

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True
    )

    X_vec = vectorizer.fit_transform(X)

    # Cross-validation
    model = LinearSVC()
    cv_scores = cross_val_score(model, X_vec, y, cv=5, scoring="accuracy")
    cv_accuracy = cv_scores.mean()

    # Train-test split for evaluation visuals
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save model + vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    return model, vectorizer, accuracy, cv_accuracy, conf_matrix, report

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="centered")

st.title("💬 Sentiment Analysis Dashboard")
st.write("Advanced NLP sentiment analysis with LinearSVC, TF-IDF, and cross-validation")

model, vectorizer, accuracy, cv_accuracy, conf_matrix, report = train_or_load_model()

st.success(f"Hold-out Test Accuracy: **{accuracy:.2%}**")
st.info(f"Cross-Validation Accuracy (5-fold): **{cv_accuracy:.2%}**")

# -------------------------------
# CONFUSION MATRIX
# -------------------------------
st.subheader("📊 Confusion Matrix")
fig_cm, ax_cm = plt.subplots()
ax_cm.imshow(conf_matrix, cmap="Blues")
ax_cm.set_title("Confusion Matrix")
ax_cm.set_xlabel("Predicted Label")
ax_cm.set_ylabel("True Label")
ax_cm.set_xticks(range(len(model.classes_)))
ax_cm.set_yticks(range(len(model.classes_)))
ax_cm.set_xticklabels(model.classes_)
ax_cm.set_yticklabels(model.classes_)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax_cm.text(j, i, conf_matrix[i, j], ha="center", va="center", color="black")
st.pyplot(fig_cm)

# -------------------------------
# CLASSIFICATION METRICS
# -------------------------------
st.subheader("📈 Classification Report")
report_df = pd.DataFrame(report).transpose().round(3)
st.dataframe(report_df)

# -------------------------------
# USER INPUT
# -------------------------------
st.subheader("📝 Analyze New Text")
user_text = st.text_area("Enter text for sentiment analysis:", height=150)

if st.button("Analyze Sentiment"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_text)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]

        st.subheader("🔍 Prediction Result")
        st.success(f"Sentiment: **{prediction}**")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("NLP Project • Streamlit • TF-IDF • LinearSVC • Cross-Validation")
