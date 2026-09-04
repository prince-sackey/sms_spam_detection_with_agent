"""
SMS Spam Detector — Streamlit deployment app.

Loads the trained Linear SVM pipeline (`best_spam_pipeline.pkl`, a single
scikit-learn Pipeline containing the TF-IDF vectorizer and the classifier)
and classifies a message as spam or ham.

IMPORTANT: the pipeline was trained on *cleaned* text (see clean_text below —
lowercased, with URLs/emails/money/phone numbers replaced by placeholder
tokens). Any new message must go through the same cleaning step before being
passed to the pipeline, or predictions will be less accurate than the
notebook's reported metrics — especially on messages containing links, prices,
or phone numbers, which is exactly the kind of content spam messages contain.

To deploy on Streamlit Community Cloud:
1. Push this file, requirements.txt, and best_spam_pipeline.pkl to a GitHub repo.
2. Go to share.streamlit.io, sign in with GitHub, and create a new app
   pointing at this repo and this file (streamlit_app.py).
3. It builds and serves automatically.
"""

import re
import string

import joblib
import streamlit as st


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' <URL> ', text)
    text = re.sub(r'\S+@\S+', ' <EMAIL> ', text)
    text = re.sub(
        r'€\s?\d+|£\s?\d+|\$\s?\d+|\d+\s?(euros?|pounds?|pence|dollars?|usd|gbp|eur)',
        ' <MONEY> ', text
    )
    text = re.sub(r'\+?\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}', ' <PHONE> ', text)
    text = re.sub(r'\b\d{4,6}\b', ' <PHONE> ', text)
    text = re.sub(r'\b\d+\b', ' <NUM> ', text)
    punct = string.punctuation.replace('<', '').replace('>', '')
    text = text.translate(str.maketrans('', '', punct))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@st.cache_resource
def load_pipeline():
    return joblib.load("best_spam_pipeline.pkl")


pipeline = load_pipeline()

st.title("SMS Spam Detector")
st.write(
    "Paste an SMS message to classify it as spam or ham. Built with a Linear SVM "
    "(TF-IDF features), selected over Logistic Regression, BiLSTM, and CNN models "
    "via 5-fold cross-validation — see the training notebook for the full comparison."
)

message = st.text_area("Message", placeholder="e.g. Free entry! Claim your prize now!")

if st.button("Classify") and message.strip():
    cleaned = clean_text(message)
    prediction = pipeline.predict([cleaned])[0]
    score = pipeline.decision_function([cleaned])[0]

    if prediction == "spam":
        st.error(f"🔴 SPAM  (score: {score:+.3f})")
    else:
        st.success(f"🟢 HAM  (score: {score:+.3f})")

    st.caption(
        "Score is the SVM's signed distance from the decision boundary, not a "
        "probability — further from zero means more confident, sign indicates class."
    )
