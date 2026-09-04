import re
import string
import json
import joblib
import streamlit as st
from groq import Groq
import os

from dotenv import load_dotenv
load_dotenv()
# ── Load model ───────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return joblib.load("best_spam_pipeline.pkl")

pipeline = load_pipeline()

# ── Groq client ──────────────────────────────────────────

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# ── Text cleaning ─────────────────────────────────────────
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

# ── Classification tool ───────────────────────────────────
def classify_sms(message: str) -> str:
    """
    Classifies an SMS message as spam or ham using a trained Linear SVM pipeline.
    Returns the prediction, confidence score, and interpretation.
    """
    if not message or not message.strip():
        return "No message provided."
    cleaned = clean_text(message)
    prediction = pipeline.predict([cleaned])[0]
    score = pipeline.decision_function([cleaned])[0]
    abs_score = abs(score)
    if abs_score > 2.0:
        confidence = "very high confidence"
    elif abs_score > 1.0:
        confidence = "high confidence"
    elif abs_score > 0.5:
        confidence = "moderate confidence"
    else:
        confidence = "low confidence — borderline case"
    return f"""
    Prediction: {prediction.upper()}
    Score: {score:+.3f}
    Confidence: {confidence}
    """

# ── Tool schema ───────────────────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "classify_sms",
            "description": "Classifies an SMS message as spam or ham.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The SMS message to classify"
                    }
                },
                "required": ["message"]
            }
        }
    }
]

# ── System prompt ─────────────────────────────────────────
system_prompt = """
You are an expert SMS spam detection agent helping users identify and deal with spam messages.

Your workflow for every message analysis:
1. Always call the classify_sms tool first before responding
2. Based on the result provide:
   - Clear verdict (SPAM or HAM)
   - Confidence level explanation
   - Why the message is likely spam or legitimate
   - What the user should do next

If SPAM advise:
   - Do not click any links
   - Do not call any numbers in the message
   - Block the sender
   - Report to network provider if serious

If HAM:
   - Confirm it appears legitimate
   - Note anything slightly suspicious if borderline

Rules:
- Always use the classify_sms tool — never guess
- Be clear and helpful
- Keep responses concise and actionable
"""

# ── Agent function ────────────────────────────────────────
def run_spam_agent(user_message, chat_history):
    messages = [{"role": "system", "content": system_prompt}]
    for human, assistant in chat_history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    response_message = response.choices[0].message

    if response_message.tool_calls:
        tool_call = response_message.tool_calls[0]
        tool_args = json.loads(tool_call.function.arguments)
        tool_result = classify_sms(tool_args["message"])

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": response_message.tool_calls
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result
        })

        final_response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=tools
        )
        return final_response.choices[0].message.content

    return response_message.content

# ── Streamlit UI ──────────────────────────────────────────
st.set_page_config(page_title="SMS Spam Detector Agent", page_icon="📨")
st.title("📨 SMS Spam Detection Agent")
st.write("Powered by Linear SVM + AI Agent. Paste any SMS message to analyse it.")

# Initialise chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for human, assistant in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(human)
    with st.chat_message("assistant"):
        st.write(assistant)

# Input box
message = st.chat_input("Paste an SMS message to check...")

if message:
    with st.chat_message("user"):
        st.write(message)
    with st.chat_message("assistant"):
        with st.spinner("Analysing message..."):
            response = run_spam_agent(message, st.session_state.chat_history)
        st.write(response)
    st.session_state.chat_history.append((message, response))