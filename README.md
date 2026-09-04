# SMS Spam Detection Agent

An AI-powered SMS spam detector that combines a trained **Linear SVM classifier** with an **AI agent** (Groq + OpenAI) to not only classify messages but explain *why* they are spam and advise users on what to do next.

## How it works

## Features

- **Real ML model** — Linear SVM trained on the SMS Spam Collection dataset, selected over Logistic Regression, BiLSTM, and CNN via 5-fold cross-validation (Spam F1: 0.942, Accuracy: 98.6%)
- **AI agent layer** — Groq-powered agent explains classifications and gives actionable advice
- **Confidence scoring** — SVM decision function interpreted as very high / high / moderate / low confidence
- **Text preprocessing** — URLs, emails, phone numbers, and money amounts replaced with placeholder tokens before classification, matching training conditions
- **Conversational interface** — chat-style UI, users can check multiple messages in one session

## Model Performance

| Model | Spam F1 | Accuracy | CV Mean F1 |
|---|---|---|---|
| **Linear SVM (selected)** | 0.942 | 98.6% | 0.971 ± 0.007 |
| Logistic Regression | 0.938 | 98.5% | 0.967 ± 0.004 |
| CNN (Conv1D) | 0.949 | 98.7% | N/A |
| BiLSTM | 0.930 | 98.3% | N/A |

Linear SVM was selected over CNN despite CNN's slightly higher raw F1 because SVM's cross-validated score is a more reliable estimate than a single train/val/test split.

## Known Limitations

- Training data (SMS Spam Collection, 2012) does not include modern phishing patterns such as delivery notification scams or telecom roaming messages — these may be misclassified
- The agent reasoning layer partially compensates by flagging suspicious elements the model may miss

## Running Locally


# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run agent_app.py
```

## Related Project

- [SMS Spam Detector](https://github.com/your-username/sms_spam_detection) — original classifier without the agent layer
- [Live demo](https://smsspamdetection-prince-sackey.streamlit.app/) — original deployment

## Tech Stack

- **Classifier:** scikit-learn Linear SVM + TF-IDF
- **Agent:** Groq API and openAI
- **Interface:** Streamlit