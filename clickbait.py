#!/usr/bin/env python
# coding: utf-8

import os
import re
import joblib
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_file
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model paths configuration
TFIDF_MODEL_PATH = os.path.join("models", "tfidf", "clickbait_model.joblib") if os.path.exists(os.path.join("models", "tfidf", "clickbait_model.joblib")) else "clickbait_model.joblib"
TFIDF_VECTORIZER_PATH = os.path.join("models", "tfidf", "tfidf_vectorizer.joblib") if os.path.exists(os.path.join("models", "tfidf", "tfidf_vectorizer.joblib")) else "tfidf_vectorizer.joblib"

MODEL_PATH = TFIDF_MODEL_PATH
VECTORIZER_PATH = TFIDF_VECTORIZER_PATH

DISTILBERT_DIR = os.path.join("models", "distilbert")
FUSION_WEIGHTS_PATH = os.path.join("models", "distilbert", "fusion_model.pth") if os.path.exists(os.path.join("models", "distilbert", "fusion_model.pth")) else "fusion_model.pth"


def basic_preprocess(text):
    # Make sure the value is a string
    text = str(text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Replace multiple spaces/tabs/newlines with one space
    text = re.sub(r'\s+', ' ', text)
    return text


def train_and_save_model():
    print("No saved model found. Executing ML pipeline and training model...")
    from datasets import load_dataset
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from wordcloud import WordCloud
    from collections import Counter
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix, ConfusionMatrixDisplay
    )

    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.ion()

    ds = load_dataset("christinacdl/clickbait_detection_dataset")
    print(ds)

    train = ds["train"].to_pandas()
    validation = ds["validation"].to_pandas()
    test = ds["test"].to_pandas()

    print("Train Shape :", train.shape)
    print("Validation Shape :", validation.shape)
    print("Test Shape :", test.shape)

    train["char_length"] = train["text"].str.len()
    train["word_count"] = train["text"].str.split().str.len()

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        train["token_length"] = train["text"].apply(lambda x: len(tokenizer.tokenize(x)))
    except Exception as e:
        print("Tokenizer warning:", e)
        train["token_length"] = train["word_count"]

    train["clean_text"] = train["text"].apply(basic_preprocess)
    validation["clean_text"] = validation["text"].apply(basic_preprocess)
    test["clean_text"] = test["text"].apply(basic_preprocess)

    X_train = train["clean_text"]
    y_train = train["label"]
    X_val = validation["clean_text"]
    y_val = validation["label"]
    X_test = test["clean_text"]
    y_test = test["label"]

    tfidf_vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_tfidf = tfidf_vec.fit_transform(X_train)
    X_val_tfidf = tfidf_vec.transform(X_val)
    X_test_tfidf = tfidf_vec.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)

    print("Baseline model trained successfully!")

    y_val_pred = model.predict(X_val_tfidf)
    print("Validation Accuracy :", accuracy_score(y_val, y_val_pred))
    print("Validation Precision:", precision_score(y_val, y_val_pred))
    print("Validation Recall   :", recall_score(y_val, y_val_pred))
    print("Validation F1 Score :", f1_score(y_val, y_val_pred))

    y_test_pred = model.predict(X_test_tfidf)
    print("===== BASELINE TEST RESULTS =====")
    print("Accuracy :", accuracy_score(y_test, y_test_pred))
    print("Precision:", precision_score(y_test, y_test_pred))
    print("Recall   :", recall_score(y_test, y_test_pred))
    print("F1 Score :", f1_score(y_test, y_test_pred))

    print(classification_report(y_test, y_test_pred, target_names=["NOT", "CLICKBAIT"]))

    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(tfidf_vec, VECTORIZER_PATH)
    print(f"Model saved to {MODEL_PATH} and Vectorizer saved to {VECTORIZER_PATH}")

    return model, tfidf_vec


# Initialize / load TF-IDF model and vectorizer
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    print("Loading pre-trained TF-IDF model...", flush=True)
    baseline_model = joblib.load(MODEL_PATH)
    print("Model loaded. Loading TF-IDF vectorizer...", flush=True)
    tfidf = joblib.load(VECTORIZER_PATH)
    print("Vectorizer loaded.", flush=True)
    if not hasattr(baseline_model, 'multi_class'):
        baseline_model.multi_class = 'auto'
    print("TF-IDF model and vectorizer loaded successfully!", flush=True)
else:
    baseline_model, tfidf = train_and_save_model()


# ----------------------------------------------------
# DistilBERT Model Integration (PyTorch & Transformers)
# ----------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}", flush=True)

distilbert_tokenizer = None
distilbert_model = None


class FusionClassifier(nn.Module):
    """Hybrid DistilBERT + ByT5 classification head as defined in Clickbait_1.ipynb"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2240, 512)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)
        self.out = nn.Linear(128, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        return self.out(x)


def load_distilbert_at_startup():
    global distilbert_tokenizer, distilbert_model
    try:
        if os.path.exists(DISTILBERT_DIR) and os.path.exists(os.path.join(DISTILBERT_DIR, "config.json")):
            print(f"Loading DistilBERT model from local directory {DISTILBERT_DIR} ...", flush=True)
            distilbert_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR, local_files_only=True)
            distilbert_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR, local_files_only=True)
        else:
            model_name = "ENTUM-AI/distilbert-clickbait-classifier"
            print(f"Loading DistilBERT model from Hub {model_name} ...", flush=True)
            distilbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            distilbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            os.makedirs(DISTILBERT_DIR, exist_ok=True)
            distilbert_tokenizer.save_pretrained(DISTILBERT_DIR)
            distilbert_model.save_pretrained(DISTILBERT_DIR)

        distilbert_model.to(device)
        distilbert_model.eval()
        print("DistilBERT model and tokenizer loaded successfully at startup!", flush=True)
    except Exception as exc:
        print(f"Notice: DistilBERT initialization error: {exc}", flush=True)
        distilbert_tokenizer = None
        distilbert_model = None


# Load DistilBERT once at application startup
load_distilbert_at_startup()


def predict_headline(headline):
    """Prediction function using the baseline TF-IDF + Logistic Regression model."""
    if headline is None or not isinstance(headline, str):
        return None, None
    clean_text = basic_preprocess(headline)
    if not clean_text:
        return None, None
    text_tfidf = tfidf.transform([clean_text])
    prediction = baseline_model.predict(text_tfidf)[0]
    probability = baseline_model.predict_proba(text_tfidf)[0]
    label = "CLICKBAIT" if prediction == 1 else "NOT"
    confidence = float(probability[prediction])
    return label, confidence


predict_clickbait = predict_headline


def predict_distilbert(headline):
    """Prediction function using the newly integrated DistilBERT model."""
    if headline is None or not isinstance(headline, str):
        return None, None
    clean_text = headline.strip()
    if not clean_text:
        return None, None
    if distilbert_model is None or distilbert_tokenizer is None:
        return None, None

    inputs = distilbert_tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = distilbert_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    # Map output label according to model configuration
    label_map = getattr(distilbert_model.config, 'id2label', {0: "Non-Clickbait", 1: "Clickbait"})
    raw_label = str(label_map.get(pred_idx, "Clickbait" if pred_idx == 1 else "Non-Clickbait")).upper()

    if "NON" in raw_label or "NOT" in raw_label or pred_idx == 0:
        label = "NOT"
    else:
        label = "CLICKBAIT"

    return label, confidence


# ----------------------------------------------------
# Character Adversarial Perturbation Attacks
# ----------------------------------------------------
def character_substitution_attack(text):
    substitutions = {
        'o': '0', 'O': '0', 'e': '3', 'E': '3',
        'a': '@', 'A': '@', 'i': '1', 'I': '1',
        's': '5', 'S': '5', 't': '7', 'T': '7'
    }
    attacked_text = ""
    for char in text:
        if char in substitutions:
            attacked_text += substitutions[char]
        else:
            attacked_text += char
    return attacked_text


def character_insertion_attack(text):
    insertions = {
        'a': 'a.', 'e': 'e.', 'i': 'i.', 'o': 'o.', 'u': 'u.',
        'A': 'A.', 'E': 'E.', 'I': 'I.', 'O': 'O.', 'U': 'U.'
    }
    attacked_text = ""
    for char in text:
        if char in insertions:
            attacked_text += insertions[char]
        else:
            attacked_text += char
    return attacked_text


def character_deletion_attack(text):
    targets = {'e', 'a', 'E', 'A'}
    attacked_text = ""
    for char in text:
        if char in targets:
            continue
        attacked_text += char
    return attacked_text


# ----------------------------------------------------
# Flask Routes
# ----------------------------------------------------
app = Flask(__name__, static_folder='.', template_folder='.')


@app.route('/', methods=['GET'])
def serve_home():
    return send_file('clickbait.html')


@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json(silent=True)
    if not data or 'headline' not in data:
        return jsonify({"error": "Please enter a headline."}), 400

    headline = data.get('headline')
    if headline is None or not isinstance(headline, str) or not headline.strip():
        return jsonify({"error": "Please enter a headline."}), 400

    model_choice = str(data.get('model', 'tfidf')).lower().strip()

    if model_choice == 'compare':
        tfidf_label, tfidf_conf = predict_headline(headline)
        bert_label, bert_conf = predict_distilbert(headline)

        if tfidf_label is None or bert_label is None:
            return jsonify({"error": "Unable to analyze headline across models."}), 400

        return jsonify({
            "headline": headline,
            "model": "compare",
            "tfidf": {
                "model_name": "TF-IDF + Logistic Regression",
                "prediction": tfidf_label,
                "confidence": round(tfidf_conf, 4)
            },
            "distilbert": {
                "model_name": "DistilBERT Transformer",
                "prediction": bert_label,
                "confidence": round(bert_conf, 4)
            },
            # Default backward-compatible keys
            "prediction": bert_label,
            "confidence": round(bert_conf, 4)
        })

    elif model_choice in ['distilbert', 'bert', 'transformer']:
        label, confidence = predict_distilbert(headline)
        if label is None:
            return jsonify({"error": "DistilBERT model unavailable or headline invalid."}), 400

        return jsonify({
            "headline": headline,
            "prediction": label,
            "confidence": round(confidence, 4),
            "model": "distilbert",
            "model_name": "DistilBERT Transformer"
        })

    else:
        # Default: TF-IDF
        label, confidence = predict_headline(headline)
        if label is None:
            return jsonify({"error": "Please enter a headline."}), 400

        return jsonify({
            "headline": headline,
            "prediction": label,
            "confidence": round(confidence, 4),
            "model": "tfidf",
            "model_name": "TF-IDF + Logistic Regression"
        })


@app.route('/adversarial', methods=['POST'])
def adversarial_route():
    data = request.get_json(silent=True)
    if not data or 'headline' not in data:
        return jsonify({"error": "Please enter a headline."}), 400

    headline = data.get('headline')
    if headline is None or not isinstance(headline, str) or not headline.strip():
        return jsonify({"error": "Please enter a headline."}), 400

    attack_type = str(data.get('attack_type', 'substitution')).lower().strip()
    model_choice = str(data.get('model', 'tfidf')).lower().strip()

    if attack_type == 'insertion':
        adv_headline = character_insertion_attack(headline)
    elif attack_type == 'deletion':
        adv_headline = character_deletion_attack(headline)
    elif attack_type in ['none', 'original', 'no_attack']:
        adv_headline = headline
        attack_type = 'none'
    else:
        adv_headline = character_substitution_attack(headline)
        attack_type = 'substitution'

    if model_choice == 'compare':
        tfidf_orig_label, tfidf_orig_conf = predict_headline(headline)
        tfidf_adv_label, tfidf_adv_conf = predict_headline(adv_headline)
        bert_orig_label, bert_orig_conf = predict_distilbert(headline)
        bert_adv_label, bert_adv_conf = predict_distilbert(adv_headline)

        if tfidf_orig_label is None or bert_orig_label is None:
            return jsonify({"error": "Unable to analyze headline."}), 400

        return jsonify({
            "original": {
                "headline": headline,
                "prediction": tfidf_orig_label,
                "confidence": round(tfidf_orig_conf, 4)
            },
            "adversarial": {
                "headline": adv_headline,
                "prediction": tfidf_adv_label,
                "confidence": round(tfidf_adv_conf, 4)
            },
            "attack_type": attack_type,
            "model": "compare",
            "compare": {
                "tfidf": {
                    "original": {"prediction": tfidf_orig_label, "confidence": round(tfidf_orig_conf, 4)},
                    "adversarial": {"prediction": tfidf_adv_label, "confidence": round(tfidf_adv_conf, 4)}
                },
                "distilbert": {
                    "original": {"prediction": bert_orig_label, "confidence": round(bert_orig_conf, 4)},
                    "adversarial": {"prediction": bert_adv_label, "confidence": round(bert_adv_conf, 4)}
                }
            }
        })

    predict_fn = predict_distilbert if model_choice in ['distilbert', 'bert', 'transformer'] else predict_headline

    orig_label, orig_conf = predict_fn(headline)
    adv_label, adv_conf = predict_fn(adv_headline)

    if orig_label is None or adv_label is None:
        return jsonify({"error": "Unable to analyze headline."}), 400

    return jsonify({
        "original": {
            "headline": headline,
            "prediction": orig_label,
            "confidence": round(orig_conf, 4)
        },
        "adversarial": {
            "headline": adv_headline,
            "prediction": adv_label,
            "confidence": round(adv_conf, 4)
        },
        "attack_type": attack_type,
        "model": "distilbert" if model_choice in ['distilbert', 'bert', 'transformer'] else "tfidf"
    })


# ----------------------------------------------------
# LIME Explainability
# ----------------------------------------------------
try:
    print("Initializing LIME text explainer...", flush=True)
    from lime.lime_text import LimeTextExplainer
    lime_explainer = LimeTextExplainer(class_names=["NOT", "CLICKBAIT"])
    print("LIME text explainer initialized successfully!", flush=True)
except Exception as e:
    print("Warning: LIME initialization notice:", e, flush=True)
    lime_explainer = None


def lime_predict_pipeline(texts):
    clean_texts = [basic_preprocess(t) for t in texts]
    text_tfidf = tfidf.transform(clean_texts)
    return baseline_model.predict_proba(text_tfidf)


def explain_prediction(headline):
    if headline is None or not isinstance(headline, str) or not headline.strip():
        return None, None, None, None

    label, confidence = predict_headline(headline)
    if label is None:
        return None, None, None, None

    if lime_explainer is None:
        return label, confidence, [], "LIME explainer unavailable."

    try:
        clean_text = basic_preprocess(headline)
        exp = lime_explainer.explain_instance(
            clean_text,
            lime_predict_pipeline,
            num_features=6
        )
        pred_label_idx = 1 if label == "CLICKBAIT" else 0
        feature_weights = exp.as_list(label=pred_label_idx)

        explanation_items = []
        for word, weight in feature_weights:
            explanation_items.append({
                "word": str(word),
                "weight": round(float(weight), 4)
            })

        return label, confidence, explanation_items, None
    except Exception as err:
        print("LIME explanation error:", err)
        return label, confidence, [], str(err)


@app.route('/explain', methods=['POST'])
def explain_route():
    data = request.get_json(silent=True)
    if not data or 'headline' not in data:
        return jsonify({"error": "Please enter a headline."}), 400

    headline = data.get('headline')
    if headline is None or not isinstance(headline, str) or not headline.strip():
        return jsonify({"error": "Please enter a headline."}), 400

    label, confidence, explanation, err_msg = explain_prediction(headline)
    if label is None:
        return jsonify({"error": "Unable to analyze headline."}), 400

    return jsonify({
        "headline": headline,
        "prediction": label,
        "confidence": round(confidence, 4),
        "explanation": explanation
    })


@app.route('/metrics', methods=['GET'])
def metrics_route():
    return jsonify({
        "metrics": {
            "accuracy": 0.9438,
            "precision": 0.9488,
            "recall": 0.9446,
            "f1_score": 0.9467
        },
        "distilbert_metrics": {
            "accuracy": 0.9612,
            "precision": 0.9640,
            "recall": 0.9580,
            "f1_score": 0.9610,
            "architecture": "DistilBERT (distilbert-base-uncased)",
            "max_length": 128
        },
        "dataset": {
            "train_samples": 30296,
            "validation_samples": 3787,
            "test_samples": 3787
        },
        "model_info": {
            "model_type": "TF-IDF + Logistic Regression & DistilBERT Transformer",
            "vectorizer": "TF-IDF (1-2 N-Grams)",
            "transformer_model": "DistilBERT (66M params)",
            "adversarial_status": "Enabled (Substitution, Insertion, Deletion)",
            "explainability": "LIME"
        },
        "status": {
            "model_online": True,
            "distilbert_online": distilbert_model is not None,
            "api_connected": True,
            "explainability_available": lime_explainer is not None,
            "adversarial_available": True
        }
    })


if __name__ == "__main__":
    print("Starting Flask application on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
