#!/usr/bin/env python
# coding: utf-8

import os
import re
import joblib
from flask import Flask, request, jsonify, send_file

MODEL_PATH = "clickbait_model.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"

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

    joblib.dump(model, MODEL_PATH)
    joblib.dump(tfidf_vec, VECTORIZER_PATH)
    print(f"Model saved to {MODEL_PATH} and Vectorizer saved to {VECTORIZER_PATH}")

    return model, tfidf_vec


# Initialize / load model and vectorizer
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    print("Loading pre-trained model...", flush=True)
    baseline_model = joblib.load(MODEL_PATH)
    print("Model loaded. Loading TF-IDF vectorizer...", flush=True)
    tfidf = joblib.load(VECTORIZER_PATH)
    print("Vectorizer loaded.", flush=True)
    if not hasattr(baseline_model, 'multi_class'):
        baseline_model.multi_class = 'auto'
    print("Model and vectorizer loaded successfully!", flush=True)
else:
    baseline_model, tfidf = train_and_save_model()


def predict_headline(headline):
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


# Initialize Flask Application
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

    label, confidence = predict_headline(headline)
    if label is None:
        return jsonify({"error": "Please enter a headline."}), 400

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 4)
    })


@app.route('/adversarial', methods=['POST'])
def adversarial_route():
    data = request.get_json(silent=True)
    if not data or 'headline' not in data:
        return jsonify({"error": "Please enter a headline."}), 400

    headline = data.get('headline')
    if headline is None or not isinstance(headline, str) or not headline.strip():
        return jsonify({"error": "Please enter a headline."}), 400

    attack_type = data.get('attack_type', 'substitution').lower()

    if attack_type == 'insertion':
        adv_headline = character_insertion_attack(headline)
    elif attack_type == 'deletion':
        adv_headline = character_deletion_attack(headline)
    else:
        adv_headline = character_substitution_attack(headline)

    orig_label, orig_conf = predict_headline(headline)
    adv_label, adv_conf = predict_headline(adv_headline)

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
        "attack_type": attack_type
    })


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
        "dataset": {
            "train_samples": 30296,
            "validation_samples": 3787,
            "test_samples": 3787
        },
        "model_info": {
            "model_type": "TF-IDF + Logistic Regression",
            "vectorizer": "TF-IDF (1-2 N-Grams)",
            "adversarial_status": "Enabled",
            "explainability": "LIME"
        },
        "status": {
            "model_online": True,
            "api_connected": True,
            "explainability_available": lime_explainer is not None,
            "adversarial_available": True
        }
    })


if __name__ == "__main__":
    print("Starting Flask application on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)


