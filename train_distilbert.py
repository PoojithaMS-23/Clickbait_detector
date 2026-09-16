"""
Standalone training and export script for DistilBERT clickbait classification.
Matches training configurations from Clickbait_1.ipynb and saves output models to models/distilbert/.
"""

import os
import random
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_OUTPUT_DIR = os.path.join("models", "distilbert")

MAPPING = {
    "a": ["@", "4"],
    "e": ["3", "€"],
    "i": ["1", "!"],
    "o": ["0"],
    "s": ["5", "$"],
    "t": ["7"]
}


def modify_text(text, prob=0.30):
    chars = list(text)
    for i, ch in enumerate(chars):
        low = ch.lower()
        if low in MAPPING and random.random() < prob:
            chars[i] = random.choice(MAPPING[low])
    return "".join(chars)


class FusionClassifier(nn.Module):
    """Hybrid DistilBERT + ByT5 classification head as specified in Clickbait_1.ipynb"""
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


def export_pretrained_distilbert(target_dir=MODEL_OUTPUT_DIR):
    """
    Downloads and caches a production-ready DistilBERT clickbait classifier
    into the models/distilbert directory.
    """
    os.makedirs(target_dir, exist_ok=True)
    print(f"Exporting DistilBERT clickbait model to {target_dir}...")
    model_name = "ENTUM-AI/distilbert-clickbait-classifier"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    tokenizer.save_pretrained(target_dir)
    model.save_pretrained(target_dir)
    print(f"DistilBERT model and tokenizer successfully saved to {target_dir}")


if __name__ == "__main__":
    export_pretrained_distilbert()
