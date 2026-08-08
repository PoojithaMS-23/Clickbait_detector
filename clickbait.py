#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#dataset


# In[1]:


from datasets import load_dataset

ds = load_dataset("christinacdl/clickbait_detection_dataset")


# In[2]:


print(ds)


# In[3]:


# =====================================================
# Import Required Libraries
# =====================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud

# Improve plot appearance
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8,5)


# In[4]:


train = ds["train"].to_pandas()
validation = ds["validation"].to_pandas()
test = ds["test"].to_pandas()

print(train.head())


# In[28]:


get_ipython().system('pip install transformers')


# In[5]:


# =====================================================
# Dataset Information
# =====================================================

print(train.info())


# In[6]:


# =====================================================
# Number of Samples
# =====================================================

print("Train Shape :", train.shape)
print("Validation Shape :", validation.shape)
print("Test Shape :", test.shape)


# In[7]:


# =====================================================
# Missing Value Analysis
# =====================================================

print(train.isnull().sum())


# In[8]:


# =====================================================
# Duplicate Analysis
# =====================================================

duplicates = train.duplicated().sum()

print("Duplicate Rows :", duplicates)


# In[9]:


# =====================================================
# Class Distribution
# =====================================================

print(train["text_label"].value_counts())


# In[10]:


sns.countplot(data=train, x="text_label")

plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Headlines")
plt.show()


# In[11]:


# =====================================================
# Character Length
# =====================================================

train["char_length"] = train["text"].str.len()

train["char_length"].describe()


# In[12]:


plt.hist(train["char_length"], bins=30)

plt.title("Character Length Distribution")
plt.xlabel("Characters")
plt.ylabel("Frequency")

plt.show()


# In[13]:


# =====================================================
# Word Count
# =====================================================

train["word_count"] = train["text"].str.split().str.len()

train["word_count"].describe()


# In[14]:


plt.hist(train["word_count"], bins=30)

plt.title("Word Count Distribution")
plt.xlabel("Words")
plt.ylabel("Frequency")

plt.show()


# In[15]:


# =====================================================
# Token Length using RoBERTa Tokenizer
# =====================================================

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

train["token_length"] = train["text"].apply(
    lambda x: len(tokenizer.tokenize(x))
)

train["token_length"].describe()


# In[16]:


plt.hist(train["token_length"], bins=30)

plt.title("Token Length Distribution")

plt.xlabel("Tokens")

plt.ylabel("Frequency")

plt.show()


# In[17]:


# =====================================================
# Vocabulary Size
# =====================================================

vocab = set()

for sentence in train["text"]:
    vocab.update(sentence.lower().split())

print("Vocabulary Size :", len(vocab))


# In[18]:


# =====================================================
# Most Frequent Words
# =====================================================

from collections import Counter

words = " ".join(train["text"]).lower().split()

counter = Counter(words)

print(counter.most_common(20))


# In[19]:


# =====================================================
# WordCloud
# =====================================================

text = " ".join(train["text"])

wc = WordCloud(
    background_color="white",
    width=900,
    height=500
).generate(text)

plt.imshow(wc)

plt.axis("off")

plt.show()


# In[20]:


# =====================================================
# Character Distribution
# =====================================================

special_chars = ['@','0','1','3','$','#','!','?','%']

counts = {}

text = " ".join(train["text"])

for char in special_chars:
    counts[char] = text.count(char)

char_df = pd.DataFrame(
    counts.items(),
    columns=["Character","Count"]
)

print(char_df)


# In[21]:


sns.barplot(
    data=char_df,
    x="Character",
    y="Count"
)

plt.title("Special Character Distribution")

plt.show()


# In[22]:


# =====================================================
# Dataset Summary
# =====================================================

print("Average Characters :", train["char_length"].mean())

print("Average Words :", train["word_count"].mean())

print("Average Tokens :", train["token_length"].mean())

print("Maximum Tokens :", train["token_length"].max())

print("Minimum Tokens :", train["token_length"].min())


# In[ ]:





# In[ ]:





# In[23]:


import transformers
import huggingface_hub
import tokenizers
import datasets

print("Transformers:", transformers.__version__)
print("HuggingFace Hub:", huggingface_hub.__version__)
print("Tokenizers:", tokenizers.__version__)
print("Datasets:", datasets.__version__)


# In[24]:


import pandas as pd
import re

def basic_preprocess(text):
    # Make sure the value is a string
    text = str(text)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Replace multiple spaces/tabs/newlines with one space
    text = re.sub(r'\s+', ' ', text)

    return text


# Apply preprocessing
train["clean_text"] = train["text"].apply(basic_preprocess)
validation["clean_text"] = validation["text"].apply(basic_preprocess)
test["clean_text"] = test["text"].apply(basic_preprocess)


# In[25]:


comparison = pd.DataFrame({
    "Original": train["text"].head(10),
    "Cleaned": train["clean_text"].head(10)
})

comparison


# In[26]:


print("Empty original texts:",
      train["text"].isna().sum())

print("Empty cleaned texts:",
      train["clean_text"].str.strip().eq("").sum())


# In[27]:


print(train[["clean_text", "label", "text_label"]].head())


# In[28]:


print(train["text_label"].value_counts())
print(validation["text_label"].value_counts())
print(test["text_label"].value_counts())


# In[29]:


print(pd.crosstab(train["label"], train["text_label"]))


# In[30]:


train.to_csv("train_preprocessed.csv", index=False)
validation.to_csv("validation_preprocessed.csv", index=False)
test.to_csv("test_preprocessed.csv", index=False)


# In[31]:


print(pd.crosstab(train["label"], train["text_label"]))


# In[32]:


X_train = train["clean_text"]
y_train = train["label"]

X_val = validation["clean_text"]
y_val = validation["label"]

X_test = test["clean_text"]
y_test = test["label"]

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))


# In[33]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),   # unigrams + bigrams
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)

# IMPORTANT:
# Only transform validation/test.
# Do NOT fit TF-IDF again on them.
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Validation TF-IDF shape:", X_val_tfidf.shape)
print("Test TF-IDF shape:", X_test_tfidf.shape)


# In[34]:


from sklearn.linear_model import LogisticRegression

baseline_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

baseline_model.fit(X_train_tfidf, y_train)

print("Baseline model trained successfully!")


# In[35]:


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

y_val_pred = baseline_model.predict(X_val_tfidf)

print("Validation Accuracy :", accuracy_score(y_val, y_val_pred))
print("Validation Precision:", precision_score(y_val, y_val_pred))
print("Validation Recall   :", recall_score(y_val, y_val_pred))
print("Validation F1 Score :", f1_score(y_val, y_val_pred))


# In[36]:


y_test_pred = baseline_model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print("===== BASELINE TEST RESULTS =====")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# In[37]:


from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=["NOT", "CLICKBAIT"]
    )
)


# In[38]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_test_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NOT", "CLICKBAIT"]
)

disp.plot()

plt.title("Baseline Logistic Regression - Confusion Matrix")
plt.show()


# In[39]:


def predict_clickbait(text):
    text_tfidf = tfidf.transform([text])
    prediction = baseline_model.predict(text_tfidf)[0]
    
    probability = baseline_model.predict_proba(text_tfidf)[0]
    
    label = "CLICKBAIT" if prediction == 1 else "NOT"
    
    return label, probability[prediction]


# In[40]:


text = "You won't believe what happened next!"

label, confidence = predict_clickbait(text)

print("Prediction:", label)
print("Confidence:", round(confidence * 100, 2), "%")


# In[41]:


original = "You won't believe what happened next!"

adversarial = "Y0u w0n't believe what happened next!"

print("Original:")
print(predict_clickbait(original))

print("\nAdversarial:")
print(predict_clickbait(adversarial))


# In[42]:


def character_substitution_attack(text):
    substitutions = {
        'o': '0',
        'O': '0',
        'e': '3',
        'E': '3',
        'a': '@',
        'A': '@',
        'i': '1',
        'I': '1',
        's': '5',
        'S': '5'
    }

    attacked_text = ""

    for char in text:
        if char in substitutions:
            attacked_text += substitutions[char]
        else:
            attacked_text += char

    return attacked_text


# In[43]:


text = "You won't believe what happened next!"

attacked = character_substitution_attack(text)

print("Original:")
print(text)

print("\nAdversarial:")
print(attacked)


# In[44]:


original_prediction = predict_clickbait(text)
adversarial_prediction = predict_clickbait(attacked)

print("Original:", original_prediction)
print("Adversarial:", adversarial_prediction)


# In[ ]:




