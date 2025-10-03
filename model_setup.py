import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModel
import torch
import re
import string
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_bert_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    return tokenizer, model

def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()


def train_and_save_model(csv_path, model_path="clf.pkl", embeddings_path="embeddings.pkl"):
    df = pd.read_csv(csv_path)
    if 'Review' not in df.columns or 'Category' not in df.columns:
        raise ValueError("CSV must contain 'Review' and 'Category' columns")
    
    tokenizer, model = load_bert_model()
    df['clean_text'] = df['Review'].apply(preprocess)


    if os.path.exists(embeddings_path):
        with open(embeddings_path, "rb") as f:
            X = pickle.load(f)
        print("Loaded cached embeddings.")
    else:
        X = np.array([get_bert_embedding(text, tokenizer, model) for text in df['clean_text']])
        with open(embeddings_path, "wb") as f:
            pickle.dump(X, f)
        print("Saved embeddings cache.")

    y = df['Category']
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    
    # Save classifier
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    
    return clf, tokenizer, model



def load_model(model_path="clf.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Saved model not found!")
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    tokenizer, model = load_bert_model()
    return clf, tokenizer, model



def predict_comment(text, clf, tokenizer, model):
    vec = get_bert_embedding(preprocess(text), tokenizer, model).reshape(1, -1)
    pred = clf.predict(vec)[0]
    proba = clf.predict_proba(vec)[0] if hasattr(clf, "predict_proba") else None
    confidence = max(proba) if proba is not None else None
    return pred, confidence
