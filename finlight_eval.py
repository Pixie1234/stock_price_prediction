#!/usr/bin/env python3
"""
Finlight API Evaluation
Generates the AAPL and JPM results from the thesis
Uses real news from Finlight API with 60% FinBERT + 40% RoBERTa sentiment
"""
import os
import sys

PROJECT_ROOT = '/home/anastasija/Diploma Thesis/ResearchPrediction/PythonProject'
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FINLIGHT API EVALUATION")
print("Using real news + 60% FinBERT + 40% RoBERTa")
print("="*70)

# ============================================================
# FINLIGHT API SETUP
# ============================================================
API_KEY = 'sk_c3944c5fd6706ce5293517e8187d05f5c275d932389841cd156ed97ecdfc04ba'

print("\n[1] Setup Finlight API...")
from finlight_client import FinlightApi, ApiConfig
from finlight_client.models import GetArticlesParams

client = FinlightApi(ApiConfig(api_key=API_KEY))

# ============================================================
# LOAD SENTIMENT MODELS
# ============================================================
print("[2] Loading FinBERT + RoBERTa...")

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

DEVICE = "cpu"

fin_tok = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
fin_mod = AutoModelForSequenceClassification.from_pretrained(
    "yiyanghkust/finbert-tone"
).to(DEVICE)
fin_mod.eval()

roberta = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=-1,
    truncation=True,
    max_length=512,
)

print("Models loaded!")

# ============================================================
# SENTIMENT FUNCTIONS (60% + 40%)
# ============================================================
def finbert_sentiment(text):
    inputs = fin_tok(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = fin_mod(**inputs)
    probs = torch.softmax(out.logits, dim=-1)[0]
    labels = ["Bearish", "Neutral", "Bullish"]
    idx = torch.argmax(probs).item()
    return labels[idx], probs[idx].item()

def roberta_sentiment(text):
    result = roberta(text[:512])[0]
    return result["label"], result["score"]

def label_to_direction(label):
    label = label.upper()
    if "BULLISH" in label or "POSITIVE" in label or label == "POS":
        return 1.0
    if "BEARISH" in label or "NEGATIVE" in label or label == "NEG":
        return -1.0
    return 0.0

def fuse_sentiment_60_40(text):
    """60% FinBERT + 40% RoBERTa fusion"""
    fin_label, fin_score = finbert_sentiment(text)
    rob_label, rob_score = roberta_sentiment(text)
    
    fin_dir = label_to_direction(fin_label)
    rob_dir = label_to_direction(rob_label)
    
    # 60% FinBERT + 40% RoBERTa
    fused = (fin_dir * fin_score * 0.60) + (rob_dir * rob_score * 0.40)
    bias = np.clip(fused, -0.20, 0.20)
    
    return float(bias), fin_label, rob_label

# ============================================================
# LOAD LSTM MODEL
# ============================================================
print("[3] Loading LSTM...")

from data_pipeline import prepare_data, SEQ_LEN, N_FEATURES
from lstm_model import load_model

# ============================================================
# EVALUATION
# ============================================================
print("[4] Running evaluation...")
print("-"*50)

results = []

for ticker in ['AAPL', 'JPM']:
    print(f"\n--- {ticker} ---")
    
    try:
        # Get news from Finlight
        params = GetArticlesParams(query=ticker, language='en', limit=5)
        articles = client.articles.fetch_articles(params)
        
        print(f"  Finlight articles: {len(articles.articles)}")
        
        if not articles.articles:
            print(f"  No news")
            continue
        
        # Get headline and compute sentiment
        headline = articles.articles[0].title
        sentiment, fin_l, rob_l = fuse_sentiment_60_40(headline)
        
        print(f"  Headline: {headline[:50]}...")
        print(f"  Sentiment: {sentiment:+.4f} (FinBERT:{fin_l}, RoBERTa:{rob_l})")
        
        # Get stock data
        data = prepare_data(ticker, years=2)
        scaled = data['scaled']
        raw = data['raw_ohlcv']
        
        train_end = int(len(scaled) * 0.70)
        
        # Evaluate
        lstm_preds = []
        prop_preds = []
        actuals = []
        
        for i in range(train_end, train_end + 30):
            if i >= len(scaled) - 1:
                break
            
            last_seq = scaled[i - SEQ_LEN:i]
            actual = raw[i + 1, 3]
            base = raw[i, 3]
            
            model_file = f"models/{ticker}_lstm_ohlcv_indicators_v4.h5"
            if not os.path.exists(model_file):
                print(f"  No model")
                break
            
            model = load_model(model_file)
            pred = model.predict(last_seq.reshape(1, SEQ_LEN, N_FEATURES), verbose=0)[0]
            
            lstm_pred = base * np.exp(pred[1])
            prop_pred = base * np.exp(pred[1] + sentiment * 0.3)
            
            lstm_preds.append(lstm_pred)
            prop_preds.append(prop_pred)
            actuals.append(actual)
        
        if len(lstm_preds) < 5:
            continue
        
        # Directional accuracy
        actual_dir = np.array(actuals[1:]) > np.array(actuals[:-1])
        lstm_dir = np.array(lstm_preds[1:]) > np.array(lstm_preds[:-1])
        lstm_acc = np.mean(lstm_dir == actual_dir)
        
        prop_dir = np.array(prop_preds[1:]) > np.array(prop_preds[:-1])
        prop_acc = np.mean(prop_dir == actual_dir)
        
        print(f"  LSTM: {lstm_acc*100:.1f}%")
        print(f"  Proposed: {prop_acc*100:.1f}%")
        
        results.append({
            'ticker': ticker,
            'sentiment': sentiment,
            'lstm': lstm_acc,
            'prop': prop_acc
        })
        
    except Exception as e:
        print(f"  Error: {e}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("RESULTS - FINLIGHT API + 60% FINBERT + 40% ROBERTA")
print("="*70)

for r in results:
    sent_label = "Bearish" if r['sentiment'] < -0.01 else "Bullish" if r['sentiment'] > 0.01 else "Neutral"
    print(f"{r['ticker']}: {sent_label} ({r['sentiment']:+.2f}) | LSTM={r['lstm']*100:.1f}% | Proposed={r['prop']*100:.1f}%")

print("\nDone!")