#!/usr/bin/env python3
"""
Kaggle Dataset Evaluation with 60% FinBERT + 40% RoBERTa
Separate script - will NOT be pushed to GitHub
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
print("KAGGLE EVALUATION - 60% FinBERT + 40% RoBERTa")
print("="*70)

# ============================================================
# LOAD KAGGLE DATA
# ============================================================
print("\n[1] Loading Kaggle data...")
df = pd.read_csv('archive/raw_analyst_ratings.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

# Check available tickers
print(f"Total headlines: {len(df)}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# ============================================================
# LOAD MODELS
# ============================================================
print("\n[2] Loading FinBERT + RoBERTa...")

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

DEVICE = "cpu"

# FinBERT
fin_tok = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
fin_mod = AutoModelForSequenceClassification.from_pretrained(
    "yiyanghkust/finbert-tone"
).to(DEVICE)
fin_mod.eval()

# RoBERTa
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
    
    # Clip to [-0.20, +0.20]
    bias = np.clip(fused, -0.20, 0.20)
    
    return float(bias), fin_label, rob_label

# Test the fusion
test_text = "AAPL reports strong earnings beat"
bias, fl, rl = fuse_sentiment_60_40(test_text)
print(f"Test fusion: {test_text[:30]}... -> bias={bias:+.4f} (FinBERT:{fl}, RoBERTa:{rl})")

# ============================================================
# STOCK DATA
# ============================================================
print("\n[3] Loading stock data...")

from data_pipeline import prepare_data, SEQ_LEN, N_FEATURES
from lstm_model import load_model

# ============================================================
# EVALUATION
# ============================================================
print("\n[4] Running evaluation...")
print("-"*50)

results = []

for ticker in ['AAPL', 'XOM', 'NVDA']:
    print(f"\n--- {ticker} ---")
    
    try:
        # Get stock data
        stock_df = df[df['stock'] == ticker].copy()
        if len(stock_df) < 10:
            print(f"  Noenough data")
            continue
            
        print(f"  Kaggle headlines: {len(stock_df)}")
        
        # Get price data - use years=2 (original)
        data = prepare_data(ticker, years=2)
        scaled = data['scaled']
        raw = data['raw_ohlcv']
        
        train_end = int(len(scaled) * 0.70)
        
        # Test on 10 days
        test_days = min(10, len(scaled) - train_end - 1)
        
        lstm_preds = []
        prop_preds = []
        actuals = []
        sentiments = []
        
        for i in range(train_end, train_end + test_days):
            # Get prediction date
            pred_idx = i
            if hasattr(raw, 'index'):
                pred_date = raw.index[pred_idx]
            else:
                pred_date = pd.Timestamp('2020-01-01')
            
            # Get headlines from 1-3 days BEFORE prediction date
            date_before = pred_date - pd.Timedelta(days=3)
            mask = (stock_df['date'] < pred_date) & (stock_df['date'] > date_before)
            ticker_headlines = stock_df[mask]['headline'].head(3).tolist()
            
            if ticker_headlines:
                # Compute 60/40 sentiment
                bias, fl, rl = fuse_sentiment_60_40(ticker_headlines[0])
                sentiments.append(bias)
            else:
                bias = 0.0
            
            # LSTM prediction
            last_seq = scaled[i - SEQ_LEN:i]
            actual = raw[i + 1, 3]
            base = raw[i, 3]
            
            model_file = f"models/{ticker}_lstm_ohlcv_indicators_v4.h5"
            if not os.path.exists(model_file):
                print(f"  No model")
                break
            
            model = load_model(model_file)
            pred = model.predict(last_seq.reshape(1, SEQ_LEN, N_FEATURES), verbose=0)[0]
            
            # Without sentiment
            lstm_pred = base * np.exp(pred[1])
            # With sentiment (60/40 fusion)
            prop_pred = base * np.exp(pred[1] + bias * 0.3)
            
            lstm_preds.append(lstm_pred)
            prop_preds.append(prop_pred)
            actuals.append(actual)
        
        if len(lstm_preds) < 3:
            print(f"  Not enough predictions")
            continue
        
        # Directional accuracy
        lstm_dir = (np.array(lstm_preds[1:]) > np.array(lstm_preds[:-1])).astype(int)
        actual_dir = (np.array(actuals[1:]) > np.array(actuals[:-1])).astype(int)
        lstm_acc = np.mean(lstm_dir == actual_dir)
        
        prop_dir = (np.array(prop_preds[1:]) > np.array(prop_preds[:-1])).astype(int)
        prop_acc = np.mean(prop_dir == actual_dir)
        
        print(f"  LSTM+Tech: {lstm_acc*100:.1f}%")
        print(f"  Proposed (60/40): {prop_acc*100:.1f}%")
        print(f"  Sentiment used: {np.mean(sentiments):.4f} avg")
        
        results.append({'ticker': ticker, 'lstm': lstm_acc, 'prop': prop_acc, 'sent': np.mean(sentiments)})
        
    except Exception as e:
        print(f"  Error: {e}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("RESULTS WITH 60% FINBERT + 40% ROBERTA")
print("="*70)

for r in results:
    print(f"{r['ticker']}: LSTM={r['lstm']*100:.1f}% | Proposed={r['prop']*100:.1f}% (sent={r['sent']:.4f})")

if results:
    lstm_avg = np.mean([r['lstm'] for r in results])
    prop_avg = np.mean([r['prop'] for r in results])
    print(f"\nAverage: LSTM={lstm_avg*100:.1f}% | Proposed={prop_avg*100:.1f}%")

print("\nDone!")
print("This file is separate - will NOT be pushed to GitHub")