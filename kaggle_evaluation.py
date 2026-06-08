#!/usr/bin/env python3
"""
Kaggle Dataset Evaluation
Separate script to evaluate using Kaggle historical news data
Does NOT modify any existing code
"""
import os
import sys

# Separate path - don't interfere with existing code
PROJECT_ROOT = '/home/anastasija/Diploma Thesis/ResearchPrediction/PythonProject'
KAGGLE_DATA = os.path.join(PROJECT_ROOT, 'archive')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'kaggle_results.txt')

os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("KAGGLE DATASET EVALUATION")
print("Using FinBERT (60%) + RoBERTa (40%)")
print("="*70)

# ============================================================
# LOAD KAGGLE DATA
# ============================================================
print("\n[1/5] Loading Kaggle dataset...")
df = pd.read_csv(os.path.join(KAGGLE_DATA, 'raw_analyst_ratings.csv'))

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

# Stock ticker to test
TICKERS = ['AAPL', 'XOM', 'NVDA']  # Only these have enough data
print(f"Available tickers with data: {TICKERS}")

# ============================================================
# LOAD SENTIMENT MODELS
# ============================================================
print("\n[2/5] Loading FinBERT + RoBERTa...")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

DEVICE = "cpu"
FINBERT_MODEL = "yiyanghkust/finbert-tone"
ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

fin_tok = AutoTokenizer.from_pretrained(FINBERT_MODEL)
fin_mod = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL).to(DEVICE)
fin_mod.eval()

roberta = pipeline(
    "sentiment-analysis",
    model=ROBERTA_MODEL,
    device=-1,
    truncation=True,
    max_length=512,
)

print("Models loaded!")

# ============================================================
# SENTIMENT FUNCTIONS
# ============================================================
def finbert_sentiment(text, fin_tok, fin_mod):
    inputs = fin_tok(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = fin_mod(**inputs)
    probs = torch.softmax(out.logits, dim=-1)[0]
    labels = ["Bearish", "Neutral", "Bullish"]
    idx = torch.argmax(probs).item()
    return labels[idx], probs[idx].item()

def roberta_sentiment(text, roberta):
    result = roberta(text[:512])[0]
    return result["label"], result["score"]

def label_to_direction(label):
    label = label.upper()
    if "BULLISH" in label or "POSITIVE" in label or label == "POS":
        return 1.0
    if "BEARISH" in label or "NEGATIVE" in label or label == "NEG":
        return -1.0
    return 0.0

def fuse_sentiment(fin_label, fin_score, rob_label, rob_score):
    """60% FinBERT + 40% RoBERTa fusion"""
    fin_dir = label_to_direction(fin_label)
    rob_dir = label_to_direction(rob_label)
    
    fin_signal = fin_dir * fin_score * 0.60
    rob_signal = rob_dir * rob_score * 0.40
    
    fused = fin_signal + rob_signal
    bias = np.clip(fused, -0.20, 0.20)  # Beta = 0.20
    
    return float(bias)

# ============================================================
# LOAD LSTM MODEL
# ============================================================
print("\n[3/5] Loading LSTM model...")

from data_pipeline import prepare_data, SEQ_LEN, N_FEATURES
from lstm_model import load_model

# ============================================================
# EVALUATION
# ============================================================
print("\n[4/5] Running evaluation...")
print("-"*50)

results = []

for ticker in TICKERS:
    try:
        ticker_df = df[df['stock'] == ticker].copy()
        
        if len(ticker_df) < 50:
            print(f"{ticker}: Not enough data")
            continue
        
        # Get stock price data
        stock_data = prepare_data(ticker, years=2)
        scaled = stock_data['scaled']
        raw = stock_data['raw_ohlcv']
        
        train_end = int(len(scaled) * 0.70)
        
        # Test on 20 days
        test_days = 20
        sent = 0.0
        
        lstm_preds = []
        prop_preds = []
        actuals = []
        
        for i in range(train_end, train_end + test_days):
            if i >= len(scaled) - 1:
                break
            
            # Get news from 1-2 days before
            test_date = pd.Timestamp(raw.index[i]) if hasattr(raw, 'index') else None
            
            # Get headlines for this ticker around test date
            mask = ticker_df['stock'] == ticker
            headlines = ticker_df[mask]['headline'].head(5).tolist()
            
            if headlines:
                # Compute sentiment
                fin_l, fin_s = finbert_sentiment(headlines[0], fin_tok, fin_mod)
                rob_l, rob_s = roberta_sentiment(headlines[0], roberta)
                sent = fuse_sentiment(fin_l, fin_s, rob_l, rob_s)
            
            # Predict
            last_seq = scaled[i - SEQ_LEN:i]
            actual = raw[i + 1, 3]
            base = raw[i, 3]
            
            model_file = f"models/{ticker}_lstm_ohlcv_indicators_v4.h5"
            if not os.path.exists(model_file):
                print(f"{ticker}: No model file")
                continue
                
            model = load_model(model_file)
            pred = model.predict(last_seq.reshape(1, SEQ_LEN, N_FEATURES), verbose=0)[0]
            
            lstm_preds.append(base * np.exp(pred[1]))
            prop_preds.append(base * np.exp(pred[1] + sent * 0.3))
            actuals.append(actual)
        
        if len(lstm_preds) < 5:
            print(f"{ticker}: Not enough predictions")
            continue
        
        # Calculate directional accuracy
        lstm_dir = np.mean(np.array(lstm_preds[1:]) > np.array(lstm_preds[:-1]))
        actual_dir = np.mean(np.array(actuals[1:]) > np.array(actuals[:-1]))
        lstm_acc = np.mean((np.array(lstm_preds[1:]) > np.array(lstm_preds[:-1])) == (np.array(actuals[1:]) > np.array(actuals[:-1])))
        
        prop_dir = np.mean(np.array(prop_preds[1:]) > np.array(prop_preds[:-1]))
        prop_acc = np.mean((np.array(prop_preds[1:]) > np.array(prop_preds[:-1])) == (np.array(actuals[1:]) > np.array(actuals[:-1])))
        
        print(f"{ticker}: LSTM={lstm_acc*100:.1f}%  |  Proposed={prop_acc*100:.1f}%")
        results.append({'ticker': ticker, 'lstm': lstm_acc, 'prop': prop_acc})
        
    except Exception as e:
        print(f"{ticker}: Error - {e}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

for r in results:
    print(f"{r['ticker']}: LSTM={r['lstm']*100:.1f}%  |  Proposed={r['prop']*100:.1f}%")

if results:
    lstm_avg = np.mean([r['lstm'] for r in results])
    prop_avg = np.mean([r['prop'] for r in results])
    print(f"\nAverage: LSTM={lstm_avg*100:.1f}%  |  Proposed={prop_avg*100:.1f}%")

print("\n[5/5] Done!")
print(f"Results saved to: {OUTPUT_FILE}")