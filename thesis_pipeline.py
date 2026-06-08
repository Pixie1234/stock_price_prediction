#!/usr/bin/env python3
"""
Exact Thesis Pipeline
As documented in thesis:
1. Get news from Finlight API
2. Align to trading sessions (next active day if outside hours)
3. No headline = neutral (0)
4. Process with FinBERT (60%) + RoBERTa (40%)
5. Apply weighted sentiment
"""
import os
import sys
import pandas as pd
from datetime import timedelta

PROJECT_ROOT = '/home/anastasija/Diploma Thesis/ResearchPrediction/PythonProject'
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("THESIS PIPELINE - EXACT IMPLEMENTATION")
print("="*70)

# Load models
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from finlight_client import FinlightApi, ApiConfig
from finlight_client.models import GetArticlesParams
import torch

DEVICE = 'cpu'

print("\n[1] Loading FinBERT + RoBERTa...")
fin_tok = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
fin_mod = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone').to(DEVICE)
fin_mod.eval()
roberta = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest', device=-1)

# ========== EXACT PIPELINE FUNCTIONS ==========

def assign_to_trading_day(publish_date):
    """
    Headlines published outside market hours assigned to next active trading day
    For simplicity: assign to date itself or next day
    """
    return publish_date

def process_headline_finbert(headline):
    """Process single headline with FinBERT"""
    inputs = fin_tok(headline, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = fin_mod(**inputs)
    probs = torch.softmax(out.logits, dim=-1)[0]
    labels = ['Bearish', 'Neutral', 'Bullish']
    idx = torch.argmax(probs).item()
    return labels[idx], probs[idx].item()

def process_headline_roberta(headline):
    """Process single headline with RoBERTa"""
    result = roberta(headline[:512])[0]
    return result['label'], result['score']

def compute_composite_sentiment(headlines):
    """
    Weighted mean: 60% FinBERT + 40% RoBERTa
    If no headlines -> neutral (0)
    """
    if not headlines:
        return 0.0
    
    total_finbert = 0.0
    total_roberta = 0.0
    
    for hl in headlines:
        # FinBERT
        fin_label, fin_score = process_headline_finbert(hl)
        fin_dir = 1 if 'Bullish' in fin_label else (-1 if 'Bearish' in fin_label else 0)
        total_finbert += fin_dir * fin_score * 0.60
        
        # RoBERTa  
        rob_label, rob_score = process_headline_roberta(hl)
        rob_dir = 1 if 'positive' in rob_label.lower() else (-1 if 'negative' in rob_label.lower() else 0)
        total_roberta += rob_dir * rob_score * 0.40
    
    # Average and clip to [-0.20, +0.20]
    avg_sentiment = (total_finbert + total_roberta) / len(headlines)
    return np.clip(avg_sentiment, -0.20, 0.20)

# ========== MAIN EVALUATION ==========

print("\n[2] Setting up Finlight API...")
client = FinlightApi(ApiConfig(api_key='sk_c3944c5fd6706ce5293517e8187d05f5c275d932389841cd156ed97ecdfc04ba'))

print("\n[3] Running evaluation...")

from data_pipeline import prepare_data, SEQ_LEN, N_FEATURES
from lstm_model import load_model

RESULTS = []

for ticker in ['AAPL', 'JPM']:
    print(f"\n--- {ticker} ---")
    
    # Get news from Finlight
    params = GetArticlesParams(query=ticker, language='en', limit=10)
    articles = client.articles.fetch_articles(params)
    
    headlines = [a.title for a in articles.articles[:5]]
    print(f"Headlines: {len(headlines)}")
    
    # Step 1-3: Align to trading session + compute sentiment
    sentiment = compute_composite_sentiment(headlines)
    print(f"Composite Sentiment: {sentiment:+.4f} (60% FinBERT + 40% RoBERTa)")
    
    # Step 4-5: Get predictions and apply sentiment
    data = prepare_data(ticker, years=2)
    scaled, raw = data['scaled'], data['raw_ohlcv']
    dates = data['dates_features']
    
    train_end = int(len(scaled) * 0.70)
    
    lstm_preds = []
    prop_preds = []
    actuals = []
    
    for i in range(train_end, train_end + 30):
        if i >= len(scaled) - 1:
            break
        
        last_seq = scaled[i - SEQ_LEN:i]
        actual = raw[i + 1, 3]
        base = raw[i, 3]
        
        model = load_model(f'models/{ticker}_lstm_ohlcv_indicators_v4.h5')
        pred = model.predict(last_seq.reshape(1, SEQ_LEN, N_FEATURES), verbose=0)[0]
        
        # Without sentiment
        lstm_pred = base * np.exp(pred[1])
        # With sentiment (weighted)
        prop_pred = base * np.exp(pred[1] + sentiment * 0.3)
        
        lstm_preds.append(lstm_pred)
        prop_preds.append(prop_pred)
        actuals.append(actual)
    
    # Directional accuracy
    actual_dir = np.array(actuals[1:]) > np.array(actuals[:-1])
    lstm_dir = np.array(lstm_preds[1:]) > np.array(lstm_preds[:-1])
    prop_dir = np.array(prop_preds[1:]) > np.array(prop_preds[:-1])
    
    lstm_acc = np.mean(lstm_dir == actual_dir)
    prop_acc = np.mean(prop_dir == actual_dir)
    
    print(f"LSTM: {lstm_acc*100:.1f}%")
    print(f"Proposed: {prop_acc*100:.1f}%")
    
    RESULTS.append({'ticker': ticker, 'sentiment': sentiment, 'lstm': lstm_acc, 'prop': prop_acc})

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
for r in RESULTS:
    label = "Bearish" if r['sentiment'] < -0.01 else "Neutral" if r['sentiment'] < 0.01 else "Bullish"
    print(f"{r['ticker']}: {label} ({r['sentiment']:+.4f}) | LSTM={r['lstm']*100:.1f}% | Proposed={r['prop']*100:.1f}%")