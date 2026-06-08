#!/usr/bin/env python3
"""
Full Model Comparison: LSTM (base) vs LSTM+Tech vs Proposed
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from data_pipeline import prepare_data, SEQ_LEN, N_FEATURES
from lstm_model import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FULL MODEL COMPARISON")
print("="*70)

# Sentiment
from finlight_client import FinlightApi, ApiConfig
from finlight_client.models import GetArticlesParams

API_KEY = 'sk_c3944c5fd6706ce5293517e8187d05f5c275d932389841cd156ed97ecdfc04ba'
DEVICE = 'cpu'

fin_tok = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
fin_mod = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone').to(DEVICE)
fin_mod.eval()
roberta = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest', device=-1)

def get_sentiment(ticker):
    client = FinlightApi(ApiConfig(api_key=API_KEY))
    params = GetArticlesParams(query=ticker, language='en', limit=3)
    articles = client.articles.fetch_articles(params)
    
    total = 0.0
    for hl in [a.title for a in articles.articles]:
        inputs = fin_tok(hl, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            out = fin_mod(**inputs)
        probs = torch.softmax(out.logits, dim=-1)[0]
        total += (1 if probs[2] > probs[0] else (-1 if probs[0] > probs[2] else 0)) * probs.max().item() * 0.60
        
        rob = roberta(hl[:512])[0]
        total += (1 if rob['label'] == 'positive' else (-1 if rob['label'] == 'negative' else 0)) * rob['score'] * 0.40
    
    return np.clip(total / 3, -0.20, 0.20)

# Prepare base LSTM data
def get_base_data(ticker):
    df = yf.download(ticker, period='2y', auto_adjust=False)
    df = df.dropna()
    returns = np.log(df / df.shift(1)).dropna()
    
    feat_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    features = returns[feat_cols].values
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    
    return scaled, df['Close'].values

results = []

for ticker in ['AAPL', 'JPM']:
    print(f"\n{'='*40}")
    print(f"{ticker}")
    print(f"{'='*40}")
    
    sentiment = get_sentiment(ticker)
    print(f"Sentiment: {sentiment:+.4f}")
    
    # LSTM+Tech data
    data = prepare_data(ticker, years=2)
    scaled_tech, raw = data['scaled'], data['raw_ohlcv']
    
    # Base LSTM data
    scaled_base, close_prices = get_base_data(ticker)
    
    train_end = int(len(scaled_tech) * 0.65)
    
    # Load models
    base_model = load_model('models/{}_lstm_v6.h5'.format(ticker))
    tech_model = load_model('models/{}_lstm_ohlcv_indicators_v4.h5'.format(ticker))
    
    base_preds, tech_preds, sent_preds, actuals = [], [], [], []
    
    for i in range(train_end, train_end + 30):
        if i >= len(scaled_tech) - 1:
            break
        
        # Base LSTM (5 features)
        base_seq = scaled_base[i - SEQ_LEN:i]
        base_pred = base_model.predict(base_seq.reshape(1, SEQ_LEN, 5), verbose=0)[0, 0]
        base_price = close_prices[i]
        base_preds.append(base_price * np.exp(base_pred))
        
        # LSTM+Tech (8 features)
        tech_seq = scaled_tech[i - SEQ_LEN:i]
        tech_pred = tech_model.predict(tech_seq.reshape(1, SEQ_LEN, N_FEATURES), verbose=0)[0, 1]
        tech_preds.append(raw[i, 3] * np.exp(tech_pred))
        
        # Proposed
        sent_preds.append(raw[i, 3] * np.exp(tech_pred + sentiment * 0.3))
        
        actuals.append(raw[i + 1, 3])
    
    # Directional accuracy
    actual_dir = np.array(actuals[1:]) > np.array(actuals[:-1])
    
    base_dir = np.array(base_preds[1:]) > np.array(base_preds[:-1])
    base_acc = np.mean(base_dir == actual_dir)
    
    tech_dir = np.array(tech_preds[1:]) > np.array(tech_preds[:-1])
    tech_acc = np.mean(tech_dir == actual_dir)
    
    sent_dir = np.array(sent_preds[1:]) > np.array(sent_preds[:-1])
    sent_acc = np.mean(sent_dir == actual_dir)
    
    print(f"\n{'Model':<30} {'Dir Acc':>10}")
    print(f"{'-'*40}")
    print(f"{'LSTM (base)':<30} {base_acc*100:>10.1f}%")
    print(f"{'LSTM + Tech Indicators':<30} {tech_acc*100:>10.1f}%")
    print(f"{'Proposed (LSTM+Tech+Sentiment)':<30} {sent_acc*100:>10.1f}%")
    
    results.append({'t': ticker, 'base': base_acc, 'tech': tech_acc, 'sent': sent_acc})

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"{'Stock':<10} {'LSTM':>10} {'LSTM+Tech':>12} {'Proposed':>12}")
print(f"{'-'*50}")
for r in results:
    print(f"{r['t']:<10} {r['base']*100:>10.1f}% {r['tech']*100:>12.1f}% {r['sent']*100:>12.1f}%")