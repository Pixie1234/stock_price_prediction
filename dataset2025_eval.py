#!/usr/bin/env python3
"""
Dataset2025 Evaluation
Uses Kaggle dataset with 2025 news - date matches our test period!
Pre-labeled sentiment - no FinBERT needed
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
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DATASET2025 EVALUATION")
print("Using pre-labeled sentiment from 2025 news")
print("="*70)

# ============================================================
# LOAD DATASET2025
# ============================================================
print("\n[1] Loading dataset2025...")
df = pd.read_csv('dataset2025/financial_news_events.csv')
df['Date'] = pd.to_datetime(df['Date'])

print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Total headlines: {len(df)}")

# Map sentiment to numeric (higher weight for clearer signal)
sentiment_map = {'Positive': 0.15, 'Neutral': 0.03, 'Negative': -0.15}
df['sentiment_score'] = df['Sentiment'].map(sentiment_map)
df['sentiment_score'] = df['sentiment_score'].fillna(0.0)

# Also try using raw values 
df['raw_sentiment'] = df['Sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
df['raw_sentiment'] = df['raw_sentiment'].fillna(0)

# ============================================================
# LOAD MODELS
# ============================================================
print("\n[2] Loading LSTM model...")

from data_pipeline import prepare_data, SEQ_LEN, N_FEATURES
from lstm_model import load_model

# ============================================================
# STOCK MAPPING
# ============================================================
# Map dataset tickers to our model tickers
ticker_map = {
    'Apple': 'AAPL',
    'Microsoft': 'MSFT', 
    'JP Morgan': 'JPM',
    'Exxon': 'XOM'
}

# ============================================================
# EVALUATION
# ============================================================
print("\n[3] Running evaluation...")
print("-"*50)

results = []

for dataset_name, model_ticker in ticker_map.items():
    print(f"\n--- {model_ticker} ---")
    
    try:
        # Get headlines for this company
        company_df = df[df['Related_Company'] == dataset_name].copy()
        
        if len(company_df) < 10:
            print(f"  Not enough data")
            continue
            
        print(f"  Headlines: {len(company_df)}")
        
        # Get stock data - use years=10 to match dataset2025 dates
        data = prepare_data(model_ticker, years=10)
        scaled = data['scaled']
        raw = data['raw_ohlcv']
        dates = data['dates_features']
        
        train_end = int(len(scaled) * 0.70)
        
        # Match test dates to news dates
        lstm_preds = []
        prop_preds = []
        actuals = []
        sentiments = []
        
        for i in range(train_end, train_end + 15):
            if i >= len(scaled) - 1:
                break
            
            # Get date for this prediction
            pred_date = dates[i]
            
            # Get news from 1-3 days BEFORE this date
            mask = (company_df['Date'] < pred_date) & (company_df['Date'] >= pred_date - pd.Timedelta(days=3))
            news = company_df[mask]
            
            if len(news) > 0:
                # Use most recent headline's sentiment
                latest = news.iloc[-1]
                sentiment = latest['raw_sentiment'] * 0.15
            else:
                sentiment = 0.0
            
            sentiments.append(sentiment)
            
            # Predict
            last_seq = scaled[i - SEQ_LEN:i]
            actual = raw[i + 1, 3]
            base = raw[i, 3]
            
            model_file = f"models/{model_ticker}_lstm_ohlcv_indicators_v4.h5"
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
            print(f"  Not enough predictions")
            continue
        
        # Directional accuracy
        actual_dir = np.array(actuals[1:]) > np.array(actuals[:-1])
        lstm_dir = np.array(lstm_preds[1:]) > np.array(lstm_preds[:-1])
        lstm_acc = np.mean(lstm_dir == actual_dir)
        
        prop_dir = np.array(prop_preds[1:]) > np.array(prop_preds[:-1])
        prop_acc = np.mean(prop_dir == actual_dir)
        
        print(f"  LSTM+Tech: {lstm_acc*100:.1f}%")
        print(f"  Proposed (2025 news): {prop_acc*100:.1f}%")
        print(f"  Avg sentiment: {np.mean(sentiments):.4f}")
        
        results.append({
            'ticker': model_ticker, 
            'lstm': lstm_acc, 
            'prop': prop_acc,
            'sent': np.mean(sentiments)
        })
        
    except Exception as e:
        print(f"  Error: {e}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("RESULTS WITH DATASET2025 (60% FINBERT + 40% RoBERTa not needed)")
print("="*70)

for r in results:
    print(f"{r['ticker']}: LSTM={r['lstm']*100:.1f}% | Proposed={r['prop']*100:.1f}% (sent={r['sent']:.4f})")

if results:
    lstm_avg = np.mean([r['lstm'] for r in results])
    prop_avg = np.mean([r['prop'] for r in results])
    print(f"\nAverage: LSTM={lstm_avg*100:.1f}% | Proposed={prop_avg*100:.1f}%")

print("\nDone! This uses 2025 news that matches test period.")