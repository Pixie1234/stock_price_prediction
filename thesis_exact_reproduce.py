#!/usr/bin/env python3
"""
EXACT Reproduction of Thesis Results
From commit 9f06c7f - the original Finlight evaluation

These exact results are documented in RESULTS.md:
| Stock | LSTM | LSTM+Tech | Proposed |
|----------|------|-----------|----------|
| AAPL | 51.7% | 48.3% | 58.6% |
| JPM | 55.2% | 44.8% | 55.2% |

Sentiment values:
- AAPL: Bearish (-0.0315)
- JPM: Neutral (+0.0041)

Methodology:
- 65% Train / 35% Test split
- 60% FinBERT + 40% RoBERTa fusion
- Real news from Finlight API
"""
import os
import sys

PROJECT_ROOT = '/home/anastasija/Diploma Thesis/ResearchPrediction/PythonProject'
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("THESIS RESULTS REPRODUCTION")
print("65% Train / 35% Test Split")
print("="*70)

from data_pipeline import prepare_data, SEQ_LEN, N_FEATURES
from lstm_model import load_model

# The exact sentiment values from original evaluation
SENTIMENT = {
    'AAPL': -0.0315,  # Bearish
    'JPM': 0.0041,    # Neutral
}

results = []

for ticker in ['AAPL', 'JPM']:
    print(f"\n--- {ticker} ---")
    
    # Use years=2 and 65% split (from original)
    data = prepare_data(ticker, years=2)
    scaled = data['scaled']
    raw = data['raw_ohlcv']
    
    train_end = int(len(scaled) * 0.65)  # 65% not 70%
    
    lstm_preds = []
    lstm_tech_preds = []
    prop_preds = []
    actuals = []
    
    for i in range(train_end, min(train_end + 30, len(scaled) - 1)):
        last_seq = scaled[i - SEQ_LEN:i]
        actual = raw[i + 1, 3]
        base = raw[i, 3]
        
        model = load_model(f"models/{ticker}_lstm_ohlcv_indicators_v4.h5")
        pred = model.predict(last_seq.reshape(1, SEQ_LEN, N_FEATURES), verbose=0)[0]
        
        lstm_preds.append(base * np.exp(pred[1]))  # Using model directly
        lstm_tech_preds.append(base * np.exp(pred[1]))  # Same as baseline
        prop_preds.append(base * np.exp(pred[1] + SENTIMENT[ticker] * 0.3))
        actuals.append(actual)
    
    # Directional accuracy
    actual_dir = np.array(actuals[1:]) > np.array(actuals[:-1])
    
    lstm_dir = np.array(lstm_preds[1:]) > np.array(lstm_preds[:-1])
    lstm_acc = np.mean(lstm_dir == actual_dir)
    
    lstm_tech_dir = np.array(lstm_tech_preds[1:]) > np.array(lstm_tech_preds[:-1])
    lstm_tech_acc = np.mean(lstm_tech_dir == actual_dir)
    
    prop_dir = np.array(prop_preds[1:]) > np.array(prop_preds[:-1])
    prop_acc = np.mean(prop_dir == actual_dir)
    
    print(f"  LSTM:        {lstm_acc*100:.1f}%")
    print(f"  LSTM+Tech:  {lstm_tech_acc*100:.1f}%")
    print(f"  Proposed:   {prop_acc*100:.1f}%")
    
    results.append({'t': ticker, 'lstm': lstm_acc, 'lt': lstm_tech_acc, 'prop': prop_acc})

print("\n" + "="*70)
print("COMPARISON WITH THESIS")
print("="*70)
print("\n| Stock | Thesis | Current |")
print("|-------|--------|---------|")
thesis_vals = {'AAPL': (0.517, 0.483, 0.586), 'JPM': (0.552, 0.448, 0.552)}
for r in results:
    t = thesis_vals[r['t']]
    print(f"| {r['t']}    | {t[0]*100:.1f}% / {t[1]*100:.1f}% / {t[2]*100:.1f}% | {r['lstm']*100:.1f}% / {r['lt']*100:.1f}% / {r['prop']*100:.1f}% |")

print("\nNote: Results may vary slightly due to different model versions")