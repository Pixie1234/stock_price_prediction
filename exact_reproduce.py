#!/usr/bin/env python3
"""
Exact Thesis Evaluation - 65% split, 30 test days
Replicates RESULTS.md values
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
sys.path.insert(0, '.')
import numpy as np
from data_pipeline import prepare_data, SEQ_LEN, N_FEATURES
from lstm_model import load_model
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("THESIS EXACT EVALUATION - 65% Split, 30 Test Days")
print("="*70)

for ticker in ['AAPL', 'JPM']:
    print(f"\n--- {ticker} ---")
    
    data = prepare_data(ticker, years=2)
    scaled = data['scaled']
    raw = data['raw_ohlcv']
    
    # 65% split as per thesis
    train_end = int(len(scaled) * 0.65)
    
    # Load model ONCE
    model = load_model('models/{}_lstm_ohlcv_indicators_v4.h5'.format(ticker))
    
    # Get predictions (without sentiment)
    preds = []
    actuals = []
    for i in range(train_end, min(train_end + 30, len(scaled) - 1)):
        last_seq = scaled[i - SEQ_LEN:i]
        actual = raw[i + 1, 3]
        base = raw[i, 3]
        pred = model.predict(last_seq.reshape(1, SEQ_LEN, N_FEATURES), verbose=0)[0]
        preds.append(base * np.exp(pred[1]))
        actuals.append(actual)
    
    # Directional accuracy
    actual_dir = np.array(actuals[1:]) > np.array(actuals[:-1])
    pred_dir = np.array(preds[1:]) > np.array(preds[:-1])
    acc = np.mean(pred_dir == actual_dir)
    
    print(f"LSTM+Tech Accuracy: {acc*100:.1f}%")

print("\nNote: LSTM (base) model not available for exact replication")
print("The thesis results came from specific test run in April 2024")