#!/usr/bin/env python3
"""
Kaggle Dataset Evaluation
Run this separately to test with Kaggle historical news data
Does not modify any existing files
"""
import os
import sys

# Add project to path
sys.path.insert(0, '/home/anastasija/Diploma Thesis/ResearchPrediction/PythonProject')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['KAGGLE_USERNAME'] = os.environ.get('KAGGLE_USERNAME', '')
os.environ['KAGGLE_KEY'] = os.environ.get('KAGGLE_KEY', '')

print("="*60)
print("KAGGLE DATASET EVALUATION")
print("="*60)
print("""
To download Kaggle datasets:
1. Go to https://www.kaggle.com/account
2. Create API token (will download kaggle.json)
3. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables

Or manually download from:
- https://www.kaggle.com/datasets/yash612/stockmarket-sentiment-dataset
- https://www.kaggle.com/datasets/willbert0/trainset-nysedat
""")

# Check if kaggle is available
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("Kaggle API authenticated!")
except Exception as e:
    print(f"Kaggle API not configured: {e}")
    print("Please download datasets manually from Kaggle website")