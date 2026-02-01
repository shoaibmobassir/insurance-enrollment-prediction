"""
Quick Test Script to Verify Pipeline Components
"""

import sys
import os

# Test imports
print("Testing imports...")
try:
    import pandas as pd
    print("✓ pandas")
except ImportError as e:
    print(f"✗ pandas: {e}")

try:
    import numpy as np
    print("✓ numpy")
except ImportError as e:
    print(f"✗ numpy: {e}")

try:
    from sklearn.model_selection import train_test_split
    print("✓ scikit-learn")
except ImportError as e:
    print(f"✗ scikit-learn: {e}")

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib")
except ImportError as e:
    print(f"✗ matplotlib: {e}")

try:
    import seaborn as sns
    print("✓ seaborn")
except ImportError as e:
    print(f"✗ seaborn: {e}")

# Test data loading
print("\nTesting data loading...")
try:
    df = pd.read_csv('data/employee_data.csv')
    print(f"✓ Data loaded: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Sample:\n{df.head(3)}")
except Exception as e:
    print(f"✗ Data loading failed: {e}")

# Test module imports
print("\nTesting module imports...")
sys.path.insert(0, 'src')

try:
    from data_processing import DataProcessor
    print("✓ DataProcessor imported")
except ImportError as e:
    print(f"✗ DataProcessor: {e}")

try:
    from model_training import ModelTrainer
    print("✓ ModelTrainer imported")
except ImportError as e:
    print(f"✗ ModelTrainer: {e}")

try:
    from model_evaluation import ModelEvaluator
    print("✓ ModelEvaluator imported")
except ImportError as e:
    print(f"✗ ModelEvaluator: {e}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
