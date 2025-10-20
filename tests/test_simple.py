"""
Basit CI/CD test dosyası
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import FeaturePreprocessor, ImbalanceHandler


def test_data_validation():
    """Test data validation"""
    # Synthetic data ile test
    np.random.seed(42)
    data = {
        'Amount': np.random.lognormal(2, 1, 100),
        'Time': np.random.randint(0, 86400, 100),
        'Class': np.random.choice([0, 1], 100, p=[0.95, 0.05])
    }
    df = pd.DataFrame(data)
    
    # Schema validation
    required_columns = ['Amount', 'Time', 'Class']
    assert all(col in df.columns for col in required_columns), 'Missing required columns'
    
    # Data type validation
    assert df['Amount'].dtype in ['int64', 'float64'], 'Amount must be numeric'
    assert df['Class'].dtype in ['int64', 'float64'], 'Class must be numeric'
    
    # Value range validation
    assert df['Amount'].min() >= 0, 'Amount cannot be negative'
    assert df['Class'].isin([0, 1]).all(), 'Class must be 0 or 1'
    
    print('✅ Data validation passed')


def test_preprocessing():
    """Test basic preprocessing"""
    # Sample data
    data = pd.DataFrame({
        'numerical_col': [1, 2, 3, 4, 5],
        'categorical_col': ['A', 'B', 'A', 'C', 'B'],
        'target': [0, 1, 0, 1, 0]
    })
    
    preprocessor = FeaturePreprocessor()
    processed_data = preprocessor.fit_transform(data, target_col='target')
    
    assert preprocessor.is_fitted
    assert processed_data is not None
    assert len(processed_data) == len(data)
    print('✅ Preprocessing test passed')


def test_imbalance_handling():
    """Test SMOTE"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.array([0] * 90 + [1] * 10)  # 90% class 0, 10% class 1
    
    X_resampled, y_resampled = ImbalanceHandler.apply_smote(X, y)
    
    # Check that minority class is upsampled
    assert len(X_resampled) > len(X)
    assert len(y_resampled) > len(y)
    
    # Check class balance
    unique, counts = np.unique(y_resampled, return_counts=True)
    assert len(unique) == 2
    assert counts[0] == counts[1]  # Should be balanced
    print('✅ SMOTE test passed')


if __name__ == "__main__":
    test_data_validation()
    test_preprocessing() 
    test_imbalance_handling()
    print('✅ All simple tests passed!')