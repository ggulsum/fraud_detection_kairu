"""
Fraud Detection Model Evaluation
Basit evaluation sınıfı
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)


class FraudEvaluator:
    """Model evaluation için basit sınıf"""
    
    def __init__(self, model=None, model_name="Model"):
        self.model = model
        self.model_name = model_name
        self.results = {}
    
    def evaluate_binary_classification(self, X_test, y_true, y_pred_proba=None):
        """Binary classification evaluation"""
        
        if y_pred_proba is None and self.model is not None:
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Binary predictions (threshold 0.5)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        results = {
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'pr_auc': average_precision_score(y_true, y_pred_proba),
            'f1_score': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        self.results = results
        return results