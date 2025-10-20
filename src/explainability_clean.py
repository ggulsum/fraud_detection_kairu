"""
Model Explainability - Clean Version
SHAP ve LIME için temiz implementasyon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import logging

# Safe imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExplainer:
    """Simple Model Explainer"""
    
    def __init__(self, model, X_train, feature_names=None, class_names=None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        self.class_names = class_names or ['Normal', 'Fraud']
        self.shap_explainer = None
        self.shap_values = None
        self.lime_explainer = None
        
    def initialize_shap(self, explainer_type='tree'):
        """SHAP explainer başlat"""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP mevcut değil")
            return False
            
        try:
            if explainer_type == 'tree':
                self.shap_explainer = shap.TreeExplainer(self.model)
            else:
                # Fallback to simple approach
                background = shap.sample(self.X_train, min(50, len(self.X_train)))
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba, background
                )
            logger.info("SHAP explainer hazır")
            return True
        except Exception as e:
            logger.error(f"SHAP hatası: {e}")
            return False
    
    def compute_shap_values(self, X_test, max_samples=50):
        """SHAP values hesapla"""
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return None, X_test
            
        # Limit samples
        X_sample = X_test.head(max_samples) if hasattr(X_test, 'head') else X_test[:max_samples]
        
        try:
            self.shap_values = self.shap_explainer.shap_values(X_sample)
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]  # Fraud class
            return self.shap_values, X_sample
        except Exception as e:
            logger.error(f"SHAP computation error: {e}")
            return None, X_sample
    
    def plot_shap_summary(self, X_test=None):
        """SHAP summary plot"""
        if not SHAP_AVAILABLE or self.shap_values is None:
            logger.warning("SHAP values mevcut değil")
            return
            
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values, X_test,
            feature_names=self.feature_names,
            plot_type="bar",
            show=False
        )
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.show()
    
    def initialize_lime(self):
        """LIME explainer başlat"""
        if not LIME_AVAILABLE:
            logger.warning("LIME mevcut değil")
            return False
            
        try:
            self.lime_explainer = LimeTabularExplainer(
                self.X_train.values if hasattr(self.X_train, 'values') else self.X_train,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode='classification'
            )
            logger.info("LIME explainer hazır")
            return True
        except Exception as e:
            logger.error(f"LIME hatası: {e}")
            return False
    
    def explain_instance_lime(self, X_test, instance_idx=0):
        """LIME ile instance açıklama"""
        if not LIME_AVAILABLE or self.lime_explainer is None:
            logger.warning("LIME mevcut değil")
            return None
            
        instance = X_test.iloc[instance_idx] if hasattr(X_test, 'iloc') else X_test[instance_idx]
        
        try:
            explanation = self.lime_explainer.explain_instance(
                instance.values if hasattr(instance, 'values') else instance,
                self.model.predict_proba,
                num_features=10
            )
            
            # Plot
            fig = explanation.as_pyplot_figure(label=1)
            plt.title(f'LIME Explanation - Instance {instance_idx}')
            plt.tight_layout()
            plt.show()
            
            return explanation
        except Exception as e:
            logger.error(f"LIME explanation error: {e}")
            return None
    
    def permutation_importance(self, X_test, y_test):
        """Permutation feature importance"""
        try:
            perm_importance = permutation_importance(
                self.model, X_test, y_test, 
                n_repeats=5, random_state=42
            )
            
            # Plot
            sorted_idx = np.argsort(perm_importance.importances_mean)[-10:]
            
            plt.figure(figsize=(10, 6))
            plt.barh(
                [self.feature_names[i] for i in sorted_idx],
                perm_importance.importances_mean[sorted_idx]
            )
            plt.xlabel('Permutation Importance')
            plt.title('Feature Importance (Permutation-based)')
            plt.tight_layout()
            plt.show()
            
            return perm_importance
        except Exception as e:
            logger.error(f"Permutation importance error: {e}")
            return None


def demo_explainability():
    """Clean explainability demo"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    logger.info("Explainability demo başlatılıyor...")
    
    # Create dataset
    X, y = make_classification(
        n_samples=500, n_features=8, n_informative=6,
        weights=[0.9, 0.1], random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(8)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Model accuracy: {model.score(X_test, y_test):.3f}")
    
    # Create explainer
    explainer = ModelExplainer(
        model, X_train, feature_names, ['Normal', 'Fraud']
    )
    
    # Try SHAP
    if explainer.initialize_shap():
        shap_values, X_sample = explainer.compute_shap_values(X_test)
        if shap_values is not None:
            explainer.plot_shap_summary(X_sample)
    
    # Try LIME
    if explainer.initialize_lime():
        explainer.explain_instance_lime(X_test, 0)
    
    # Permutation importance (always works)
    explainer.permutation_importance(X_test, y_test)
    
    print("✅ Explainability demo tamamlandı!")
    return explainer


if __name__ == "__main__":
    demo_explainability()