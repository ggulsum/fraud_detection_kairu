"""
Ana Fraud Detection Pipeline
Training, inference ve deployment için end-to-end pipeline
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
import logging
import argparse
from datetime import datetime
import warnings

# Local imports
from preprocessing import FeaturePreprocessor, ImbalanceHandler
from .evaluation import FraudEvaluator
try:
    from explainability_clean import ModelExplainer
except ImportError:
    print("⚠️  Explainability module import hatası")
    ModelExplainer = None
from outlier_detection import OutlierDetector

warnings.filterwarnings('ignore')


class FraudDetectionPipeline:
    """End-to-end Fraud Detection Pipeline"""
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Args:
            config_path (str): Configuration dosyası yolu
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_mlflow()
        
        # Pipeline components
        self.preprocessor = None
        self.models = {}
        self.evaluators = {}
        self.explainer = None
        
        # Data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        logger.info("Fraud Detection Pipeline initialized")
    
    def _load_config(self, config_path):
        """Configuration dosyasını yükle"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            # Default config if file not found
            return self._get_default_config()
    
    def _get_default_config(self):
        """Default configuration"""
        return {
            'data': {'test_size': 0.3, 'random_state': 42},
            'preprocessing': {'scaling_method': 'robust', 'encoding_method': 'onehot'},
            'models': {
                'random_forest': {'n_estimators': 100, 'random_state': 42},
                'logistic_regression': {'random_state': 42},
                'isolation_forest': {'contamination': 0.05, 'random_state': 42}
            },
            'evaluation': {'min_roc_auc': 0.7, 'min_pr_auc': 0.3}
        }
    
    def _setup_logging(self):
        """Logging setup"""
        global logger
        
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(level=log_level, format=log_format)
        logger = logging.getLogger(__name__)
        
        # File logging
        if log_config.get('file_logging', False):
            os.makedirs('logs', exist_ok=True)
            file_handler = logging.FileHandler(log_config.get('log_file', 'logs/pipeline.log'))
            file_handler.setFormatter(logging.Formatter(log_format))
            logger.addHandler(file_handler)
    
    def _setup_mlflow(self):
        """MLflow setup"""
        mlflow_config = self.config.get('mlflow', {})
        
        # Set tracking URI
        tracking_uri = mlflow_config.get('tracking_uri', 'sqlite:///mlflow.db')
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        experiment_name = mlflow_config.get('experiment_name', 'fraud_detection')
        mlflow.set_experiment(experiment_name)
        
        # Auto-logging
        if mlflow_config.get('autolog', {}).get('sklearn', True):
            mlflow.sklearn.autolog()
        
        logger.info(f"MLflow configured - Experiment: {experiment_name}")
    
    def load_data(self, data_path=None, synthetic=True, download_with_kagglehub=False):
        """
        Veri yükleme
        
        Args:
            data_path (str): Veri dosyası yolu
            synthetic (bool): Synthetic data kullan
            download_with_kagglehub (bool): KaggleHub ile otomatik indirme
        """
        if download_with_kagglehub:
            logger.info("KaggleHub ile Credit Card Fraud dataset indiriliyor...")
            try:
                import kagglehub
                path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
                csv_file = os.path.join(path, "creditcard.csv")
                if os.path.exists(csv_file):
                    data = pd.read_csv(csv_file)
                    logger.info(f"KaggleHub dataset yüklendi: {data.shape}")
                else:
                    logger.warning("KaggleHub'dan CSV bulunamadı, synthetic data kullanılıyor")
                    data = self._generate_synthetic_data()
            except Exception as e:
                logger.error(f"KaggleHub indirme hatası: {e}, synthetic data kullanılıyor")
                data = self._generate_synthetic_data()
        elif synthetic or data_path is None:
            logger.info("Synthetic fraud data oluşturuluyor...")
            data = self._generate_synthetic_data()
        else:
            logger.info(f"Veri yükleniyor: {data_path}")
            data = pd.read_csv(data_path)
        
        # Data validation
        self._validate_data(data)
        
        # Train-test split
        X = data.drop('Class', axis=1)
        y = data['Class']
        
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y if self.config['data'].get('stratify', True) else None
        )
        
        logger.info(f"Data loaded - Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        logger.info(f"Class distribution - Train: {np.bincount(self.y_train)}, Test: {np.bincount(self.y_test)}")
    
    def _generate_synthetic_data(self, n_samples=5000):
        """Synthetic fraud data oluştur"""
        np.random.seed(self.config['data']['random_state'])
        
        # Normal transactions (95%)
        n_normal = int(n_samples * 0.95)
        normal_data = {
            'Amount': np.random.lognormal(2, 1, n_normal),
            'Time': np.random.randint(0, 86400, n_normal),
            'Merchant_Category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail'], n_normal),
            'Customer_Age': np.random.randint(18, 80, n_normal),
            'Is_Weekend': np.random.choice([0, 1], n_normal, p=[0.7, 0.3]),
            'Transaction_Count_Day': np.random.poisson(3, n_normal),
            'Class': np.zeros(n_normal)
        }
        
        # Fraud transactions (5%)
        n_fraud = n_samples - n_normal
        fraud_data = {
            'Amount': np.random.lognormal(4, 2, n_fraud),  # Higher amounts
            'Time': np.random.randint(0, 86400, n_fraud),
            'Merchant_Category': np.random.choice(['online', 'atm', 'international'], n_fraud),
            'Customer_Age': np.random.randint(25, 60, n_fraud),
            'Is_Weekend': np.random.choice([0, 1], n_fraud, p=[0.4, 0.6]),  # More weekend fraud
            'Transaction_Count_Day': np.random.poisson(8, n_fraud),  # More transactions
            'Class': np.ones(n_fraud)
        }
        
        # Combine
        data = {}
        for key in normal_data.keys():
            data[key] = np.concatenate([normal_data[key], fraud_data[key]])
        
        df = pd.DataFrame(data)
        
        # Add some missing values
        missing_indices = np.random.choice(df.index, int(len(df) * 0.02))
        df.loc[missing_indices, 'Customer_Age'] = np.nan
        
        return df
    
    def _validate_data(self, data):
        """Data validation"""
        validation_config = self.config.get('data', {}).get('validation', {})
        
        # Required columns
        required_cols = validation_config.get('required_columns', ['Amount', 'Time', 'Class'])
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Class values
        if 'Class' in data.columns:
            valid_classes = validation_config.get('class_values', [0, 1])
            invalid_classes = data['Class'].unique()
            invalid_classes = [c for c in invalid_classes if c not in valid_classes]
            if invalid_classes:
                raise ValueError(f"Invalid class values: {invalid_classes}")
        
        # Amount validation
        if 'Amount' in data.columns:
            min_amount = validation_config.get('amount_min', 0)
            max_amount = validation_config.get('amount_max', 1000000)
            if data['Amount'].min() < min_amount or data['Amount'].max() > max_amount:
                logger.warning(f"Amount values outside expected range [{min_amount}, {max_amount}]")
        
        logger.info("Data validation completed")
    
    def preprocess_data(self):
        """Data preprocessing"""
        logger.info("Data preprocessing başlatılıyor...")
        
        preprocessing_config = self.config.get('preprocessing', {})
        
        # Initialize preprocessor
        self.preprocessor = FeaturePreprocessor(
            scaling_method=preprocessing_config.get('scaling_method', 'robust'),
            encoding_method=preprocessing_config.get('encoding_method', 'onehot')
        )
        
        # Preprocess training data
        train_data = pd.concat([self.X_train, self.y_train], axis=1)
        train_processed = self.preprocessor.fit_transform(train_data, target_col='Class')
        
        self.X_train_processed = train_processed.drop('Class', axis=1)
        self.y_train_processed = train_processed['Class']
        
        # Preprocess test data
        test_data = pd.concat([self.X_test, self.y_test], axis=1)
        test_processed = self.preprocessor.transform(test_data, target_col='Class')
        
        self.X_test_processed = test_processed.drop('Class', axis=1)
        self.y_test_processed = test_processed['Class']
        
        logger.info(f"Preprocessing completed - Features: {self.X_train_processed.shape[1]}")
        
        # Handle class imbalance
        imbalance_config = self.config.get('imbalance', {})
        method = imbalance_config.get('method', 'smote')
        
        if method == 'smote':
            self.X_train_balanced, self.y_train_balanced = ImbalanceHandler.apply_smote(
                self.X_train_processed, self.y_train_processed,
                sampling_strategy=imbalance_config.get('sampling_strategy', 'auto'),
                random_state=imbalance_config.get('random_state', 42)
            )
        elif method == 'adasyn':
            self.X_train_balanced, self.y_train_balanced = ImbalanceHandler.apply_adasyn(
                self.X_train_processed, self.y_train_processed
            )
        else:
            self.X_train_balanced = self.X_train_processed
            self.y_train_balanced = self.y_train_processed
        
        logger.info(f"Class balancing completed - Final training size: {len(self.X_train_balanced)}")
    
    def train_models(self):
        """Model training"""
        logger.info("Model training başlatılıyor...")
        
        models_config = self.config.get('models', {})
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                'data_size': len(self.X_train_balanced),
                'n_features': self.X_train_balanced.shape[1],
                'preprocessing_method': self.config.get('preprocessing', {}).get('scaling_method', 'robust')
            })
            
            for model_name, model_params in models_config.items():
                logger.info(f"Training {model_name}...")
                
                # Initialize model
                if model_name == 'random_forest':
                    model = RandomForestClassifier(**model_params)
                elif model_name == 'logistic_regression':
                    model = LogisticRegression(**model_params)
                elif model_name == 'isolation_forest':
                    model = IsolationForest(**model_params)
                elif model_name == 'lof':
                    model = LocalOutlierFactor(**model_params)
                else:
                    logger.warning(f"Unknown model: {model_name}")
                    continue
                
                # Train model
                if model_name in ['isolation_forest', 'lof']:
                    # Unsupervised models
                    model.fit(self.X_train_balanced)
                else:
                    # Supervised models
                    model.fit(self.X_train_balanced, self.y_train_balanced)
                
                self.models[model_name] = model
                
                # Cross validation for supervised models
                if model_name not in ['isolation_forest', 'lof']:
                    cv_scores = cross_val_score(
                        model, self.X_train_balanced, self.y_train_balanced,
                        cv=5, scoring='roc_auc'
                    )
                    mlflow.log_metric(f"{model_name}_cv_roc_auc_mean", cv_scores.mean())
                    mlflow.log_metric(f"{model_name}_cv_roc_auc_std", cv_scores.std())
                    logger.info(f"{model_name} CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                
                logger.info(f"{model_name} training completed")
            
            logger.info("All models trained successfully")
    
    def evaluate_models(self):
        """Model evaluation"""
        logger.info("Model evaluation başlatılıyor...")
        
        evaluation_config = self.config.get('evaluation', {})
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}...")
            
            # Initialize evaluator
            evaluator = FraudEvaluator(model, model_name)
            
            # Evaluate
            if model_name in ['isolation_forest', 'lof']:
                # Outlier detection evaluation
                detector = OutlierDetector()
                detector.scaler = self.preprocessor.scaler
                
                if model_name == 'isolation_forest':
                    detector.isolation_forest = model
                    predictions = detector.predict_isolation_forest(self.X_test_processed)
                else:
                    detector.lof = model
                    predictions = detector.predict_lof(self.X_test_processed)
                
                # Create dummy probabilities for evaluation
                y_pred_proba = predictions.astype(float)
                results = evaluator.evaluate_binary_classification(
                    self.X_test_processed, self.y_test_processed,
                    y_pred_proba=y_pred_proba, y_pred=predictions
                )
            else:
                # Supervised model evaluation
                results = evaluator.evaluate_binary_classification(
                    self.X_test_processed, self.y_test_processed
                )
            
            self.evaluators[model_name] = evaluator
            
            # Log metrics to MLflow
            with mlflow.start_run(nested=True):
                mlflow.log_params(model.get_params() if hasattr(model, 'get_params') else {})
                mlflow.log_metrics({
                    f"{model_name}_roc_auc": results['roc_auc'],
                    f"{model_name}_pr_auc": results['pr_auc'],
                    f"{model_name}_f1_score": results['f1_score'],
                    f"{model_name}_precision": results['precision'],
                    f"{model_name}_recall": results['recall']
                })
                
                # Log model
                if model_name not in ['isolation_forest', 'lof']:
                    mlflow.sklearn.log_model(model, f"{model_name}_model")
            
            # Print evaluation report
            evaluator.print_evaluation_report()
            
            # Check performance thresholds
            min_roc_auc = evaluation_config.get('min_roc_auc', 0.7)
            min_pr_auc = evaluation_config.get('min_pr_auc', 0.3)
            
            if results['roc_auc'] < min_roc_auc:
                logger.warning(f"{model_name} ROC-AUC ({results['roc_auc']:.4f}) below threshold ({min_roc_auc})")
            
            if results['pr_auc'] < min_pr_auc:
                logger.warning(f"{model_name} PR-AUC ({results['pr_auc']:.4f}) below threshold ({min_pr_auc})")
        
        logger.info("Model evaluation completed")
    
    def explain_models(self, model_name='random_forest'):
        """Model explainability"""
        if ModelExplainer is None:
            logger.warning("ModelExplainer mevcut değil - explainability atlanıyor")
            return None, None
            
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return
        
        logger.info(f"Explaining {model_name}...")
        
        # Initialize explainer
        self.explainer = ModelExplainer(
            self.models[model_name],
            self.X_train_balanced,
            feature_names=list(self.X_train_processed.columns),
            class_names=['Normal', 'Fraud']
        )
        
        # SHAP analysis
        explainer_config = self.config.get('explainability', {}).get('shap', {})
        self.explainer.initialize_shap(
            explainer_type=explainer_config.get('explainer_type', 'auto'),
            max_evals=explainer_config.get('max_evals', 100)
        )
        
        shap_values, X_sample = self.explainer.compute_shap_values(
            self.X_test_processed,
            max_samples=explainer_config.get('max_samples', 100)
        )
        
        # Generate plots
        self.explainer.plot_shap_summary(X_sample)
        self.explainer.plot_shap_waterfall(X_sample, 0)
        
        # Global feature importance
        importance = self.explainer.global_feature_importance(X_sample)
        
        # Fraud pattern analysis
        fraud_patterns = self.explainer.analyze_fraud_patterns(
            X_sample, self.y_test_processed[:len(X_sample)]
        )
        
        logger.info("Model explanation completed")
        return importance, fraud_patterns
    
    def save_models(self, save_path="models/"):
        """Model ve preprocessor kaydetme"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save preprocessor
        joblib.dump(self.preprocessor, os.path.join(save_path, 'preprocessor.pkl'))
        logger.info("Preprocessor saved")
        
        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(save_path, f'{model_name}_model.pkl')
            joblib.dump(model, model_path)
            logger.info(f"{model_name} model saved to {model_path}")
        
        # Save feature names
        feature_info = {
            'feature_names': list(self.X_train_processed.columns),
            'n_features': len(self.X_train_processed.columns),
            'preprocessing_config': self.config.get('preprocessing', {})
        }
        joblib.dump(feature_info, os.path.join(save_path, 'feature_info.pkl'))
        
        logger.info(f"All models saved to {save_path}")
    
    def load_models(self, load_path="models/"):
        """Model ve preprocessor yükleme"""
        try:
            # Load preprocessor
            self.preprocessor = joblib.load(os.path.join(load_path, 'preprocessor.pkl'))
            logger.info("Preprocessor loaded")
            
            # Load models
            model_files = [f for f in os.listdir(load_path) if f.endswith('_model.pkl')]
            for model_file in model_files:
                model_name = model_file.replace('_model.pkl', '')
                model_path = os.path.join(load_path, model_file)
                self.models[model_name] = joblib.load(model_path)
                logger.info(f"{model_name} model loaded")
            
            # Load feature info
            feature_info = joblib.load(os.path.join(load_path, 'feature_info.pkl'))
            logger.info(f"Feature info loaded - {feature_info['n_features']} features")
            
            logger.info(f"All models loaded from {load_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict(self, data, model_name='random_forest'):
        """Prediction"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded")
        
        # Preprocess data
        data_processed = self.preprocessor.transform(data)
        
        # Make prediction
        model = self.models[model_name]
        
        if model_name in ['isolation_forest', 'lof']:
            # Outlier detection
            predictions = model.predict(data_processed)
            probabilities = np.where(predictions == -1, 0.8, 0.2)  # Convert to pseudo-probabilities
            return predictions, probabilities
        else:
            # Supervised model
            predictions = model.predict(data_processed)
            probabilities = model.predict_proba(data_processed)[:, 1]
            return predictions, probabilities
    
    def run_full_pipeline(self, data_path=None, save_models=True, use_kagglehub=False):
        """Full pipeline execution"""
        logger.info("Full Fraud Detection Pipeline başlatılıyor...")
        
        try:
            # 1. Load data
            self.load_data(data_path, synthetic=(data_path is None and not use_kagglehub), 
                          download_with_kagglehub=use_kagglehub)
            
            # 2. Preprocess
            self.preprocess_data()
            
            # 3. Train models
            self.train_models()
            
            # 4. Evaluate models
            self.evaluate_models()
            
            # 5. Explain best model
            best_model = self._find_best_model()
            self.explain_models(best_model)
            
            # 6. Save models
            if save_models:
                self.save_models()
            
            logger.info("Full pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False
    
    def _find_best_model(self):
        """En iyi modeli bul (ROC-AUC'ye göre)"""
        best_model = None
        best_score = 0
        
        for model_name, evaluator in self.evaluators.items():
            if evaluator.results and 'roc_auc' in evaluator.results:
                roc_auc = evaluator.results['roc_auc']
                if roc_auc > best_score:
                    best_score = roc_auc
                    best_model = model_name
        
        logger.info(f"Best model: {best_model} (ROC-AUC: {best_score:.4f})")
        return best_model or 'random_forest'


def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(description='Fraud Detection Pipeline')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--data', help='Data file path (optional, uses synthetic if not provided)')
    parser.add_argument('--mode', choices=['train', 'predict', 'explain'], default='train', help='Pipeline mode')
    parser.add_argument('--model', default='random_forest', help='Model name for prediction/explanation')
    parser.add_argument('--load_models', action='store_true', help='Load existing models')
    parser.add_argument('--save_models', action='store_true', help='Save trained models')
    parser.add_argument('--use_kagglehub', action='store_true', help='Download data with KaggleHub')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FraudDetectionPipeline(args.config)
    
    if args.mode == 'train':
        # Training mode
        success = pipeline.run_full_pipeline(args.data, args.save_models, args.use_kagglehub)
        sys.exit(0 if success else 1)
        
    elif args.mode == 'predict':
        # Prediction mode
        if args.load_models:
            pipeline.load_models()
        
        # Demo prediction with synthetic data
        pipeline.load_data(synthetic=True)
        pipeline.preprocess_data()
        
        predictions, probabilities = pipeline.predict(
            pipeline.X_test_processed.head(), args.model
        )
        
        print("Sample Predictions:")
        for i, (pred, prob) in enumerate(zip(predictions[:5], probabilities[:5])):
            print(f"Sample {i}: Prediction={pred}, Probability={prob:.4f}")
    
    elif args.mode == 'explain':
        # Explanation mode
        if args.load_models:
            pipeline.load_models()
        
        # Load data and explain
        pipeline.load_data(synthetic=True)
        pipeline.preprocess_data()
        
        importance, patterns = pipeline.explain_models(args.model)
        
        print("Top 10 Important Features:")
        for i, (feature, score) in enumerate(list(importance.items())[:10]):
            print(f"{i+1}. {feature}: {score:.4f}")


if __name__ == "__main__":
    main()