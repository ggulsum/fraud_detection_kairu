"""
Fraud Detection Pipeline Test Suite
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import FraudDetectionPipeline
from preprocessing import FeaturePreprocessor, ImbalanceHandler


class TestFraudDetectionPipeline:
    """Pipeline test class"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample test data"""
        np.random.seed(42)
        data = {
            'Amount': np.random.lognormal(2, 1, 100),
            'Time': np.random.randint(0, 86400, 100),
            'Merchant_Category': np.random.choice(['grocery', 'gas'], 100),
            'Class': np.random.choice([0, 1], 100, p=[0.9, 0.1])
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def pipeline(self):
        """Pipeline instance"""
        return FraudDetectionPipeline()
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization"""
        assert pipeline.config is not None
        assert pipeline.preprocessor is None
        assert pipeline.models == {}
        assert pipeline.evaluators == {}
    
    def test_synthetic_data_generation(self, pipeline):
        """Test synthetic data generation"""
        data = pipeline._generate_synthetic_data(1000)
        
        # Check data structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 1000
        assert 'Class' in data.columns
        assert 'Amount' in data.columns
        
        # Check class distribution
        class_counts = data['Class'].value_counts()
        assert len(class_counts) == 2
        assert class_counts[0] > class_counts[1]  # More normal than fraud
    
    def test_data_validation(self, pipeline, sample_data):
        """Test data validation"""
        # Valid data should pass
        pipeline._validate_data(sample_data)
        
        # Invalid data should raise error
        invalid_data = sample_data.copy()
        invalid_data['Class'] = invalid_data['Class'] + 2  # Invalid class values
        
        with pytest.raises(ValueError):
            pipeline._validate_data(invalid_data)
    
    def test_data_loading(self, pipeline):
        """Test data loading"""
        pipeline.load_data(synthetic=True)
        
        assert pipeline.X_train is not None
        assert pipeline.X_test is not None
        assert pipeline.y_train is not None
        assert pipeline.y_test is not None
        
        # Check train/test split
        total_size = len(pipeline.X_train) + len(pipeline.X_test)
        test_ratio = len(pipeline.X_test) / total_size
        assert 0.2 <= test_ratio <= 0.4  # Roughly 30% test size
    
    def test_preprocessing(self, pipeline):
        """Test preprocessing"""
        pipeline.load_data(synthetic=True)
        pipeline.preprocess_data()
        
        assert pipeline.preprocessor is not None
        assert pipeline.X_train_processed is not None
        assert pipeline.X_test_processed is not None
        assert pipeline.X_train_balanced is not None
        assert pipeline.y_train_balanced is not None
        
        # Check that balanced data has more samples (due to SMOTE)
        assert len(pipeline.X_train_balanced) >= len(pipeline.X_train_processed)
    
    def test_model_training(self, pipeline):
        """Test model training"""
        pipeline.load_data(synthetic=True)
        pipeline.preprocess_data()
        
        with patch('mlflow.start_run'):
            with patch('mlflow.log_params'):
                with patch('mlflow.log_metric'):
                    pipeline.train_models()
        
        # Check that models are trained
        assert len(pipeline.models) > 0
        assert 'random_forest' in pipeline.models
        assert 'logistic_regression' in pipeline.models
    
    def test_model_evaluation(self, pipeline):
        """Test model evaluation"""
        pipeline.load_data(synthetic=True)
        pipeline.preprocess_data()
        
        with patch('mlflow.start_run'):
            with patch('mlflow.log_params'):
                with patch('mlflow.log_metric'):
                    pipeline.train_models()
                    pipeline.evaluate_models()
        
        # Check that evaluators are created
        assert len(pipeline.evaluators) > 0
        for evaluator in pipeline.evaluators.values():
            assert evaluator.results is not None
            assert 'roc_auc' in evaluator.results
            assert 'pr_auc' in evaluator.results
    
    def test_model_prediction(self, pipeline):
        """Test model prediction"""
        pipeline.load_data(synthetic=True)
        pipeline.preprocess_data()
        
        with patch('mlflow.start_run'):
            with patch('mlflow.log_params'):
                with patch('mlflow.log_metric'):
                    pipeline.train_models()
        
        # Test prediction
        test_sample = pipeline.X_test.head(5)
        predictions, probabilities = pipeline.predict(test_sample, 'random_forest')
        
        assert len(predictions) == 5
        assert len(probabilities) == 5
        assert all(pred in [0, 1] for pred in predictions)
        assert all(0 <= prob <= 1 for prob in probabilities)
    
    def test_model_save_load(self, pipeline, tmp_path):
        """Test model saving and loading"""
        pipeline.load_data(synthetic=True)
        pipeline.preprocess_data()
        
        with patch('mlflow.start_run'):
            with patch('mlflow.log_params'):
                with patch('mlflow.log_metric'):
                    pipeline.train_models()
        
        # Save models
        save_path = str(tmp_path / "models")
        pipeline.save_models(save_path)
        
        # Check files are created
        assert os.path.exists(os.path.join(save_path, 'preprocessor.pkl'))
        assert os.path.exists(os.path.join(save_path, 'random_forest_model.pkl'))
        
        # Load models in new pipeline
        new_pipeline = FraudDetectionPipeline()
        success = new_pipeline.load_models(save_path)
        assert success
        assert new_pipeline.preprocessor is not None
        assert len(new_pipeline.models) > 0


class TestFeaturePreprocessor:
    """Feature preprocessor tests"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for preprocessing"""
        return pd.DataFrame({
            'numerical_col': [1, 2, 3, 4, 5],
            'categorical_col': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = FeaturePreprocessor()
        assert preprocessor.scaling_method == 'standard'
        assert preprocessor.encoding_method == 'onehot'
        assert not preprocessor.is_fitted
    
    def test_feature_identification(self, sample_data):
        """Test feature type identification"""
        preprocessor = FeaturePreprocessor()
        preprocessor.identify_features(sample_data)
        
        assert 'numerical_col' in preprocessor.numerical_features
        assert 'categorical_col' in preprocessor.categorical_features
        assert 'target' in preprocessor.numerical_features  # Treated as numerical
    
    def test_fit_transform(self, sample_data):
        """Test fit and transform"""
        preprocessor = FeaturePreprocessor()
        processed_data = preprocessor.fit_transform(sample_data, target_col='target')
        
        assert preprocessor.is_fitted
        assert processed_data is not None
        assert len(processed_data) == len(sample_data)
    
    def test_transform_only(self, sample_data):
        """Test transform after fit"""
        preprocessor = FeaturePreprocessor()
        
        # Fit first
        preprocessor.fit_transform(sample_data, target_col='target')
        
        # Then transform new data
        new_data = sample_data.copy()
        transformed = preprocessor.transform(new_data, target_col='target')
        
        assert transformed is not None
        assert len(transformed) == len(new_data)


class TestImbalanceHandler:
    """Imbalance handler tests"""
    
    @pytest.fixture
    def imbalanced_data(self):
        """Imbalanced dataset"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.array([0] * 90 + [1] * 10)  # 90% class 0, 10% class 1
        return X, y
    
    def test_analyze_imbalance(self, imbalanced_data):
        """Test imbalance analysis"""
        X, y = imbalanced_data
        
        # Should not raise an error
        info = ImbalanceHandler.analyze_imbalance(y, target_names=['Normal', 'Fraud'])
        
        assert 'classes' in info
        assert 'counts' in info
        assert 'imbalance_ratio' in info
        assert info['imbalance_ratio'] > 1
    
    def test_smote_application(self, imbalanced_data):
        """Test SMOTE application"""
        X, y = imbalanced_data
        
        X_resampled, y_resampled = ImbalanceHandler.apply_smote(X, y)
        
        # Check that minority class is upsampled
        assert len(X_resampled) > len(X)
        assert len(y_resampled) > len(y)
        
        # Check class balance
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert len(unique) == 2
        assert counts[0] == counts[1]  # Should be balanced




if __name__ == "__main__":
    pytest.main([__file__, "-v"])