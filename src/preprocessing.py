"""
Feature Scaling ve Encoding utilities
Fraud detection için veri ön işleme araçları
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeaturePreprocessor:
    """Fraud Detection için kapsamlı feature preprocessing sınıfı"""
    
    def __init__(self, scaling_method='standard', encoding_method='onehot'):
        """
        Args:
            scaling_method (str): 'standard', 'robust', 'minmax'
            encoding_method (str): 'onehot', 'label', 'ordinal'
        """
        self.scaling_method = scaling_method
        self.encoding_method = encoding_method
        
        # Scaler seçimi
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaling_method: 'standard', 'robust', veya 'minmax'")
        
        # Encoder seçimi
        if encoding_method == 'onehot':
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        elif encoding_method == 'label':
            self.encoder = LabelEncoder()
        elif encoding_method == 'ordinal':
            self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', 
                                        unknown_value=-1)
        else:
            raise ValueError("encoding_method: 'onehot', 'label', veya 'ordinal'")
        
        self.numerical_features = []
        self.categorical_features = []
        self.encoded_feature_names = []
        self.is_fitted = False
        
    def identify_features(self, df):
        """
        Numerical ve categorical featureları otomatik tespit et
        
        Args:
            df (pd.DataFrame): Input dataframe
        """
        # Numerical features
        self.numerical_features = df.select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()
        
        # Categorical features
        self.categorical_features = df.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        logger.info(f"Numerical features: {len(self.numerical_features)}")
        logger.info(f"Categorical features: {len(self.categorical_features)}")
        
    def handle_missing_values(self, df, strategy='mean', categorical_strategy='most_frequent'):
        """
        Eksik değerleri doldur
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategy (str): Numerical için strateji
            categorical_strategy (str): Categorical için strateji
            
        Returns:
            pd.DataFrame: İşlenmiş dataframe
        """
        df_processed = df.copy()
        
        # Numerical missing values
        if self.numerical_features:
            num_imputer = SimpleImputer(strategy=strategy)
            df_processed[self.numerical_features] = num_imputer.fit_transform(
                df_processed[self.numerical_features]
            )
        
        # Categorical missing values
        if self.categorical_features:
            cat_imputer = SimpleImputer(strategy=categorical_strategy)
            df_processed[self.categorical_features] = cat_imputer.fit_transform(
                df_processed[self.categorical_features]
            )
        
        logger.info("Eksik değerler dolduruldu")
        return df_processed
    
    def detect_outliers(self, df, method='iqr', threshold=1.5):
        """
        Outlier detection
        
        Args:
            df (pd.DataFrame): Input dataframe
            method (str): 'iqr' veya 'zscore'
            threshold (float): Threshold değeri
            
        Returns:
            pd.DataFrame: Outlier information
        """
        outlier_info = {}
        
        for col in self.numerical_features:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_info[col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(df) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > threshold]
                outlier_info[col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(df) * 100,
                    'threshold': threshold
                }
        
        return outlier_info
    
    def create_features(self, df):
        """
        Feature engineering
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Feature engineered dataframe
        """
        df_new = df.copy()
        
        # Transaction amount için log transformation
        if 'Amount' in df_new.columns:
            df_new['Amount_log'] = np.log1p(df_new['Amount'])
        
        # Time features (eğer Time kolonu varsa)
        if 'Time' in df_new.columns:
            df_new['Hour'] = (df_new['Time'] // 3600) % 24
            df_new['DayOfWeek'] = (df_new['Time'] // (3600 * 24)) % 7
        
        # Interaction features (ilk 2 numerical feature)
        if len(self.numerical_features) >= 2:
            feat1, feat2 = self.numerical_features[:2]
            df_new[f'{feat1}_{feat2}_ratio'] = df_new[feat1] / (df_new[feat2] + 1e-8)
            df_new[f'{feat1}_{feat2}_product'] = df_new[feat1] * df_new[feat2]
        
        logger.info("Feature engineering tamamlandı")
        return df_new
    
    def fit_transform(self, df, target_col=None):
        """
        Fit ve transform işlemlerini birlikte yap
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target kolonu (exclude edilecek)
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        df_processed = df.copy()
        
        # Target kolonu varsa çıkar
        if target_col and target_col in df_processed.columns:
            target = df_processed[target_col]
            df_processed = df_processed.drop(columns=[target_col])
        
        # Feature types belirle
        self.identify_features(df_processed)
        
        # Missing values handle et
        df_processed = self.handle_missing_values(df_processed)
        
        # Feature engineering
        df_processed = self.create_features(df_processed)
        
        # Feature types güncelle
        self.identify_features(df_processed)
        
        # Numerical scaling
        if self.numerical_features:
            df_processed[self.numerical_features] = self.scaler.fit_transform(
                df_processed[self.numerical_features]
            )
            logger.info(f"Numerical features scaled with {self.scaling_method}")
        
        # Categorical encoding
        if self.categorical_features:
            if self.encoding_method == 'onehot':
                encoded_data = self.encoder.fit_transform(
                    df_processed[self.categorical_features]
                )
                
                # Feature names al
                if hasattr(self.encoder, 'get_feature_names_out'):
                    self.encoded_feature_names = self.encoder.get_feature_names_out(
                        self.categorical_features
                    ).tolist()
                else:
                    self.encoded_feature_names = [
                        f"{cat}_{val}" for cat in self.categorical_features 
                        for val in self.encoder.categories_[
                            self.categorical_features.index(cat)
                        ]
                    ]
                
                # Encoded dataframe oluştur
                encoded_df = pd.DataFrame(
                    encoded_data, 
                    columns=self.encoded_feature_names,
                    index=df_processed.index
                )
                
                # Categorical kolonları çıkar ve encoded olanları ekle
                df_processed = df_processed.drop(columns=self.categorical_features)
                df_processed = pd.concat([df_processed, encoded_df], axis=1)
                
            else:  # label veya ordinal
                for col in self.categorical_features:
                    df_processed[col] = self.encoder.fit_transform(df_processed[col])
            
            logger.info(f"Categorical features encoded with {self.encoding_method}")
        
        self.is_fitted = True
        
        # Target kolonu varsa geri ekle
        if target_col and 'target' in locals():
            df_processed[target_col] = target
        
        return df_processed
    
    def transform(self, df, target_col=None):
        """
        Sadece transform (fit edilmiş preprocessor için)
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target kolonu
            
        Returns:
            pd.DataFrame: Transformed dataframe
        """
        if not self.is_fitted:
            raise ValueError("Önce fit_transform metodunu çağırın")
        
        df_processed = df.copy()
        
        # Target kolonu varsa çıkar
        if target_col and target_col in df_processed.columns:
            target = df_processed[target_col]
            df_processed = df_processed.drop(columns=[target_col])
        
        # Missing values handle et
        df_processed = self.handle_missing_values(df_processed)
        
        # Feature engineering
        df_processed = self.create_features(df_processed)
        
        # Numerical scaling
        if self.numerical_features:
            # Mevcut featurelar için transform
            existing_num_features = [f for f in self.numerical_features 
                                   if f in df_processed.columns]
            if existing_num_features:
                df_processed[existing_num_features] = self.scaler.transform(
                    df_processed[existing_num_features]
                )
        
        # Categorical encoding
        if self.categorical_features:
            existing_cat_features = [f for f in self.categorical_features 
                                   if f in df_processed.columns]
            
            if existing_cat_features and self.encoding_method == 'onehot':
                encoded_data = self.encoder.transform(
                    df_processed[existing_cat_features]
                )
                
                encoded_df = pd.DataFrame(
                    encoded_data, 
                    columns=self.encoded_feature_names,
                    index=df_processed.index
                )
                
                df_processed = df_processed.drop(columns=existing_cat_features)
                df_processed = pd.concat([df_processed, encoded_df], axis=1)
                
            elif existing_cat_features:
                for col in existing_cat_features:
                    df_processed[col] = self.encoder.transform(df_processed[col])
        
        # Target kolonu varsa geri ekle
        if target_col and 'target' in locals():
            df_processed[target_col] = target
        
        return df_processed
    
    def visualize_distributions(self, df_original, df_processed, target_col=None):
        """
        Original vs processed data distributions
        
        Args:
            df_original (pd.DataFrame): Original data
            df_processed (pd.DataFrame): Processed data
            target_col (str): Target column name
        """
        # Numerical features için karşılaştırma
        numerical_cols = [col for col in self.numerical_features 
                         if col in df_original.columns][:4]  # İlk 4 tanesini al
        
        if numerical_cols:
            fig, axes = plt.subplots(2, len(numerical_cols), 
                                   figsize=(4*len(numerical_cols), 8))
            
            for i, col in enumerate(numerical_cols):
                # Original
                axes[0, i].hist(df_original[col], bins=50, alpha=0.7, 
                              color='blue', edgecolor='black')
                axes[0, i].set_title(f'Original - {col}')
                axes[0, i].set_ylabel('Frequency')
                
                # Processed
                if col in df_processed.columns:
                    axes[1, i].hist(df_processed[col], bins=50, alpha=0.7, 
                                  color='red', edgecolor='black')
                    axes[1, i].set_title(f'Processed - {col}')
                    axes[1, i].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.show()
    
    def get_feature_info(self):
        """Feature bilgilerini döndür"""
        info = {
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'encoded_feature_names': self.encoded_feature_names,
            'scaling_method': self.scaling_method,
            'encoding_method': self.encoding_method
        }
        return info


class ImbalanceHandler:
    """Class imbalance handling için araçlar"""
    
    @staticmethod
    def analyze_imbalance(y, target_names=None):
        """
        Class imbalance analizi
        
        Args:
            y (array-like): Target values
            target_names (list): Class isimleri
        """
        unique, counts = np.unique(y, return_counts=True)
        
        print("CLASS DISTRIBUTION")
        print("="*30)
        for i, (cls, count) in enumerate(zip(unique, counts)):
            name = target_names[i] if target_names else f"Class {cls}"
            percentage = count / len(y) * 100
            print(f"{name}: {count} ({percentage:.2f}%)")
        
        # Imbalance ratio
        imbalance_ratio = max(counts) / min(counts)
        print(f"\nImbalance Ratio: {imbalance_ratio:.2f}")
        
        # Visualization
        plt.figure(figsize=(8, 6))
        labels = target_names if target_names else [f"Class {cls}" for cls in unique]
        colors = ['lightblue', 'salmon'][:len(unique)]
        
        plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors)
        plt.title('Class Distribution')
        plt.show()
        
        return {'classes': unique, 'counts': counts, 'imbalance_ratio': imbalance_ratio}
    
    @staticmethod
    def apply_smote(X, y, sampling_strategy='auto', random_state=42):
        """
        SMOTE uygula
        
        Args:
            X (array-like): Features
            y (array-like): Target
            sampling_strategy: SMOTE strategy
            random_state (int): Random seed
            
        Returns:
            tuple: (X_resampled, y_resampled)
        """
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        logger.info(f"SMOTE uygulandı: {len(X)} -> {len(X_resampled)} samples")
        return X_resampled, y_resampled
    
    @staticmethod
    def apply_adasyn(X, y, sampling_strategy='auto', random_state=42):
        """ADASYN uygula"""
        adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        
        logger.info(f"ADASYN uygulandı: {len(X)} -> {len(X_resampled)} samples")
        return X_resampled, y_resampled
    
    @staticmethod
    def apply_smotetomek(X, y, sampling_strategy='auto', random_state=42):
        """SMOTETomek uygula"""
        smotetomek = SMOTETomek(sampling_strategy=sampling_strategy, 
                               random_state=random_state)
        X_resampled, y_resampled = smotetomek.fit_resample(X, y)
        
        logger.info(f"SMOTETomek uygulandı: {len(X)} -> {len(X_resampled)} samples")
        return X_resampled, y_resampled


def demo_preprocessing(data="/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta4/fraud-detection/data/processed/dataset_with_anomaly_scores_raw.csv"):
    # 1) Veriyi oku
    df = pd.read_csv(data)
    assert "Class" in df.columns, "Hedef kolon 'Class' bulunamadı."
    
    print("Original Dataset Info:")
    print(df.info())

    # 2) Sınıf dağılımı (bilgilendirme)
    print("\nClass Distribution:")
    ImbalanceHandler.analyze_imbalance(df['Class'], ['Normal', 'Fraud'])

    # 3) Train/Test ayrımı
    if "split" in df.columns:
        # split varsa onu kullan
        feature_cols = [c for c in df.columns if c not in ("Class", "split")]
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        test_df  = df[df["split"] == "test"].reset_index(drop=True)

        X_train = train_df[feature_cols].copy()
        y_train = train_df["Class"].astype(int).copy()
        X_test  = test_df[feature_cols].copy()
        y_test  = test_df["Class"].astype(int).copy()
    else:
        # split yoksa stratified split yap
        X = df.drop(columns=["Class"], errors="ignore")
        y = df["Class"].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

    # 4) Preprocessing (robust + onehot)
    preprocessor = FeaturePreprocessor(scaling_method='robust', encoding_method='onehot')

    # Fit transform on train
    X_train_processed = preprocessor.fit_transform(
        pd.concat([X_train, y_train.rename("Class")], axis=1), target_col='Class'
    )
    y_train_processed = X_train_processed['Class'].astype(int)
    X_train_processed = X_train_processed.drop('Class', axis=1)

    # Transform test
    X_test_processed = preprocessor.transform(
        pd.concat([X_test, y_test.rename("Class")], axis=1), target_col='Class'
    )
    y_test_processed = X_test_processed['Class'].astype(int)
    X_test_processed = X_test_processed.drop('Class', axis=1)

    print(f"\nProcessed Train Shape: {X_train_processed.shape}")
    print(f"Processed Test Shape: {X_test_processed.shape}")

    # 5) SMOTE (opsiyonel)
    try:
        X_train_balanced, y_train_balanced = ImbalanceHandler.apply_smote(
            X_train_processed, y_train_processed
        )
    except Exception as e:
        logger.warning(f"SMOTE atlandı: {e}")
        X_train_balanced, y_train_balanced = X_train_processed, y_train_processed

    # 6) Görsel karşılaştırma (mevcut fonksiyonunla)
    preprocessor.visualize_distributions(X_train, X_train_processed)

    # 7) Çıktı dosyaları
    from pathlib import Path
    DATA_DIR = Path("./data/processed"); DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_out = DATA_DIR / "train_processed_supervised.csv"
    test_out  = DATA_DIR / "test_processed_supervised.csv"
    full_out  = DATA_DIR / "dataset_processed_supervised.csv"

    # Train/Test ayrı CSV
    pd.concat([X_train_processed.reset_index(drop=True), y_train_processed.reset_index(drop=True)], axis=1)\
      .to_csv(train_out, index=False)
    pd.concat([X_test_processed.reset_index(drop=True),  y_test_processed.reset_index(drop=True)],  axis=1)\
      .to_csv(test_out, index=False)

    # Tek CSV (split etiketiyle)
    if "split" in df.columns:
        train_merge = pd.concat([
            X_train_processed.reset_index(drop=True),
            y_train_processed.reset_index(drop=True),
            pd.Series(["train"]*len(y_train_processed), name="split")
        ], axis=1)
        test_merge  = pd.concat([
            X_test_processed.reset_index(drop=True),
            y_test_processed.reset_index(drop=True),
            pd.Series(["test"]*len(y_test_processed), name="split")
        ], axis=1)
        full_df = pd.concat([train_merge, test_merge], axis=0, ignore_index=True)
    else:
        # split yoksa tek CSV'yi split kolonsuz kaydet
        full_df = pd.concat([
            pd.concat([X_train_processed.reset_index(drop=True), y_train_processed.reset_index(drop=True)], axis=1),
            pd.concat([X_test_processed.reset_index(drop=True),  y_test_processed.reset_index(drop=True)],  axis=1)
        ], axis=0, ignore_index=True)

    full_df.to_csv(full_out, index=False)

    print(f"[OK] Kaydedildi → {train_out}")
    print(f"[OK] Kaydedildi → {test_out}")
    print(f"[OK] Kaydedildi → {full_out}")

    return {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train_processed,
        'y_test': y_test_processed,
        'X_train_balanced': X_train_balanced,
        'y_train_balanced': y_train_balanced,
        'preprocessor': preprocessor
    }


if __name__ == "__main__":
    demo_results = demo_preprocessing()