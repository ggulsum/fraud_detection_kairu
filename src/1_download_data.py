#!/usr/bin/env python3
"""
Credit Card Fraud Detection Dataset Download
Sadece en uygun dataset ile çalışmak için basitleştirilmiş versiyon
"""

import os
import kagglehub
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_creditcard_fraud():
    """
    Credit Card Fraud Detection dataset indir
    Fraud detection öğrenmek için en uygun dataset
    """
    try:
        logger.info("Credit Card Fraud Detection dataset indiriliyor...")
        
        # Download dataset
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        logger.info(f"Dataset indirildi: {path}")
        
        # Load and inspect data
        csv_file = os.path.join(path, "creditcard.csv")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Class distribution:\n{df['Class'].value_counts()}")
            
            # Copy to project data directory
            data_dir = "data/raw"
            os.makedirs(data_dir, exist_ok=True)
            
            target_path = os.path.join(data_dir, "creditcard_fraud.csv")
            df.to_csv(target_path, index=False)
            logger.info(f"Data kopyalandı: {target_path}")
            
            return target_path, df
        else:
            logger.error("creditcard.csv dosyası bulunamadı")
            return None, None
            
    except Exception as e:
        logger.error(f"Credit card dataset download hatası: {e}")
        return None, None


def main():
    """Ana fonksiyon - Credit Card Fraud dataset indir"""
    print("🔽 Credit Card Fraud Detection Dataset Download")
    print("="*60)
    print("Bu dataset fraud detection öğrenmek için en uygun seçenektir:")
    print("• 284,807 işlem")  
    print("• %0.172 fraud oranı (gerçekçi imbalance)")
    print("• PCA ile önceden işlenmiş (V1-V28 features)")
    print("• Time, Amount, Class kolonları")
    print("="*60)
    
    # Create data directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Download Credit Card Fraud dataset
    result, df = download_creditcard_fraud()
    
    if result and df is not None:
        print(f"\n✅ Credit Card Fraud dataset hazır!")
        print(f"📁 Dosya konumu: {result}")
        print(f"📊 Dataset boyutu: ~150MB")
        
        # Dataset özeti
        print(f"\n📈 Dataset Özeti:")
        print(f"   Satır sayısı: {len(df):,}")
        print(f"   Kolon sayısı: {len(df.columns)}")
        print(f"   Normal işlem: {len(df[df['Class']==0]):,} (%{len(df[df['Class']==0])/len(df)*100:.2f})")
        print(f"   Fraud işlem: {len(df[df['Class']==1]):,} (%{len(df[df['Class']==1])/len(df)*100:.3f})")
        print(f"   Eksik değer: {df.isnull().sum().sum()}")
        
        # Feature bilgileri
        print(f"\n🔍 Feature Bilgileri:")
        print(f"   V1-V28: PCA ile dönüştürülmüş gizli features")
        print(f"   Time: İşlem zamanı (saniye)")
        print(f"   Amount: İşlem tutarı") 
        print(f"   Class: 0=Normal, 1=Fraud")
        
        print(f"\n🚀 Kullanım örnekleri:")
        print(f"   # Gerçek data ile training")
        print(f"   python src/pipeline.py --data {result} --save_models")
        print(f"   ")
        print(f"   # KaggleHub ile otomatik")
        print(f"   python src/pipeline.py --use_kagglehub --save_models")
        print(f"   ")
        print(f"   # Demo çalıştırma")
        print(f"   python run_demo.py")
        
        # Feature dağılımları
        print(f"\n📊 İlk 5 satır:")
        print(df.head())
        
        print(f"\n💰 Amount istatistikleri:")
        print(df['Amount'].describe())
        
    else:
        print("\n❌ Credit Card dataset indirilemedi")
        print("💡 Alternatif: Synthetic data kullanabilirsin")
        print("   python run_demo.py")
    
    print(f"\n🎯 Bu dataset ile öğrenebileceğin konular:")
    print(f"   • Isolation Forest & LOF outlier detection")
    print(f"   • Feature scaling (robust scaler önerilir)")
    print(f"   • Imbalanced data handling (SMOTE)")
    print(f"   • ROC-AUC vs PR-AUC karşılaştırması") 
    print(f"   • SHAP ile feature importance")
    print(f"   • Threshold optimization")
    print(f"   • Business cost analysis")
    
    print(f"\n📚 Next Steps:")
    print(f"   1. Dataset indirme tamamlandı ✅")
    print(f"   2. python run_demo.py ile demo'yu çalıştır")
    print(f"   3. Gerçek data ile training yap")
    print(f"   4. SHAP ile model açıklaması analiz et")


if __name__ == "__main__":
    main()