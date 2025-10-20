# 🏦 Credit Card Fraud Detection - Dataset Hikayesi

## 📖 Hikaye

Büyük bir Avrupa bankası, 2013 yılında müşterilerinin kredi kartı işlemlerinde artan fraud vakalarıyla karşı karşıya kaldı. Günde yüz binlerce işlem gerçekleşirken, bunların sadece %0.172'si sahte işlemdi - ancak bu küçük oran bile bankanın milyonlarca euro zarar etmesine neden oluyordu.

### 🎯 İş Problemi

1. **Gerçek Zamanlı Tespit**: Fraud işlemler gerçekleşirken anında tespit edilmeli
2. **Dengesiz Veri**: %99.83 normal, %0.17 fraud - klasik accuracy yanıltıcı
3. **Yanlış Pozitif Maliyeti**: İyi müşteriyi engellemek → müşteri kaybı
4. **Yanlış Negatif Maliyeti**: Fraud'ı kaçırmak → direkt finansal zarar
5. **Gizlilik**: Müşteri bilgileri korunmalı, feature'lar gizlenmeli

### 🔒 Gizlilik ve Güvenlik

Banka müşteri gizliliğini korumak için **Principal Component Analysis (PCA)** kullanarak tüm hassas feature'ları dönüştürdü:

- **V1-V28**: PCA ile dönüştürülmüş gizli özellikler
- **Time**: İşlem zamanı (ilk işlemden itibaren saniye)
- **Amount**: İşlem tutarı (euro cinsinden)
- **Class**: 0=Normal, 1=Fraud (hedef değişken)

---

## 📊 Dataset Özellikleri

### Boyut ve Dağılım
- **Toplam İşlem**: 284,807 işlem
- **Zaman Aralığı**: 2 gün (172,800 saniye)
- **Normal İşlem**: 284,315 (%99.827)
- **Fraud İşlem**: 492 (%0.173)

### Feature Profili
```
- V1-V28: PCA dönüştürülmüş sayısal özellikler
- Time: 0-172792 saniye arası
- Amount: 0-25691.16 euro arası
- Class: 0 (Normal) / 1 (Fraud)
```

### Gerçek Dünya Temsili
Bu dataset gerçek fraud detection zorluklarını mükemmel şekilde yansıtır:

✅ **Aşırı dengesiz dağılım** - Production fraud rate'ini temsil eder  
✅ **Gizlilik korunmuş** - PCA ile sensitive data maskelenmiş  
✅ **Zaman faktörü** - Temporal patterns mevcut  
✅ **Farklı tutar aralıkları** - Mikro ve makro işlemler  
✅ **Temiz veri** - Missing value yok, preprocessed  

---

## 🎯 Fraud Patterns ve İnsigtler

### 1. Zaman Bazlı Patterns
```python
# Gece saatlerinde fraud riski artar
fraud_by_hour = data[data['Class']==1]['Time'].apply(lambda x: (x/3600) % 24)
# Peak fraud hours: 23:00-05:00
```

### 2. Tutar Bazlı Patterns
```python
# Fraud işlemler genellikle küçük tutarlarda
fraud_amounts = data[data['Class']==1]['Amount']
normal_amounts = data[data['Class']==0]['Amount']
# Fraud median: ~22 euro, Normal median: ~27 euro
```

### 3. PCA Component Insights
```python
# V4, V11, V12, V14 genellikle en discriminative features
# V1, V2, V3 zaman bazlı patterns içerebilir
# V10, V16, V17 tutar-related features olabilir
```

---

## 🚨 Business Impact Senaryoları

### Senaryo 1: Konservatif Yaklaşım
- **Threshold**: 0.1 (düşük)
- **Sonuç**: %95 fraud yakalanır, %8 false positive
- **Maliyet**: Yüksek müşteri deneyimi sorunu

### Senaryo 2: Dengeli Yaklaşım  
- **Threshold**: 0.3 (optimal)
- **Sonuç**: %85 fraud yakalanır, %2 false positive
- **Maliyet**: En iyi cost-benefit ratio

### Senaryo 3: Agresif Yaklaşım
- **Threshold**: 0.7 (yüksek)
- **Sonuç**: %60 fraud yakalanır, %0.5 false positive
- **Maliyet**: Yüksek finansal kayıp

---

## 🔬 Model Development Journey

### Fase 1: Baseline Models
```python
# Logistic Regression: ROC-AUC ~0.93, PR-AUC ~0.65
# Random Forest: ROC-AUC ~0.95, PR-AUC ~0.75
# XGBoost: ROC-AUC ~0.96, PR-AUC ~0.80
```

### Fase 2: Outlier Detection
```python
# Isolation Forest: ROC-AUC ~0.85 (unsupervised)
# Local Outlier Factor: ROC-AUC ~0.80
# One-Class SVM: ROC-AUC ~0.75
```

### Fase 3: Advanced Techniques
```python
# SMOTE + XGBoost: PR-AUC ~0.85
# Cost-sensitive learning: Precision-Recall balance
# Ensemble methods: Voting classifier stability
```

---

## 📈 Expected Performance Benchmarks

### Outlier Detection (Unsupervised)
- **Isolation Forest**: ROC-AUC: 0.82-0.88
- **LOF**: ROC-AUC: 0.78-0.85  
- **Ensemble**: ROC-AUC: 0.85-0.90

### Supervised Learning
- **Logistic Regression**: ROC-AUC: 0.93+, PR-AUC: 0.65+
- **Random Forest**: ROC-AUC: 0.95+, PR-AUC: 0.75+
- **Gradient Boosting**: ROC-AUC: 0.96+, PR-AUC: 0.80+

### Business Metrics
- **Optimal threshold**: 0.3-0.4
- **Cost reduction**: 60-80% vs random checking
- **False positive rate**: <3% for customer satisfaction

---

## 🎓 Öğrenme Hedefleri

### Teknik Öğrenimler
1. **Imbalanced Learning**: SMOTE, undersampling, cost-sensitive
2. **Outlier Detection**: IF, LOF, ensemble methods
3. **Evaluation Metrics**: PR-AUC > ROC-AUC for imbalanced data
4. **Feature Engineering**: PCA interpretation, time features
5. **Model Explainability**: SHAP, LIME for fraud investigation

### Business Öğrenimler  
1. **Cost-Benefit Analysis**: FP vs FN trade-off optimization
2. **Threshold Tuning**: Business constraint integration
3. **Real-time Scoring**: Latency vs accuracy balance
4. **Model Monitoring**: Drift detection, performance degradation
5. **Regulatory Compliance**: Model explainability requirements

---

## 🔮 Production Considerations

### Real-time Inference
```python
# Target latency: <100ms
# Feature preprocessing: cached pipelines
# Model serving: REST API + batch scoring
# Fallback mechanism: rule-based backup
```

### Model Monitoring
```python
# Data drift: Feature distribution monitoring
# Model drift: Performance metric tracking  
# Business metrics: Cost per transaction
# Alert thresholds: Performance degradation >5%
```

### Regulatory Requirements
```python
# Model explainability: SHAP values per prediction
# Audit trail: Decision logging and versioning
# Bias detection: Fairness across customer segments
# Documentation: Model cards and governance
```

---

## 📚 Dataset Kullanım Rehberi

### 1. Veri İndirme
```bash
# Kaggle CLI ile
kaggle datasets download -d mlg-ulb/creditcardfraud

# Manuel indirme
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

### 2. Veri Yükleme
```python
import pandas as pd
df = pd.read_csv('data/raw/creditcard.csv')
print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {df['Class'].mean():.3%}")
```

### 3. Temel EDA
```python
# Class distribution
print(df['Class'].value_counts())

# Amount statistics by class
print(df.groupby('Class')['Amount'].describe())

# Time patterns
df['Hour'] = (df['Time'] // 3600) % 24
fraud_by_hour = df[df['Class']==1]['Hour'].value_counts().sort_index()
```

---

## ⚠️ Etik ve Güvenlik Notları

### Veri Gizliliği
- Bu dataset halihazırda anonimleştirilmiş (PCA ile)
- Gerçek müşteri bilgileri içermez
- Eğitim ve araştırma amaçlı kullanım onaylanmış

### Model Fairness
- Fraud detection modelleri bias içerebilir
- Farklı müşteri segmentlerinde performans test edilmeli
- Adil ve eşit treatment sağlanmalı

### Responsible AI
- Model kararları açıklanabilir olmalı
- İnsan oversight mekanizması bulunmalı
- Hatalı tahminlerin düzeltme süreci olmalı

---

**🎯 Bu dataset ile geliştirilecek fraud detection sistemi, gerçek dünya koşullarında kullanılmaya hazır, production-grade bir çözüm prototipi olacaktır.**

---

*Bu hikaye, Credit Card Fraud Detection projesinin sadece "bir ML egzersizi" değil, gerçek hayattaki finansal güvenlik sistemlerinin bir prototipi olduğunu vurgulamak için hazırlanmıştır.*