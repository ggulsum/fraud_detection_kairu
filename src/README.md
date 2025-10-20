# Fraud Detection Pipeline

## 📌 Genel Bakış

[pipeline.py](src/pipeline.py), **dolandırıcılık (fraud)** tespiti için tasarlanmış uçtan uca (**end-to-end**) bir veri işleme ve modelleme akışını (**pipeline**) yönetir.
Bu dosya tek başına çalıştırılabilir bir Python modülüdür ve aşağıdaki adımları gerçekleştirir:

* Veri yükleme ve doğrulama
* Ön işleme (scaling, encoding, feature engineering)
* Veri dengesizliği giderme (SMOTE, ADASYN, vb.)
* Model eğitimi (Random Forest, Logistic Regression, Isolation Forest, LOF)
* Model değerlendirme (ROC-AUC, F1, Precision, Recall, PR-AUC)
* Model kaydetme ve yükleme
* Tahmin ve açıklanabilirlik (Explainability)
* Tam pipeline çalıştırma (end-to-end)

Kod, modüler bir yapıya sahiptir ve aşağıdaki dış modülleri kullanır:

* `FeaturePreprocessor`
* `ImbalanceHandler`
* `FraudEvaluator`
* `OutlierDetector`
* `ModelExplainer`

---

## Class: `FraudDetectionPipeline`

### 1️⃣ Constructor

```python
def __init__(self, config_path="config/config.yaml"):
```

**Amaç:** Pipeline konfigürasyonunu yükler, loglama ve MLflow ayarlarını yapar.

**Parametre:**

* `config_path` — YAML formatında pipeline ayar dosyasının yolu.

**Yapılan İşlemler:**

* Config dosyası okunur.
* Logging sistemi başlatılır.
* MLflow bağlantısı kurulur (deney kaydı için).

---

### 2️⃣ `load_data()`

```python
def load_data(self, data_path=None, synthetic=True, download_with_kagglehub=False):
```

**Amaç:** Eğitim için veri yüklemek.

**Parametreler:**

* `data_path`: Dosya yolu (CSV)
* `synthetic`: `True` ise sentetik veri oluşturur
* `download_with_kagglehub`: `True` ise KaggleHub üzerinden veri indirir

**Dönüş:**
`self.data` → *pandas DataFrame*

Ek: `_validate_data()` ile şema ve eksik değer kontrolü yapılır.

**KaggleHub kullanımı:**

```bash
python src/pipeline.py --mode train --use_kagglehub
```
---

### 3️⃣ `_generate_synthetic_data()`

**Amaç:** Gerçek veri yoksa test amaçlı rastgele sahte veri üretir.
**Dönüş:** `DataFrame` (amount, time, class, vb. sütunlar içerir)

---

### 4️⃣ `_validate_data()`

**Amaç:** Veri kalitesini ve bütünlüğünü kontrol eder.

**Kontroller:**

* Eksik değer yüzdesi (`missing_threshold`)
* Gerekli sütunların varlığı
* Sayısal değer aralıkları (ör. `amount > 0`)
* `class` değerlerinin uygunluğu (`0/1`)

Hatalı durumda: `ValueError` fırlatır ve işlem durur.

---

### 5️⃣ `preprocess_data()`

**Amaç:** Veri üzerinde ön işleme yapmak.

**İçerik:**

* Kategorik/sayısal sütun ayrımı
* Ölçekleme (`robust`, `standard`, `minmax`)
* Encoding (`onehot`, `label`)
* Eksik veri doldurma (imputation)
* Veri dengesizliği giderme (SMOTE, ADASYN, vb.)

**Kullanır:**

* `FeaturePreprocessor`
* `ImbalanceHandler`

**Dönüş:**
`self.X_train`, `self.X_test`, `self.y_train`, `self.y_test`

---

### 6️⃣ `train_models()`

**Amaç:** Config’de tanımlı tüm modelleri eğitir.

**Desteklenen modeller:**

* `RandomForestClassifier`
* `LogisticRegression`
* `IsolationForest`
* `LocalOutlierFactor`

**Dönüş:**
`self.models` → `{ model_adı: model_nesnesi }`

MLflow aktifse, her model için ayrı bir “run” olarak kaydeder.

---

### 7️⃣ `evaluate_models()`

**Amaç:** Her modeli test verisinde değerlendirir.

**Kullanır:** 
* `FraudEvaluator`
* `OutlierDetector` (özellikle Isolation Forest ve LOF için)

**Metrikler:**

* ROC-AUC
* PR-AUC
* F1-score
* Precision, Recall

**Dönüş:**
`{ model_adı: metrik_sonuçları }`

En iyi modeli `_find_best_model()` fonksiyonu seçer.

---

### 8️⃣ `save_models()` / `load_models()`

**Amaç:** Eğitilen modelleri ve ön işleme nesnelerini kaydetmek veya yeniden yüklemek.

**Kayıt edilen dosyalar:**

* `preprocessor.pkl`
* `<model_adı>_model.pkl`
* `feature_info.pkl`

**Kullanılan araç:** `joblib`

---

### 9️⃣ `predict()`

```python
def predict(self, data, model_name="random_forest"):
```

**Amaç:** Yeni veriler üzerinde tahmin yapmak.

**Parametreler:**

* `data`: DataFrame formatında yeni veri
* `model_name`: Kullanılacak modelin adı

**Dönüş:**
`predictions` → `numpy array` veya `DataFrame`

---

### 🔟 `explain_models()`

**Amaç:** Modelin kararlarını açıklamak (SHAP / LIME tabanlı).

Not: `explainability_clean.py` modülü mevcutsa `ModelExplainer` çağrılır, yoksa uyarı verip atlanır.

**Girdi:** Model adı
**Dönüş:** Grafik veya açıklama çıktısı 

---

### 1️⃣1️⃣ `run_full_pipeline()`

**Amaç:** Tek komutla tüm süreci baştan sona çalıştırmak:

1. Veri yükle
2. Ön işle
3. Modelleri eğit
4. Değerlendir
5. En iyi modeli kaydet

**Dönüş:** En iyi modelin adı ve metrikleri.
Fonksiyon, tüm adımlar başarıyla tamamlandığında True, hata durumunda False döndürür.

---

## CLI (Komut Satırı) Kullanımı

Pipeline, doğrudan komut satırından çalıştırılabilir:

```bash
python src/pipeline.py
```

---

## Özet

Bu dosya:

* Fraud tespiti için tamamen otomatik bir modelleme akışı sunar.
* Her adımı modüler, açıklanabilir ve yeniden kullanılabilir olacak şekilde tasarlanmıştır.
* CLI’dan veya kod içinden kolayca çalıştırılabilir.
* MLflow ile deney takibi yapılabilir.
* SHAP tabanlı açıklamalarla model davranışı analiz edilebilir.

---

