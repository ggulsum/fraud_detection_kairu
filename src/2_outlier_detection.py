"""
Modül 1 — Outlier Detection (RAW / scale yok)
Kaynak (HAM CSV): /Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta4/fraud-detection/data/raw/creditcard_fraud.csv
Çıktılar:
  - ./data/processed/dataset_with_anomaly_scores_raw.csv
  - ./data/processed/outlier_meta_raw.json
Not: Hiçbir ölçekleme/encoding yok. IF & LOF ham değerlerle çalışır.
"""

from pathlib import Path
import os, json
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

DATA_DIR = Path("./data/processed"); DATA_DIR.mkdir(parents=True, exist_ok=True)

# >>>>> HAM VERİ YOLU (senin verdiğin yol) <<<<<
RAW_PATH = Path("/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta4/fraud-detection/data/raw/creditcard_fraud.csv")

OUT_CSV  = DATA_DIR / "dataset_with_anomaly_scores_raw.csv"
OUT_META = DATA_DIR / "outlier_meta_raw.json"

def choose_threshold_by_f1(y_true, scores):
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-9)
    best_idx = int(np.argmax(f1))
    thr_choice = float(thr[max(0, min(best_idx-1, len(thr)-1))]) if len(thr) > 0 else 0.0
    return {"threshold": thr_choice, "precision": float(prec[best_idx]),
            "recall": float(rec[best_idx]), "f1": float(f1[best_idx])}

def main():
    assert RAW_PATH.exists(), f"Ham veri bulunamadı: {RAW_PATH}"
    df = pd.read_csv(RAW_PATH)
    assert "Class" in df.columns, "Hedef kolon 'Class' bulunamadı."

    # split yoksa oluştur (stratified)
    if "split" not in df.columns:
        y_tmp = df["Class"].astype(int).values
        idx_train, idx_test = train_test_split(
            np.arange(len(df)), test_size=0.30, random_state=42, stratify=y_tmp
        )
        split = np.array(["train"]*len(df), dtype=object); split[idx_test] = "test"
        df["split"] = split

    feature_cols = [c for c in df.columns if c not in ("Class","split")]
    train = df[df["split"]=="train"].reset_index(drop=True)
    test  = df[df["split"]=="test"].reset_index(drop=True)

    X_train = train[feature_cols].values
    X_test  = test[feature_cols].values
    y_test  = test["Class"].astype(int).values

    print(f"[OK] Kaynak: {RAW_PATH}")
    print(f"[OK] Train: {X_train.shape} | Test: {X_test.shape} | Test fraud oranı: {y_test.mean():.6f}")

    # --- Isolation Forest (ham) ---
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)
    iso.fit(X_train)
    if_scores_train = -iso.decision_function(X_train)  # yüksek = anomali
    if_scores_test  = -iso.decision_function(X_test)

    if_thr = choose_threshold_by_f1(y_test, if_scores_test)
    if_alarm_test = (if_scores_test >= if_thr["threshold"]).astype(int)

    if_roc = float(roc_auc_score(y_test, if_scores_test))
    if_ap  = float(average_precision_score(y_test, if_scores_test))
    print(f"[IF] ROC-AUC={if_roc:.4f} | PR-AUC(AP)={if_ap:.4f} | "
          f"Eşik≈{if_thr['threshold']:.6f} | P={if_thr['precision']:.3f} R={if_thr['recall']:.3f} F1={if_thr['f1']:.3f} "
          f"| Alarm oranı={if_alarm_test.mean():.4f}")

    # --- LOF (novelty=True, ham) ---
    lof = LocalOutlierFactor(n_neighbors=35, contamination="auto", novelty=True)
    lof.fit(X_train)
    lof_scores_train = -lof.score_samples(X_train)     # yüksek = anomali
    lof_scores_test  = -lof.score_samples(X_test)

    lof_thr = choose_threshold_by_f1(y_test, lof_scores_test)
    lof_alarm_test = (lof_scores_test >= lof_thr["threshold"]).astype(int)

    lof_roc = float(roc_auc_score(y_test, lof_scores_test))
    lof_ap  = float(average_precision_score(y_test, lof_scores_test))
    print(f"[LOF] ROC-AUC={lof_roc:.4f} | PR-AUC(AP)={lof_ap:.4f} | "
          f"Eşik≈{lof_thr['threshold']:.6f} | P={lof_thr['precision']:.3f} R={lof_thr['recall']:.3f} F1={lof_thr['f1']:.3f} "
          f"| Alarm oranı={lof_alarm_test.mean():.4f}")

    # --- Skor/Alarm kolonlarını yaz ---
    df_out = df.copy()
    df_out["if_score"]  = np.nan; df_out["lof_score"] = np.nan
    df_out.loc[df_out["split"]=="train","if_score"]  = if_scores_train
    df_out.loc[df_out["split"]=="train","lof_score"] = lof_scores_train
    df_out.loc[df_out["split"]=="test","if_score"]   = if_scores_test
    df_out.loc[df_out["split"]=="test","lof_score"]  = lof_scores_test

    df_out["if_alarm"]  = 0; df_out["lof_alarm"] = 0
    df_out.loc[df_out["split"]=="test","if_alarm"]  = if_alarm_test
    df_out.loc[df_out["split"]=="test","lof_alarm"] = lof_alarm_test

    df_out.to_csv(OUT_CSV, index=False)
    print(f"[OK] Kaydedildi → {OUT_CSV}")

    meta = {
        "input_file": str(RAW_PATH),
        "output_file": str(OUT_CSV),
        "n_train": int(len(train)), "n_test": int(len(test)),
        "iforest": {"roc_auc": if_roc, "pr_auc_ap": if_ap, **if_thr},
        "lof":     {"roc_auc": lof_roc, "pr_auc_ap": lof_ap, **lof_thr, "n_neighbors": 35},
        "notes": [
            "Ölçekleme yok; skor yönleri normalize: yüksek skor = daha anomali.",
            "Eşikler PR eğrisinde F1’i maksimize eden noktadan seçildi.",
            "Skorlar train+test için üretildi; testte alarm etiketleri yazıldı."
        ]
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] Meta kayıt → {OUT_META}")

if __name__ == "__main__":
    main()


#[IF] ROC-AUC=0.9480 | PR-AUC(AP)=0.1381 | Eşik≈0.148798 | P=0.253 R=0.270 F1=0.261 | Alarm oranı=0.0019
'''
🔹 **[IF]**
Bu sonuçlar **Isolation Forest (IF)** modeli için.

---

🔹 **ROC-AUC = 0.9480**

* ROC-AUC, modelin **normal vs fraud ayırt etme becerisini** ölçer.
* 1’e ne kadar yakınsa, o kadar iyi ayırt ediyor demektir.
* 0.948 ≈ **çok yüksek** → IF, skor üretmede gayet iyi.

---

🔹 **PR-AUC (AP) = 0.1381**

* PR-AUC, “Precision-Recall eğrisi altındaki alan.”
* Fraud çok az olduğu için **PR-AUC daha önemlidir**.
* 0.1381 → düşük sayılır. Çünkü dengesiz veri setinde (fraud %0.17) precision-recall işleri zor.
* Yani ROC iyi ama fraud yakalamada pratikte o kadar da şahane değil.

---

🔹 **Eşik ≈ 0.148798**

* Isolation Forest her işleme bir **“anomaly score”** veriyor.
* Skor ↑ → daha anormal (fraud olasılığı daha fazla).
* Biz bir noktada “şuradan sonrası fraud olsun” diye eşik koyuyoruz.
* Bu eşik **F1 skorunu en çok artıran nokta**.

---

🔹 **P = 0.253 (Precision)**

* “Model fraud dediğinde, %25’i gerçekten fraud.”
* Yani alarm verdiği her 4 işlemin 3’ü aslında normal → **çok false alarm** var.

---

🔹 **R = 0.270 (Recall)**

* “Gerçek fraud’ların %27’sini yakaladı.”
* Yani fraud’ların büyük kısmını kaçırıyor.

---

🔹 **F1 = 0.261**

* Precision ve Recall’un dengesi.
* 0.26 düşük → model bu eşikte hem çok alarm veriyor hem de çoğunu kaçırıyor.

---

🔹 **Alarm oranı = 0.0019**

* Tüm işlemlerin sadece %0.19’u fraud alarmı aldı.
* Veri setinde fraud %0.17 civarında → model azıcık fazla alarm veriyor.

---

📌 **Özet:**

* Isolation Forest güzel anomaly score üretiyor (ROC yüksek).
* Ama doğrudan alarm mekanizması (threshold ile etiketleme) çok başarılı değil → Precision/Recall düşük.
* Bu yüzden **en iyi kullanım şekli**, `if_score`’u **ek bir feature** olarak supervised modele sokmak.
* Böylece Logistic Regression, XGBoost gibi modeller `if_score` + diğer değişkenleri birlikte kullanıp daha iyi Precision/Recall dengesine ulaşır.

---

'''