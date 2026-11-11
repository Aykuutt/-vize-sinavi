import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# 0. Setup and Paths
BASE_DIR = r"C:\Users\Lenovo\Desktop\Vize_Sinavi"
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True

csv_path = os.path.join(BASE_DIR, "ai4i2020.csv")
df = pd.read_csv(csv_path)

print(f"Data shape: {df.shape}")
print("Columns:", list(df.columns))

# ==============================================================================
# SORU 1: VERİYİ TANIMA VE PROBLEMİ ANLAMA
# ==============================================================================
print("\n" + "="*60)
print("SORU 1 ANALİZİ BAŞLIYOR...")
print("="*60)

# a) Dağılım İncelemesi
target_col = 'Machine failure'
fail_counts = df[target_col].value_counts()
fail_pcts = df[target_col].value_counts(normalize=True) * 100

print(f"\n--- Soru 1(a): Hedef Değişken Dağılımı ({target_col}) ---")
print(f"Arızasız Makine (0): {fail_counts[0]} adet ({fail_pcts[0]:.2f}%)")
print(f"Arızalı Makine (1):  {fail_counts[1]} adet ({fail_pcts[1]:.2f}%)")
imbalance_ratio = fail_counts[0] / fail_counts[1]
print(f"Dengesizlik Oranı (0 / 1): {imbalance_ratio:.2f}:1")

# Görselleştirme: Dağılım Grafiği (Bar & Donut)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar Chart
sns.barplot(x=fail_counts.index.astype(str), y=fail_counts.values, ax=axes[0], palette=['#2ecc71', '#e74c3c'])
axes[0].set_title("Makine Arıza Durumu Dağılımı (Frekans)", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Arıza Durumu (0: Normal, 1: Arızalı)", fontsize=11)
axes[0].set_ylabel("Makine Sayısı (Adet)", fontsize=11)
for i, v in enumerate(fail_counts.values):
    axes[0].text(i, v + 100, f"{v} ({fail_pcts.iloc[i]:.1f}%)", ha='center', fontweight='bold', fontsize=11)

# Donut Chart
axes[1].pie(fail_counts.values, labels=['Normal (0)', 'Arızalı (1)'], autopct='%1.1f%%',
            startangle=45, colors=['#2ecc71', '#e74c3c'], explode=(0, 0.12),
            wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2))
axes[1].set_title("Makine Arıza Oranı (Sınıf Dengesizliği)", fontsize=13, fontweight='bold')

plt.savefig(os.path.join(FIG_DIR, "soru1a_hedef_dagilimi.png"), dpi=300)
plt.close()

# b) Sayısal Değişkenler Açısından Karşılaştırma
numeric_features = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

print("\n--- Soru 1(b): Sayısal Değişkenlerin Arıza Durumuna Göre Özeti ---")
comparison_stats = df.groupby(target_col)[numeric_features].agg(['mean', 'std', 'median'])
print(comparison_stats.T)

# Karşılaştırma Kutuları (Boxplots)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, col in enumerate(numeric_features):
    sns.boxplot(x=target_col, y=col, data=df, ax=axes[idx], palette=['#2ecc71', '#e74c3c'])
    axes[idx].set_title(f"{col} - Arıza Karşılaştırması", fontsize=12, fontweight='bold')
    axes[idx].set_xlabel("Arıza Durumu (0: Normal, 1: Arızalı)")
    axes[idx].set_ylabel(col)

# 6. kutucuğa Sıcaklık Farkı (Process Temp - Air Temp) ekleyelim (Isı yayılım hatası için kritik gösterge)
df['Temp Difference [K]'] = df['Process temperature [K]'] - df['Air temperature [K]']
sns.boxplot(x=target_col, y='Temp Difference [K]', data=df, ax=axes[5], palette=['#2ecc71', '#e74c3c'])
axes[5].set_title("Sıcaklık Farkı [K] (İşlem - Hava)", fontsize=12, fontweight='bold')
axes[5].set_xlabel("Arıza Durumu (0: Normal, 1: Arızalı)")
axes[5].set_ylabel("Fark [K]")

plt.savefig(os.path.join(FIG_DIR, "soru1b_sayisal_karsilastirma_kutu.png"), dpi=300)
plt.close()

# c) İki Boyutlu İlişkiler ve Kritik Arıza Bölgeleri (Torque vs Speed, Tool wear vs Torque)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.scatterplot(
    data=df, x='Rotational speed [rpm]', y='Torque [Nm]',
    hue=target_col, palette=['#27ae60', '#c0392b'], alpha=0.6,
    style=target_col, markers=['o', 'X'], ax=axes[0]
)
axes[0].set_title("Tork ve Dönme Hızı İlişkisi (Güç Arızaları)", fontsize=13, fontweight='bold')
axes[0].axhline(60, color='red', linestyle='--', alpha=0.5, label='Yüksek Tork Riski (>60 Nm)')
axes[0].axvline(1350, color='blue', linestyle='--', alpha=0.5, label='Düşük Devir Riski (<1350 rpm)')
axes[0].legend()

sns.scatterplot(
    data=df, x='Tool wear [min]', y='Torque [Nm]',
    hue=target_col, palette=['#27ae60', '#c0392b'], alpha=0.6,
    style=target_col, markers=['o', 'X'], ax=axes[1]
)
axes[1].set_title("Alet Aşınması ve Tork İlişkisi (Aşırı Yük Arızaları)", fontsize=13, fontweight='bold')
axes[1].axvline(200, color='purple', linestyle='--', alpha=0.5, label='Kritik Aşınma Eşiği (>200 dk)')
axes[1].legend()

plt.savefig(os.path.join(FIG_DIR, "soru1c_ariza_kumelenmesi.png"), dpi=300)
plt.close()

# Korelasyon Matrisi
plt.figure(figsize=(10, 8))
corr_cols = numeric_features + ['Temp Difference [K]', 'Machine failure']
corr_matrix = df[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", cbar=True, vmin=-1, vmax=1)
plt.title("Sensör Değişkenleri ve Arıza Arasındaki Korelasyon Matrisi", fontsize=13, fontweight='bold')
plt.savefig(os.path.join(FIG_DIR, "soru1c_korelasyon_matrisi.png"), dpi=300)
plt.close()

# ==============================================================================
# SORU 2: ARIZA TAHMİN MODELLERİNİN GELİŞTİRİLMESİ
# ==============================================================================
print("\n" + "="*60)
print("SORU 2 MODELLEME BAŞLIYOR...")
print("="*60)

# Hocanın kuralı:
# TWF, HDF, PWF, OSF, RNF KESİNLİKLE KULLANILMAYACAK.
# UDI ve Product ID çıkarılacak.
# Kullanılacak özellikler: 'Type', 'Air temperature [K]', 'Process temperature [K]',
#                         'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'

exclude_cols = ['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Temp Difference [K]']
feature_cols = [c for c in df.columns if c not in exclude_cols]
print("Kullanılan Bağımsız Değişkenler (X):", feature_cols)

X_raw = df[feature_cols].copy()
y = df[target_col].copy()

# 'Type' değişkenini One-Hot Encode yapalım (L, M, H)
X = pd.get_dummies(X_raw, columns=['Type'], drop_first=True)
print("One-Hot Encoding sonrası sütunlar:", list(X.columns))

# a) Veriyi Eğitim ve Test Kümelerine Ayırma
# Stratify=y kullanıyoruz çünkü %3.4'lük azınlık sınıfın her iki kümede aynı oranda olması şart!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Eğitim Kümesi: {X_train.shape[0]} örnek (Arıza: {y_train.sum()} adet, %{y_train.mean()*100:.2f})")
print(f"Test Kümesi:   {X_test.shape[0]} örnek (Arıza: {y_test.sum()} adet, %{y_test.mean()*100:.2f})")

# Ön İşleme: Ölçeklendirme (StandardScaler)
# KNN ve Lojistik Regresyon mesafe/katsayı temelli olduğu için ölçeklendirme zorunludur!
# Data leakage önlemek için scaler SADECE X_train üzerinde fit edilir, sonra hem train hem test transform edilir.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# b) 3 Algoritmanın Eğitimi ve Karşılaştırılması
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_split=10)
}

results = []
trained_models = {}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, model) in enumerate(models.items()):
    # Model eğitimi (KNN ve LogReg scaled veride, DT ağacı unscaled veya scaled fark etmez ama scaled ile tutarlı veriyoruz)
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    
    # Test tahminleri
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Metrikler
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.0
    
    cm = confusion_matrix(y_test, y_pred)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1]
    })
    
    # Confusion Matrix Görselleştirme
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                annot_kws={"size": 14, "weight": "bold"})
    axes[i].set_title(f"{name}\nAccuracy: {acc:.3f} | Recall: {rec:.3f}", fontsize=12, fontweight='bold')
    axes[i].set_xlabel("Tahmin Edilen Sınıf (Predicted)", fontsize=11)
    axes[i].set_ylabel("Gerçek Sınıf (Actual)", fontsize=11)
    axes[i].set_xticklabels(['Normal (0)', 'Arıza (1)'])
    axes[i].set_yticklabels(['Normal (0)', 'Arıza (1)'])

plt.savefig(os.path.join(FIG_DIR, "soru2b_confusion_matrices.png"), dpi=300)
plt.close()

# Sonuçları DataFrame yapalım
results_df = pd.DataFrame(results)
print("\n--- Soru 2(b): Model Performans Tablosu ---")
print(results_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "TP", "FN", "FP"]].to_string(index=False))

# Performans Metrikleri Karşılaştırma Grafiği
fig, ax = plt.subplots(figsize=(10, 6))
metrics_plot_df = pd.melt(results_df, id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                           var_name='Metrik', value_name='Skor')
sns.barplot(data=metrics_plot_df, x='Model', y='Skor', hue='Metrik', palette='tab10', ax=ax)
ax.set_title("Modellerin Performans Metrikleri Karşılaştırması", fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.set_ylabel("Skor", fontsize=12)
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height + 0.02),
                    ha='center', va='bottom', fontsize=9, rotation=0)

plt.savefig(os.path.join(FIG_DIR, "soru2b_model_karsilastirma.png"), dpi=300)
plt.close()

# Decision Tree Öznitelik Önemleri (Feature Importance)
dt_model = trained_models["Decision Tree"]
feat_importances = pd.Series(dt_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(9, 5))
sns.barplot(x=feat_importances.values, y=feat_importances.index, palette='viridis')
plt.title("Karar Ağacı (Decision Tree) Öznitelik Önem Düzeyleri", fontsize=13, fontweight='bold')
plt.xlabel("Önem Skoru (Gini Importance)", fontsize=11)
for i, v in enumerate(feat_importances.values):
    plt.text(v + 0.01, i, f"{v:.3f}", va='center', fontweight='bold')
plt.xlim(0, max(feat_importances.values) + 0.1)
plt.savefig(os.path.join(FIG_DIR, "soru2d_dt_feature_importance.png"), dpi=300)
plt.close()

# Karar Ağacı Yapısının İlk Seviyelerini Görselleştirme
plt.figure(figsize=(20, 10))
plot_tree(dt_model, max_depth=3, feature_names=list(X.columns), class_names=['Normal', 'Arıza'],
          filled=True, rounded=True, fontsize=10)
plt.title("Karar Ağacı Kural Yapısı (İlk 3 Seviye)", fontsize=14, fontweight='bold')
plt.savefig(os.path.join(FIG_DIR, "soru2d_karar_agaci_semasi.png"), dpi=300)
plt.close()

# ==============================================================================
# SORU 3 DESTEKLEYİCİ ANALİZ: EŞİK DEĞERİ (THRESHOLD) VE MALİYET TİCARETİ
# ==============================================================================
print("\n" + "="*60)
print("SORU 3 ANALİZİ BAŞLIYOR...")
print("="*60)

# Decision Tree ve Logistic Regression için eşik değeri değişimiyle Recall - Precision değişimi
thresholds = np.linspace(0.05, 0.95, 19)
thresh_results = []

y_proba_dt = dt_model.predict_proba(X_test_scaled)[:, 1]

for t in thresholds:
    y_pred_t = (y_proba_dt >= t).astype(int)
    cm_t = confusion_matrix(y_test, y_pred_t)
    rec_t = recall_score(y_test, y_pred_t, zero_division=0)
    prec_t = precision_score(y_test, y_pred_t, zero_division=0)
    f1_t = f1_score(y_test, y_pred_t, zero_division=0)
    thresh_results.append({
        'Threshold': t,
        'Recall (Arıza Yakalama)': rec_t,
        'Precision (Alarm Doğruluğu)': prec_t,
        'F1-Score': f1_t,
        'False Negatives (Kaçan Arıza)': cm_t[1, 0],
        'False Positives (Yanlış Alarm)': cm_t[0, 1]
    })

thresh_df = pd.DataFrame(thresh_results)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(thresh_df['Threshold'], thresh_df['Recall (Arıza Yakalama)'], 'r-o', label='Recall (Arıza Yakalama Oranı)', linewidth=2)
ax1.plot(thresh_df['Threshold'], thresh_df['Precision (Alarm Doğruluğu)'], 'b--s', label='Precision (Alarm İsabeti)', linewidth=2)
ax1.plot(thresh_df['Threshold'], thresh_df['F1-Score'], 'g-.^', label='F1-Score', linewidth=2)
ax1.set_xlabel("Karar Eşik Değeri (Classification Threshold)", fontsize=11)
ax1.set_ylabel("Skor (0-1)", fontsize=11)
ax1.set_title("Eşik Değerine Göre Recall vs Precision Değişimi (Maliyet & Risk Analizi)", fontsize=13, fontweight='bold')
ax1.legend(loc='center left')
ax1.grid(True, alpha=0.3)
plt.savefig(os.path.join(FIG_DIR, "soru3b_esik_degeri_tradeoff.png"), dpi=300)
plt.close()

# Overfitting İncelemesi: Train vs Test skorları karşılaştırması
train_acc_dt = accuracy_score(y_train, dt_model.predict(X_train_scaled))
test_acc_dt = accuracy_score(y_test, dt_model.predict(X_test_scaled))
train_f1_dt = f1_score(y_train, dt_model.predict(X_train_scaled))
test_f1_dt = f1_score(y_test, dt_model.predict(X_test_scaled))

print(f"\nDecision Tree (max_depth=6):")
print(f"Train Accuracy: {train_acc_dt:.4f} | Test Accuracy: {test_acc_dt:.4f}")
print(f"Train F1:       {train_f1_dt:.4f} | Test F1:       {test_f1_dt:.4f}")

# Aşırı derin ağaç (Overfitted Tree) simülasyonu
unpruned_dt = DecisionTreeClassifier(random_state=42) # no depth limit
unpruned_dt.fit(X_train_scaled, y_train)
train_f1_unpruned = f1_score(y_train, unpruned_dt.predict(X_train_scaled))
test_f1_unpruned = f1_score(y_test, unpruned_dt.predict(X_test_scaled))

print(f"\nSınırsız Derinlik Karar Ağacı (Aşırı Öğrenme / Overfitting Örneği):")
print(f"Train F1: {train_f1_unpruned:.4f} (Mükemmel ezberleme)")
print(f"Test F1:  {test_f1_unpruned:.4f} (Genelleme kaybı!)")

# Sonuçları CSV olarak kaydet
results_df.to_csv(os.path.join(BASE_DIR, "model_sonuclari.csv"), index=False)
comparison_stats.to_csv(os.path.join(BASE_DIR, "sayisal_karsilastirma_istatistikleri.csv"))
thresh_df.to_csv(os.path.join(BASE_DIR, "esik_degeri_analizi.csv"), index=False)

print("\n" + "="*60)
print("TÜM ANALİZLER VE GRAFİKLER BAŞARIYLA TAMAMLANDI!")
print(f"Grafikler klasörü: {FIG_DIR}")
print("="*60)
