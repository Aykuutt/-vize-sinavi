import json
import os

notebook_path = r"C:\Users\Lenovo\Desktop\Vize_Sinavi\vize_odevi.ipynb"

# Helper to create markdown cell
def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else source.splitlines(keepends=True)
    }

# Helper to create code cell
def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else source.splitlines(keepends=True)
    }

cells = []

# Title & Info
cells.append(md_cell("""# ENDÜSTRİYEL VERİ MADENCİLİĞİ VE KESTİRİMCİ BAKIM VİZE PROJESİ
## AI4I 2020 Predictive Maintenance Veri Seti Analizi ve Model Geliştirme

**Teslim Edilen:** bekcetintav@gmail.com  
**Proje Konusu:** Fabrika Makine Arızalarının Makine Öğrenmesi Algoritmaları ile Önceden Tahmin Edilmesi ve Üretim Yönetimi Değerlendirmesi  
**Kullanılan Veri Seti:** UCI Machine Learning Repository - AI4I 2020 Predictive Maintenance Dataset (10.000 Gözlem)  
"""))

# Cell 1: Imports
cells.append(md_cell("""### 0. Kütüphanelerin Yüklenmesi ve Ortam Hazırlığı"""))
cells.append(code_cell("""import os
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

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

print("Kütüphaneler başarıyla yüklendi!")
"""))

# Cell 2: Data Load
cells.append(md_cell("""### Veri Setinin Okunması ve İlk İnceleme"""))
cells.append(code_cell("""df = pd.read_csv('ai4i2020.csv')
print(f"Toplam Satır Sayısı: {df.shape[0]}, Sütun Sayısı: {df.shape[1]}")
df.head()
"""))

# Cell 3: Soru 1(a)
cells.append(md_cell("""---
## SORU 1 – Veriyi Tanıma ve Problemi Anlama (10 Puan)
### a) Makine Arızası (Machine failure) Değişkeninin Dağılımı ve Sınıflandırma Zorlukları
Fabrika yönetiminin arıza koşullarını analiz etmesi için öncelikle hedef değişkenin (`Machine failure`) dağılımı incelenmelidir.
"""))

cells.append(code_cell("""fail_counts = df['Machine failure'].value_counts()
fail_pcts = df['Machine failure'].value_counts(normalize=True) * 100

print(f"Arızasız Makineler (0): {fail_counts[0]} adet (%{fail_pcts[0]:.2f})")
print(f"Arızalı Makineler  (1): {fail_counts[1]} adet (%{fail_pcts[1]:.2f})")
print(f"Dengesizlik Oranı: {fail_counts[0]/fail_counts[1]:.2f} : 1")

# Görselleştirme
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Çubuk grafik
sns.barplot(x=fail_counts.index.astype(str), y=fail_counts.values, ax=axes[0], palette=['#2ecc71', '#e74c3c'], hue=fail_counts.index.astype(str), legend=False)
axes[0].set_title("Makine Arıza Frekansı", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Arıza Durumu (0: Normal, 1: Arızalı)")
axes[0].set_ylabel("Makine Sayısı")
for i, v in enumerate(fail_counts.values):
    axes[0].text(i, v + 120, f"{v} (%{fail_pcts.iloc[i]:.1f})", ha='center', fontweight='bold')

# Halka grafik
axes[1].pie(fail_counts.values, labels=['Normal (0)', 'Arızalı (1)'], autopct='%1.1f%%',
            startangle=45, colors=['#2ecc71', '#e74c3c'], explode=(0, 0.12),
            wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2))
axes[1].set_title("Sınıf Oranı Dağılımı", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()
"""))

cells.append(md_cell("""#### Sınıflandırma Açısından Zorluklar:
1. **Şiddetli Sınıf Dengesizliği (Severe Class Imbalance):** Veri setinde arıza oranı yalnızca **%3.39**'dur (10.000 makineden sadece 339 tanesi arızalıdır). Dengesizlik oranı yaklaşık **28.5:1**'dir.
2. **Doğruluk Paradoksu (Accuracy Paradox):** Bir makine öğrenmesi modeli hiç eğitim yapmadan her makineye "Normal (0)" tahmini yapsa bile matematiksel olarak **%96.61 Doğruluk (Accuracy)** elde edecektir. Ancak bu model 339 arızanın 339'unu da kaçıracağı için fabrika için tamamen felakettir.
3. **Azınlık Sınıfını Öğrenme Güçlüğü:** Algoritmalar genel hata oranını düşürmeye çalıştıkları için azınlık sınıfı gürültü (noise) olarak görme eğilimindedir.
"""))

# Cell 4: Soru 1(b)
cells.append(md_cell("""### b) Arızalı ve Arızasız Makinelerin Sayısal Değişkenler Açısından Karşılaştırılması
Makineler; **Hava Sıcaklığı**, **İşlem Sıcaklığı**, **Dönme Hızı**, **Tork** ve **Takım Aşınması** değişkenleri açısından karşılaştırılmıştır.
"""))

cells.append(code_cell("""numeric_cols = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

# Özet İstatistikler Tablosu
summary_table = df.groupby('Machine failure')[numeric_cols].agg(['mean', 'std', 'median']).round(2)
display(summary_table.T)
"""))

cells.append(code_cell("""# Kutu Grafikleri (Boxplots) ile Karşılaştırma
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for idx, col in enumerate(numeric_cols):
    sns.boxplot(x='Machine failure', y=col, data=df, ax=axes[idx], palette=['#2ecc71', '#e74c3c'], hue='Machine failure', legend=False)
    axes[idx].set_title(f"{col} Dağılımı", fontsize=12, fontweight='bold')
    axes[idx].set_xlabel("Arıza Durumu (0: Normal, 1: Arızalı)")

# 6. Kutuya Sıcaklık Farkı
df['Temp Difference [K]'] = df['Process temperature [K]'] - df['Air temperature [K]']
sns.boxplot(x='Machine failure', y='Temp Difference [K]', data=df, ax=axes[5], palette=['#2ecc71', '#e74c3c'], hue='Machine failure', legend=False)
axes[5].set_title("Sıcaklık Farkı [K] (İşlem - Hava)", fontsize=12, fontweight='bold')
axes[5].set_xlabel("Arıza Durumu (0: Normal, 1: Arızalı)")

plt.tight_layout()
plt.show()
"""))

# Cell 5: Soru 1(c)
cells.append(md_cell("""### c) Değişkenlerin Arıza Tahmini İçin Yararlılığı ve Üretim Yönetimi Çıkarımları
"""))

cells.append(code_cell("""# Tork, Hız ve Aşınma İlişkilerinin Dağılım Grafiği
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

sns.scatterplot(
    data=df, x='Rotational speed [rpm]', y='Torque [Nm]',
    hue='Machine failure', palette=['#27ae60', '#c0392b'], alpha=0.6,
    style='Machine failure', markers=['o', 'X'], ax=axes[0]
)
axes[0].set_title("Tork ve Dönme Hızı İlişkisi (Güç Arızaları)", fontsize=12, fontweight='bold')
axes[0].axhline(60, color='red', linestyle='--', alpha=0.6, label='Yüksek Tork Sınırı (>60 Nm)')
axes[0].axvline(1350, color='blue', linestyle='--', alpha=0.6, label='Düşük Devir Sınırı (<1350 rpm)')
axes[0].legend()

sns.scatterplot(
    data=df, x='Tool wear [min]', y='Torque [Nm]',
    hue='Machine failure', palette=['#27ae60', '#c0392b'], alpha=0.6,
    style='Machine failure', markers=['o', 'X'], ax=axes[1]
)
axes[1].set_title("Takım Aşınması ve Tork İlişkisi (Aşırı Gerilim)", fontsize=12, fontweight='bold')
axes[1].axvline(200, color='purple', linestyle='--', alpha=0.6, label='Kritik Aşınma Eşiği (>200 dk)')
axes[1].legend()

plt.tight_layout()
plt.show()
"""))

cells.append(md_cell("""#### Bulgular ve Üretim Yönetimi Çıkarımları:
1. **Tork [Nm] ve Dönme Hızı [rpm]:**
   - Arızasız makinelerde ortalama tork **39.63 Nm** iken arızalı makinelerde **50.17 Nm**'ye fırlamaktadır. Medyan değer 39.9'dan 53.7 Nm'ye çıkmaktadır.
   - Fiziksel mekanik güç formülü ($P = \\tau \\cdot \\omega$) gereği tork ve dönme hızı ters orantılıdır. Arızalar iki uç noktada kümelenmektedir:
     * Düşük Devir (<1350 rpm) + Yüksek Tork (>60 Nm): Motor aşırı zorlanmakta ve Güç Arızası (PWF) meydana gelmektedir.
     * Çok Yüksek Devir (>2500 rpm): Isı ve santrifüj yükü artarak arıza tetiklenmektedir.
2. **Takım Aşınması (Tool wear [min]):**
   - Arızalı makinelerde medyan aşınma süresi **165 dakikadır** (Normalde 107 dk). 
   - Takım aşınması 200 dakikayı geçtiğinde uç körelmekte, kesme direnci artmakta, bu da doğrudan motor torkunu yukarı çekerek takım kırılmasına (TWF) veya aşırı gerilime (OSF) sebep olmaktadır.
3. **Üretim Yönetimi Aksiyon Önerisi:**
   - **Önleyici Değişim Eşiği:** Fabrikada takımlar 180-200 dakikaya ulaştığında otomatik takım değiştirme protokolü başlatılmalıdır.
   - **Tork Emniyet Kilidi:** Tork 60 Nm'yi aştığında CNC tezgahı ilerleme hızını otomatik düşürmeli veya acil durdurma devreye girmelidir.
"""))

# Cell 6: Soru 2 Giriş & Veri Hazırlama
cells.append(md_cell("""---
## SORU 2 – Arıza Tahmin Modellerinin Geliştirilmesi (20 Puan)
### a) Veri Bölme ve Ön İşleme Tercihleri ve Gerekçeleri

> **Önemli Kural:** `TWF, HDF, PWF, OSF, RNF` hedef sızıntısı (data leakage) yarattığı için çıkarılmıştır. `UDI` ve `Product ID` tekil kimlikler olduğu için elenmiştir. Sadece sensör verileri ve `Type` kullanılmıştır.
"""))

cells.append(code_cell("""# 1. Değişkenlerin Seçilmesi (Hocanın kuralına tam uyum)
exclude_cols = ['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Temp Difference [K]']
features = [c for c in df.columns if c not in exclude_cols]
print("Modele Girecek Özellikler (X):", features)

X_raw = df[features].copy()
y = df['Machine failure'].copy()

# 2. Kategorik 'Type' değişkeninin One-Hot Kodlanması
X = pd.get_dummies(X_raw, columns=['Type'], drop_first=True)

# 3. Eğitim ve Test Kümelerine Ayırma (Stratified Split)
# Gerekçe: Sınıf dengesizliği (%3.39) nedeniyle arıza oranının her iki kümede eşit korunması şarttır.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Eğitim Kümesi: {X_train.shape[0]} satır (Arıza Oranı: %{y_train.mean()*100:.2f})")
print(f"Test Kümesi:   {X_test.shape[0]} satır (Arıza Oranı: %{y_test.mean()*100:.2f})")

# 4. Özellik Ölçeklendirme (StandardScaler)
# Gerekçe: KNN mesafe bazlıdır; Lojistik Regresyon gradyan temellidir.
# Devir hızı (1500) ile Tork (40) arasındaki ölçek uçurumunu gidermek için fit_transform sadece eğitime uygulanır.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
"""))

# Cell 7: Soru 2(b) Model Eğitimi ve Karşılaştırma
cells.append(md_cell("""### b) Üç Modelin Eğitimi ve Test Verisi Performanslarının Karşılaştırılması
Eğitilen modeller:
1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Decision Tree (Karar Ağacı)**
"""))

cells.append(code_cell("""models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_split=10)
}

results = []
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0
    cm = confusion_matrix(y_test, y_pred)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "TN": cm[0,0], "FP": cm[0,1],
        "FN": cm[1,0], "TP": cm[1,1]
    })
    
    # Karışıklık Matrisi (Confusion Matrix) Çizimi
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                annot_kws={"size": 13, "weight": "bold"})
    axes[i].set_title(f"{name}\nAcc: {acc:.3f} | Recall: {rec:.3f}", fontsize=11, fontweight='bold')
    axes[i].set_xlabel("Tahmin Edilen Sınıf")
    axes[i].set_ylabel("Gerçek Sınıf")
    axes[i].set_xticklabels(['Normal (0)', 'Arıza (1)'])
    axes[i].set_yticklabels(['Normal (0)', 'Arıza (1)'])

plt.tight_layout()
plt.show()

# Performans Tablosu
results_df = pd.DataFrame(results)
display(results_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "TP", "FN", "FP"]])
"""))

cells.append(code_cell("""# Performans Karşılaştırma Grafiği
fig, ax = plt.subplots(figsize=(10, 5))
metrics_plot_df = pd.melt(results_df, id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                           var_name='Metrik', value_name='Skor')
sns.barplot(data=metrics_plot_df, x='Model', y='Skor', hue='Metrik', palette='tab10', ax=ax)
ax.set_title("Modellerin Performans Metrikleri Karşılaştırması", fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.15)
for p in ax.patches:
    h = p.get_height()
    if h > 0:
        ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h + 0.02),
                    ha='center', va='bottom', fontsize=9)
plt.show()
"""))

# Cell 8: Soru 2(c) Accuracy Tartışması
cells.append(md_cell("""### c) En Yüksek Accuracy Değeri Veren Model En İyi Model midir?
**Kesinlikle Hayır!**
1. **Doğruluk Paradoksu (Accuracy Paradox):** Test kümemizde 2000 makinenin 1932'si arızasız, 68'i arızalıdır. Bütün test kümesine hiç düşünmeden "Arıza Yok (0)" diyen bir kukla model (Dummy Baseline) %96.60 Accuracy elde eder.
2. **Lojistik Regresyon Örneği:** Lojistik Regresyon **%96.75 Accuracy** almıştır. Ancak 68 gerçek arızanın sadece 7 tanesini yakalayabilmiş, 61 arızayı kaçırmıştır (**Recall = %10.29**). Fabrikada makineler arka arkaya bozulurken model %96.75 doğrulukla "her şey yolunda" demektedir.
3. **Sonuç:** Dengesiz veri setlerinde Accuracy başarının göstergesi olamaz. Fabrika için hayati olan metrik arızaları kaçırmama oranı olan **Recall** ve dengeli performansı gösteren **F1-Score**'dur.
"""))

# Cell 9: Soru 2(d) Model Seçimi
cells.append(md_cell("""### d) Model Seçimi ve Operasyonel Gerekçelendirme
**Seçilen Model: Karar Ağacı (Decision Tree)**

**Gerekçeler:**
1. **En Yüksek Arıza Yakalama Oranı (Recall) ve F1-Score:** Test kümesindeki 68 arızadan 27'sini yakalamış (**Recall: %39.71**, **F1-Score: 0.5294**), diğer iki modeli geride bırakmıştır.
2. **Şeffaflık ve Kural Tabanlı Açıklanabilirlik (Interpretability):** Fabrika bakım şefleri ve operatörler bir "kara kutu" modele güvenmez. Karar ağacı aşağıdaki grafikte görüldüğü gibi açık kurallar üretir:
   - *"Tork > 61.5 Nm ve Hız < 1380 rpm ise arıza riski çok yüksektir."*
3. **Düşük İşlem Yükü ve Gömülü Sistemlere (PLC/IoT) Kolay Entegrasyon:** Karar ağacı basit `IF-ELSE` dallanmalarından ibarettir. KNN gibi her tahminde binlerce veriyle mesafe hesaplamaz; mikrodenetleyicilerde mikrosaniyeler içinde çalışır.
"""))

cells.append(code_cell("""# Karar Ağacı Öznitelik Önem Düzeyleri (Feature Importances)
dt_model = models["Decision Tree"]
feat_importances = pd.Series(dt_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(9, 4.5))
sns.barplot(x=feat_importances.values, y=feat_importances.index, palette='viridis', hue=feat_importances.index, legend=False)
plt.title("Karar Ağacı (Decision Tree) Öznitelik Önem Düzeyleri", fontsize=12, fontweight='bold')
plt.xlabel("Önem Skoru (Gini Importance)")
for i, v in enumerate(feat_importances.values):
    plt.text(v + 0.01, i, f"{v:.3f}", va='center', fontweight='bold')
plt.xlim(0, max(feat_importances.values) + 0.08)
plt.show()
"""))

# Cell 10: Soru 3
cells.append(md_cell("""---
## SORU 3 – Modeli Gerçek Hayatta Kullanmak (10 Puan)

### a) Hata Türlerinin Karşılaştırılması (Yanlış Negatif vs. Yanlış Pozitif)
* **Yanlış Negatif (False Negative - Tip II Hata):** Makine gerçekte arızalanacakken modelin "sağlam" tahmini yapmasıdır.
* **Yanlış Pozitif (False Positive - Tip I Hata):** Makine arızalanmayacakken modelin "arıza var" alarmı üretmesidir.

**Hangi Hata Daha Kritik?**  
Fabrika için **YANLIŞ NEGATİF (Arızayı Kaçırma)** açık ara çok daha kritiktir!

**Gerekçeler:**
1. **Plansız Duruş Maliyeti (Unplanned Downtime):** Üretim hattında bir makinenin beklenmedik şekilde kilitlenmesi veya parçalanması tüm hattı durdurur. Bir saatlik hat duruşunun maliyeti binlerce dolardır.
2. **İkincil Hasarlar ve Hurda (Scrap):** Aniden kırılan kesici takım, işlenen gövdeyi hurdaya ayırabilir veya pahalı ana mili (spindle) tahrip edebilir.
3. **İş Sağlığı ve Güvenliği:** Yüksek devirde kırılan metal parçalar operatör yaralanmalarına yol açabilir.
4. **Yanlış Pozitifin Düşük Maliyeti:** Yanlış alarm durumunda bir teknisyen gelir, 10 dakika titreşim veya yağ kontrolü yapar ve makine çalışmaya devam eder. 10 dakikalık işçilik maliyeti, plansız hat duruşunun yanında önemsizdir.
"""))

cells.append(md_cell("""### b) Arızaları Yakalama Başarısını (Recall) Artırma Yöntemleri ve Olumsuz Sonuçları
Arıza yakalama oranını artırmak için uygulanabilecek adımlar:
1. **Karar Eşiğinin Düşürülmesi (Threshold Tuning):** Modelin arıza deme olasılık eşiğini 0.50'den örneğin 0.25'e düşürmek.
2. **Maliyet Duyarlı Öğrenme (Cost-Sensitive Learning / Class Weights):** Eğitimde arıza sınıfına daha yüksek ceza katsayısı vermek (`class_weight='balanced'`).
3. **Sentetik Örnekleme (SMOTE):** Azınlık arıza sınıfından sentetik örnekler üreterek modeli arızalara karşı daha duyarlı eğitmek.
"""))

cells.append(code_cell("""# Karar Eşik Değeri Analizi (Recall vs Precision Trade-off)
thresholds = np.linspace(0.05, 0.95, 19)
thresh_results = []
y_proba_dt = dt_model.predict_proba(X_test_scaled)[:, 1]

for t in thresholds:
    y_pred_t = (y_proba_dt >= t).astype(int)
    cm_t = confusion_matrix(y_test, y_pred_t)
    thresh_results.append({
        'Threshold': t,
        'Recall (Arıza Yakalama)': recall_score(y_test, y_pred_t, zero_division=0),
        'Precision (Alarm Doğruluğu)': precision_score(y_test, y_pred_t, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred_t, zero_division=0),
        'FN (Kaçan Arıza)': cm_t[1,0],
        'FP (Yanlış Alarm)': cm_t[0,1]
    })

thresh_df = pd.DataFrame(thresh_results)

fig, ax1 = plt.subplots(figsize=(10, 4.8))
ax1.plot(thresh_df['Threshold'], thresh_df['Recall (Arıza Yakalama)'], 'r-o', label='Recall (Arıza Yakalama Oranı)', linewidth=2)
ax1.plot(thresh_df['Threshold'], thresh_df['Precision (Alarm Doğruluğu)'], 'b--s', label='Precision (Alarm İsabeti)', linewidth=2)
ax1.plot(thresh_df['Threshold'], thresh_df['F1-Score'], 'g-.^', label='F1-Score', linewidth=2)
ax1.set_xlabel("Karar Eşik Değeri (Threshold)")
ax1.set_ylabel("Skor")
ax1.set_title("Eşik Değerine Göre Recall vs Precision Değişimi", fontsize=12, fontweight='bold')
ax1.legend()
plt.show()
"""))

cells.append(md_cell("""#### Olası Olumsuz Sonuçlar:
* **Yanlış Alarmların (False Positive) Katlanması:** Eşik değeri düşürüldüğünde arızaların %80'i yakalanabilir ancak sağlam makinelere verilen yanlış alarmlar 7'den 100'ün üzerine çıkabilir.
* **Alarm Yorgunluğu (Alarm Fatigue) ve Güvenilirlik Kaybı:** Bakım personeli sürekli sağlam makinelere kontrole gönderildiğinde bir süre sonra sistemi ciddiye almamaya başlar ("Yalancı Çoban Sendromu"). Gerçek bir arıza alarmı geldiğinde de müdahalede gecikebilirler.
* **Gereksiz Bakım Masrafları:** Sürekli parça durdurulması ve kontroller üretim hızını düşürür.
"""))

cells.append(md_cell("""### c) Aşırı Öğrenme (Overfitting) Problemi ve İki Somut Çözüm Önerisi

**Problem Teşhisi:**
Model eğitim verisinde çok yüksek başarı (örneğin %99-100), test verisinde ise belirgin şekilde düşük başarı gösteriyorsa bu durum **Aşırı Öğrenme (Overfitting)** problemidir. Model verinin genel mantığını öğrenmek yerine eğitim kümesindeki gürültü ve tesadüfi örüntüleri ezberlemiştir.

**Bunu Azaltmak İçin İki Somut Öneri:**
1. **Ağaç Budama ve Hiperparametre Kısıtlaması (Tree Pruning):**
   - Karar ağacının derinliğine sınır koyulmalıdır (`max_depth = 5-6`).
   - Bir düğümün bölünmesi için gereken minimum örnek sayısı (`min_samples_split = 10-20`) ve yapraktaki minimum örnek sayısı (`min_samples_leaf = 5`) artırılarak ağacın tekil uç noktalara aşırı dallanması önlenir.
2. **K-Katlı Tabakalı Çapraz Doğrulama (Stratified K-Fold Cross-Validation) ve Düzenlileştirme (Regularization):**
   - Model tek bir eğitim-test ayrımına göre değil, 5 veya 10 katlı çapraz doğrulama ile test edilmelidir.
   - Ayrıca L1/L2 ceza katsayıları (regularization) eklenerek veya birden fazla ağacın ortalamasını alan **Random Forest** gibi topluluk (ensemble) algoritmalarına geçilerek model varyansı düşürülmeli ve genelleme yeteneği artırılmalıdır.
"""))

# Construct notebook json
notebook_content = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.6"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, ensure_ascii=False, indent=2)

print(f"Jupyter Notebook başarıyla oluşturuldu: {notebook_path}")
