# ENDÜSTRİYEL VERİ MADENCİLİĞİ VE KESTİRİMCİ BAKIM VİZE PROJE RAPORU
**Konu:** AI4I 2020 Predictive Maintenance Veri Seti ile Makine Arıza Tahmini ve Üretim Yönetimi Değerlendirmesi  
**Teslim Edilen:** bekcetintav@gmail.com  
**Veri Seti:** UCI Machine Learning Repository (AI4I 2020 Predictive Maintenance Dataset - 10.000 Kayıt)  

---

## YÖNETİCİ ÖZETİ (EXECUTIVE SUMMARY)
Bu çalışma, modern bir üretim tesisindeki frezeleme ve işleme merkezlerinden toplanan 10.000 adet makine sensör kaydını inceleyerek kestirimci bakım (predictive maintenance) stratejilerini optimize etmeyi amaçlamaktadır. Üretim yönetiminin en temel hedefleri; plansız duruş sürelerini (unplanned downtime) sıfıra indirmek, bakım maliyetlerini minimize etmek ve iş güvenliğini sağlamaktır. 

Çalışma kapsamında verideki **%3.39'luk aşırı sınıf dengesizliği** incelenmiş; `UDI, Product ID` ve hedef sızıntısına (data leakage) yol açan arıza göstergeleri (`TWF, HDF, PWF, OSF, RNF`) veri setinden çıkarılmıştır. **Logistic Regression**, **K-Nearest Neighbors (KNN)** ve **Decision Tree (Karar Ağacı)** algoritmaları tabakalı (stratified) veri bölme ve ölçeklendirme adımlarından sonra eğitilmiş; fabrika operasyonları açısından **Karar Ağacı (Decision Tree)** en uygun model olarak seçilmiştir.

---

## BÖLÜM 1: VERİYİ TANIMA VE PROBLEMİ ANLAMA (SORU 1 - 10 PUAN)

### 1.1. Makine Arızası (Machine failure) Dağılımı ve Sınıflandırma Zorlukları (Soru 1.a)
Veri setinde yer alan hedef değişken `Machine failure` ikili (binary) bir değişkendir:
* **0 (Arıza Yok / Normal Durum):** 9.661 adet (**%96.61**)
* **1 (Arıza Var / Arızalı Durum):** 339 adet (**%3.39**)
* **Dengesizlik Oranı (Imbalance Ratio):** Yaklaşık **28.5 : 1**

```
Hedef Değişken Dağılımı:
[Normal Makineler (0)] ████████████████████████████████████████ 96.61% (9661)
[Arızalı Makineler (1)] █ 3.39% (339)
```

#### Sınıflandırma Açısından İçerdiği Temel Zorluklar:
1. **Doğruluk Paradoksu (Accuracy Paradox):**
   Geleneksel makine öğrenmesi algoritmaları toplam hata fonksiyonunu küçültmeye odaklanır. Eğer model hiçbir sensör verisine bakmaksızın fabrikadaki her makineye körü körüne *"Arıza Yok (0)"* tahmini yaparsa, test verisinde matematiksel olarak **%96.61 Accuracy (Doğruluk)** skoru elde edecektir. Ancak bu model gerçekte meydana gelen 339 arızanın tamamını kaçıracağı için fabrikayı plansız duruş felaketine sürükleyecektir.
2. **Azınlık Sınıf Bilgi Yetersizliği ve Gürültü Riski:**
   339 arıza örneği, 5 farklı arıza türünün (TWF, HDF, PWF, OSF, RNF) karmaşık kombinasyonlarını barındırmaktadır. Model, sınırlı sayıdaki bu örnekleri genel kurallar yerine eğitim kümesinin "gürültüsü (noise)" veya "uç değerleri (outliers)" olarak algılama eğilimi taşır.
3. **Metrik Seçimi Zorunluluğu:**
   Sınıf dengesizliği nedeniyle `Accuracy` metriği tek başına anlamsızdır. Değerlendirmede fabrikanın arızaları kaçırmama yeteneğini ölçen **Recall (Duyarlılık)**, alarm isabetini ölçen **Precision (Kesinlik)** ve bunların harmonik ortalaması olan **F1-Score** ile **ROC-AUC** metrikleri esas alınmalıdır.

---

### 1.2. Sayısal Değişkenler Açısından Karşılaştırma (Soru 1.b)
Arızalı ve arızasız makineler üretim süreçlerini doğrudan belirleyen 5 sayısal sensör değişkeni ve türetilen sıcaklık farkı açısından karşılaştırılmıştır:

| Değişken | Arıza = 0 (Normal) Ortalama ± Std | Arıza = 0 Medyan | Arıza = 1 (Arızalı) Ortalama ± Std | Arıza = 1 Medyan | Fiziksel & Operasyonel Farklılık |
|---|---|---|---|---|---|
| **Tool wear [min] (Aşınma)** | 106.69 ± 62.95 | 107.0 dk | **143.78 ± 72.76** | **165.0 dk** | Arızalı takımlar medyan olarak **%54 daha fazla** aşınmıştır. |
| **Torque [Nm] (Tork)** | 39.63 ± 9.47 | 39.9 Nm | **50.17 ± 16.37** | **53.7 Nm** | Arızalı makinelerde tork ortalaması belirgin yüksek ve varyansı 2 kat geniştir. |
| **Rotational speed [rpm] (Hız)** | 1540.26 ± 167.39 | 1507 rpm | **1496.49 ± 384.94** | **1365 rpm** | Arızalı makinelerin standart sapması devasadır (düşük devirde boğulma veya aşırı hız). |
| **Air temperature [K]** | 299.97 ± 1.99 | 300.0 K | **300.89 ± 2.07** | **301.6 K** | Arızalı makineler yaklaşık 1 K daha sıcak ortamda çalışmaktadır. |
| **Process temp [K]** | 309.99 ± 1.49 | 310.0 K | **310.29 ± 1.36** | **310.4 K** | İşlem sıcaklığı arızalı grupta hafif yükselmektedir. |
| **Sıcaklık Farkı [K]** | 10.02 ± 1.00 | 10.0 K | **9.40 ± 1.32** | **9.3 K** | Sıcaklık farkının düşmesi ısı tahliyesini (HDF) zorlaştırmaktadır. |

#### Grafiksel Bulgular:
* `figures/soru1b_sayisal_karsilastirma_kutu.png`: Kutu grafiklerinde (Boxplots) özellikle **Takım Aşınması (Tool wear)** ve **Tork (Torque)** değişkenlerinde arızalı grubun çeyreklikler arası aralığının (IQR) normal gruptan belirgin şekilde yukarı kaydığı görülmektedir.
* `figures/soru1c_ariza_kumelenmesi.png`: Tork - Devir ve Aşınma - Tork scatter grafiklerinde arızaların rastgele dağılmadığı, belirli kritik eşik bölgelerinde kümelendiği görülmektedir.

---

### 1.3. Değişkenlerin Yararlılığı ve Üretim Yönetimi Çıkarımları (Soru 1.c)

#### En Yararlı Değişkenler ve Mühendislik Gerekçeleri:
1. **Tork [Nm] ve Dönme Hızı [rpm] (Mekanik Güç Çifti):**
   * *Mekanik Gerekçe:* Talaşlı imalatta mekanik güç formülü $P = \tau \times \omega = \text{Torque} \times \left(\frac{2\pi \times \text{Speed}}{60}\right)$ şeklindedir. Dağılım analizinde arızaların iki uçta toplandığı tespit edilmiştir:
     1. **Aşırı Tork / Düşük Devir Bölgesi ($\text{Tork} > 60\text{ Nm}, \text{Hız} < 1350\text{ rpm}$):** Motor yüksek kesme direncini karşılamak için aşırı tork üretmekte, bu durum hem elektrik motorunda aşırı güç çekilmesine (PWF - Power Failure) hem de mekanik mili zorlayarak aşırı gerilim kırılmasına (OSF - Overstrain Failure) sebep olmaktadır.
     2. **Yüksek Devir Bölgesi ($> 2500\text{ rpm}$):** Yataklarda aşırı sürtünme ve kontrolsüz ısınma meydana gelmektedir.
   * Korelasyon analizinde Tork, arıza ile en yüksek pozitif korelasyona (**r = 0.191**) sahip değişkendir.
2. **Takım Aşınması [min] (Tool wear):**
   * *Gerekçe:* Takım aşınması kümülatif (birikimli) bir hasardır. 180-200 dakikayı geçen takımlarda kesici geometrisi bozulmakta, kesme kuvvetleri artmakta ve doğrudan tork sıçramalarına yol açmaktadır. Arızalı makinelerde medyan aşınma 165 dakikadır.

#### Üretim Yönetimi Açısından Anlamlı Çıkarımlar ve Eylem Planı:
* **Önleyici Değiştirme Politikası:** Fabrika bakım birimi, takımların çalışma süresini takip ederek **180-200 dakika bandına ulaşan takımları henüz arıza meydana gelmeden vardiya aralarında değiştirmelidir**. Bu basit kural, takım kırılmalarını (TWF) ve aşırı gerilim arızalarını (OSF) doğrudan engelleyecektir.
* **Otomatik Yük Sınırlayıcı (CNC Feed-rate Override):** Sensörlerden okunan anlık tork değeri 60 Nm sınırına yaklaştığında CNC kontrol ünitesi ilerleme hızını otomatik olarak düşürmeli; tork 65 Nm'yi aştığında acil güvenli durdurma tetiklenmelidir.

---

## BÖLÜM 2: ARIZA TAHMİN MODELLERİNİN GELİŞTİRİLMESİ (SORU 2 - 20 PUAN)

### 2.1. Veri Bölme ve Ön İşleme Tercihleri ve Gerekçeleri (Soru 2.a)

#### Değişken Seçimi ve Veri Sızıntısının (Data Leakage) Önlenmesi:
* **Kullanılmayan Değişkenler:**
  * `UDI` ve `Product ID`: Bu alanlar yapay sıra numarası ve ürün barkodudur. Arıza fiziğiyle hiçbir nedensel bağlantısı yoktur; modele verilmesi ezberlemeye (overfitting) yol açar.
  * `TWF, HDF, PWF, OSF, RNF`: Bu değişkenler arızanın türünü belirten **sonuç göstergeleridir**. Arıza gerçekleştiğinde tetiklenirler. Bu değişkenlerin modele verilmesi doğrudan **Data Leakage (Hedef Sızıntısı)** oluşturur ve gerçek hayat senaryosunda (arıza henüz olmadan tahmin etme) fabrikada bulunamayacak bilgilerin modele hileli şekilde verilmesi anlamına gelir.
* **Kullanılan Bağımsız Değişkenler ($X$):**
  * `Type` (Kategorik: H, M, L kalite seviyesi)
  * `Air temperature [K]`
  * `Process temperature [K]`
  * `Rotational speed [rpm]`
  * `Torque [Nm]`
  * `Tool wear [min]`

#### Ön İşleme Tercihleri:
1. **Kategorik Kodlama (One-Hot Encoding):** `Type` değişkeni kukla değişkenlere dönüştürülmüştür (`Type_L`, `Type_M`).
2. **Tabakalı Veri Bölme (Stratified Train-Test Split - %80 / %20):**
   * Toplam 10.000 veri; 8.000 Eğitim (Train), 2.000 Test olarak ayrılmıştır.
   * *Gerekçe:* Arıza oranı %3.39 gibi çok düşük bir düzeyde olduğu için standart rastgele bölme yapıldığında test kümesinde arıza oranı kayabilir. `stratify=y` kullanılarak hem eğitim (%3.39 arıza, 271 adet) hem test (%3.40 arıza, 68 adet) kümesinde gerçek dünya oranları korunmuştur.
3. **Standartlaştırma / Ölçeklendirme (StandardScaler):**
   * *Gerekçe:* K-En Yakın Komşu (KNN) Öklid mesafesi hesaplar; Dönme Hızı (~1500 rpm) ile Tork (~40 Nm) arasındaki sayısal mertebe farkı, KNN modelinin torku tamamen göz ardı etmesine neden olur. Lojistik Regresyon da gradyan güncellemesi ve L2 ceza teriminde öznitelik ölçeklerine duyarlıdır.
   * *Data Leakage Güvencesi:* `StandardScaler` yalnızca `X_train` üzerinde fit edilmiş (`scaler.fit_transform`), `X_test` ise eğitimden öğrenilen ortalama ve standart sapma ile dönüştürülmüştür (`scaler.transform`).

---

### 2.2. Modellerin Test Verisi Üzerindeki Performans Karşılaştırması (Soru 2.b)
Test kümesinde yer alan **2.000 makine kaydı** (1.932 Normal, 68 Arızalı) üzerinde elde edilen test performans sonuçları:

| Model | Accuracy (Doğruluk) | Precision (Kesinlik) | Recall (Duyarlılık) | F1-Score | ROC-AUC | TP (Doğru Arıza) | FN (Kaçan Arıza) | FP (Yanlış Alarm) | TN (Doğru Normal) |
|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.9675 | 0.6364 | 0.1029 | 0.1772 | 0.8995 | 7 | 61 | 4 | 1928 |
| **K-Nearest Neighbors (KNN)** | 0.9740 | 0.8333 | 0.2941 | 0.4348 | 0.8292 | 20 | 48 | 4 | 1928 |
| **Decision Tree (max_depth=6)** | **0.9760** | **0.7941** | **0.3971** | **0.5294** | **0.9366** | **27** | **41** | **7** | **1925** |

*Görsel Kanıtlar:*
* Modellerin karışıklık matrisleri `figures/soru2b_confusion_matrices.png` dosyasında yer almaktadır.
* Metrik karşılaştırma çubuk grafiği `figures/soru2b_model_karsilastirma.png` dosyasında sunulmuştur.

---

### 2.3. En Yüksek Accuracy Veren Model En İyi Model midir? (Soru 2.c)
**Cevap: Kesinlikle Hayır!**

#### Gerekçeli Tartışma:
1. **Kör Tahminci (Dummy Baseline) Kıyaslaması:**
   Test kümesinde 2.000 makinenin 1.932'si sağlamdır. Eğer hiçbir makine öğrenmesi yapmayıp tüm makinelere "Arıza Yok (0)" deseydik, elde edeceğimiz Accuracy tam **%96.60** (1932 / 2000) olacaktı.
2. **Lojistik Regresyonun Aldatıcı Başarısı:**
   Lojistik Regresyon modeli **%96.75 Accuracy** almıştır. İlk bakışta mükemmel bir skor gibi görünse de detaylar incelendiğinde:
   * Test verisinde gerçekten bozulan 68 makinenin **yalnızca 7 tanesini** yakalayabilmiştir.
   * **61 adet arızayı tamamen kaçırmıştır (FN = 61, Recall = %10.29).**
   * Yani bu model fabrikaya kurulursa makinelerin %90'ı göz göre göre bozulacak, üretim duracak ancak model "%96.75 doğruyum" diyecektir.
3. **Sonuç:**
   Sınıf dengesizliği içeren kestirimci bakım problemlerinde **Accuracy metriği tek başına yanıltıcı ve tehlikelidir**. Üretim yönetiminin odaklanması gereken metrik, arızaları yakalama oranı olan **Recall** ve hassasiyet ile dengesini kuran **F1-Score**'dur.

---

### 2.4. Model Seçimi ve Operasyonel Gerekçelendirme (Soru 2.d)
**Fabrika İçin Seçilen Model: KARAR AĞACI (DECISION TREE)**

#### Seçim Gerekçeleri (Metriksel ve Operasyonel):
1. **En Yüksek Arıza Yakalama Performansı:**
   Karar Ağacı, test verisinde 27 arızayı tespit ederek **%39.71 Recall** ve **0.5294 F1-Score** ile diğer modellerin belirgin şekilde önüne geçmiştir. ROC-AUC skoru da **0.9366** ile en yüksek seviyededir.
2. **Açıklanabilirlik ve Beyaz Kutu (White-Box) Avantajı:**
   KNN ve Lojistik Regresyon modelleri matematiksel uzaklık ve sigmoid fonksiyonları ile çalıştığı için fabrika operatörlerine doğrudan bir fiziksel mantık sunamaz. Karar Ağacı ise şeffaf kurallar üretir (`figures/soru2d_karar_agaci_semasi.png`):
   * *Kural Örneği:* `IF Torque > 61.5 Nm AND Rotational speed <= 1380 rpm THEN Arıza Riski = YÜKSEK (%85)`
   * *Operasyonel Yarar:* Bakım mühendisi makinenin neden arıza sinyali verdiğini anlar (örneğin aşırı tork nedeniyle mil zorlanması) ve doğru alet çantasıyla makinenin başına gider.
3. **Endüstriyel Gömülü Sistemlere (PLC / Edge IoT) Kolay Entegrasyon:**
   Karar Ağacı, basit `IF-ELSE` mantıksal bloklarına derlenebilir. KNN algoritması her yeni tahminde 8.000 adet eğitim verisiyle Öklid mesafesi hesaplamak zorundayken, Karar Ağacı CNC tezgahının üzerindeki düşük maliyetli bir mikrodenetleyicide veya PLC ünitesinde mikrosaniyeler içinde çalışır; sıfır ağ gecikmesiyle gerçek zamanlı koruma sağlar.

---

## BÖLÜM 3: MODELİ GERÇEK HAYATTA KULLANMAK (SORU 3 - 10 PUAN)

### 3.1. Hata Türlerinin Karşılaştırılması ve Kritiklik Değerlendirmesi (Soru 3.a)
Problemdeki iki temel hata türü şunlardır:
* **Yanlış Negatif (False Negative - Tip II Hata):** Makine arızalanacakken modelin *"Normal/Arıza Yok"* tahmini yapması.
* **Yanlış Pozitif (False Positive - Tip I Hata):** Makine sağlamken modelin *"Arıza Olacak"* alarmı vermesi.

#### Hangi Hata Fabrika İçin Daha Kritik?
Fabrika operasyonları açısından **YANLIŞ NEGATİF (False Negative) AÇIK ARA ÇOK DAHA KRİTİKTİR.**

```
HATA TÜRLERİ VE MALİYET KIYASLAMASI:
┌───────────────────────────────────────┬───────────────────────────────────────┐
│     YANLIŞ NEGATİF (ARIZAYI KAÇIRMA)  │     YANLIŞ POZİTİF (YANLIŞ ALARM)     │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ • Plansız hat duruşu (saatlerce)      │ • 10-15 dakikalık teknisyen kontrolü  │
│ • İşlenen parçanın hurdaya gitmesi    │ • Küçük iş gücü kaybı                 │
│ • Fener milinde (spindle) ağır hasar  │ • Makine kapatılmadan kontrol imkanı  │
│ • Müşteriye teslimat gecikme cezası   │                                       │
│ • Operatör can güvenliği riski        │                                       │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ Maliyet: 50.000 TL - 500.000 TL+      │ Maliyet: 200 TL - 500 TL (İşçilik)    │
└───────────────────────────────────────┴───────────────────────────────────────┘
```

**Mühendislik Gerekçelendirmesi:**
Talaşlı imalatta yüksek devirde çalışan bir freze ucunun işleme esnasında kırılması durumunda parça fırlayabilir, makine gövdesi delinebilir ve ana mil rulmanları dağılabilir. Bu durum fabrikanın haftalık üretim programını kilitler. Oysa yanlış bir arıza alarmında bakım görevlisi el tipi titreşim ölçüm cihazıyla makinenin yanına gider, 10 dakika ölçüm alır, bir anormallik olmadığını teyit eder ve tezgah çalışmaya devam eder. Aradaki maliyet farkı en az 100-200 kat mertebesindedir.

---

### 3.2. Arıza Yakalama Başarısını (Recall) Artırma Yöntemleri ve Olası Olumsuz Sonuçları (Soru 3.b)

#### Arıza Yakalama Başarısını Artıracak Değişiklikler:
1. **Sınıf Ağırlıklandırması (Class Weight / Cost-Sensitive Learning):**
   Model eğitiminde arıza sınıfına çoğunluk sınıfının 10-20 katı ceza katsayısı verilebilir (`class_weight={0: 1, 1: 15}`). Model bir arızayı kaçırdığında algoritma çok daha büyük kayıp (loss) hesaplar.
2. **Karar Eşik Değerinin (Classification Threshold) Düşürülmesi:**
   Varsayılan sınıflandırma olasılık eşiği $0.50$'dir. Bu eşik değeri $0.20$ veya $0.25$ seviyesine çekildiğinde, model arıza ihtimalini en ufak biçimde hissettiğinde dahi alarm üretecektir.
3. **Sentetik Azınlık Aşırı Örnekleme (SMOTE):**
   Eğitim verisindeki arıza örnekleri sentetik olarak artırılarak arızaların karar uzayındaki temsili genişletilebilir.

#### Bu Değişikliklerin Olası Olumsuz Sonuçları:
* **Yanlış Pozitiflerin (Yanlış Alarmların) Patlaması:**
  Eşik değeri 0.50'den 0.20'ye çekildiğinde Recall %40'tan %80'e çıkabilir; ancak yanlış alarmlar (FP) 7'den 120'ye fırlar (`figures/soru3b_esik_degeri_tradeoff.png`).
* **Alarm Yorgunluğu (Alarm Fatigue) ve Yalancı Çoban Sendromu:**
  Bakım teknisyenleri günde 15-20 defa sağlam makineye kontrole gönderildiğinde bir süre sonra sistemin alarmlarını ciddiye almamaya başlar. Bu durum, gelecekte meydana gelecek gerçek bir arıza alarmının da "yine yanlış alarmdır" denilerek göz ardı edilmesine ve felakete yol açabilir.
* **Gereksiz Bakım ve Duruş Maliyetleri:**
  Her alarmda tezgahın durdurulması, hattın toplam verimliliğini (OEE - Overall Equipment Effectiveness) düşürür.

---

### 3.3. Aşırı Öğrenme (Overfitting) Problemi ve İki Somut Öneri (Soru 3.c)

#### Problem Teşhisi:
Modelin eğitim verisinde çok yüksek (örneğin %99-100 veya F1 = 1.00), test verisinde ise belirgin biçimde düşük başarı göstermesi durumunda teşhis **AŞIRI ÖĞRENME (OVERFITTING / EZBERLEME)** problemidir. Model eğitim verisindeki genel mekanik kuralları öğrenmek yerine, veri setindeki rastgele dalgalanmaları ve gürültüyü (noise) ezberlemiştir.

#### Bunu Azaltmak İçin İki Somut Öneri:

##### Öneri 1: Ağaç Budama ve Karmaşıklık Kısıtlaması (Tree Pruning & Hyperparameter Constraints)
* **Uygulama:** Karar ağacının serbestçe dallanmasına izin verilmemelidir.
  * `max_depth = 5 veya 6` ile ağaç derinliği sınırlandırılmalıdır.
  * `min_samples_split = 15-20` belirlenerek sadece yeterli sayıda örnek barındıran düğümlerin bölünmesine izin verilmelidir.
  * `min_samples_leaf = 5-10` tanımlanarak tek bir aykırı veriye özel yaprak oluşması engellenmelidir.
* **Etki:** Yapılan simülasyonda sınırsız karar ağacı eğitimde $F1 = 1.000$ verirken budanmış ağaç genelleme başarısını korumuştur.

##### Öneri 2: Tabakalı K-Katlı Çapraz Doğrulama (Stratified K-Fold Cross-Validation) ve Topluluk (Ensemble) Modelleri
* **Uygulama:**
  * Model tek bir şanslı veya şanssız eğitim-test ayrımına göre değil, 5 katlı Tabakalı Çapraz Doğrulama (`StratifiedKFold(n_splits=5)`) ile eğitilmeli ve hiperparametreler `GridSearchCV` ile optimize edilmelidir.
  * Tek bir karar ağacının yüksek varyansını (variance) kırmak için birden fazla ağacın oy çokluğuyla karar verdiği **Random Forest** veya gradyan artıran **XGBoost/LightGBM** gibi ensemble yöntemlerine geçilmelidir. Bu sayede model tek bir ağacın ezberleme hatasından kurtularak kararlı hale gelir.

---

## BÖLÜM 4: DOSYA DİZİN YAPISI VE TESLİM DETAYLARI

Proje klasörü: `C:\Users\Lenovo\Desktop\Vize_Sinavi\`
* `ai4i2020.csv`: Kullanılan orijinal veri seti.
* `run_analysis.py`: Tüm veri hazırlama, modelleme, metrik hesaplama ve grafik çizimlerini içeren ana Python kodu.
* `vize_odevi.ipynb`: Açıklamalı, hücre bazlı çalıştırılabilir Jupyter Notebook dosyası.
* `VIZE_RAPORU.md`: Detaylı vize raporu (bu belge).
* `VIZE_RAPORU.html`: Tarayıcıdan açılıp PDF/Yazıcı çıktısı alınabilen profesyonel formatlı rapor.
* `figures/`: Raporlanan tüm yüksek çözünürlüklü grafikler:
  * `soru1a_hedef_dagilimi.png`
  * `soru1b_sayisal_karsilastirma_kutu.png`
  * `soru1c_ariza_kumelenmesi.png`
  * `soru1c_korelasyon_matrisi.png`
  * `soru2b_confusion_matrices.png`
  * `soru2b_model_karsilastirma.png`
  * `soru2d_dt_feature_importance.png`
  * `soru2d_karar_agaci_semasi.png`
  * `soru3b_esik_degeri_tradeoff.png`
* `model_sonuclari.csv`: Modellerin metrik tablosu.
* `sayisal_karsilastirma_istatistikleri.csv`: Arıza durumuna göre sensör istatistikleri.

---
**Raporu Hazırlayan:** Antigravity AI & Öğrenci  
**Tarih:** 11 Kasım 2026
