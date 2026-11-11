import os
import base64

base_dir = r"C:\Users\Lenovo\Desktop\Vize_Sinavi"
fig_dir = os.path.join(base_dir, "figures")
html_path = os.path.join(base_dir, "VIZE_RAPORU.html")

def img_to_base64(fig_name):
    p = os.path.join(fig_dir, fig_name)
    if os.path.exists(p):
        with open(p, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
    return ""

img_1a = img_to_base64("soru1a_hedef_dagilimi.png")
img_1b = img_to_base64("soru1b_sayisal_karsilastirma_kutu.png")
img_1c = img_to_base64("soru1c_ariza_kumelenmesi.png")
img_1corr = img_to_base64("soru1c_korelasyon_matrisi.png")
img_2cm = img_to_base64("soru2b_confusion_matrices.png")
img_2comp = img_to_base64("soru2b_model_karsilastirma.png")
img_2imp = img_to_base64("soru2d_dt_feature_importance.png")
img_2tree = img_to_base64("soru2d_karar_agaci_semasi.png")
img_3tradeoff = img_to_base64("soru3b_esik_degeri_tradeoff.png")

html_content = f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vize Proje Raporu - Predictive Maintenance</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.65;
            color: #2d3748;
            background-color: #f7fafc;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1050px;
            margin: 0 auto;
            background: #ffffff;
            padding: 45px 55px;
            border-radius: 12px;
            box-shadow: 0 4px 25px rgba(0,0,0,0.06);
        }}
        h1 {{
            color: #1a365d;
            font-size: 26px;
            border-bottom: 3px solid #3182ce;
            padding-bottom: 12px;
            margin-top: 0;
            text-align: center;
        }}
        h2 {{
            color: #2b6cb0;
            font-size: 20px;
            margin-top: 35px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 8px;
        }}
        h3 {{
            color: #2d3748;
            font-size: 16px;
            margin-top: 22px;
        }}
        .meta-box {{
            background: #ebf8ff;
            border-left: 5px solid #3182ce;
            padding: 15px 20px;
            border-radius: 0 8px 8px 0;
            margin: 20px 0 30px 0;
            font-size: 14.5px;
        }}
        .meta-box strong {{
            color: #2c5282;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }}
        th, td {{
            padding: 12px 14px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background-color: #edf2f7;
            color: #2d3748;
            font-weight: 700;
        }}
        tr:hover {{
            background-color: #f7fafc;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
        .badge-success {{ background: #c6f6d5; color: #22543d; }}
        .badge-danger {{ background: #fed7d7; color: #742a2a; }}
        .badge-warning {{ background: #feebc8; color: #744210; }}
        .badge-info {{ background: #bee3f8; color: #2a4365; }}
        
        .image-box {{
            text-align: center;
            margin: 25px 0;
            background: #fcfcfd;
            border: 1px solid #e2e8f0;
            padding: 15px;
            border-radius: 8px;
        }}
        .image-box img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .image-caption {{
            font-size: 13px;
            color: #718096;
            margin-top: 8px;
            font-weight: 600;
        }}
        .callout {{
            background: #fffaf0;
            border-left: 4px solid #dd6b20;
            padding: 14px 18px;
            margin: 18px 0;
            border-radius: 0 6px 6px 0;
            font-size: 14px;
        }}
        .highlight {{
            background: #f7fafc;
            border: 1px solid #cbd5e0;
            padding: 14px 18px;
            border-radius: 6px;
            margin: 15px 0;
        }}
        ul, ol {{
            padding-left: 24px;
        }}
        li {{
            margin-bottom: 8px;
        }}
        @media print {{
            body {{
                background-color: #ffffff;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
                padding: 15px;
            }}
            .image-box {{
                page-break-inside: avoid;
            }}
            table {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>

<div class="container">
    <h1>ENDÜSTRİYEL VERİ MADENCİLİĞİ VE KESTİRİMCİ BAKIM VİZE PROJE RAPORU</h1>
    
    <div class="meta-box">
        <p><strong>Ödev Konusu:</strong> AI4I 2020 Predictive Maintenance Veri Seti Arıza Tahmini ve Üretim Yönetimi Analizi</p>
        <p><strong>Teslim Edilecek E-Posta:</strong> bekcetintav@gmail.com</p>
        <p><strong>Veri Seti:</strong> UCI Machine Learning Repository (10.000 Gözlem, Frezeleme & Talaşlı İmalat)</p>
        <p><strong>Rapor Tarihi:</strong> 11 Kasım 2026</p>
    </div>

    <!-- BÖLÜM 1 -->
    <h2>SORU 1 – Veriyi Tanıma ve Problemi Anlama (10 Puan)</h2>

    <h3>a) Makine Arızası (Machine failure) Dağılımı ve Sınıflandırma Zorlukları</h3>
    <p>Veri setindeki hedef değişken <code>Machine failure</code> ikili (binary) bir sınıftır. 10.000 kaydın dağılımı aşağıdaki gibidir:</p>
    
    <table>
        <thead>
            <tr>
                <th>Sınıf Durumu</th>
                <th>Açıklama</th>
                <th>Örnek Sayısı (Adet)</th>
                <th>Yüzde Dağılımı (%)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><span class="badge badge-success">0 (Normal)</span></td>
                <td>Arızasız çalışan makineler</td>
                <td><strong>9.661</strong></td>
                <td><strong>%96.61</strong></td>
            </tr>
            <tr>
                <td><span class="badge badge-danger">1 (Arızalı)</span></td>
                <td>Arıza yaşayan makineler</td>
                <td><strong>339</strong></td>
                <td><strong>%3.39</strong></td>
            </tr>
        </tbody>
    </table>
    
    <div class="image-box">
        <img src="{img_1a}" alt="Hedef Değişken Dağılımı">
        <div class="image-caption">Şekil 1: Hedef Değişken Dağılımı ve Aşırı Sınıf Dengesizliği (28.5 : 1)</div>
    </div>

    <div class="highlight">
        <strong>Veri Setinin Sınıflandırma Açısından İçerdiği Zorluklar:</strong>
        <ol>
            <li><strong>Aşırı Sınıf Dengesizliği (Class Imbalance):</strong> Arızalı makineler tüm verinin yalnızca %3.39'unu oluşturmaktadır. Dengesizlik oranı yaklaşık 28.5 kat büyüktür.</li>
            <li><strong>Doğruluk Paradoksu (Accuracy Paradox):</strong> Bir model hiçbir makine öğrenmesi yapmadan her makineye "Arızasız (0)" tahmini yapsa bile teorik olarak <strong>%96.61 Doğruluk (Accuracy)</strong> alacaktır. Ancak bu model 339 arızanın tamamını kaçıracağı için endüstriyel açıdan felakettir.</li>
            <li><strong>Azınlık Sınıfını Öğrenme Güçlüğü:</strong> Sadece 339 örnek, 5 farklı mekanik arıza tipinin karmaşık etkileşimlerini öğrenmek için çok dardır; modeller bu örnekleri "gürültü" sayma eğilimi gösterir.</li>
        </ol>
    </div>

    <h3>b) Arızalı ve Arızasız Makinelerin Sayısal Değişkenler Açısından Karşılaştırılması</h3>
    <p>Makineler 5 temel sensör metriği ve türetilen sıcaklık farkı değişkeni açısından kıyaslanmıştır:</p>

    <table>
        <thead>
            <tr>
                <th>Sensör Değişkeni</th>
                <th>Normal (0) Ortalama ± Std</th>
                <th>Normal (0) Medyan</th>
                <th>Arızalı (1) Ortalama ± Std</th>
                <th>Arızalı (1) Medyan</th>
                <th>Endüstriyel Karşılaştırma</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Tool wear [min]</strong></td>
                <td>106.69 ± 62.95</td>
                <td>107.0 dk</td>
                <td><strong>143.78 ± 72.76</strong></td>
                <td><strong>165.0 dk</strong></td>
                <td>Arızalı grupta aşınma süresi medyan olarak %54 daha yüksektir.</td>
            </tr>
            <tr>
                <td><strong>Torque [Nm]</strong></td>
                <td>39.63 ± 9.47</td>
                <td>39.9 Nm</td>
                <td><strong>50.17 ± 16.37</strong></td>
                <td><strong>53.7 Nm</strong></td>
                <td>Arızalı grupta tork ortalaması belirgin yüksek, varyans 2 kat geniştir.</td>
            </tr>
            <tr>
                <td><strong>Rotational speed [rpm]</strong></td>
                <td>1540.26 ± 167.39</td>
                <td>1507 rpm</td>
                <td><strong>1496.49 ± 384.94</strong></td>
                <td><strong>1365 rpm</strong></td>
                <td>Arızalarda devir sapması 385 rpm'ye fırlar (aşırı hız veya boğulma).</td>
            </tr>
            <tr>
                <td><strong>Air temp [K]</strong></td>
                <td>299.97 ± 1.99</td>
                <td>300.0 K</td>
                <td><strong>300.89 ± 2.07</strong></td>
                <td><strong>301.6 K</strong></td>
                <td>Arızalı makineler yaklaşık 1 K daha yüksek hava sıcaklığında çalışmıştır.</td>
            </tr>
            <tr>
                <td><strong>Process temp [K]</strong></td>
                <td>309.99 ± 1.49</td>
                <td>310.0 K</td>
                <td><strong>310.29 ± 1.36</strong></td>
                <td><strong>310.4 K</strong></td>
                <td>İşlem sıcaklığı arızalı grupta hafif yükselmektedir.</td>
            </tr>
            <tr>
                <td><strong>Sıcaklık Farkı [K]</strong></td>
                <td>10.02 ± 1.00</td>
                <td>10.0 K</td>
                <td><strong>9.40 ± 1.32</strong></td>
                <td><strong>9.3 K</strong></td>
                <td>Fark azaldığında ısı tahliyesi zorlaşmakta ve HDF arızası doğmaktadır.</td>
            </tr>
        </tbody>
    </table>

    <div class="image-box">
        <img src="{img_1b}" alt="Sayısal Değişkenler Kutu Grafikleri">
        <div class="image-caption">Şekil 2: Arızalı (1) vs Normal (0) Makinelerin Sensör Dağılımları (Boxplots)</div>
    </div>

    <h3>c) Değişkenlerin Yararlılığı ve Üretim Yönetimi Çıkarımları</h3>
    <div class="image-box">
        <img src="{img_1c}" alt="Arıza Kümeleri">
        <div class="image-caption">Şekil 3: Tork - Devir ve Aşınma - Tork İkili Dağılımları (Kritik Arıza Eşikleri)</div>
    </div>

    <div class="highlight">
        <strong>Üretim Yönetimi ve Bakım Stratejisi Çıkarımları:</strong>
        <ul>
            <li><strong>Tork ve Dönme Hızı (Mekanik Güç Riski):</strong> Güç bağıntısı (P = Tork &times; Devir) gereği arızalar iki uç noktada kümelenmiştir:
                <br>• <em>Düşük Devir (&lt;1350 rpm) + Yüksek Tork (&gt;60 Nm):</em> Motor ağır talaş kaldırma direncinde aşırı tork üretmekte; bu durum elektrik motorunu aşırı yüklemekte (PWF) ve mili kırılmaya zorlamaktadır (OSF). Tork arıza ile en yüksek pozitif korelasyona (<strong>r = 0.191</strong>) sahiptir.
            </li>
            <li><strong>Takım Aşınması (Tool wear):</strong> Aşınma kümülatif bir yıpranmadır. 200 dakikayı geçen takımlarda kesici geometrisi bozulmakta, kesme sürtünmesi torku fırlatarak takım kırılmasına (TWF) sebep olmaktadır.</li>
            <li><strong>Operasyonel Eylem Planı:</strong> Fabrika bakım yönetimi takımları 180-200 dakika bandında otomatik önleyici bakımla değiştirmeli, CNC kontrol ünitelerine 60 Nm üzeri tork limit koruması koymalıdır.</li>
        </ul>
    </div>

    <!-- BÖLÜM 2 -->
    <h2>SORU 2 – Arıza Tahmin Modellerinin Geliştirilmesi (20 Puan)</h2>

    <h3>a) Veri Bölme ve Ön İşleme Tercihleri ve Gerekçeleri</h3>
    <div class="callout">
        <strong>Önemli Kural Takibi:</strong> <code>TWF, HDF, PWF, OSF, RNF</code> arızanın sonuç göstergeleri olduğu için modele kesinlikle dahil edilmemiştir (Data Leakage önlenmiştir). <code>UDI</code> ve <code>Product ID</code> gereksiz kimlik alanları olduğu için çıkarılmıştır.
    </div>
    <ul>
        <li><strong>Tabakalı Veri Bölme (Stratified Split %80 - %20):</strong> 8.000 satır eğitim, 2.000 satır test olarak ayrılmıştır. Sınıf dengesizliği nedeniyle <code>stratify=y</code> kullanılarak hem eğitimde (%3.39) hem testte (%3.40) arıza oranı eşit korunmuştur.</li>
        <li><strong>Öznitelik Ölçeklendirme (StandardScaler):</strong> KNN Öklid mesafesi hesapladığı için Devir (~1500 rpm) ile Tork (~40 Nm) arasındaki ölçek farkını gidermek şarttır. Ölçekleyici data leakage olmaması için SADECE eğitim kümesiyle fit edilmiştir.</li>
    </ul>

    <h3>b) Üç Algoritmanın Test Performanslarının Karşılaştırılması</h3>
    <p>2.000 test kaydı (1.932 Normal, 68 Arızalı) üzerinde elde edilen metrikler:</p>

    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>ROC-AUC</th>
                <th>TP (Yakalanan)</th>
                <th>FN (Kaçan)</th>
                <th>FP (Yanlış Alarm)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Logistic Regression</strong></td>
                <td>0.9675</td>
                <td>0.6364</td>
                <td>0.1029</td>
                <td>0.1772</td>
                <td>0.8995</td>
                <td>7</td>
                <td>61</td>
                <td>4</td>
            </tr>
            <tr>
                <td><strong>K-Nearest Neighbors (KNN)</strong></td>
                <td>0.9740</td>
                <td>0.8333</td>
                <td>0.2941</td>
                <td>0.4348</td>
                <td>0.8292</td>
                <td>20</td>
                <td>48</td>
                <td>4</td>
            </tr>
            <tr>
                <td><strong>Decision Tree (max_depth=6)</strong></td>
                <td><strong>0.9760</strong></td>
                <td><strong>0.7941</strong></td>
                <td><strong>0.3971</strong></td>
                <td><strong>0.5294</strong></td>
                <td><strong>0.9366</strong></td>
                <td><strong>27</strong></td>
                <td><strong>41</strong></td>
                <td><strong>7</strong></td>
            </tr>
        </tbody>
    </table>

    <div class="image-box">
        <img src="{img_2cm}" alt="Confusion Matrices">
        <div class="image-caption">Şekil 4: Modellerin Test Verisi Karışıklık Matrisleri (Confusion Matrices)</div>
    </div>

    <div class="image-box">
        <img src="{img_2comp}" alt="Model Kıyaslama">
        <div class="image-caption">Şekil 5: Modellerin Performans Metrikleri Karşılaştırma Grafiği</div>
    </div>

    <h3>c) En Yüksek Accuracy Değeri Veren Model En İyi Model midir?</h3>
    <p><strong>Cevap: Kesinlikle Hayır!</strong></p>
    <div class="highlight">
        <p>Veri setimizde test kümesinin %96.60'ı arızasızdır. Hiçbir şey bilmeden her şeye "0" diyen tembel bir model %96.60 Accuracy alır.</p>
        <p>Lojistik Regresyon modeli <strong>%96.75 Accuracy</strong> almış gibi görünse de gerçekleşen 68 arızanın <strong>61 tanesini kaçırmıştır (Recall = %10.29)</strong>. Fabrikada makineler kırılırken model %96.75 doğrulukla övünmektedir. Dolayısıyla dengesiz kestirimci bakım verilerinde Accuracy metriği tek başına yanıltıcıdır; fabrikayı kurtaracak olan <strong>Recall</strong> ve <strong>F1-Score</strong>'dur.</p>
    </div>

    <h3>d) Model Seçimi ve Operasyonel Gerekçelendirme</h3>
    <p><strong>Fabrika İçin Seçilen Model: KARAR AĞACI (DECISION TREE)</strong></p>
    
    <div class="image-box">
        <img src="{img_2imp}" alt="Feature Importance">
        <div class="image-caption">Şekil 6: Karar Ağacı Öznitelik Önem Düzeyleri (Feature Importance)</div>
    </div>

    <div class="highlight">
        <strong>Seçim Gerekçeleri:</strong>
        <ol>
            <li><strong>En Yüksek Arıza Yakalama (Recall & F1):</strong> 68 arızadan 27'sini yakalayarak (%39.71 Recall) ve 0.5294 F1-Score ile rakiplerini açık ara geride bırakmıştır.</li>
            <li><strong>Açıklanabilirlik (Explainability):</strong> Karar Ağacı kara kutu değildir. Bakım ekibine doğrudan kurallar sunar: <em>"Tork &gt; 61.5 Nm ve Hız &lt; 1380 rpm ise arıza riski %85'tir."</em> Operatör arızanın mekanik nedenini anlar.</li>
            <li><strong>PLC ve Gömülü Sistem Uyumluluğu:</strong> Basit IF-ELSE kod bloklarına dönüştürülerek CNC kontrol ünitelerinde mikrosaniyeler içinde gecikmesiz çalışır.</li>
        </ol>
    </div>

    <!-- BÖLÜM 3 -->
    <h2>SORU 3 – Modeli Gerçek Hayatta Kullanmak (10 Puan)</h2>

    <h3>a) Hata Türlerinin Karşılaştırılması (Yanlış Negatif vs. Yanlış Pozitif)</h3>
    <div class="highlight">
        <p><strong>Kritik Olan Hata: YANLIŞ NEGATİF (False Negative - Tip II Hata) AÇIK ARA ÇOK DAHA KRİTİKTİR!</strong></p>
        <ul>
            <li><strong>Yanlış Negatif Maliyeti (Arızayı Kaçırma):</strong> Model makineye "sağlam" der ancak makine aniden parçalanır. Tüm hat durur (plansız duruş), işlenen gövde hurda olur, fener mili hasar görür, müşteriye teslimat gecikir ve operatör güvenliği tehlikeye girer. Maliyet: On binlerce / yüz binlerce lira!</li>
            <li><strong>Yanlış Pozitif Maliyeti (Yanlış Alarm):</strong> Model sağlam makineye "arıza" der. Bakım teknisyeni gider, makineyi 10 dakika kontrol eder, bir şey olmadığını görüp üretime devam ettirir. Maliyet: 15 dakikalık işçilik mesaisi.</li>
        </ul>
    </div>

    <h3>b) Arızaları Yakalama Başarısını (Recall) Artırma Yöntemleri ve Olumsuz Sonuçları</h3>
    <div class="image-box">
        <img src="{img_3tradeoff}" alt="Eşik Değeri Analizi">
        <div class="image-caption">Şekil 7: Karar Eşik Değerine Göre Recall vs Precision Değişimi (Maliyet & Risk Ticareti)</div>
    </div>

    <div class="highlight">
        <strong>Uygulanacak Değişiklikler:</strong>
        <ol>
            <li><strong>Karar Eşiğinin Düşürülmesi (Threshold Tuning):</strong> Eşik 0.50'den 0.20-0.25 seviyesine çekilebilir.</li>
            <li><strong>Sınıf Ağırlıklandırması (Class Weights):</strong> Arıza sınıfına 10-15 kat ceza katsayısı verilerek model arızalara aşırı duyarlı hale getirilebilir.</li>
        </ol>
        <strong>Olası Olumsuz Sonuçlar:</strong>
        <ul>
            <li><strong>Yanlış Alarmların (False Positive) Fırlaması:</strong> Eşik düştüğünde arızaların %80'i yakalanır ancak yanlış alarmlar 7'den 100'ün üzerine çıkar.</li>
            <li><strong>Alarm Yorgunluğu (Alarm Fatigue) ve Yalancı Çoban Sendromu:</strong> Bakım ekibi sürekli boş alarmlara koşturulursa bir süre sonra sistemi ciddiye almaz ve gerçek bir arıza alarmında da müdahalede gecikebilir.</li>
            <li><strong>Gereksiz Hat Duruşları ve İş Gücü Kaybı:</strong> Sürekli kontrol üretim verimini düşürür.</li>
        </ul>
    </div>

    <h3>c) Aşırı Öğrenme (Overfitting) Problemi ve İki Somut Çözüm Önerisi</h3>
    <div class="highlight">
        <p><strong>Problem Teşhisi: AŞIRI ÖĞRENME (OVERFITTING)</strong><br>
        Model eğitimde %100'e yakın başarı gösterip testte çöküyorsa verinin temel kurallarını değil, eğitimdeki gürültü ve tesadüfi örüntüleri ezberlemiştir.</p>
        
        <strong>İki Somut Çözüm Önerisi:</strong>
        <ol>
            <li><strong>Ağaç Budama ve Hiperparametre Kısıtlaması (Tree Pruning):</strong> Karar ağacının derinliği sınırlandırılmalıdır (<code>max_depth=5</code> veya <code>6</code>). Düğüm bölünmesi için gereken minimum örnek sayısı (<code>min_samples_split=15</code>) ve yaprak başına minimum örnek (<code>min_samples_leaf=5</code>) kuralı koyularak tekil aykırı değerleri ezberleyen derin yaprakların oluşması engellenir.</li>
            <li><strong>Tabakalı K-Katlı Çapraz Doğrulama (Stratified K-Fold CV) ve Ensemble (Topluluk) Yöntemleri:</strong> Veri 5 eşit tabakaya bölünerek K-Fold CV ile eğitilmeli; ayrıca tek bir karar ağacının yüksek varyansını kırmak için birden fazla ağacın ortalamasını alan <strong>Random Forest</strong> veya <strong>XGBoost</strong> gibi ensemble modellere geçilmelidir.</li>
        </ol>
    </div>

    <div class="meta-box" style="margin-top: 40px; text-align: center;">
        <p><strong>Tüm analiz kodları, veri seti ve Jupyter Notebook dosyası Vize_Sinavi klasöründe hazırdır.</strong></p>
        <p>Rapor Sonu | Başarılar Dileriz.</p>
    </div>

</div>

</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"HTML Rapor oluşturuldu: {html_path}")
