# Kestirimci Bakım (Predictive Maintenance) Vize Projesi

Bu repo, UCI AI4I 2020 Predictive Maintenance veri seti kullanılarak makine arızalarının tahmin edilmesi ve üretim yönetimi açısından değerlendirilmesini içeren vize ödevini barındırmaktadır.

## 📂 Dosya Yapısı
* `ai4i2020.csv`: Orijinal veri seti (10.000 kayıt).
* `run_analysis.py`: Veri işleme, modelleme ve görselleştirmeleri üreten ana Python scripti.
* `vize_odevi.ipynb`: Açıklamalı Jupyter Notebook dosyası.
* `VIZE_RAPORU.md`: Markdown formatında kapsamlı akademik & endüstriyel vize raporu.
* `VIZE_RAPORU.html`: Tarayıcıdan açılıp PDF olarak kaydedilebilen modern ve şık HTML rapor.
* `figures/`: Yüksek çözünürlüklü tüm grafik ve görselleştirmeler:
  - `soru1a_hedef_dagilimi.png`: Arıza frekansı ve sınıf dengesizliği.
  - `soru1b_sayisal_karsilastirma_kutu.png`: Sensörlerin kutu grafikleri (Boxplots).
  - `soru1c_ariza_kumelenmesi.png`: Tork-Hız ve Aşınma-Tork arıza kümeleri.
  - `soru1c_korelasyon_matrisi.png`: Sensör korelasyon ısı haritası.
  - `soru2b_confusion_matrices.png`: Modellerin karışıklık matrisleri.
  - `soru2b_model_karsilastirma.png`: Accuracy, Precision, Recall, F1 karşılaştırması.
  - `soru2d_dt_feature_importance.png`: Karar ağacı öznitelik önem dereceleri.
  - `soru2d_karar_agaci_semasi.png`: Karar ağacı şeması (ilk 3 seviye kural ağacı).
  - `soru3b_esik_degeri_tradeoff.png`: Eşik değerine göre Recall-Precision değişimi.

## 🚀 Çalıştırma
Tüm analizleri baştan koşturmak için terminalde:
```bash
python run_analysis.py
```
veya Jupyter Notebook ile:
```bash
jupyter notebook vize_odevi.ipynb
```

---

## ✉️ Hocaya Gönderilecek E-Posta Taslağı
* **Kime:** bekcetintav@gmail.com
* **Konu:** [Öğrenci No] - [Ad Soyad] - Predictive Maintenance Vize Ödevi Teslimi
* **İçerik:**
> Sayın Hocam,
>
> AI4I 2020 Predictive Maintenance veri seti üzerine hazırladığım vize ödevi raporunu ve Python kodlarımı ekte ve GitHub repomda bilginize sunarım.
>
> Projede Soru 1, Soru 2 ve Soru 3 kapsamındaki tüm maddeler (sınıf dengesizliği analizi, veri sızıntısını önleme, modellerin eğitimi ve karşılaştırılması, endüstriyel maliyet değerlendirmeleri) eksiksiz olarak incelenmiş ve raporlanmıştır.
>
> Saygılarımla,  
> [Adınız Soyadınız]  
> [Öğrenci Numaranız]
