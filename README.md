#  Hayvan Sınıflandırma Projesi

Bu proje, farklı hayvan türlerini sınıflandırmak için geliştirilmiş bir görüntü işleme ve makine öğrenmesi sistemidir. Derin öğrenme temelli bu sistem, kullanıcıdan alınan bir hayvan görselini analiz ederek **kedi, köpek, fil, at** gibi kategorilere ayırır. Arka planda TensorFlow/Keras kütüphanesi kullanılarak eğitilmiş bir CNN modeli çalışır. Kullanıcı dostu bir arayüz için **Streamlit** entegrasyonu mevcuttur.

---

##  Özellikler

- CNN mimarisi ile görüntü sınıflandırma
- `Animals-10` veri kümesi ile eğitim
- Eğitim sonrası model `.keras` uzantısıyla kaydedilir
- Eğitim yapılmışsa model yeniden eğitilmeden doğrudan yüklenir
- Eğitim sırasında doğruluk ve kayıp grafikleri çizilir
- Türkçe sınıf isimleri ile tahmin sonucu sunulur
- Görsel yükleyerek tahmin yapmaya imkân tanıyan **Streamlit** arayüzü

---

##  Klasör Yapısı
  yapay_zeka_proje/
├── app.py
├── model.py
├── train.py
├── utils/
│   └── preprocessing.py
├── models/
│   └── model.pth
├── dataset/
│   └── seg_train / seg_test
├── screenshot.png
└── README.md

 # Veri klasör yapısı tf.keras.utils.image_dataset_from_directory() fonksiyonuna uygun olarak aşağıdaki gibidir:

  raw-img/
  ├── cane/
  ├── gatto/
  ├── elefante/
  ├── mucca/
  ├── cavallo/
  ├── gallina/
  ├── pecora/
  ├── farfalla/
  ├── ragno/
  └── scoiattolo/

# Projeyi başlatmak için:

streamlit run app/interface.py

- Görsel yükleyin veya sürükleyip bırakın.
- Sistem tahmin sonucunu anında gösterir.
- Türkçe sınıf ismi ve güven yüzdesi sunulur.


# Gerekli Python kütüphanelerini yüklemek için:

- tensorflow
- streamlit
- numpy
- pillow
- matplotlib
- scikit-learn
