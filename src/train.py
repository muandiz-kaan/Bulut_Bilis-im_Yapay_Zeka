# src/train.py

import tensorflow as tf
from preprocess import load_data

train_ds, val_ds, class_names = load_data()

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),  # 🔥 Ekstra derinlik
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # ✅ Aşırı öğrenmeyi azalt
    tf.keras.layers.Dense(len(class_names), activation='softmax')


    
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_ds, validation_data=val_ds, epochs=10)

# Tam yolu kullanarak model kaydetme
model.save('/Users/bilalkaanak/Desktop/Bulut_Bilişim_Yapay_Zeka/model/animal_model.keras')

print("✅ Model kaydedildi.")
