# src/preprocess.py

import tensorflow as tf

def load_data(data_dir='/Users/bilalkaanak/Desktop/Bulut_Bilişim_Yapay_Zeka/data/archive/raw-img', img_size=(128, 128), batch_size=32):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True  # ✅ Karıştırma açık
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )

    class_names = train_ds.class_names

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds, class_names
