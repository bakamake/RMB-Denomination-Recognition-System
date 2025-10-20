import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os


# 设置参数
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = ".\RMB-Dataset\RMBDataset"  # 数据集路径

# 数据加载
def load_data():
    data_dir = os.path.abspath(DATA_DIR)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"类别: {class_names}")

    # 数据预处理
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names

# 模型构建
def create_model(num_classes):
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# 训练模型
def train_model():
    print("加载数据...")
    train_ds, val_ds, class_names = load_data()

    print("构建模型...")
    model = create_model(len(class_names))
    model.summary()

    print("开始训练...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # 保存模型
    model.save('rmb_classifier.h5')
    print("模型已保存为 rmb_classifier.h5")

    return model, history, class_names

if __name__ == "__main__":
    model, history, class_names = train_model()
    print(f"训练完成！模型可以识别: {class_names}")