import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import cv2
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')


# 设置参数
IMG_SIZE = 224
BATCH_SIZE = 64  # 增大batch size
EPOCHS = 50      # 增加训练轮数
DATA_DIR = "./RMB-Dataset/RMBDataset"  # 数据集路径,记得在本目录下执行文件
MODEL_SAVE_PATH = 'rmb_classifier_with_features.h5'
FEATURES_SAVE_PATH = 'feature_extractor.pkl'

# 高级训练参数
TRAINING_PARAMS = {
    # 优化器参数
    'optimizer': 'adam',  # 'adam', 'sgd', 'rmsprop'
    'learning_rate': 0.001,
    'weight_decay': 1e-4,  # L2正则化

    # 学习率调度
    'lr_scheduler': 'reduce_on_plateau',  # 'reduce_on_plateau', 'cosine', 'step', None
    'lr_factor': 0.5,
    'lr_patience': 10,
    'lr_min': 1e-7,

    # 早停机制
    'early_stopping': True,
    'early_stopping_patience': 15,
    'early_stopping_monitor': 'val_accuracy',

    # 模型检查点
    'checkpoint': True,
    'checkpoint_monitor': 'val_accuracy',

    # 数据增强（特征空间）
    'feature_noise': 0.01,  # 添加噪声增强
    'feature_dropout': 0.1,  # 特征dropout
}


class FeatureExtractor:
    """从qt_camera.py移植的特征提取类"""

    @staticmethod
    def extract_features(image):
        """提取纸币的所有特征"""
        # 确保图像是CV2格式（BGR）
        if isinstance(image, tf.Tensor):
            image = image.numpy()
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            if len(image.shape) == 4:
                image = image[0]  # 去掉batch维度
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 1. Harris角点检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_float = np.float32(gray)
        dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
        dst = cv2.dilate(dst, None)
        harris_corners = np.sum(dst > 0.01 * dst.max())

        # 转换为uint8（Shi-Tomasi、ORB、Canny需要）
        gray_uint8 = gray.astype(np.uint8)

        # 2. Shi-Tomasi角点检测
        corners_st = cv2.goodFeaturesToTrack(gray_uint8, maxCorners=100, qualityLevel=0.01, minDistance=10)
        shi_tomasi_corners = len(corners_st) if corners_st is not None else 0

        # 3. ORB特征点检测
        orb = cv2.ORB_create(nfeatures=1000)
        keypoints, descriptors = orb.detectAndCompute(gray_uint8, None)
        orb_features = len(keypoints)

        # 4. 轮廓检测和几何特征
        edges = cv2.Canny(gray_uint8, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        aspect_ratio = 0
        area_ratio = 0

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # 计算长宽比
            rect = cv2.boundingRect(largest_contour)
            x, y, w, h = rect
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0

            # 计算面积占比
            frame_area = image.shape[0] * image.shape[1]
            contour_area = cv2.contourArea(largest_contour)
            area_ratio = contour_area / frame_area if frame_area > 0 else 0

        # 5. Canny边缘统计
        edges_count = np.sum(edges > 0)

        # 6. 颜色直方图特征（简化版）
        hist_b = cv2.calcHist([image], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [32], [0, 256])

        # 组合所有特征
        features = np.array([
            harris_corners,
            shi_tomasi_corners,
            orb_features,
            aspect_ratio,
            area_ratio,
            edges_count,
            *hist_b.flatten(),
            *hist_g.flatten(),
            *hist_r.flatten()
        ])

        return features

    @staticmethod
    def extract_denomination_scores(features):
        """基于特征预测面额（从qt_camera.py移植的predict_denomination逻辑）"""
        if len(features) < 6:
            return "未知", {}

        harris_corners = features[0]
        shi_tomasi_corners = features[1]
        orb_features = features[2]
        aspect_ratio = features[3]
        area_ratio = features[4]

        # 人民币各面额的典型特征
        denomination_features = {
            '1': {
                'aspect_ratio_range': (1.5, 1.8),
                'area_ratio_range': (0.15, 0.25),
                'corner_range': (15, 40),
                'feature_range': (50, 150)
            },
            '5': {
                'aspect_ratio_range': (1.5, 1.8),
                'area_ratio_range': (0.15, 0.25),
                'corner_range': (15, 40),
                'feature_range': (50, 150)
            },
            '10': {
                'aspect_ratio_range': (1.7, 2.0),
                'area_ratio_range': (0.20, 0.30),
                'corner_range': (20, 50),
                'feature_range': (80, 200)
            },
            '20': {
                'aspect_ratio_range': (1.7, 2.0),
                'area_ratio_range': (0.20, 0.30),
                'corner_range': (20, 50),
                'feature_range': (80, 200)
            },
            '50': {
                'aspect_ratio_range': (1.8, 2.1),
                'area_ratio_range': (0.25, 0.35),
                'corner_range': (25, 60),
                'feature_range': (100, 250)
            },
            '100': {
                'aspect_ratio_range': (1.8, 2.1),
                'area_ratio_range': (0.25, 0.35),
                'corner_range': (25, 60),
                'feature_range': (100, 250)
            }
        }

        # 计算匹配分数
        scores = {}
        for denom, feat_ranges in denomination_features.items():
            score = 0

            # 长宽比匹配
            ar_min, ar_max = feat_ranges['aspect_ratio_range']
            if ar_min <= aspect_ratio <= ar_max:
                score += 25

            # 面积占比匹配
            area_min, area_max = feat_ranges['area_ratio_range']
            if area_min <= area_ratio <= area_max:
                score += 25

            # 角点数量匹配
            corner_min, corner_max = feat_ranges['corner_range']
            if corner_min <= shi_tomasi_corners <= corner_max:
                score += 25

            # 特征点数量匹配
            feat_min, feat_max = feat_ranges['feature_range']
            if feat_min <= orb_features <= feat_max:
                score += 25

            scores[denom] = score

        # 选择分数最高的面额
        if scores:
            best_denomination = max(scores, key=scores.get)
            best_score = scores[best_denomination]

            if best_score < 40:
                return "未知", scores

            return best_denomination, scores

        return "未知", {}

def load_and_extract_features():
    """加载数据并提取特征"""
    data_dir = os.path.abspath(DATA_DIR)

    # 加载所有图像数据（不使用batch）
    all_data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=None,
        shuffle=False
    )

    class_names = all_data.class_names
    print(f"类别: {class_names}")
    print(f"总类别数: {len(class_names)}")

    # 分离图像和标签
    images = []
    labels = []

    print("正在提取图像特征...")
    for image, label in all_data:
        # 提取特征
        features = FeatureExtractor.extract_features(image.numpy())
        images.append(features)
        labels.append(label.numpy())

        # 进度显示
        if len(images) % 50 == 0:
            print(f"已处理: {len(images)} 张图像")

    X = np.array(images)
    y = np.array(labels)

    print(f"特征提取完成！")
    print(f"特征维度: {X.shape}")
    print(f"标签维度: {y.shape}")

    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, class_names, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    """分割训练集和验证集"""
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_val, y_train, y_val

def create_feature_based_model(input_dim, num_classes):
    """创建基于特征的模型（用于处理提取的特征）"""
    # 获取训练参数
    params = TRAINING_PARAMS

    # 选择优化器
    if params['optimizer'] == 'adam':
        optimizer = keras.optimizers.Adam(
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
    elif params['optimizer'] == 'sgd':
        optimizer = keras.optimizers.SGD(
            learning_rate=params['learning_rate'],
            momentum=0.9,
            weight_decay=params['weight_decay']
        )
    elif params['optimizer'] == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
    else:
        optimizer = 'adam'

    # 创建模型（更深层，添加更多正则化）
    model = keras.Sequential([
        layers.InputLayer(input_shape=(input_dim,)),
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(params['weight_decay'])),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(params['weight_decay'])),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(params['weight_decay'])),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(params['weight_decay'])),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_cnn_model(num_classes):
    """创建CNN模型（用于对比）"""
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

def get_training_callbacks():
    """获取训练回调函数"""
    params = TRAINING_PARAMS
    callbacks = []

    # 早停
    if params['early_stopping']:
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=params['early_stopping_monitor'],
            patience=params['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)

    # 学习率调度 - ReduceLROnPlateau
    if params['lr_scheduler'] == 'reduce_on_plateau':
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=params['lr_factor'],
            patience=params['lr_patience'],
            min_lr=params['lr_min'],
            verbose=1
        )
        callbacks.append(lr_scheduler)

    # 余弦退火调度（可选）
    elif params['lr_scheduler'] == 'cosine':
        # 使用CosineDecay，需要先计算总步数
        # 这里简化处理，实际应该根据数据量计算
        pass

    # 模型检查点
    if params['checkpoint']:
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH.replace('.h5', '_best.h5'),
            monitor=params['checkpoint_monitor'],
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)

    return callbacks


def train_feature_based_model():
    """训练基于特征的模型"""
    print("=" * 50)
    print("训练基于特征的模型")
    print("=" * 50)

    # 显示训练参数
    params = TRAINING_PARAMS
    print("\n训练参数:")
    print(f"  优化器: {params['optimizer']}")
    print(f"  学习率: {params['learning_rate']}")
    print(f"  L2正则化: {params['weight_decay']}")
    print(f"  早停: {params['early_stopping']}")
    print(f"  学习率调度: {params['lr_scheduler']}")
    print("")

    # 加载数据并提取特征
    X, y, class_names, scaler = load_and_extract_features()

    # 分割数据
    X_train, X_val, y_train, y_val = split_data(X, y)

    print(f"\n训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")

    # 构建模型
    print("\n构建特征模型...")
    model = create_feature_based_model(X_train.shape[1], len(class_names))
    model.summary()

    # 获取回调函数
    callbacks = get_training_callbacks()

    # 训练模型
    print("\n开始训练特征模型...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1  # 显示训练进度
    )

    # 保存最终模型
    model.save(MODEL_SAVE_PATH)
    print(f"\n✓ 模型已保存为 {MODEL_SAVE_PATH}")

    # 保存特征标准化器
    with open(FEATURES_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ 特征标准化器已保存为 {FEATURES_SAVE_PATH}")

    # 打印最佳结果
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    print(f"\n🏆 最佳验证准确率: {best_val_acc:.4f} (第{best_epoch}轮)")

    return model, history, class_names


def train_cnn_model():
    """训练CNN模型（对比用）"""
    print("=" * 50)
    print("训练CNN模型（对比）")
    print("=" * 50)

    # 重新加载原始图像数据
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
    print(f"\n类别: {class_names}")

    # 数据预处理
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # 构建模型
    print("\n构建CNN模型...")
    model = create_cnn_model(len(class_names))
    model.summary()

    # 训练模型
    print("\n开始训练CNN模型...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # 保存模型
    model.save('rmb_cnn_model.h5')
    print("\nCNN模型已保存为 rmb_cnn_model.h5")

    return model, history, class_names

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("RMB面额识别系统 - 特征提取模型训练")
    print("=" * 60)
    print("\n请选择训练模式:")
    print("1. 基于特征的模型（推荐）- 使用Harris角点、ORB特征、几何特征等")
    print("2. CNN模型 - 传统卷积神经网络")
    print("3. 两种模型都训练（对比）")

    choice = input("\n请输入选择 (1/2/3): ").strip()

    if choice == "1":
        model, history, class_names = train_feature_based_model()
        print(f"\n✓ 特征模型训练完成！")
        print(f"✓ 模型可以识别: {class_names}")
        print(f"\n文件保存:")
        print(f"  - 模型文件: {MODEL_SAVE_PATH}")
        print(f"  - 特征标准化器: {FEATURES_SAVE_PATH}")

    elif choice == "2":
        model, history, class_names = train_cnn_model()
        print(f"\n✓ CNN模型训练完成！")
        print(f"✓ 模型可以识别: {class_names}")

    elif choice == "3":
        print("\n正在训练两种模型，请稍候...")
        model1, history1, class_names = train_feature_based_model()
        print("\n\n")
        model2, history2, _ = train_cnn_model()

        print("\n" + "=" * 60)
        print("两种模型训练完成！")
        print(f"✓ 特征模型文件: {MODEL_SAVE_PATH}")
        print(f"✓ 特征标准化器: {FEATURES_SAVE_PATH}")
        print(f"✓ CNN模型文件: rmb_cnn_model.h5")
        print("=" * 60)

    else:
        print("无效选择！")


if __name__ == "__main__":
    main()


def quick_test_model():
    """快速测试模型性能"""
    import matplotlib.pyplot as plt

    print("\n" + "=" * 50)
    print("快速模型测试")
    print("=" * 50)

    # 加载数据并提取特征
    X, y, class_names, scaler = load_and_extract_features()

    # 分割数据
    X_train, X_val, y_train, y_val = split_data(X, y)

    # 训练模型
    model = create_feature_based_model(X_train.shape[1], len(class_names))

    # 简单训练几轮（用于验证）
    print("\n训练模型进行快速测试...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,  # 只训练10轮快速测试
        batch_size=BATCH_SIZE,
        verbose=0
    )

    # 评估模型
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n✓ 验证集准确率: {val_acc:.4f}")

    # 预测几个样本
    print("\n样本预测:")
    for i in range(min(5, len(X_val))):
        pred = model.predict(X_val[i:i+1], verbose=0)
        class_idx = np.argmax(pred[0])
        true_idx = y_val[i]
        confidence = pred[0][class_idx]
        print(f"  样本 {i+1}: 预测={class_names[class_idx]} (置信度={confidence:.2f}), 实际={class_names[true_idx]}")

    print("\n快速测试完成！")

    # 可视化训练历史（简单版）
    print("\n训练历史:")
    print(f"  初始验证准确率: {history.history['val_accuracy'][0]:.4f}")
    print(f"  最终验证准确率: {history.history['val_accuracy'][-1]:.4f}")
    print(f"  最佳验证准确率: {max(history.history['val_accuracy']):.4f}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("RMB面额识别系统 - 特征提取模型训练")
    print("=" * 60)
    print("\n请选择训练模式:")
    print("1. 基于特征的模型（推荐）- 使用Harris角点、ORB特征、几何特征等")
    print("2. CNN模型 - 传统卷积神经网络")
    print("3. 快速测试 - 训练10轮快速验证模型效果")
    print("4. 两种模型都训练（对比）")

    choice = input("\n请输入选择 (1/2/3/4): ").strip()

    if choice == "1":
        model, history, class_names = train_feature_based_model()
        print(f"\n✓ 特征模型训练完成！")
        print(f"✓ 模型可以识别: {class_names}")
        print(f"\n文件保存:")
        print(f"  - 模型文件: {MODEL_SAVE_PATH}")
        print(f"  - 特征标准化器: {FEATURES_SAVE_PATH}")

    elif choice == "2":
        model, history, class_names = train_cnn_model()
        print(f"\n✓ CNN模型训练完成！")
        print(f"✓ 模型可以识别: {class_names}")

    elif choice == "3":
        quick_test_model()

    elif choice == "4":
        print("\n正在训练两种模型，请稍候...")
        model1, history1, class_names = train_feature_based_model()
        print("\n\n")
        model2, history2, _ = train_cnn_model()

        print("\n" + "=" * 60)
        print("两种模型训练完成！")
        print(f"✓ 特征模型文件: {MODEL_SAVE_PATH}")
        print(f"✓ 特征标准化器: {FEATURES_SAVE_PATH}")
        print(f"✓ CNN模型文件: rmb_cnn_model.h5")
        print("=" * 60)

    else:
        print("无效选择！")


if __name__ == "__main__":
    main()