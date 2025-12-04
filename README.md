# RMB Denomination Recognition System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

一个基于计算机视觉技术的人民币面额智能识别系统，能够实时识别1元、5元、10元、20元、50元、100元人民币，适用于银行、零售、安防等多个场景。

![Demo](assets/demo.gif)

## ✨ 特性

- 🎯 **高精度识别**：采用先进的图像处理算法，识别准确率高
- 🚀 **实时处理**：支持摄像头实时视频流识别，帧率达10+ FPS
- 💰 **全面面额支持**：支持1元、5元、10元、20元、50元、100元全面额识别
- 🔍 **多窗口显示**：提供原始画面、边缘检测、背景去除等多种可视化窗口
- 📊 **置信度评估**：实时显示识别结果置信度，确保识别可靠性
- 🎨 **可视化界面**：友好的实时显示界面，直观展示识别结果
- 🛡️ **鲁棒性强**：对光照变化、角度变化具有一定适应性

## 🛠️ 技术栈

- **主要框架**：Python 3.8+
- **图像处理**：OpenCV 4.x
- **科学计算**：NumPy
- **可选后端**：TensorFlow 2.x / Keras
- **其他**：图像处理算法库

## 📋 项目结构

```
RMB-Denomination-Recognition-System/
├── rmb_classifier.py                 # 模型训练脚本
├── rmb_live_classifier.py            # 实时识别应用
├── rmb_classifier.h5                 # 训练好的模型文件 (255MB)
├── test/
│   ├── try-camera.py                 # 摄像头测试脚本
│   └── try-tensorflow.py             # 环境测试脚本
├── RMB-Dataset/                      # 训练数据集
│   └── RMBDataset/
│       ├── 1/                        # 1元纸币图像 (134张)
│       ├── 5/                        # 5元纸币图像 (98张)
│       ├── 10/                       # 10元纸币图像 (156张)
│       ├── 20/                       # 20元纸币图像 (99张)
│       ├── 50/                       # 50元纸币图像 (167张)
│       └── 100/                      # 100元纸币图像 (132张)
├── 框架搭建.md                       # 系统框架搭建文档
├── 图像算法分析报告.md               # 图像处理算法分析报告
├── AGENTS.md                         # 仓库管理指南
└── README.md                         # 项目说明文档
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- OpenCV 4.x
- NumPy
- Web摄像头或USB摄像头
- 可选：TensorFlow 2.x (用于模型训练)

### 安装依赖

#### 使用 pip/pipx
```bash
# 创建虚拟环境
python -m venv rmb-env
source rmb-env/bin/activate  # Linux/Mac
# rmb-env\Scripts\activate  # Windows

# 安装依赖
pip install opencv-python numpy
# 如需训练模型，请安装：
pip install tensorflow
```

#### 使用 uv (推荐)
```bash
# 使用 uv 创建并激活虚拟环境
uv venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# 安装依赖
uv sync
```

### 使用方法

#### 1. 训练模型 (可选)

```bash
python rmb_classifier.py
```

或使用 uv：

```bash
uv run rmb_classifier.py
```

#### 2. 实时识别

确保摄像头连接正常，然后运行：

```bash
python rmb_live_classifier.py
```

或使用 uv：

```bash
uv run rmb_live_classifier.py
```

#### 3. 测试摄像头

在运行主程序前，可先测试摄像头是否正常工作：

```bash
python test/try-camera.py
```

## 📊 系统参数

- **输入尺寸**：224×224 像素
- **处理帧率**：10+ FPS
- **置信度阈值**：30%
- **支持面额**：6种 (1, 5, 10, 20, 50, 100元)
- **数据集规模**：786张标注图片
- **模型大小**：255MB

## 🔧 核心功能

### 图像预处理

- RGB色彩空间转换
- 图像尺寸标准化
- 像素值归一化处理

### 多种边缘检测方案

系统实现了多种图像处理算法：

1. **基础Canny边缘检测** - 125/175阈值设置
2. **边缘可视化增强** - 绿色边缘叠加显示
3. **轮廓长度过滤** - 自动过滤噪声和小轮廓
4. **多级阈值融合** - 综合多种阈值的优化方案

### 智能背景去除

集成智能前景分割算法：
- 自动前景/背景分离
- 适配复杂背景环境
- 无需手动初始化

### 特征点检测

- Harris角点检测算法
- Shi-Tomasi角点优化
- ORB特征描述符提取

## 🎯 应用场景

- **金融行业**：银行ATM机、智能柜台纸币识别
- **零售商店**：自助收银系统、无人商店
- **安防监控**：大额现金交易监控、反洗钱识别
- **教育领域**：金融科技教学演示
- **工业应用**：现金处理设备集成

## 📝 算法分析

### 技术特点

- **多算法融合**：结合边缘检测、轮廓分析、特征提取等多种技术
- **实时性优化**：针对实时处理场景进行算法优化
- **鲁棒性设计**：对环境光变化、拍摄角度有一定适应性
- **可视化友好**：提供多窗口实时显示，便于调试和观察

### 性能表现

- 在标准光照条件下识别准确率高
- 处理速度满足实时应用需求
- 支持多种摄像头设备
- 内存占用合理

## 🔜 未来规划

- [ ] 多币种支持 (美元、欧元等)
- [ ] 移动端性能优化
- [ ] 边缘设备部署适配
- [ ] 云端协同处理
- [ ] 识别速度进一步优化

## 🐛 已知问题

- 在光线不足环境下识别准确率下降
- 部分折叠或严重污损纸币识别困难
- 需要纸币完整出现在画面中

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 👨‍💻 作者

- **bakamake** - *初始开发* - [GitHub](https://github.com/bakamake)

## 🙏 致谢

感谢所有为计算机视觉和图像处理技术做出贡献的开发者们！

---

⭐ 如果这个项目对你有帮助，请给个 Star 支持一下！

---

## 🔬 特征提取模型 (新功能)

### 模型文件打包方式

#### 1. 基于特征的模型文件

训练完成后会生成以下文件：

##### `rmb_classifier_with_features.h5`
- **文件类型**: Keras HDF5 模型文件
- **内容**: 训练好的神经网络模型权重
- **输入维度**: 104 维特征向量（包含角点、ORB特征、几何特征、颜色直方图）
- **输出**: 各面额的分类概率
- **大小**: 约 1-5 MB

##### `feature_extractor.pkl`
- **文件类型**: Python pickle 序列化文件
- **内容**: scikit-learn 的 StandardScaler 对象
- **用途**: 对输入特征进行标准化处理
- **重要性**: **必须在推理时使用相同的标准化器**
- **大小**: 约 几KB

#### 2. CNN模型文件

##### `rmb_cnn_model.h5`
- **文件类型**: Keras HDF5 模型文件
- **内容**: 传统卷积神经网络模型
- **输入**: 224x224 RGB 图像
- **输出**: 各面额的分类概率
- **大小**: 约 10-50 MB

### 如何使用模型文件

#### 方法1: 单独加载
```python
import tensorflow as tf
import pickle
import cv2
import numpy as np

# 加载模型和标准化器
model = tf.keras.models.load_model('rmb_classifier_with_features.h5')
with open('feature_extractor.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 提取并标准化特征
def predict_denomination(image):
    # 提取特征（来自 FeatureExtractor.extract_features）
    features = FeatureExtractor.extract_features(image)

    # 标准化
    features_scaled = scaler.transform(features.reshape(1, -1))

    # 预测
    prediction = model.predict(features_scaled)
    class_idx = np.argmax(prediction[0])

    return class_names[class_idx], prediction[0]
```

#### 方法2: 创建推理模块
```python
class RMBDenominationPredictor:
    def __init__(self, model_path, scaler_path):
        self.model = tf.keras.models.load_model(model_path)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

    def predict(self, image):
        features = FeatureExtractor.extract_features(image)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.model.predict(features_scaled, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        return class_idx, confidence

# 使用
predictor = RMBDenominationPredictor(
    'rmb_classifier_with_features.h5',
    'feature_extractor.pkl'
)
result = predictor.predict(image)
```

#### 方法3: 打包为单一文件
```python
import joblib

# 打包模型和标准化器
joblib.dump({
    'model': model,
    'scaler': scaler,
    'class_names': class_names
}, 'rmb_predictor.pkl')

# 使用
loaded = joblib.load('rmb_predictor.pkl')
model = loaded['model']
scaler = loaded['scaler']
class_names = loaded['class_names']
```

### 特征说明

系统提取的特征维度为 104 维：
1. **前6维**: 基础特征
   - Harris角点数量
   - Shi-Tomasi角点数量
   - ORB特征点数量
   - 长宽比
   - 面积占比
   - Canny边缘像素数量

2. **后98维**: 颜色直方图（每通道32维）
   - 蓝色通道直方图: 32维
   - 绿色通道直方图: 32维
   - 红色通道直方图: 32维

### 训练特征提取模型

运行训练脚本：
```bash
python rmb_classifier.py
```

选择训练模式：
- `1`: 基于特征的模型（推荐）- 使用Harris角点、ORB特征、几何特征等
- `2`: CNN模型 - 传统卷积神经网络
- `3`: 两种模型都训练（对比）

### 模型优势

1. **轻量级**: 特征模型文件小，加载快
2. **可解释性强**: 特征基于图像的几何和视觉特性
3. **兼容 qt_camera.py**: 使用相同的特征提取逻辑
4. **高效率**: 推理速度快，适合实时应用

### 注意事项

1. **必须保存标准化器**: `feature_extractor.pkl` 文件必须与模型文件一起保存
2. **特征提取顺序**: 必须与训练时使用相同的特征提取流程
3. **数据预处理**: 输入图像需要调整为 224x224 像素
4. **环境依赖**: 需要安装 `opencv-python`, `scikit-learn`, `tensorflow` 等库
