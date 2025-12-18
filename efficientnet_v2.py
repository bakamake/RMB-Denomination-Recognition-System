import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import efficientnet_v2

# --- 1. 显存优化 (针对 RTX 3060 6G) ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- 2. 加载模型 (只加载一次) ---
print("⏳ 正在加载 EfficientNetV2-S，请稍候...")
model = tf.keras.applications.EfficientNetV2S(
    weights='imagenet',
    include_top=True
)
print("✅ 模型加载成功！")

# --- 3. 打开摄像头 ---
# 0 通常是默认摄像头。如果有多个摄像头，尝试改 1 或 2
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

# 设置摄像头分辨率 (可选，降低分辨率可以提高 FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("🚀 开始视频流识别，按 'Q' 键退出...")

while True:
    # 1. 读取一帧
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法接收帧 (流结束?). Exiting ...")
        break

    # 2. 预处理 (为了喂给模型)
    # A. 缩放：模型需要 384x384
    input_img = cv2.resize(frame, (384, 384))
    
    # B. 颜色转换：OpenCV 是 BGR，模型需要 RGB (这一步非常关键！)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    
    # C. 增加维度：变成 (1, 384, 384, 3)
    input_img = np.expand_dims(input_img, axis=0)

    # 3. 推理 (Predict)
    # verbose=0 防止控制台疯狂刷屏进度条
    preds = model.predict(input_img, verbose=0)
    
    # 4. 解码结果 (获取概率最高的)
    decoded = efficientnet_v2.decode_predictions(preds, top=1)[0][0]
    class_name = decoded[1]  # 类别名称
    confidence = decoded[2]  # 置信度 (概率)

    # 5. 可视化：把结果画在画面上
    # 格式化文本：例如 "tabby_cat: 85.4%"
    text = f"{class_name}: {confidence:.1%}"
    
    # 在原图(frame)上写字
    # 参数：图, 文本, 坐标, 字体, 大小, 颜色(BGR), 线宽
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # 显示画面
    cv2.imshow('Real-time Recognition (Press Q to exit)', frame)

    # 6. 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. 释放资源 ---
cap.release()
cv2.destroyAllWindows()