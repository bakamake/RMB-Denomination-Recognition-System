import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import os

class RMBClassifier:
    def __init__(self, model_path='rmb_classifier.h5'):
        self.model_path = model_path
        self.model = None
        self.class_names = ['1', '5', '10', '20', '50', '100']
        self.img_size = 224
        self.confidence_threshold = 0.3

        # 加载模型
        self.load_model()

    def load_model(self):
        """加载训练好的模型"""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"模型 {self.model_path} 加载成功！")
            print(f"可识别类别: {self.class_names}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

    def preprocess_image(self, img):
        """预处理图像以适应模型输入"""
        # 转换为RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 调整大小
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))

        # 归一化到0-1范围
        img_normalized = img_resized.astype(np.float32) / 255.0

        # 添加batch维度
        img_batch = np.expand_dims(img_normalized, axis=0)

        return img_batch

    def predict(self, img):
        """对图像进行预测"""
        if self.model is None:
            return None, 0.0

        # 预处理图像
        processed_img = self.preprocess_image(img)

        # 进行预测
        predictions = self.model.predict(processed_img, verbose=0)

        # 获取预测结果
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]

        # 如果置信度低于阈值，返回未知
        if confidence < self.confidence_threshold:
            return "未知", confidence

        predicted_class = self.class_names[predicted_class_idx]

        return predicted_class, confidence

def main():

    # 初始化分类器
    try:
        classifier = RMBClassifier()
    except Exception as e:
        print(f"初始化分类器失败: {e}")
        return

    # 打开视频流
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("未接收到数据流，请确认摄像头0可用")
        cap.release()
        return

    print("视频流连接成功！开始识别RMB...")
    print("按 'q' 键退出程序")

    # FPS计算变量
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    count = 0

    while True:
        ret, frame = cap.read()

        if ret:
            # 进行RMB识别
            predicted_class, confidence = classifier.predict(frame)

            # 计算FPS
            fps_counter += 1
            if fps_counter >= 10:  # 每10帧更新一次FPS
                current_fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            

            # #抽帧保存图片
            # count += 1
            # if count % 30 == 0:  # 每30帧保存1张
            #     cv2.imwrite(f'data/frame_{count//30:06d}.jpg', frame)

            # 在画面上显示结果
            display_frame = frame

            # 绘制识别结果
            result_text = f"Denomination: {predicted_class} Yuan"
            confidence_text = f"Confidence: {confidence:.2%}"
            fps_text = f"FPS: {current_fps:.1f}"

            # 设置文本样式
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            # 根据预测结果设置颜色
            if predicted_class == "can't recognition":
                color = (0, 0, 255)  # 红色表示未知
            else:
                color = (0, 255, 0)  # 绿色表示识别成功

            # 绘制文本背景
            cv2.rectangle(display_frame, (10, 10), (300, 120), (0, 0, 0), -1)

            # 绘制文本
            cv2.putText(display_frame, result_text, (20, 40), font, font_scale, color, thickness)
            cv2.putText(display_frame, confidence_text, (20, 70), font, font_scale, color, thickness)
            cv2.putText(display_frame, fps_text, (20, 100), font, font_scale, (255, 255, 255), thickness)

            # 显示画面
            cv2.imshow("RMB-live", display_frame)
            
            # # 边缘检测1
            # # canny算法是输出单通道二值图像，只有0（黑）和255（白）,只有黑白,125,175是一个偏平均的区域
            # canny_frame = cv2.Canny(frame, 125, 175)
            # # 后续分离三通道需要先转rgb,本质是三原色三通道增加相同的量,合成结果仍然为白色和黑色
            # canny_rgb_frame = cv2.cvtColor(canny_frame , cv2.COLOR_GRAY2RGB)
            # cv2.imshow("Canny RGB Frame", canny_rgb_frame)
            # # 分离三通道,使后续合并的边缘框突出于较为通道均衡的原视频流,用户容易看出(色盲就没办法了)
            # # 我们只取绿色通道,所以只有绿色通道命名完整
            # b,green_edge,r = cv2.split(canny_rgb_frame)
            # add_green_edge = cv2.cvtColor(green_edge, cv2.COLOR_GRAY2RGB)
            # #合并两个彩色通道
            # add_frame = cv2.add(add_green_edge,frame)
            # cv2.imshow("add", add_frame)
            # error: 仍然为白色边缘
            # 原因分析, 原理搞错了,只要是单通道就是黑白,分离完再转rgb仍然为黑白,
            # 用数学来说, 分离前：RGB图像（三维数组也是矩阵，通道在矩阵中的位置决定颜色）
            # 分离后二维矩阵通道失去了数学结构上的在三维矩阵内部的位置属性,也就失去了颜色属性
            # 我们需要一个三维矩阵,但是只有一个为canny(0,255)的通道,其他均为(0),这样才行


            # 边缘检测2
            # canny算法是输出单通道二值图像，只有0（黑）和255（白）,只有黑白,125,
        
            canny_frame = cv2.Canny(frame, 125, 175)
            green_edge = np.zeros_like(frame)
            green_edge[canny_frame == 255] = [0, 255, 0]
            #合并两个彩色通道
            add_frame = cv2.add(green_edge,frame)
            cv2.imshow("add", add_frame)

        else:
            print('等待数据......')
            time.sleep(0.1)

        # 检查退出键
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")

if __name__ == "__main__":
    main()