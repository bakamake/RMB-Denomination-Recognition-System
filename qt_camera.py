#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt6 + OpenCV 摄像头Frame显示示例
集成rmb_live_classifier.py的功能到Qt界面
"""

import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QSlider,
    QGroupBox, QTextEdit, QSizePolicy, QComboBox,
    QFormLayout
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont

# 尝试导入TensorFlow（如果模型存在）
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("警告: TensorFlow未安装，面额识别功能将受限")

class CameraThread(QThread):
    """摄像头读取线程，避免阻塞UI"""
    frame_ready = pyqtSignal(np.ndarray)  # 发出frame就绪信号

    def __init__(self):
        super().__init__()
        self.cap = None
        self.running = False

    def run(self):
        """线程主循环"""
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("错误：无法打开摄像头0")
            return

        print("摄像头连接成功！")
        self.running = True

        # FPS计算变量
        fps_counter = 0
        fps_start_time = cv2.getTickCount()

        while self.running:
            ret, frame = self.cap.read()

            if ret:
                # 计算FPS
                fps_counter += 1
                if fps_counter >= 30:  # 每30帧计算一次FPS
                    fps_time = (cv2.getTickCount() - fps_start_time) / cv2.getTickFrequency()
                    current_fps = fps_counter / fps_time
                    fps_counter = 0
                    fps_start_time = cv2.getTickCount()
                    print(f"FPS: {current_fps:.2f}")

                # 发出frame信号
                self.frame_ready.emit(frame)
            else:
                print("读取帧失败")
                break

        self.cap.release()

    def stop(self):
        """停止线程"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.quit()
        self.wait()


class ImageProcessor:
    """图像处理类 - 将rmb_live_classifier.py的功能封装到这里"""

    @staticmethod
    def apply_canny_edge_detection(frame, threshold1=125, threshold2=175):
        """Canny边缘检测"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)
        return edges

    @staticmethod
    def add_green_edges(original_frame, edges):
        """在原图上添加绿色边缘"""
        # 转换为3通道的边缘图像
        edge_colored = np.zeros_like(original_frame)
        edge_colored[edges == 255] = [0, 255, 0]  # 绿色边缘
        # 合并原图和边缘
        combined = cv2.add(edge_colored, original_frame)
        return combined

    @staticmethod
    def fuse_canny_edges(frame, weak_thresh=(50, 100), strong_thresh=(150, 250)):
        """多级Canny阈值融合"""
        # 弱边缘和强边缘
        canny_weak = cv2.Canny(frame, weak_thresh[0], weak_thresh[1])
        canny_strong = cv2.Canny(frame, strong_thresh[0], strong_thresh[1])

        # 融合边缘
        fused = np.zeros_like(canny_strong)
        fused[canny_strong == 255] = 255  # 保留所有强边缘

        # 在弱边缘中保留长轮廓
        contours, _ = cv2.findContours(canny_weak, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.arcLength(contour, False) > 30:
                cv2.drawContours(fused, [contour], -1, 255, 1)

        return fused

    @staticmethod
    def apply_grabcut_segmentation(frame, rect=None):
        """GrabCut前景分割 - 优化版，减少迭代次数"""
        h, w = frame.shape[:2]

        # 预处理：降噪
        blurred = cv2.GaussianBlur(frame, (3, 3), 0)  # 降低内核大小

        # 初始化mask和模型
        mask = np.zeros((h, w), np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # 如果没有指定矩形，使用图像中央区域
        if rect is None:
            rect = (w//6, h//6, w*2//3, h*2//3)

        # 先用更小的矩形，提高速度
        rect_small = (
            int(rect[0] + rect[2] * 0.1),
            int(rect[1] + rect[3] * 0.1),
            int(rect[2] * 0.8),
            int(rect[3] * 0.8)
        )

        # 应用GrabCut - 只迭代2次（原来是5次）
        cv2.grabCut(blurred, mask, rect_small, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)

        # 处理mask：0和2是背景，1和3是前景
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result = frame * mask2[:, :, np.newaxis]

        return result

    @staticmethod
    def apply_harris_corner_detection(frame):
        """Harris角点检测"""
        # 转换灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        # Harris角点检测
        dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        dst = cv2.dilate(dst, None)  # 增强角点

        # 创建结果图像
        result = frame.copy()
        # 标红角点
        result[dst > 0.01 * dst.max()] = [0, 0, 255]

        return result

    @staticmethod
    def apply_orb_feature_detection(frame):
        """ORB特征检测"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 创建ORB检测器
        orb = cv2.ORB_create(nfeatures=1000)

        # 检测关键点和描述符
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        # 绘制关键点
        result = cv2.drawKeypoints(frame, keypoints, None,
                                   color=(0, 255, 0),  # 绿色
                                   flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        return result, len(keypoints)

    @staticmethod
    def apply_contour_detection(frame):
        """轮廓检测和过滤"""
        # 转换灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 查找轮廓
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建结果图像
        result = frame.copy()

        # 绘制轮廓
        for i, contour in enumerate(contours):
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            # 只显示面积大于100的轮廓
            if area > 100:
                # 绘制轮廓
                cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
                # 标记轮廓编号
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(result, str(i), (cx, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return result, len(contours)

    @staticmethod
    def apply_fast_segmentation(frame):
        """快速分割 - 基于颜色和边缘的简单分割"""
        # 转换到HSV颜色空间（更稳定）
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 定义人民币颜色范围（偏黄绿色）
        # 人民币的典型HSV范围
        lower_color1 = np.array([15, 30, 30])    # 浅黄色
        upper_color1 = np.array([85, 255, 255])  # 黄绿色
        lower_color2 = np.array([0, 0, 150])     # 深色（数字部分）
        upper_color2 = np.array([180, 255, 255])

        # 创建掩码
        mask1 = cv2.inRange(hsv, lower_color1, upper_color1)
        mask2 = cv2.inRange(hsv, lower_color2, upper_color2)
        color_mask = cv2.bitwise_or(mask1, mask2)

        # 形态学操作：去除噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

        # 查找最大轮廓
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # 创建最终掩码
            final_mask = np.zeros_like(color_mask)
            cv2.fillPoly(final_mask, [largest_contour], 255)

            # 应用掩码
            result = cv2.bitwise_and(frame, frame, mask=final_mask)
            return result

        # 如果没找到轮廓，返回原图
        return frame

    @staticmethod
    def detect_bill_features(frame):
        """检测纸币特征点（角点、特征点、几何特征）"""
        # 创建结果图像（彩色）
        result = frame.copy()

        # 1. 检测角点
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_float = np.float32(gray)  # Harris需要float32

        # Harris角点
        dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
        dst = cv2.dilate(dst, None)
        corners = (dst > 0.01 * dst.max()).astype(np.uint8) * 255

        # Shi-Tomasi角点
        corners_st = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

        # 2. ORB特征点 - 需要uint8类型的灰度图
        gray_uint8 = gray.astype(np.uint8)  # ORB需要uint8
        orb = cv2.ORB_create(nfeatures=1000)
        keypoints, descriptors = orb.detectAndCompute(gray_uint8, None)

        # 3. 查找轮廓分析几何特征
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bill_info = {
            'harris_corners': np.sum(corners > 0),
            'shi_tomasi_corners': len(corners_st) if corners_st is not None else 0,
            'orb_features': len(keypoints),
            'contours': len(contours),
            'dominant_contour': None,
            'aspect_ratio': 0,
            'area_ratio': 0
        }

        # 分析主要轮廓
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            bill_info['dominant_contour'] = largest_contour

            # 计算外接矩形和长宽比
            rect = cv2.boundingRect(largest_contour)
            x, y, w, h = rect
            aspect_ratio = max(w, h) / min(w, h)
            bill_info['aspect_ratio'] = aspect_ratio

            # 计算面积占比
            frame_area = frame.shape[0] * frame.shape[1]
            contour_area = cv2.contourArea(largest_contour)
            area_ratio = contour_area / frame_area
            bill_info['area_ratio'] = area_ratio

            # 绘制轮廓
            cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 2)

        # 绘制角点（红色）
        result[dst > 0.01 * dst.max()] = [0, 0, 255]

        # 绘制Shi-Tomasi角点（蓝色小圆点）
        if corners_st is not None:
            for corner in corners_st:
                x, y = corner[0]
                cv2.circle(result, (int(x), int(y)), 3, (255, 0, 0), -1)

        # 绘制ORB特征点（绿色）
        cv2.drawKeypoints(result, keypoints, result,
                         color=(0, 255, 0),  # 绿色
                         flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        # 不在图像上绘制文字，只返回数据和特征图
        return result, bill_info

    @staticmethod
    def predict_denomination(bill_info):
        """基于特征值预测纸币面额"""
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

        aspect_ratio = bill_info['aspect_ratio']
        area_ratio = bill_info['area_ratio']
        corner_count = bill_info['shi_tomasi_corners']
        feature_count = bill_info['orb_features']

        # 计算匹配分数
        scores = {}
        for denom, features in denomination_features.items():
            score = 0

            # 长宽比匹配
            ar_min, ar_max = features['aspect_ratio_range']
            if ar_min <= aspect_ratio <= ar_max:
                score += 25

            # 面积占比匹配
            area_min, area_max = features['area_ratio_range']
            if area_min <= area_ratio <= area_max:
                score += 25

            # 角点数量匹配
            corner_min, corner_max = features['corner_range']
            if corner_min <= corner_count <= corner_max:
                score += 25

            # 特征点数量匹配
            feat_min, feat_max = features['feature_range']
            if feat_min <= feature_count <= feat_max:
                score += 25

            scores[denom] = score

        # 选择分数最高的面额
        if scores:
            best_denomination = max(scores, key=scores.get)
            best_score = scores[best_denomination]

            # 如果分数太低，标记为未知
            if best_score < 10:
                return "未知", scores

            return best_denomination, scores

        return "未知", {}

    @staticmethod
    def frame_to_qpixmap(frame):
        """将OpenCV frame转换为QPixmap"""
        # OpenCV使用BGR格式，Qt使用RGB，需要转换
        if len(frame.shape) == 3:
            height, width, channels = frame.shape
            bytes_per_line = channels * width

            # BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimage = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

            return QPixmap.fromImage(qimage)
        else:
            # 灰度图
            height, width = frame.shape
            qimage = QImage(frame.data, width, height, width, QImage.Format.Format_Grayscale8)
            return QPixmap.fromImage(qimage)


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.camera_thread = None
        self.processor = ImageProcessor()
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("RMB识别系统 - PyQt6版本")
        self.setGeometry(100, 100, 1200, 800)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)

        # 左侧：视频显示区域
        left_panel = QVBoxLayout()

        # 视频显示标签
        self.video_label = QLabel("等待摄像头...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid gray;
                background-color: black;
                color: white;
                font-size: 16px;
            }
        """)
        left_panel.addWidget(self.video_label)

        # 处理后的视频显示
        self.processed_label = QLabel("处理后的视频")
        self.processed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_label.setMinimumSize(640, 480)
        self.processed_label.setStyleSheet("""
            QLabel {
                border: 2px solid gray;
                background-color: black;
                color: white;
                font-size: 16px;
            }
        """)
        left_panel.addWidget(self.processed_label)

        main_layout.addLayout(left_panel)

        # 右侧：控制面板
        right_panel = QVBoxLayout()

        # 控制按钮组
        control_group = QGroupBox("摄像头控制")
        control_layout = QVBoxLayout(control_group)

        self.start_btn = QPushButton("开始摄像头")
        self.start_btn.clicked.connect(self.start_camera)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止摄像头")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)

        right_panel.addWidget(control_group)

        # 处理模式选择组
        mode_group = QGroupBox("处理模式")
        mode_layout = QFormLayout(mode_group)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "原始视频",
            "Canny边缘检测",
            "绿色边缘增强",
            "多级Canny融合",
            "GrabCut分割",
            "快速颜色分割",
            "纸币特征检测",
            "面额识别",
            "Harris角点检测",
            "ORB特征检测",
            "轮廓检测"
        ])
        self.mode_combo.currentTextChanged.connect(self.change_mode)
        mode_layout.addRow("选择模式:", self.mode_combo)

        self.processing_info = QLabel("模式信息: 无")
        self.processing_info.setWordWrap(True)
        mode_layout.addRow("当前状态:", self.processing_info)

        right_panel.addWidget(mode_group)

        # 边缘检测参数组
        edge_group = QGroupBox("Canny边缘参数")
        edge_layout = QVBoxLayout(edge_group)

        # 阈值1滑块
        self.threshold1_label = QLabel("阈值1: 125")
        self.threshold1_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold1_slider.setRange(0, 300)
        self.threshold1_slider.setValue(125)
        self.threshold1_slider.valueChanged.connect(self.update_threshold1)
        edge_layout.addWidget(self.threshold1_label)
        edge_layout.addWidget(self.threshold1_slider)

        # 阈值2滑块
        self.threshold2_label = QLabel("阈值2: 175")
        self.threshold2_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold2_slider.setRange(0, 300)
        self.threshold2_slider.setValue(175)
        self.threshold2_slider.valueChanged.connect(self.update_threshold2)
        edge_layout.addWidget(self.threshold2_label)
        edge_layout.addWidget(self.threshold2_slider)

        edge_group.setVisible(False)  # 默认隐藏，需要时显示

        right_panel.addWidget(edge_group)

        # 保存引用
        self.edge_group = edge_group

        # 详细信息组（纸币特征和面额识别）
        detail_group = QGroupBox("详细信息")
        detail_layout = QVBoxLayout(detail_group)

        self.detail_text = QTextEdit()
        self.detail_text.setMaximumHeight(200)
        self.detail_text.setReadOnly(True)
        detail_layout.addWidget(self.detail_text)

        right_panel.addWidget(detail_group)

        # 基础信息组
        info_group = QGroupBox("系统日志")
        info_layout = QVBoxLayout(info_group)

        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(150)
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)

        right_panel.addWidget(info_group)

        main_layout.addLayout(right_panel)

        # 添加弹簧
        main_layout.addStretch()
        right_panel.addStretch()

        # 状态变量
        self.current_mode = "原始视频"
        self.threshold1 = 125
        self.threshold2 = 175
        self.feature_count = 0
        self.contour_count = 0
        self.frame_counter = 0  # 帧计数器，用于跳帧处理
        self.grabcut_interval = 10  # GrabCut每10帧处理一次
        self.last_grabcut_result = None  # 缓存上一次的GrabCut结果
        self.bill_info = None  # 纸币特征信息
        self.denomination = "未知"  # 识别的面额
        self.denomination_scores = {}  # 面额识别分数

    def start_camera(self):
        """开始摄像头"""
        if self.camera_thread is None or not self.camera_thread.isRunning():
            self.camera_thread = CameraThread()
            self.camera_thread.frame_ready.connect(self.update_frame)
            self.camera_thread.start()

            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.info_text.append("摄像头已启动")

    def stop_camera(self):
        """停止摄像头"""
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread = None

            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.video_label.setText("摄像头已停止")
            self.processed_label.setText("处理后的视频")
            self.info_text.append("摄像头已停止")

    def update_frame(self, frame):
        """更新帧显示"""
        # 显示原始视频
        pixmap = self.processor.frame_to_qpixmap(frame)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

        # 增加帧计数器
        self.frame_counter += 1

        # 根据当前模式处理并显示
        processed_frame = None
        info_text = ""

        if self.current_mode == "原始视频":
            processed_frame = frame
            info_text = "显示原始摄像头画面"
            self.edge_group.setVisible(False)

        elif self.current_mode == "Canny边缘检测":
            edges = self.processor.apply_canny_edge_detection(
                frame, self.threshold1, self.threshold2
            )
            processed_frame = edges
            info_text = f"Canny边缘检测 - 阈值1: {self.threshold1}, 阈值2: {self.threshold2}"
            self.edge_group.setVisible(True)

        elif self.current_mode == "绿色边缘增强":
            edges = self.processor.apply_canny_edge_detection(
                frame, self.threshold1, self.threshold2
            )
            processed_frame = self.processor.add_green_edges(frame, edges)
            info_text = f"绿色边缘增强 - 在原图上叠加Canny边缘"
            self.edge_group.setVisible(True)

        elif self.current_mode == "多级Canny融合":
            edges = self.processor.fuse_canny_edges(frame)
            processed_frame = edges
            info_text = "多级Canny融合 - 结合强弱边缘，突出主线"
            self.edge_group.setVisible(False)

        elif self.current_mode == "GrabCut分割":
            # 只有每10帧才处理一次GrabCut，避免卡顿
            if self.frame_counter % self.grabcut_interval == 0:
                try:
                    print("执行GrabCut分割...")
                    self.last_grabcut_result = self.processor.apply_grabcut_segmentation(frame)
                    info_text = f"GrabCut前景分割 - 每{self.grabcut_interval}帧更新一次（优化版）"
                except Exception as e:
                    print(f"GrabCut处理错误: {e}")
                    self.last_grabcut_result = frame
                    info_text = "GrabCut处理失败，显示原图"
            else:
                # 使用缓存的结果，如果没有缓存则使用当前帧
                if self.last_grabcut_result is None:
                    self.last_grabcut_result = frame

            processed_frame = self.last_grabcut_result
            self.edge_group.setVisible(False)

        elif self.current_mode == "快速颜色分割":
            processed_frame = self.processor.apply_fast_segmentation(frame)
            info_text = "快速颜色分割 - 基于人民币颜色特征，速度快"
            self.edge_group.setVisible(False)

        elif self.current_mode == "纸币特征检测":
            processed_frame, self.bill_info = self.processor.detect_bill_features(frame)
            info_text = f"纸币特征检测 - Harris角点、ORB特征、几何特征"
            self.edge_group.setVisible(False)

        elif self.current_mode == "面额识别":
            processed_frame, self.bill_info = self.processor.detect_bill_features(frame)
            if self.bill_info and self.bill_info['aspect_ratio'] > 0:
                self.denomination, self.denomination_scores = self.processor.predict_denomination(self.bill_info)
            info_text = f"面额识别 - 识别结果: {self.denomination}元"
            self.edge_group.setVisible(False)

        elif self.current_mode == "Harris角点检测":
            processed_frame = self.processor.apply_harris_corner_detection(frame)
            info_text = "Harris角点检测 - 检测图像中的角点（红色标记）"
            self.edge_group.setVisible(False)

        elif self.current_mode == "ORB特征检测":
            processed_frame, self.feature_count = self.processor.apply_orb_feature_detection(frame)
            info_text = f"ORB特征检测 - 检测到 {self.feature_count} 个特征点（绿色标记）"
            self.edge_group.setVisible(False)

        elif self.current_mode == "轮廓检测":
            processed_frame, self.contour_count = self.processor.apply_contour_detection(frame)
            info_text = f"轮廓检测 - 检测到 {self.contour_count} 个轮廓（绿色轮廓，蓝色编号）"
            self.edge_group.setVisible(False)

        # 显示处理后的帧
        if processed_frame is not None:
            processed_pixmap = self.processor.frame_to_qpixmap(processed_frame)
            scaled_processed_pixmap = processed_pixmap.scaled(
                self.processed_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.processed_label.setPixmap(scaled_processed_pixmap)

            # 更新状态信息
            self.processing_info.setText(f"模式信息: {info_text}")

            # 更新详细信息显示
            self.update_details_info()

            # 记录到信息区域
            if self.current_mode in ["ORB特征检测", "轮廓检测", "纸币特征检测", "面额识别"]:
                self.info_text.append(info_text)

    def update_threshold1(self, value):
        """更新阈值1"""
        self.threshold1 = value
        self.threshold1_label.setText(f"阈值1: {value}")

    def update_threshold2(self, value):
        """更新阈值2"""
        self.threshold2 = value
        self.threshold2_label.setText(f"阈值2: {value}")

    def change_mode(self, mode):
        """改变处理模式"""
        self.current_mode = mode
        self.frame_counter = 0  # 重置帧计数器
        self.last_grabcut_result = None  # 清除GrabCut缓存
        self.bill_info = None  # 清除纸币特征信息
        self.denomination = "未知"  # 重置面额识别
        self.denomination_scores = {}  # 清空分数
        self.detail_text.clear()  # 清空详细信息
        self.info_text.append(f"切换模式: {mode}")
        print(f"切换到模式: {mode}")

    def update_details_info(self):
        """更新详细信息显示"""
        if self.bill_info and self.current_mode in ["纸币特征检测", "面额识别"]:
            detail_info = []

            # 特征检测信息
            if self.current_mode == "纸币特征检测":
                detail_info.append("=" * 30)
                detail_info.append("纸币特征分析:")
                detail_info.append(f"Harris角点: {self.bill_info['harris_corners']}")
                detail_info.append(f"Shi-Tomasi角点: {self.bill_info['shi_tomasi_corners']}")
                detail_info.append(f"ORB特征点: {self.bill_info['orb_features']}")
                detail_info.append(f"轮廓数量: {self.bill_info['contours']}")
                if self.bill_info['aspect_ratio'] > 0:
                    detail_info.append(f"长宽比: {self.bill_info['aspect_ratio']:.2f}")
                    detail_info.append(f"面积占比: {self.bill_info['area_ratio']:.2%}")
                else:
                    detail_info.append("长宽比: 未检测到纸币")
                    detail_info.append("面积占比: 未检测到纸币")
                detail_info.append("")

            # 面额识别信息
            elif self.current_mode == "面额识别":
                detail_info.append(f"识别结果: {self.denomination}元")

            # 更新显示
            if detail_info:
                self.detail_text.setText("\n".join(detail_info))

    def closeEvent(self, event):
        """关闭事件"""
        self.stop_camera()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()