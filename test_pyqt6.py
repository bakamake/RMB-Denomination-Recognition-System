#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt6 基本功能测试
测试依赖是否正确安装
"""

import sys
import numpy as np

# 测试导入
print("测试1: 导入PyQt6模块...")
try:
    from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
    from PyQt6.QtCore import QTimer
    from PyQt6.QtGui import QPixmap, QImage
    print("✓ PyQt6 导入成功")
except ImportError as e:
    print(f"✗ PyQt6 导入失败: {e}")
    sys.exit(1)

print("\n测试2: 导入OpenCV...")
try:
    import cv2
    print(f"✓ OpenCV 导入成功，版本: {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV 导入失败: {e}")
    sys.exit(1)

print("\n测试3: 导入TensorFlow...")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow 导入成功，版本: {tf.__version__}")
except ImportError as e:
    print(f"✗ TensorFlow 导入失败: {e}")
    sys.exit(1)

print("\n测试4: 创建模拟frame...")
try:
    # 创建一个640x480的模拟摄像头frame
    mock_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    print(f"✓ 模拟frame创建成功，形状: {mock_frame.shape}")
except Exception as e:
    print(f"✗ 模拟frame创建失败: {e}")
    sys.exit(1)

print("\n测试5: OpenCV BGR -> RGB转换...")
try:
    rgb_frame = cv2.cvtColor(mock_frame, cv2.COLOR_BGR2RGB)
    print(f"✓ 颜色空间转换成功，形状: {rgb_frame.shape}")
except Exception as e:
    print(f"✗ 颜色空间转换失败: {e}")
    sys.exit(1)

print("\n测试6: QImage创建...")
try:
    height, width, channels = rgb_frame.shape
    bytes_per_line = channels * width
    qimage = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    print(f"✓ QImage创建成功，大小: {width}x{height}")
except Exception as e:
    print(f"✗ QImage创建失败: {e}")
    sys.exit(1)

print("\n测试7: QPixmap创建...")
try:
    pixmap = QPixmap.fromImage(qimage)
    print(f"✓ QPixmap创建成功")
except Exception as e:
    print(f"✗ QPixmap创建失败: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("所有测试通过！✓")
print("PyQt6 + OpenCV + TensorFlow环境配置正确")
print("="*50)

# 如果在有显示器的环境中，可以尝试创建GUI
print("\n提示: 在无头环境中无法显示GUI窗口")
print("要运行完整应用，请在有显示器的环境中执行:")
print("  uv run python qt_camera_demo.py")
