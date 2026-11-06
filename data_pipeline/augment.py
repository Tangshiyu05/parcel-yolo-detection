# -*- coding: utf-8 -*-
"""
离线增强：对软包/信封等少数类做过采样与拼接，缓解类别不平衡。
训练时的在线 Mosaic/Mixup 已在 config 中开启，本文件用于离线扩充。
"""
import cv2
import numpy as np


def load_image(path: str, size: int = 640):
    img = cv2.imread(path)
    return cv2.resize(img, (size, size))


def mosaic_four(images, size: int = 640):
    """四合一拼接，模拟堆叠多目标场景。"""
    half = size // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:half, :half] = images[0][:half, :half]
    canvas[:half, half:] = images[1][:half, half:]
    canvas[half:, :half] = images[2][half:, :half]
    canvas[half:, half:] = images[3][half:, half:]
    return canvas


def hsv_jitter(img, h_gain=0.015, s_gain=0.7, v_gain=0.4):
    """HSV 色彩扰动，模拟分拣线光照变化。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] += np.random.uniform(-h_gain, h_gain) * 180
    hsv[..., 1] += np.random.uniform(-s_gain, s_gain) * 255
    hsv[..., 2] += np.random.uniform(-v_gain, v_gain) * 255
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
