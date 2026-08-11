import cv2 
import numpy as np
from typing import Tuple

class BaseAlgor:
    @staticmethod
    def binarizef(img: np.ndarray, threshold: float = 0.5, max_val: float = 255.0) -> np.ndarray:
        return (img > threshold * max_val).astype(np.float32) * 255
    
    @staticmethod
    def invert(img: np.ndarray) -> np.ndarray:
        # 二值反转 白变黑 黑变白
        if img.max() > 1.0: return (255.0 - img).astype(np.float32)
        return (1.0 - img).astype(np.float32)

    @staticmethod
    def standardize(img: np.ndarray, 
                    mean: float | None = None, sigma: float | None = None) -> np.ndarray:
        # 标准化
        if mean is None: mean = float(np.mean(img))
        if sigma is None: sigma = float(np.std(img))
        s_sigma = float(sigma)
        if np.isscalar(sigma): safe_sigma = max(s_sigma, 1e-6)
        else: safe_sigma = np.where(sigma < 1e-6, 1.0, sigma)
        return (img - mean) / safe_sigma
    
    @staticmethod
    def normalize(img: np.ndarray,
                  min_val: float | None = None, max_val: float | None = None) -> np.ndarray:
        # 归一化
        if min_val is None: min_val = float(np.min(img))
        if max_val is None: max_val = float(np.max(img))
        min_val = float(min_val)
        max_val = float(max_val)
        range_val = max_val - min_val
        if range_val < 1e-6:                                        # 除零保护
            return np.zeros_like(img)
        return (img - min_val) / range_val
    
    @staticmethod
    def f1tof255(img: np.ndarray) -> np.ndarray:
        return (img * 255).astype(np.float32)

    @staticmethod
    def f255tof1(img: np.ndarray) -> np.ndarray:
        return img.astype(np.float32) / 255.0
    
    @staticmethod
    def stand_nor_to255(img: np.ndarray) -> np.ndarray:
        save_img = BaseAlgor.standardize(img)
        save_img = BaseAlgor.normalize(save_img)
        save_img = BaseAlgor.f1tof255(save_img)
        return save_img
    
    @staticmethod
    def erode(img: np.ndarray, size: int = 3) -> np.ndarray:
        # 腐蚀
        if size < 1:
            size = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        if img.dtype not in (np.uint8, np.float32, np.float64):
            img = img.astype(np.float32)
        return cv2.erode(img, kernel, iterations=1)

    @staticmethod
    def dilate(img: np.ndarray, size: int = 3) -> np.ndarray:
        # 膨胀
        if size < 1:
            size = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        if img.dtype not in (np.uint8, np.float32, np.float64):
            img = img.astype(np.float32)
        return cv2.dilate(img, kernel, iterations=1)

    @staticmethod
    def opening(img: np.ndarray, size: int = 3) -> np.ndarray:
        # 开运算 先腐蚀后膨胀
        if size < 1:
            size = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        if img.dtype not in (np.uint8, np.float32, np.float64):
            img = img.astype(np.float32)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

    @staticmethod
    def closing(img: np.ndarray, size: int = 3) -> np.ndarray:
        # 闭运算 先膨胀后腐蚀
        if size < 1:
            size = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        if img.dtype not in (np.uint8, np.float32, np.float64):
            img = img.astype(np.float32)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
