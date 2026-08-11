import cv2 
import numpy as np
from typing import Tuple

class BaseAlgor:
    @staticmethod
    def binarize(img: np.ndarray, threshold: float = 0.5, max_val: float = 255.0) -> np.ndarray:
        return (img > threshold * max_val).astype(np.uint8) * 255

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
