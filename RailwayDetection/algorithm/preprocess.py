import cv2 
import numpy as np
from typing import Tuple

class Preprocess:

    @staticmethod
    def getshift(img: np.ndarray, ref: np.ndarray) -> Tuple[float, float, float]:
        # 返回两张图的平移量
        shift, response = cv2.phaseCorrelate(img, ref)
        dx, dy = shift
        return dx, dy, response

    @staticmethod
    def align(img: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, float]:
        # 按参考图对齐图像
        dx, dy, response = Preprocess.getshift(img, ref)
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype = np.float32)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0])), response