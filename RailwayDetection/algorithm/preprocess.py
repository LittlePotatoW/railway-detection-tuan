import cv2 
import numpy as np
from typing import Tuple

class Preprocess:

    @staticmethod
    def get_shift(img: np.ndarray, ref: np.ndarray) -> Tuple[float, float, float]:
        # 返回两张图的平移量
        shift, response = cv2.phaseCorrelate(img, ref)
        dx, dy = shift
        return dx, dy, response

    @staticmethod
    def align_trans(img: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, float]:
        # 按参考图对齐图像 平移变换
        dx, dy, response = Preprocess.get_shift(img, ref)
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype = np.float32)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0])), response
    
    @staticmethod
    def match_hist(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
        # 直方图匹配 按参考图对齐明暗
        def cdf(img: np.ndarray) -> np.ndarray:
            # 计算图像的累积分布函数
            hist, _ = np.histogram(img, bins=256, range=(0, 256))
            cdf = np.cumsum(hist) / hist.sum()
            return cdf
        cdf_img, cdf_ref = cdf(img), cdf(ref)
        lut = np.interp(cdf_img, cdf_ref, np.arange(256))
        return lut[img.astype(np.int32)]
    
    @staticmethod
    def match_linear(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
        # 线性变换 按参考图对齐明暗
        return (img - img.mean()) / (img.std() + 1e-6) * ref.std() + ref.mean()