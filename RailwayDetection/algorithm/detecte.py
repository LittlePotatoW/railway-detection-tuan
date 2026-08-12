import numpy as np
import cv2

class ScoreCal:

    @staticmethod
    def  Z_score(img: np.ndarray, K: int, eps: float, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        t_mean = cv2.boxFilter(img, -1, (K, K))
        z = np.abs(t_mean - mu) / (sigma + eps)
        return z

class BoxCal:

    @staticmethod
    def threshold(z : np.ndarray, T: float = 4.0, mask: np.ndarray | None = None)-> np.ndarray:
        # 阈值分割
        if mask is None: binary = (z > T).astype(np.uint8) * 255
        else: binary = ((z > T) & (mask > 0)).astype(np.uint8) * 255
        return binary
    
    @staticmethod
    def morphological_cleanup(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # 开运算
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)# 闭运算
        return cleaned

    @staticmethod
    def extract_boxes(binary: np.ndarray, min_area: int = 30) -> list[tuple[int, int, int, int]]:
        # 提取连通域
        n, _labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        boxes = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area >= min_area: boxes.append((x, y, w, h))
        return boxes