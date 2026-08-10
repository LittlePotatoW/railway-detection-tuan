import numpy as np
import cv2

class NormalModel:
    def __init__(self, normal_imgs: list[np.ndarray] | None = None, K: int = 16):
        self.K = K
        self.sum: np.ndarray | None = None       # Σx，float64 累加器
        self.sum_sq: np.ndarray | None = None    # Σx²，float64 累加器
        self.count: int = 0
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None
        if normal_imgs is not None:
            self.add(normal_imgs)

    def add(self, images: np.ndarray | list[np.ndarray]) -> None:
        if isinstance(images, np.ndarray) and images.ndim == 2:
            images = [images]
        for img in images:
            f = img.astype(np.float64)
            if self.sum is None:
                self.sum = f
                self.sum_sq = f * f
            else:
                self.sum += f
                assert self.sum_sq is not None 
                self.sum_sq += f * f
        self.count += len(images)

    def build(self):
        if isinstance(self.sum, np.ndarray) and isinstance(self.sum_sq, np.ndarray):
            mean_im = (self.sum / self.count).astype(np.float32)
            mean_sq = (self.sum_sq / self.count).astype(np.float32)
            mu = cv2.boxFilter(mean_im, -1, (self.K, self.K))
            e2 = cv2.boxFilter(mean_sq, -1, (self.K, self.K))
            sigma = np.sqrt(np.maximum(e2 - mu**2, 0))
            self.mu, self.sigma = mu, sigma

    def get_mean(self):
        return self.mu
    def get_sigma(self):
        return self.sigma