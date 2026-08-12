import numpy as np
import cv2
from pathlib import Path


class NormalModel:

    def __init__(self, images: list[np.ndarray] | np.ndarray | None = None,
                 masks: list[np.ndarray] | np.ndarray | None = None,
                 K: int = 16):
        self.K = K
        self.sum: np.ndarray | None = None              # Σ(img*mask)，float64
        self.sum_sq: np.ndarray | None = None           # Σ(img²*mask)，float64
        self.count: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None
        if images is not None: self.add(images, masks)

    def add(self, images: np.ndarray | list[np.ndarray],
            masks: np.ndarray | list[np.ndarray] | None = None) -> None:
        if isinstance(images, np.ndarray): images = [images]
        if masks is None:
            from ..process import Pipeline              # 延迟导入 避免循环依赖
            masks = [Pipeline.make_mask(img) for img in images]
        elif isinstance(masks, np.ndarray): masks = [masks]
        if len(masks) != len(images):
            raise ValueError(f"masks 数量 {len(masks)} 与 images 数量 {len(images)} 不一致")
        for img in images:                              # 对训练图预处理
            from ..process import Pipeline              # 延迟导入 避免循环依赖
            img = Pipeline.preprocess(img)

        for img, m in zip(images, masks): # 统计条件 掩膜内（m>0）图像非纯黑（f>0）
            f = img.astype(np.float64)
            mf = ((np.asarray(m) > 0) & (f > 0)).astype(np.float64)
            if self.sum is None:
                self.sum = f * mf
                self.sum_sq = (f * f) * mf
                self.count = mf.copy()
            else:
                assert self.sum is not None
                assert self.sum_sq is not None
                assert self.count is not None
                self.sum += f * mf
                self.sum_sq += (f * f) * mf
                self.count += mf

    def build(self) -> None:
        # 统计计算 μ σ
        if self.sum is None or self.count is None: return
        K = self.K
        block_sum = cv2.boxFilter(self.sum, -1, (K, K))
        assert self.sum_sq is not None
        block_sq = cv2.boxFilter(self.sum_sq, -1, (K, K))
        block_cnt = cv2.boxFilter(self.count, -1, (K, K))

        with np.errstate(divide="ignore", invalid="ignore"):
            mu_all = block_sum / block_cnt              # 窗口内有效像素的均值
            e2_all = block_sq / block_cnt
        sigma_all = np.sqrt(np.maximum(e2_all - mu_all**2, 0))

        covered = block_cnt > 0
        self.mu = np.where(covered, mu_all, 0.0).astype(np.float32)
        self.sigma = np.where(covered, sigma_all, 0.0).astype(np.float32)

    def get_mean(self) -> np.ndarray | None:
        return self.mu

    def get_sigma(self) -> np.ndarray | None:
        return self.sigma

    def save(self, path: str | Path) -> None:
        arrays = {"K": np.asarray(self.K)}
        for key, arr in [("count", self.count), ("sum", self.sum),
                         ("sum_sq", self.sum_sq), ("mu", self.mu),
                         ("sigma", self.sigma)]:
            if arr is not None:
                arrays[key] = arr
        np.savez_compressed(path, allow_pickle=False, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "NormalModel":
        data = np.load(path)
        model = cls.__new__(cls)
        model.K = int(data["K"])
        model.count = data["count"] if "count" in data.files else None
        model.sum = data["sum"] if "sum" in data.files else None
        model.sum_sq = data["sum_sq"] if "sum_sq" in data.files else None
        model.mu = data["mu"] if "mu" in data.files else None
        model.sigma = data["sigma"] if "sigma" in data.files else None
        return model
