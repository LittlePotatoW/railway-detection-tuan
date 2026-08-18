from PIL import Image
import numpy as np
import cv2

def PIL2numpyf(img: Image.Image) -> np.ndarray:
    return np.array(img, dtype=np.float32)

def numpy2PIL(img: np.ndarray) -> Image.Image:
    if img.dtype == np.float32: img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def PIL2gray(img: Image.Image) -> Image.Image:
    return img.convert('L')


def array255topil(arr: np.ndarray) -> Image.Image:
    # 将 0~255 的 np.ndarray 转为 PIL 图像
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def array255toheatmap(arr: np.ndarray, mask: np.ndarray | None = None) -> Image.Image:
    # 映射为 jet 彩色热力图 mask 非空时只统计掩膜内像素
    a = np.asarray(arr, dtype=np.float32)
    if mask is not None:
        m = np.asarray(mask) > 0
        valid = a[m]
        if valid.size == 0: return Image.fromarray(np.zeros((*a.shape, 3), dtype=np.uint8))
        lo, hi = float(valid.min()), float(valid.max())
        norm = np.zeros_like(a)
        if hi - lo >= 1e-6: norm[m] = (a[m] - lo) / (hi - lo)
        gray = (norm * 255).astype(np.uint8)
        bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb[~m] = (0, 0, 0)
        return Image.fromarray(rgb)
    lo, hi = float(a.min()), float(a.max())
    norm = np.zeros_like(a) if hi - lo < 1e-6 else (a - lo) / (hi - lo)
    gray = (norm * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
