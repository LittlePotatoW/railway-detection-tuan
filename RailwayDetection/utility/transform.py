from PIL import Image
import numpy as np

def PIL2numpyf(img: Image.Image) -> np.ndarray:
    return np.array(img, dtype=np.float32)

def numpy2PIL(img: np.ndarray) -> Image.Image:
    if img.dtype == np.float32: img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def PIL2gray(img: Image.Image) -> Image.Image:
    return img.convert('L')
