from pathlib import Path

from PIL import Image
import cv2
import numpy as np

class ImageLoader:
    @staticmethod
    def usePIL(path: Path | str) -> Image.Image:
        return Image.open(path)                     # RGB

    @staticmethod
    def useCV2(path: Path | str) -> np.ndarray | None:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR




