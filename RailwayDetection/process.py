from RailwayDetection import *
import numpy as np
from typing import Tuple, Dict, List, Set
from pathlib import Path

DIRECT_FILTER_REDIUS: int = 4

IMG_SUFFIX: Set = {".png", ".jpg", ".jpeg", ".bmp"}

class Pipeline:

    @staticmethod
    def make_mask(img: np.ndarray) -> np.ndarray:
        s_img = BaseAlgor.binarizef(img, 0.3)
        s_img = Preprocess.set_border_black(s_img)
        s_img = BaseAlgor.opening(s_img, 7)
        s_img = Preprocess.fill_from_center_f(s_img)
        return s_img

    @staticmethod
    def align(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, Tuple[float, float], Tuple[float, float]]:
        mask = Pipeline.make_mask(img)
        dx, angle, (aL, bL), (aR, bR) = Preprocess.get_fix_dx_angle(mask)
        img = Preprocess.rotate_image(img, angle)
        img = Preprocess.translate_image(img, dx)
        mask = Preprocess.rotate_image(mask, angle, is_mask=True)
        mask = Preprocess.translate_image(mask, dx, is_mask=True)
        return img, mask, dx, angle, (aL, bL), (aR, bR)

    @staticmethod
    def preprocess(img: np.ndarray) -> np.ndarray:
        save_img = BaseAlgor.stand_nor_to255(img)
        save_img = Preprocess.directional_filter_frequency(save_img, 0, DIRECT_FILTER_REDIUS)
        return save_img

class OperModel:
    @staticmethod
    def get_new_model(K: int | None = None) -> NormalModel:
        if K is None: return NormalModel()
        else: return NormalModel(K = K)
    
    @staticmethod
    def load_model(path: str | Path) -> NormalModel:
        return NormalModel.load(path)
    
    @staticmethod
    def save_model(model: NormalModel, path: str | Path) -> None:
        model.save(path)
    
    @staticmethod
    def train(folder: str | Path,
          model: NormalModel | None = None,
          save_path: str | Path | None = None) -> NormalModel:
        # 从文件夹加载所有图片用于训练模型。
        folder = Path(folder)
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_SUFFIX)

        if model is None: model = NormalModel()

        for f in files:
            img = PIL2numpyf(PIL2gray(ImageLoader.usePIL(f)))          # 加载图片并转为 灰度图 np.ndarray
            if model.sum is not None and model.sum.shape != img.shape:
                raise ValueError(f"图片尺寸 {img.shape} 与模型已训练的尺寸 {model.sum.shape} 不一致")
            aligned, mask, *_ = Pipeline.align(img)   # 每张图独立扶正+居中，拿到自己的掩膜
            model.add(Pipeline.preprocess(aligned), masks=mask)

        model.build()
        if save_path: model.save(save_path)
        return model
