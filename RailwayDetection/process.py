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
        s_img = BaseAlgor.opening(s_img,7)
        s_img = Preprocess.fill_from_center_f(s_img)
        return s_img

    @staticmethod
    def align(img: np.ndarray, ref_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        mask = Pipeline.make_mask(img)
        dx, dy, response = Preprocess.get_shift(mask, ref_mask)
        return Preprocess.align_trans_s(img, dx, dy), mask, dx, dy, response
    
    @staticmethod
    def align_debug(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        ref_mask = ImageLoader.usePIL(r"C:\Users\小土豆\Desktop\铁轨异物\detection\ref-instance\mask.png")
        ref_mask = PIL2gray(ref_mask)
        ref_mask = PIL2numpyf(ref_mask)
        mask = Pipeline.make_mask(img)
        dx, dy, response = Preprocess.get_shift(mask, ref_mask)
        return Preprocess.align_trans_s(img, dx, dy), mask, dx, dy, response

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
    def train(folder: str | Path, mask: np.ndarray | str | Path | None = None,
            model: NormalModel | None = None,
            save_path: str | Path | None = None) -> NormalModel:
        # 从文件夹加载所有图片用于训练模型。
        folder = Path(folder)
        exts = IMG_SUFFIX
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
        if not files:
            raise FileNotFoundError(f"文件夹里没有图片: {folder}")
        
        if mask is None:                # 自动用第一张图生成对齐参考掩膜
            first = PIL2numpyf(PIL2gray(ImageLoader.usePIL(files[0])))
            ref_mask = Pipeline.make_mask(first)
        elif isinstance(mask, (str, Path)): ref_mask = PIL2numpyf(PIL2gray(ImageLoader.usePIL(mask)))
        elif isinstance(mask, np.ndarray): ref_mask = mask
        if model is None:
            model = NormalModel()

        for f in files:
            img = PIL2numpyf(PIL2gray(ImageLoader.usePIL(f)))
            if model.sum is not None and model.sum.shape != img.shape:
                raise ValueError(f"图片尺寸 {img.shape} 与模型已训练的尺寸 {model.sum.shape} 不一致")
            aligned, *_ = Pipeline.align(img, ref_mask)      # 对齐
            model.add(Pipeline.preprocess(aligned))          # 预处理

        model.build()                                        # 算 μ、σ

        if save_path: model.save(save_path)
        return model