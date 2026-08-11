from RailwayDetection import *
import numpy as np
from typing import Tuple
from pathlib import Path

DIRECT_FILTER_REDIUS: int = 4

class Pipeline:
    @staticmethod
    def align(img: np.ndarray, ref_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        s_img = BaseAlgor.binarizef(img, 0.3)
        s_img = BaseAlgor.opening(s_img,7)
        s_img = Preprocess.fill_from_center_f(s_img)
        mask = s_img
        dx, dy, response = Preprocess.get_shift(s_img, ref_mask)
        return Preprocess.align_trans_s(img, dx, dy), mask, dx, dy, response
    
    @staticmethod
    def align_debug(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        ref_mask = ImageLoader.usePIL(r"C:\Users\小土豆\Desktop\铁轨异物\detection\ref-instance\mask.png")
        ref_mask = PIL2gray(ref_mask)
        ref_mask = PIL2numpyf(ref_mask)
        s_img = BaseAlgor.binarizef(img, 0.3)
        s_img = BaseAlgor.opening(s_img,7)
        s_img = Preprocess.fill_from_center_f(s_img)
        mask = s_img
        dx, dy, response = Preprocess.get_shift(s_img, ref_mask)
        return Preprocess.align_trans_s(img, dx, dy), mask, dx, dy, response

    @staticmethod
    def preprocess(img: np.ndarray) -> np.ndarray:
        save_img = BaseAlgor.stand_nor_to255(img)
        save_img = Preprocess.directional_filter_frequency(save_img, 0, DIRECT_FILTER_REDIUS)
        return save_img

class OperModel:
    @staticmethod
    def get_new_model() -> NormalModel:
        return NormalModel()
    @staticmethod
    def load_model(path: str | Path) -> NormalModel:
        return NormalModel.load(path)