from RailwayDetection import *
import numpy as np

class Pipeline:
    @staticmethod
    def preprocess(img: np.ndarray) -> np.ndarray:
        save_img = BaseAlgor.stand_nor_to255(img)
        return save_img