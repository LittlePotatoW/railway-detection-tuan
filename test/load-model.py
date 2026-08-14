import sys
from pathlib import Path

# 把 detection 目录加进导入搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RailwayDetection import *
from debug_tools import *

MODEL_PATH = r"test\model.npz"

model = OperModel.load_model(MODEL_PATH)

nor_img_s = model.get_sigma()
nor_img_m = model.get_mean()

show([nor_img_s, nor_img_m])


