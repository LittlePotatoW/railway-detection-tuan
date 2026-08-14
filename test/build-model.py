import sys
from pathlib import Path

# 把 detection 目录加进导入搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RailwayDetection import *


IMAGE_DIR_PATH = r"nor_img"

model = OperModel.train(IMAGE_DIR_PATH)

model.save("model.npz")




