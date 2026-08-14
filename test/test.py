import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# 把 detection 目录加进导入搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RailwayDetection import *
from debug_tools import *

MODEL_PATH = r"model.npz"

model = OperModel.load_model(MODEL_PATH)

img = Pipeline.load_image(r"C:\Users\小土豆\Desktop\铁轨异物\缺陷照片\01.png")
imgl = [img]

aligned, mask, dx, angle, (aL, bL), (aR, bR) = Pipeline.align(img)

feat = Pipeline.preprocess(aligned)
z = Pipeline.score_cal(feat, model)
boxes = Pipeline.detecte(z, mask=mask)

print("矫正: dx =", round(dx, 2), " angle =", round(angle, 2))
print("检测框:", boxes)
rail_z = z[mask > 0]
print("轨道区 z: 最大 =", round(float(rail_z.max()), 2),
      " 99%分位 =", round(float(np.percentile(rail_z, 99)), 2))

imgl.append(feat)
imgl.append(draw_boxes(aligned, boxes))
show(imgl)

z_disp = np.where(mask > 0, z, 0.0)
vmax_z = float(np.percentile(rail_z, 99.9))
fig, ax = plt.subplots(figsize=(6, 8))
im = ax.imshow(z_disp, cmap="jet", vmin=0, vmax=vmax_z)
fig.colorbar(im, ax=ax, fraction=0.046, label="z")
ax.set_title("z 异常分数热力图（轨道区域内）", fontsize=12)
ax.axis("off")
plt.tight_layout()
plt.show()