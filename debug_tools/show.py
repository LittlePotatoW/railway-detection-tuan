import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def show(images, cols=3, figsize=(12, 10)):
    """通用显示：接受单张/多张图像（灰度、彩色、PIL、float32、uint8 均可）。"""
    if isinstance(images, Image.Image):
        images = [images]
    elif isinstance(images, np.ndarray):
        images = [images] if images.ndim <= 3 else list(images)
    elif not isinstance(images, (list, tuple)):
        images = list(images)

    n = len(images)
    if n == 0:
        print("No images to show.")
        return

    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    for i in range(rows * cols):
        ax = axes[i]
        if i >= n:
            ax.axis("off")
            continue
        im = np.asarray(images[i])
        if im.ndim == 2:
            if im.max() <= 1.0:
                ax.imshow(im, cmap="gray")
            else:
                ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        else:
            if im.max() > 1.0:
                im = np.clip(im, 0, 255).round().astype(np.uint8)
            ax.imshow(im)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def _to_pil_rgb(img: np.ndarray) -> Image.Image:
    """np.ndarray（灰度/彩色，float32 0~255 或 0~1，uint8）→ PIL RGB。"""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.clip(arr, 0, 255).round().astype(np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")
    if arr.ndim == 3:
        if arr.max() > 1.0:
            arr = np.clip(arr, 0, 255).round().astype(np.uint8)
        else:
            arr = (np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
    raise ValueError(f"不支持的数组维度: {arr.ndim}")


def draw_boxes(img: np.ndarray, boxes: list,
               color=(0, 0, 255), linewidth: int = 2) -> Image.Image:
    """在图像上画框，返回 PIL 图像（不改原数组）。默认蓝色。

    boxes: [(x, y, w, h), ...]
    """
    out = _to_pil_rgb(img)
    draw = ImageDraw.Draw(out)
    for (x, y, w, h) in boxes:
        draw.rectangle([x, y, x + w, y + h], outline=color, width=linewidth)
    return out


def draw_lines(img: np.ndarray, lines: list,
               color=(0, 0, 255), linewidth: int = 2) -> Image.Image:
    """在图像上画线，返回 PIL 图像（不改原数组）。默认蓝色。

    lines: 每条线是 (a, b)（x = a + b*y）或两点 ((x1,y1),(x2,y2))
    """
    out = _to_pil_rgb(img)
    draw = ImageDraw.Draw(out)
    h = out.size[1]
    for line in lines:
        if isinstance(line[0], (tuple, list, np.ndarray)):
            (x1, y1), (x2, y2) = line
        else:
            a, b = line                       # x = a + b*y
            x1, y1, x2, y2 = a, 0.0, a + b * (h - 1), h - 1.0
        draw.line([(x1, y1), (x2, y2)], fill=color, width=linewidth)
    return out
