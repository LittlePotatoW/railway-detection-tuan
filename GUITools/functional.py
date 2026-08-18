from pathlib import Path
from typing import Callable, Tuple

import numpy as np
from PIL import Image, ImageDraw

from RailwayDetection import OperModel, Pipeline


def train_model(model_path: str | Path | None,
                save_path: str | Path,
                image_folder: str | Path,
                log: Callable[[str], None] | None = None):

    def _log(message: str) -> None:
        if log is not None: log(message)

    if not str(image_folder).strip():
        _log("Error: Training image folder path is empty.")
        return None
    
    if str(save_path) == "":
        _log("Error: Model save path is empty.")
        return None

    folder = Path(image_folder)
    if not folder.is_dir():
        _log(f"Error: Training image folder does not exist: {folder}")
        return None

    try:
        model = None
        model_path_str = "" if model_path is None else str(model_path).strip()
        if model_path_str:
            model_file = Path(model_path_str)
            if not model_file.is_file():
                _log(f"Error: Model file does not exist: {model_file}")
                return None
            _log(f"Loading existing model: {model_file}")
            model = OperModel.load_model(model_file)
            _log("Continuing training from the existing model")
        else:
            _log("Training from scratch (no existing model)")

        _log(f"Training images folder: {folder}")
        model = OperModel.train(folder, model=model, save_path=save_path)
        mean = model.get_mean()
        _log(f"Model training finished, shape: {None if mean is None else mean.shape}")
        _log(f"Model saved to: {save_path}")
        return model
    except Exception as exc:
        _log(f"Model training failed: {exc}")
        return None


def detect(model_path: str | Path,
           image_path: str | Path,
           log: Callable[[str], None] | None = None) -> Tuple[Image.Image, list, dict] | None:
    def _log(message: str) -> None:
        if log is not None: log(message)

    if not str(image_path).strip() or not Path(image_path).is_file():
        _log(f"Error: Image file does not exist: {image_path}")
        return None
    if not str(model_path).strip() or not Path(model_path).is_file():
        _log(f"Error: Model file does not exist: {model_path}")
        return None

    try:
        _log(f"Loading model: {model_path}")
        model = OperModel.load_model(model_path)

        _log(f"Loading image: {image_path}")
        img = Pipeline.load_image(image_path)
        _log(f"Aligning image, shape: {img.shape}")
        aligned, mask, dx, angle, (aL, bL), (aR, bR) = Pipeline.align(img)
        _log(f"Aligned, dx = {dx:.2f}, angle = {angle:.2f} deg")

        feat = Pipeline.preprocess(aligned)
        z = Pipeline.score_cal(feat, model)
        boxes = Pipeline.detecte(z, mask=mask)
        _log(f"Detection finished, {len(boxes)} anomaly region(s)")

        rail_z = z[mask > 0]
        info = {
            "image_size": aligned.shape,
            "dx": float(dx),
            "angle": float(angle),
            "max_z": float(rail_z.max()) if rail_z.size else 0.0,
            "z_p99": float(np.percentile(rail_z, 99)) if rail_z.size else 0.0,
        }
        return _draw_boxes(aligned, boxes), boxes, info
    except Exception as exc:
        _log(f"Detection failed: {exc}")
        return None


def _draw_boxes(img: np.ndarray, boxes: list,
               color=(0, 0, 255), linewidth: int = 2) -> Image.Image:
    # 在图像上画框 返回 PIL 图像
    out = _to_pil_rgb(img)
    draw = ImageDraw.Draw(out)
    for (x, y, w, h) in boxes: draw.rectangle([x, y, x + w, y + h], outline=color, width=linewidth)
    return out

def _to_pil_rgb(img: np.ndarray) -> Image.Image:
    arr = np.clip(np.asarray(img), 0, 255).round().astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")