import cv2 
import numpy as np
from typing import Tuple

DIRECTED_FILTER_SPATIAL_ELONGATION: float = 3.0
DIRECTED_FILTER_FREQUENCY_AXIS_TO_IMG: int = 4      # 频域掩膜（椭圆）的长轴长度设置为图像宽度几分之一

class Preprocess:

    @staticmethod
    def get_fix_dx_angle(mask: np.ndarray) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
        # 输入图像掩膜 返回图像的x方向平移量和旋转量 即 将铁轨移至中间
        # 逐行扫描
        h, w = mask.shape
        ys, xs = np.where(mask > 0)

        left = np.full(h, w, dtype=np.float64)
        right = np.full(h, -1.0, dtype=np.float64)
        np.minimum.at(left, ys, xs)
        np.maximum.at(right, ys, xs)

        valid = np.where((left < w) & (right >= 0))[0]
        bL, aL = np.polyfit(valid, left[valid], 1)
        bR, aR = np.polyfit(valid, right[valid], 1)

        aC, bC = (aL + aR) / 2, (bL + bR) / 2

        angle_deg = -np.degrees(np.arctan(bC))

        cx, cy = w / 2.0, h / 2.0
        dx = ((cx - aC) - bC * cy) / np.sqrt(1.0 + bC**2)

        return dx, angle_deg, (aL, bL), (aR, bR)
    
    @staticmethod
    def rotate_image(img: np.ndarray, angle_deg: float, center=None, is_mask: bool = False) -> np.ndarray:
        # 绕图像中心旋转
        h, w = img.shape[:2]
        if center is None:
            center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        flags = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        return cv2.warpAffine(img, M, (w, h), flags=flags)
    
    @staticmethod
    def translate_image(img: np.ndarray, dx: float, dy: float = 0.0, is_mask: bool = False) -> np.ndarray:
        # 平移图像
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype = np.float32)
        flags = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=flags)
    
    @staticmethod
    def match_hist(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
        # 直方图匹配 按参考图对齐明暗
        def cdf(img: np.ndarray) -> np.ndarray:
            # 计算图像的累积分布函数
            hist, _ = np.histogram(img, bins=256, range=(0, 256))
            cdf = np.cumsum(hist) / hist.sum()
            return cdf
        cdf_img, cdf_ref = cdf(img), cdf(ref)
        lut = np.interp(cdf_img, cdf_ref, np.arange(256))
        return lut[img.astype(np.int32)]
    
    @staticmethod
    def match_linear(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
        # 线性变换匹配 按参考图对齐明暗
        return (img - img.mean()) / (img.std() + 1e-6) * ref.std() + ref.mean()
    
    @staticmethod
    def directional_filter_spatial(img: np.ndarray, angle: float = 90, 
                                   kernel_size: int = 15, sigma: float = 3.0) -> np.ndarray:
        # 空域定向滤波 通过旋转高斯核实现指定方向的平滑
        if kernel_size % 2 == 0: kernel_size += 1               # 确保卷积核大小为奇数
        half = kernel_size // 2                                 # 生成网格坐标
        y, x = np.ogrid[-half:half+1, -half:half+1]
        theta = np.radians(angle)                               # 旋转坐标
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = x * cos_t - y * sin_t
        y_rot = x * sin_t + y * cos_t
        sigma_y = sigma / DIRECTED_FILTER_SPATIAL_ELONGATION    # 构造各向异性高斯
        kernel = np.exp(-(x_rot**2 / (2 * sigma**2) + y_rot**2 / (2 * sigma_y**2)))
        kernel = kernel / np.sum(kernel)                        # 归一化
        # 应用卷积 边界填充采用镜像
        filtered = cv2.filter2D(img, -1, kernel.astype(np.float32), borderType=cv2.BORDER_REFLECT)
        return filtered
    
    @staticmethod
    def directional_filter_frequency(img: np.ndarray, angle: float = 90,
                                     radius: int = 10) -> np.ndarray:
        # 频域定向滤波 通过旋转高通滤波器实现指定方向的滤波
        rows, cols = img.shape
        f = np.fft.fft2(img)                                    # 傅里叶变换
        fshift = np.fft.fftshift(f)
        mask = np.ones((rows, cols), dtype=np.float32)          # 创建掩膜
        crow, ccol = rows // 2, cols // 2
        axes = (cols // DIRECTED_FILTER_FREQUENCY_AXIS_TO_IMG, radius) # 滤波范围
        # OpenCV 的 ellipse 要求角度顺时针为正，但此处角度与噪声方向一致
        cv2.ellipse(mask, (ccol, crow), axes, angle, 0, 360, 0, -1)
        mask[crow, ccol] = 1.0                                  # 中心点为1
        fshift_filtered = fshift * mask                         # 应用掩膜
        f_ishift = np.fft.ifftshift(fshift_filtered)            # 逆变换
        img_filtered = np.fft.ifft2(f_ishift)
        img_filtered = np.real(img_filtered)
        # 由于浮点误差，可能有极小的负值或超出，可做裁剪
        img_filtered = np.clip(img_filtered, 0, 255)
        return img_filtered.astype(img.dtype)
    
    @staticmethod
    def set_border_black(img: np.ndarray, border: int | tuple[int, int, int, int] = 2) -> np.ndarray:
        if isinstance(border, int): top = bottom = left = right = border
        else: top, bottom, left, right = border
        h, w = img.shape[:2]
        top, bottom, left, right = max(0, top), max(0, bottom), max(0, left), max(0, right)
        top, bottom, left, right = min(h, top), min(h, bottom), min(w, left), min(w, right)

        out = img.copy()
        if top > 0: out[:top, :] = 0
        if bottom > 0: out[-bottom:, :] = 0
        if left > 0: out[:, :left] = 0
        if right > 0: out[:, -right:] = 0
        return out
    
    @staticmethod
    def fill_from_center_f(binary: np.ndarray, connectivity: int = 8) -> np.ndarray:
        # 从中心开始填充二值图 返回0-255 float32
        h, w = binary.shape
        cy, cx = h // 2, w // 2
        if binary[cy, cx] == 0:
            return np.zeros((h, w), dtype=np.float32)
        bi = (binary > 0).astype(np.uint8) * 255
        n, labels = cv2.connectedComponents(bi, connectivity=connectivity)
        center_label = labels[cy, cx]
        return (labels == center_label).astype(np.float32) * 255