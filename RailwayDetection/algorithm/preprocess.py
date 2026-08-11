import cv2 
import numpy as np
from typing import Tuple

DIRECTED_FILTER_SPATIAL_ELONGATION: float = 3.0
DIRECTED_FILTER_FREQUENCY_AXIS_TO_IMG: int = 4      # 频域掩膜（椭圆）的长轴长度设置为图像宽度几分之一

class Preprocess:

    @staticmethod
    def get_shift(img: np.ndarray, ref: np.ndarray) -> Tuple[float, float, float]:
        # 返回两张图的平移量
        shift, response = cv2.phaseCorrelate(img, ref)
        dx, dy = shift
        return dx, dy, response

    @staticmethod
    def align_trans(img: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, float]:
        # 按参考图对齐图像 平移变换
        dx, dy, response = Preprocess.get_shift(img, ref)
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype = np.float32)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0])), response
    
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