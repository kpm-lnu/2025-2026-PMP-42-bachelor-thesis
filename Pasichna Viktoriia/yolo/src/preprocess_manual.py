import math
from typing import Dict
import numpy as np
import cv2

def ensure_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)

def to_float_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    b = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    r = image[:, :, 2].astype(np.float32)
    return 0.114 * b + 0.587 * g + 0.299 * r

def gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    gray_u8 = ensure_uint8(gray)
    return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)

def pad_reflect(image: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    if image.ndim == 2:
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="reflect")

def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    kernel = np.flipud(np.fliplr(kernel.astype(np.float32)))
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = pad_reflect(image, ph, pw)
    out = np.zeros_like(image, dtype=np.float32)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded[y:y + kh, x:x + kw]
            out[y, x] = float(np.sum(region * kernel))
    return out

def convolve_color(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return convolve2d(image, kernel)
    channels = [convolve2d(image[:, :, c], kernel) for c in range(image.shape[2])]
    return np.stack(channels, axis=-1)


def grayscale_manual(image: np.ndarray) -> np.ndarray:
    return ensure_uint8(to_float_gray(image))

def mean_filter_manual(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    return ensure_uint8(convolve_color(image, kernel))

def gaussian_kernel(kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)

def gaussian_blur_manual(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    kernel = gaussian_kernel(kernel_size, sigma)
    return ensure_uint8(convolve_color(image, kernel))

def median_filter_manual(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    pad = kernel_size // 2
    if image.ndim == 2:
        padded = pad_reflect(image, pad, pad)
        out = np.zeros_like(image, dtype=np.float32)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                out[y, x] = np.median(padded[y:y+kernel_size, x:x+kernel_size])
        return ensure_uint8(out)
    channels = []
    for c in range(image.shape[2]):
        channels.append(median_filter_manual(image[:, :, c], kernel_size))
    return np.stack(channels, axis=-1)

def sobel_manual(image: np.ndarray) -> np.ndarray:
    gray = to_float_gray(image)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = convolve2d(gray, kx)
    gy = convolve2d(gray, ky)
    mag = np.sqrt(gx**2 + gy**2)
    mag = (mag / (mag.max() + 1e-8)) * 255.0
    return gray_to_bgr(mag)

def roberts_cross_manual(image: np.ndarray) -> np.ndarray:
    gray = to_float_gray(image)
    kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    padded = np.pad(gray, ((0, 1), (0, 1)), mode="reflect")
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            region = padded[y:y+2, x:x+2]
            gx[y, x] = np.sum(region * kx)
            gy[y, x] = np.sum(region * ky)
    mag = np.sqrt(gx**2 + gy**2)
    mag = (mag / (mag.max() + 1e-8)) * 255.0
    return gray_to_bgr(mag)

def laplacian_manual(image: np.ndarray) -> np.ndarray:
    gray = to_float_gray(image)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    lap = convolve2d(gray, kernel)
    lap = np.abs(lap)
    lap = (lap / (lap.max() + 1e-8)) * 255.0
    return gray_to_bgr(lap)

def dog_manual(image: np.ndarray, sigma1: float = 1.0, sigma2: float = 2.0) -> np.ndarray:
    gray = to_float_gray(image)
    g1 = convolve2d(gray, gaussian_kernel(5, sigma1))
    g2 = convolve2d(gray, gaussian_kernel(9, sigma2))
    dog = np.abs(g1 - g2)
    dog = (dog / (dog.max() + 1e-8)) * 255.0
    return gray_to_bgr(dog)

def log_manual(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    gray = to_float_gray(image)
    blur = convolve2d(gray, gaussian_kernel(5, sigma))
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    log_img = np.abs(convolve2d(blur, kernel))
    log_img = (log_img / (log_img.max() + 1e-8)) * 255.0
    return gray_to_bgr(log_img)


def _gradient_and_direction(gray: np.ndarray):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = convolve2d(gray, kx)
    gy = convolve2d(gray, ky)
    magnitude = np.hypot(gx, gy)
    direction = np.arctan2(gy, gx)
    return magnitude, direction

def _non_max_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    H, W = magnitude.shape
    Z = np.zeros((H, W), dtype=np.float32)
    angle = direction * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            q = 255
            r = 255

            # Angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            # Angle 45
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            # Angle 90
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            # Angle 135
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                Z[i, j] = magnitude[i, j]
            else:
                Z[i, j] = 0
    return Z

def _double_threshold(img: np.ndarray, low_ratio: float = 0.05, high_ratio: float = 0.15):
    high = img.max() * high_ratio
    low = high * low_ratio / max(high_ratio, 1e-8)
    res = np.zeros_like(img, dtype=np.uint8)
    weak = np.uint8(75)
    strong = np.uint8(255)
    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img >= low) & (img < high))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res, weak, strong

def _hysteresis(img: np.ndarray, weak: int, strong: int = 255):
    H, W = img.shape
    out = img.copy()
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if out[i, j] == weak:
                if np.any(out[i - 1:i + 2, j - 1:j + 2] == strong):
                    out[i, j] = strong
                else:
                    out[i, j] = 0
    return out

def canny_manual(image: np.ndarray) -> np.ndarray:
    gray = to_float_gray(image)
    blur = convolve2d(gray, gaussian_kernel(5, 1.0))
    magnitude, direction = _gradient_and_direction(blur)
    suppressed = _non_max_suppression(magnitude, direction)
    thresholded, weak, strong = _double_threshold(suppressed, 0.05, 0.15)
    edges = _hysteresis(thresholded, weak, strong)
    return gray_to_bgr(edges)

def gaussian_canny_manual(image: np.ndarray) -> np.ndarray:
    blur = gaussian_blur_manual(image, 5, 1.0)
    return canny_manual(blur)

def gray_gaussian_manual(image: np.ndarray) -> np.ndarray:
    gray = grayscale_manual(image)
    blur = gaussian_blur_manual(gray, 5, 1.0)
    return gray_to_bgr(blur)

def gaussian_sobel_manual(image: np.ndarray) -> np.ndarray:
    blur = gaussian_blur_manual(image, 5, 1.0)
    return sobel_manual(blur)

def gaussian_laplacian_manual(image: np.ndarray) -> np.ndarray:
    blur = gaussian_blur_manual(image, 5, 1.0)
    return laplacian_manual(blur)

METHODS: Dict[str, callable] = {
    "original": lambda img: img,
    "gray": grayscale_manual,
    "mean": mean_filter_manual,
    "gaussian": gaussian_blur_manual,
    "median": median_filter_manual,
    "sobel": sobel_manual,
    "roberts": roberts_cross_manual,
    "laplacian": laplacian_manual,
    "dog": dog_manual,
    "log": log_manual,
    "canny": canny_manual,
    "gaussian_canny": gaussian_canny_manual,
    "gray_gaussian": gray_gaussian_manual,
    "gaussian_sobel": gaussian_sobel_manual,
    "gaussian_laplacian": gaussian_laplacian_manual,
}

def apply_method(image: np.ndarray, method: str) -> np.ndarray:
    if method not in METHODS:
        raise ValueError(f"Unknown preprocessing method: {method}")
    return METHODS[method](image)