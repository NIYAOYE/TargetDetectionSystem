from typing import Optional, Tuple
import math

import cv2
import numpy as np


DEFAULT_SCATTER_PEAK_LIMIT = 16
MIN_TARGET_AREA = 10


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    if image.ndim == 2:
        return image

    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0]
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _to_gray_uint8(image: np.ndarray) -> Optional[np.ndarray]:
    """Single-channel uint8 view of any crop — OTSU/CLAHE below need 8-bit."""
    if image is None or image.size == 0:
        return None

    gray = _to_grayscale(image)
    if gray.size == 0:
        return None
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def _segment_core(
    gray_uint8: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Classical SAR target segmentation: denoise → enhance → OTSU → morphology
    → largest external contour. Returns (denoised, filled_mask, main_contour);
    mask/contour are None when nothing is found. Shared by feature extraction
    and the target-cutout renderer so both stay in sync.
    """
    ksize = 3 if min(gray_uint8.shape) < 32 else 5
    denoised = cv2.medianBlur(gray_uint8, ksize)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    binary = cv2.dilate(binary, kernel_open, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return denoised, None, None

    main_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray_uint8)
    cv2.drawContours(mask, [main_contour], -1, 255, thickness=-1)
    return denoised, mask, main_contour


def segment_target_mask(image_crop: np.ndarray) -> Optional[np.ndarray]:
    """Binary (0/255) uint8 mask of the dominant target in a crop, or None.

    Same segmentation the RF feature extractor uses, exposed for cutting the
    target out of its background for display.
    """
    gray = _to_gray_uint8(image_crop)
    if gray is None or float(gray.std()) < 1.0:
        # A flat patch has no target/background contrast to segment.
        return None

    _, mask, main_contour = _segment_core(gray)
    if main_contour is None or cv2.contourArea(main_contour) < MIN_TARGET_AREA:
        return None
    return mask


def crop_detection_patch(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
) -> np.ndarray:
    if image is None or image.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    image_height, image_width = image.shape[:2]
    x, y, w, h = bbox

    x1 = max(0, int(np.floor(x)))
    y1 = max(0, int(np.floor(y)))
    x2 = min(image_width, int(np.ceil(x + w)))
    y2 = min(image_height, int(np.ceil(y + h)))

    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0), dtype=image.dtype)

    return image[y1:y2, x1:x2].copy()


def PeakExtractX(imgdata: np.ndarray, max_num: int) -> Tuple[int, float]:
    if imgdata is None or imgdata.size == 0 or max_num <= 0:
        return 0, 0.0

    imgdata = _to_grayscale(imgdata).astype(np.float32, copy=False)
    if imgdata.ndim != 2 or min(imgdata.shape[:2]) < 3:
        return 0, 0.0

    meanvalue = 0.55 * float(np.max(imgdata))
    center = imgdata[1:-1, 1:-1]
    is_max = center > meanvalue

    neighbors = [
        imgdata[0:-2, 0:-2],
        imgdata[0:-2, 1:-1],
        imgdata[0:-2, 2:],
        imgdata[1:-1, 0:-2],
        imgdata[1:-1, 2:],
        imgdata[2:, 0:-2],
        imgdata[2:, 1:-1],
        imgdata[2:, 2:],
    ]

    for neighbor in neighbors:
        is_max &= center >= neighbor

    peak_mask = np.zeros_like(imgdata, dtype=bool)
    peak_mask[1:-1, 1:-1] = is_max

    y_coords, x_coords = np.nonzero(peak_mask)
    if len(x_coords) == 0:
        return 0, 0.0

    intensities = imgdata[y_coords, x_coords]
    peaks = np.column_stack((x_coords, y_coords, intensities))
    sort_idx = np.argsort(peaks[:, 2])[::-1]
    peaks = peaks[sort_idx]

    if len(peaks) > max_num:
        peaks = peaks[:max_num]

    # Mean of the selected peaks' raw intensities. (Normalising to sum=1 first
    # would make this always equal 1/count — a useless duplicate of the count.)
    return int(peaks.shape[0]), float(np.mean(peaks[:, 2]))


def computeHuMoments(img_cut: np.ndarray) -> Tuple[float, float]:
    if img_cut is None or img_cut.size == 0:
        return 0.0, 0.0

    gray_image = _to_grayscale(img_cut).astype(np.float32, copy=False)
    if gray_image.ndim != 2 or gray_image.size == 0:
        return 0.0, 0.0

    moments = cv2.moments(gray_image)
    hu = cv2.HuMoments(moments)

    h1 = float(hu[0][0])
    h2 = float(hu[1][0])

    if h1 != 0:
        h1 = -math.copysign(1.0, h1) * math.log10(abs(h1))
    if h2 != 0:
        h2 = -math.copysign(1.0, h2) * math.log10(abs(h2))

    return float(h1), float(h2)


def extract_scatter_features_from_patch(
    image_patch: np.ndarray,
    max_num: int = DEFAULT_SCATTER_PEAK_LIMIT,
) -> dict[str, float | int]:
    peak_count, peak_mean_amplitude = PeakExtractX(image_patch, max_num=max_num)
    hu_moment_1, hu_moment_2 = computeHuMoments(image_patch)

    return {
        "scatter_point_count": int(peak_count),
        "scatter_mean_amplitude": float(peak_mean_amplitude),
        "hu_moment_1": float(hu_moment_1),
        "hu_moment_2": float(hu_moment_2),
    }


def extract_scatter_features_from_bbox(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    max_num: int = DEFAULT_SCATTER_PEAK_LIMIT,
) -> dict[str, float | int]:
    patch = crop_detection_patch(image, bbox)
    return extract_scatter_features_from_patch(patch, max_num=max_num)


def extract_sar_features(image_crop: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
    del bbox
    gray_image = _to_gray_uint8(image_crop)
    if gray_image is None:
        return np.zeros(8, dtype=np.float32)

    denoised_image, mask, main_contour = _segment_core(gray_image)
    if main_contour is None:
        return np.zeros(8, dtype=np.float32)

    area = cv2.contourArea(main_contour)
    if area < MIN_TARGET_AREA:
        return np.zeros(8, dtype=np.float32)

    perimeter = cv2.arcLength(main_contour, True)
    rect = cv2.minAreaRect(main_contour)
    (_, _), (width_raw, height_raw), _ = rect

    length = max(width_raw, height_raw)
    width = min(width_raw, height_raw)
    aspect_ratio = length / width if width > 0 else 0

    pixel_values = denoised_image[mask == 255]

    if pixel_values.size > 0:
        mean_intensity = np.mean(pixel_values)
        max_intensity = np.max(pixel_values)
        std_intensity = np.std(pixel_values)
    else:
        mean_intensity = 0
        max_intensity = 0
        std_intensity = 0

    return np.array(
        [
            float(area),
            float(perimeter),
            float(length),
            float(width),
            float(aspect_ratio),
            float(mean_intensity),
            float(max_intensity),
            float(std_intensity),
        ],
        dtype=np.float32,
    )

