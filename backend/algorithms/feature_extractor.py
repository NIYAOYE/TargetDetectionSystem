from typing import Tuple
import math

import cv2
import numpy as np


DEFAULT_SCATTER_PEAK_LIMIT = 16


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
    min_area_threshold = 10

    if image_crop is None or image_crop.size == 0:
        return np.zeros(8, dtype=np.float32)

    if image_crop.ndim == 3:
        if image_crop.shape[2] >= 3:
            gray_image = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image_crop[:, :, 0]
    else:
        gray_image = image_crop

    # OTSU thresholding below requires 8-bit input; 16-bit SAR rasters would
    # otherwise throw and silently disable classification for the whole image.
    if gray_image.dtype != np.uint8:
        gray_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    ksize = 3 if min(gray_image.shape) < 32 else 5
    denoised_image = cv2.medianBlur(gray_image, ksize)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(denoised_image)

    _, binary_image = cv2.threshold(
        enhanced_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_open)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_close)
    binary_image = cv2.dilate(binary_image, kernel_open, iterations=1)

    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return np.zeros(8, dtype=np.float32)

    main_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main_contour)
    if area < min_area_threshold:
        return np.zeros(8, dtype=np.float32)

    perimeter = cv2.arcLength(main_contour, True)
    rect = cv2.minAreaRect(main_contour)
    (_, _), (width_raw, height_raw), _ = rect

    length = max(width_raw, height_raw)
    width = min(width_raw, height_raw)
    aspect_ratio = length / width if width > 0 else 0

    mask = np.zeros_like(gray_image)
    cv2.drawContours(mask, [main_contour], -1, 255, thickness=-1)
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

