from typing import Iterator, Tuple
import math

import cv2
import numpy as np

try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class ImageSlicer:
    def __init__(
        self,
        image_path: str,
        patch_size: int = 640,
        overlap: int = 64,
        padding: int = 0,
    ):
        if patch_size <= overlap:
            raise ValueError("patch_size must be greater than overlap")

        self.image_path = image_path
        self.patch_size = patch_size
        self.overlap = overlap
        self.padding = padding

        self._load_metadata()

        self.stride = self.patch_size - self.overlap
        self.n_patches_h = max(1, math.ceil(max(self.height - self.overlap, 1) / self.stride))
        self.n_patches_w = max(1, math.ceil(max(self.width - self.overlap, 1) / self.stride))
        self.total_patches = self.n_patches_h * self.n_patches_w

    def _load_metadata(self):
        if HAS_RASTERIO and self.image_path.lower().endswith((".tif", ".tiff")):
            with rasterio.open(self.image_path) as src:
                self.height = src.height
                self.width = src.width
                self.channels = src.count
                self.dtype = src.dtypes[0]
            return

        img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {self.image_path}")
        self.height, self.width = img.shape[:2]
        self.channels = img.shape[2] if len(img.shape) == 3 else 1
        self.dtype = img.dtype
        self._cached_image = img

    def _load_full_image(self) -> np.ndarray:
        if hasattr(self, "_cached_image"):
            return self._cached_image

        if HAS_RASTERIO and self.image_path.lower().endswith((".tif", ".tiff")):
            with rasterio.open(self.image_path) as src:
                img = src.read()
                if img.shape[0] == 1:
                    return img[0]
                return np.transpose(img, (1, 2, 0))

        return cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)

    def _convert_to_3channel(self, patch: np.ndarray) -> np.ndarray:
        if patch.ndim == 2:
            return cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        if patch.shape[2] == 1:
            return cv2.cvtColor(patch[:, :, 0], cv2.COLOR_GRAY2BGR)
        if patch.shape[2] == 3:
            return patch
        return cv2.cvtColor(patch[:, :, 0], cv2.COLOR_GRAY2BGR)

    def get_patch(self, row: int, col: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        y_start = min(row * self.stride, max(self.height - self.patch_size, 0))
        x_start = min(col * self.stride, max(self.width - self.patch_size, 0))
        y_end = min(y_start + self.patch_size, self.height)
        x_end = min(x_start + self.patch_size, self.width)

        y_start_pad = max(0, y_start - self.padding)
        x_start_pad = max(0, x_start - self.padding)
        y_end_pad = min(self.height, y_end + self.padding)
        x_end_pad = min(self.width, x_end + self.padding)

        full_img = self._load_full_image()
        patch = full_img[y_start_pad:y_end_pad, x_start_pad:x_end_pad]

        if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
            if patch.ndim == 2:
                padded = np.zeros((self.patch_size, self.patch_size), dtype=patch.dtype)
                padded[: patch.shape[0], : patch.shape[1]] = patch
            else:
                padded = np.zeros(
                    (self.patch_size, self.patch_size, patch.shape[2]),
                    dtype=patch.dtype,
                )
                padded[: patch.shape[0], : patch.shape[1], :] = patch
            patch = padded

        return self._convert_to_3channel(patch), (y_start, y_end, x_start, x_end)

    def __iter__(self) -> Iterator[Tuple[int, int, np.ndarray, Tuple[int, int, int, int]]]:
        for row in range(self.n_patches_h):
            for col in range(self.n_patches_w):
                patch, bbox = self.get_patch(row, col)
                yield row, col, patch, bbox

    def __len__(self) -> int:
        return self.total_patches


class CoordinateMapper:
    def __init__(self, patch_size: int, overlap: int, padding: int = 0):
        self.patch_size = patch_size
        self.overlap = overlap
        self.padding = padding
        self.stride = patch_size - overlap

    def patch_to_global(
        self,
        patch_bbox: Tuple[float, float, float, float],
        patch_row: int,
        patch_col: int,
    ) -> Tuple[float, float, float, float]:
        x_patch, y_patch, w, h = patch_bbox
        x_global = x_patch + patch_col * self.stride
        y_global = y_patch + patch_row * self.stride
        return (x_global, y_global, w, h)

    def global_to_patch(
        self,
        global_bbox: Tuple[float, float, float, float],
        patch_row: int,
        patch_col: int,
    ) -> Tuple[float, float, float, float]:
        x_global, y_global, w, h = global_bbox
        x_patch = x_global - patch_col * self.stride
        y_patch = y_global - patch_row * self.stride
        return (x_patch, y_patch, w, h)

