from pathlib import Path
from typing import Iterable, List
import csv
import os

import cv2
import numpy as np

from backend.algorithms.feature_extractor import crop_detection_patch, segment_target_mask
from backend.core.base_detector import Detection

# Clean backdrop for the target cutout — matches the UI canvas (#0a0f16), BGR.
SEGMENT_BACKGROUND_BGR = (22, 15, 10)


def image_dimensions(path: Path) -> tuple[int, int] | None:
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio

            with rasterio.open(path) as src:
                return int(src.width), int(src.height)
        except Exception:
            return None

    # Read only the image header instead of decoding the full image — SAR
    # rasters can be hundreds of MB and this runs for every file on /api/files.
    try:
        from PIL import Image

        with Image.open(path) as img:
            width, height = img.size
        return int(width), int(height)
    except Exception:
        return None


def list_files(directory: Path, extensions: set[str], include_dimensions: bool = False) -> list[dict]:
    files = []
    if not directory.exists():
        return files

    for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        stat = path.stat()
        item = {
            "name": path.name,
            "display_name": path.name,
            "kind": path.suffix.lower().lstrip("."),
            "size": stat.st_size,
            "modified_at": stat.st_mtime,
        }
        if include_dimensions:
            dimensions = image_dimensions(path)
            if dimensions is not None:
                item["width"], item["height"] = dimensions
        files.append(item)
    return files


def resolve_storage_file(directory: Path, name: str, extensions: set[str]) -> Path:
    if not name or Path(name).name != name:
        raise FileNotFoundError("Invalid file name")

    candidate = (directory / name).resolve()
    base = directory.resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise FileNotFoundError("Invalid file path") from exc

    if candidate.suffix.lower() not in extensions or not candidate.is_file():
        raise FileNotFoundError(f"File not found: {name}")
    return candidate


def load_image_array(image_path: Path) -> np.ndarray:
    if image_path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio

            with rasterio.open(image_path) as src:
                img = src.read()
                if img.shape[0] == 1:
                    return img[0]
                return np.transpose(img, (1, 2, 0))
        except ImportError:
            pass

    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path.name}")
    return img


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    image = normalize_to_uint8(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)


def to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    image = normalize_to_uint8(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image[:, :, :3].copy()


def preview_png_bytes(image_path: Path, max_size: int = 2048) -> tuple[bytes, dict[str, int]]:
    image = load_image_array(image_path)
    rgb = to_rgb_uint8(image)
    full_height, full_width = rgb.shape[:2]
    preview_width = full_width
    preview_height = full_height

    if max_size > 0:
        largest_side = max(full_width, full_height)
        if largest_side > max_size:
            scale = max_size / largest_side
            preview_width = max(1, int(round(full_width * scale)))
            preview_height = max(1, int(round(full_height * scale)))
            rgb = cv2.resize(rgb, (preview_width, preview_height), interpolation=cv2.INTER_AREA)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("Failed to encode preview image")
    return buffer.tobytes(), {
        "width": int(full_width),
        "height": int(full_height),
        "preview_width": int(preview_width),
        "preview_height": int(preview_height),
    }


def segment_target_on_background(
    image_path: Path,
    bbox: tuple[float, float, float, float],
    background_bgr: tuple[int, int, int] = SEGMENT_BACKGROUND_BGR,
    feather: float = 1.5,
) -> bytes:
    """Cut the detected target out of its crop and composite it onto a clean
    solid background, returning PNG bytes.

    The target mask comes from the same SAR segmentation used for RF features.
    Edges are feathered so the cutout doesn't look pasted. If segmentation finds
    nothing, the raw crop is shown rather than a blank tile.
    """
    image = load_image_array(image_path)
    crop = crop_detection_patch(image, bbox)
    if crop.size == 0:
        raise ValueError("Detection bbox falls outside the image")

    crop_bgr = to_bgr_uint8(crop)
    height, width = crop_bgr.shape[:2]

    mask = segment_target_mask(crop)
    if mask is None:
        alpha = np.ones((height, width), dtype=np.float32)
    else:
        alpha = mask.astype(np.float32) / 255.0
        if feather > 0:
            alpha = cv2.GaussianBlur(alpha, (0, 0), feather)

    alpha = alpha[:, :, np.newaxis]
    background = np.empty_like(crop_bgr)
    background[:] = background_bgr
    composite = (crop_bgr.astype(np.float32) * alpha + background.astype(np.float32) * (1.0 - alpha))
    composite = composite.astype(np.uint8)

    ok, buffer = cv2.imencode(".png", composite)
    if not ok:
        raise ValueError("Failed to encode segment image")
    return buffer.tobytes()


def detection_to_dict(index: int, detection: Detection) -> dict:
    return {
        "id": index,
        "bbox": [float(value) for value in detection.bbox],
        "confidence": float(detection.confidence),
        "class_id": int(detection.class_id),
        "class_name": detection.class_name,
        "source": getattr(detection, "source", "auto"),
        "rf_result": int(getattr(detection, "rf_result", -1)),
        "scatter_point_count": int(getattr(detection, "scatter_point_count", 0)),
        "scatter_mean_amplitude": float(getattr(detection, "scatter_mean_amplitude", 0.0)),
        "hu_moment_1": float(getattr(detection, "hu_moment_1", 0.0)),
        "hu_moment_2": float(getattr(detection, "hu_moment_2", 0.0)),
    }


def detections_to_dicts(detections: Iterable[Detection]) -> List[dict]:
    return [detection_to_dict(index, detection) for index, detection in enumerate(detections)]


def detection_color_bgr(detection: Detection) -> tuple[int, int, int]:
    source = getattr(detection, "source", "auto")
    rf_result = getattr(detection, "rf_result", -1)
    if source == "manual":
        return (255, 150, 0)
    if rf_result == 1:
        return (0, 255, 0)
    if rf_result == 0:
        return (0, 0, 255)
    return (0, 255, 255)


def compose_detection_overlay(image_path: Path, detections: list[Detection]) -> np.ndarray:
    image = to_bgr_uint8(load_image_array(image_path))

    for detection in detections:
        x, y, w, h = [int(round(value)) for value in detection.bbox]
        color = detection_color_bgr(detection)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        text_y = max(15, y - 5)
        cv2.putText(image, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image


def rf_result_text(detection: Detection) -> str:
    if getattr(detection, "source", "auto") == "manual":
        return "手动添加"
    rf_result = getattr(detection, "rf_result", -1)
    if rf_result == 1:
        return "真目标"
    if rf_result == 0:
        return "虚警"
    return "未分类"


def write_detection_results_csv(save_path: Path, detections: list[Detection]) -> None:
    with save_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "ID",
                "类别",
                "置信度",
                "X",
                "Y",
                "宽度",
                "高度",
                "来源",
                "散射点数",
                "平均散射强度",
                "Hu矩1",
                "Hu矩2",
                "RF结果",
            ]
        )
        for index, detection in enumerate(detections):
            x, y, w, h = detection.bbox
            writer.writerow(
                [
                    index,
                    detection.class_name,
                    f"{detection.confidence:.3f}",
                    int(x),
                    int(y),
                    int(w),
                    int(h),
                    getattr(detection, "source", "auto"),
                    int(getattr(detection, "scatter_point_count", 0)),
                    f"{float(getattr(detection, 'scatter_mean_amplitude', 0.0)):.6f}",
                    f"{float(getattr(detection, 'hu_moment_1', 0.0)):.6f}",
                    f"{float(getattr(detection, 'hu_moment_2', 0.0)):.6f}",
                    rf_result_text(detection),
                ]
            )


def export_bundle(
    output_root: Path,
    image_path: Path,
    image_name: str,
    detections: list[Detection],
) -> tuple[Path, Path, Path]:
    image_stem = Path(image_name).stem or "export"
    bundle_dir = output_root / image_stem
    bundle_dir.mkdir(parents=True, exist_ok=True)

    image_save_path = bundle_dir / f"{image_stem}_detected.png"
    csv_save_path = bundle_dir / f"{image_stem}_results.csv"

    overlay = compose_detection_overlay(image_path, detections)
    if not cv2.imwrite(str(image_save_path), overlay):
        raise ValueError("Failed to write overlay image")
    write_detection_results_csv(csv_save_path, detections)
    return bundle_dir, image_save_path, csv_save_path


def output_url(output_root: Path, path: Path) -> str:
    relative = path.resolve().relative_to(output_root.resolve())
    return "/outputs/" + "/".join(part for part in relative.parts)
