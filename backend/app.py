from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import (
    CLASSIFIER_EXTENSIONS,
    CLASSIFIER_MODEL_DIR,
    DEFAULT_CONFIG,
    DEMO_DETECTOR_NAME,
    DETECTOR_EXTENSIONS,
    DETECTOR_MODEL_DIR,
    IMAGE_DIR,
    IMAGE_EXTENSIONS,
    OUTPUTS_DIR,
    STATIC_DIR,
    ensure_storage_dirs,
    load_config,
)
from backend.image_io import (
    export_bundle,
    list_files,
    output_url,
    preview_png_bytes,
    resolve_storage_file,
)
from backend.job_store import JobStore
from backend.schemas import (
    AppConfigResponse,
    DetectRequest,
    ExportResponse,
    FilesResponse,
    ManualDetectionRequest,
    UploadResponse,
)

ensure_storage_dirs()
app_config = load_config()
overlap = int(app_config.get("image_processing", {}).get("overlap", DEFAULT_CONFIG["image_processing"]["overlap"]))
jobs = JobStore(overlap=overlap)

app = FastAPI(title="SAR Target Detection Browser Server")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config", response_model=AppConfigResponse)
def get_config():
    yolo = app_config.get("yolo", {})
    image_processing = app_config.get("image_processing", {})
    ui = app_config.get("ui", {})
    random_forest = app_config.get("random_forest", {})
    return AppConfigResponse(
        patch_size=int(image_processing.get("patch_size", 640)),
        overlap=int(image_processing.get("overlap", 64)),
        nms_threshold=float(image_processing.get("nms_threshold", 0.45)),
        conf_threshold=float(yolo.get("conf_threshold", 0.25)),
        iou_threshold=float(yolo.get("iou_threshold", 0.45)),
        device=str(yolo.get("device", "auto")),
        default_rf_enabled=bool(random_forest.get("enabled", True) and ui.get("default_rf_enabled", True)),
    )


@app.get("/api/files", response_model=FilesResponse)
def get_files():
    detectors = [
        {
            "name": DEMO_DETECTOR_NAME,
            "display_name": "demo-synthetic",
            "kind": "demo",
            "size": 0,
            "modified_at": None,
        }
    ]
    detectors.extend(list_files(DETECTOR_MODEL_DIR, DETECTOR_EXTENSIONS))
    return FilesResponse(
        images=list_files(IMAGE_DIR, IMAGE_EXTENSIONS, include_dimensions=True),
        detectors=detectors,
        classifiers=list_files(CLASSIFIER_MODEL_DIR, CLASSIFIER_EXTENSIONS),
    )


@app.get("/api/images/{image_name}/preview.png")
def get_image_preview(image_name: str):
    try:
        image_path = resolve_storage_file(IMAGE_DIR, image_name, IMAGE_EXTENSIONS)
        content, metadata = preview_png_bytes(image_path)
        return Response(
            content=content,
            media_type="image/png",
            headers={
                "X-Image-Width": str(metadata["width"]),
                "X-Image-Height": str(metadata["height"]),
                "X-Preview-Width": str(metadata["preview_width"]),
                "X-Preview-Height": str(metadata["preview_height"]),
                "Cache-Control": "no-store",
            },
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/jobs/detect")
def create_detection_job(request: DetectRequest):
    try:
        image_path = resolve_storage_file(IMAGE_DIR, request.image_name, IMAGE_EXTENSIONS)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    detector_path = None
    if request.detector_name != DEMO_DETECTOR_NAME:
        try:
            detector_path = resolve_storage_file(DETECTOR_MODEL_DIR, request.detector_name, DETECTOR_EXTENSIONS)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    classifier_path = None
    if request.classifier_name:
        try:
            classifier_path = resolve_storage_file(CLASSIFIER_MODEL_DIR, request.classifier_name, CLASSIFIER_EXTENSIONS)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    if request.patch_size <= overlap:
        raise HTTPException(status_code=400, detail="patch_size must be greater than overlap")

    return jobs.create_job(request, image_path, detector_path, classifier_path)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return jobs.summary(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@app.get("/api/jobs/{job_id}/detections")
def get_detections(job_id: str):
    try:
        return jobs.detection_dicts(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@app.post("/api/jobs/{job_id}/detections")
def add_manual_detection(job_id: str, request: ManualDetectionRequest):
    try:
        return jobs.add_manual_detection(job_id, request.bbox, request.class_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/jobs/{job_id}/detections/{index}")
def delete_detection(job_id: str, index: int):
    try:
        return jobs.delete_detection(job_id, index)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    except IndexError as exc:
        raise HTTPException(status_code=404, detail="Detection not found") from exc


@app.post("/api/jobs/{job_id}/export", response_model=ExportResponse)
def export_results(job_id: str):
    try:
        record = jobs.get_record(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    if not record.detections:
        raise HTTPException(status_code=400, detail="No detections to export")

    try:
        output_dir, image_path, csv_path = export_bundle(
            OUTPUTS_DIR,
            record.image_path,
            record.request.image_name,
            record.detections,
        )
        return ExportResponse(
            output_dir=str(output_dir),
            image_url=output_url(OUTPUTS_DIR, image_path),
            csv_url=output_url(OUTPUTS_DIR, csv_path),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# File management — upload & delete
# ---------------------------------------------------------------------------

CATEGORY_MAP = {
    "images": (IMAGE_DIR, IMAGE_EXTENSIONS),
    "detectors": (DETECTOR_MODEL_DIR, DETECTOR_EXTENSIONS),
    "classifiers": (CLASSIFIER_MODEL_DIR, CLASSIFIER_EXTENSIONS),
}


def _validate_category(category: str) -> tuple[Path, set[str]]:
    entry = CATEGORY_MAP.get(category)
    if entry is None:
        raise HTTPException(status_code=400, detail=f"Unknown category: {category}")
    return entry


def _validate_extension(filename: str, extensions: set[str]) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in extensions:
        allowed = ", ".join(sorted(extensions))
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed}")
    return suffix


def _sanitize_filename(filename: str) -> str:
    """Keep only safe characters — reject path separators and dangerous chars."""
    name = Path(filename).name
    if name != filename or not name:
        raise HTTPException(status_code=400, detail="Invalid file name")
    return name


@app.post("/api/files/upload/{category}", response_model=UploadResponse)
async def upload_file(category: str, file: UploadFile = File(...)):
    directory, extensions = _validate_category(category)
    safe_name = _sanitize_filename(file.filename)
    _validate_extension(safe_name, extensions)

    dest = (directory / safe_name).resolve()
    try:
        dest.relative_to(directory.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file path")

    try:
        content = await file.read()
        dest.write_bytes(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}") from exc

    return UploadResponse(filename=safe_name, category=category, message="Uploaded successfully")


@app.delete("/api/files/{category}/{filename}", response_model=UploadResponse)
def delete_file(category: str, filename: str):
    directory, extensions = _validate_category(category)
    safe_name = _sanitize_filename(filename)

    try:
        file_path = resolve_storage_file(directory, safe_name, extensions)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if category == "detectors" and safe_name == DEMO_DETECTOR_NAME:
        raise HTTPException(status_code=400, detail="Cannot delete the built-in demo detector")

    try:
        file_path.unlink()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {exc}") from exc

    return UploadResponse(filename=safe_name, category=category, message="Deleted successfully")
