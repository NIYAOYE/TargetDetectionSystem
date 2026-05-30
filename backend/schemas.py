from typing import List, Optional

from pydantic import BaseModel, Field


class FileInfo(BaseModel):
    name: str
    display_name: str
    kind: str
    size: int = 0
    modified_at: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None


class FilesResponse(BaseModel):
    images: List[FileInfo]
    detectors: List[FileInfo]
    classifiers: List[FileInfo]


class AppConfigResponse(BaseModel):
    patch_size: int
    overlap: int
    nms_threshold: float
    conf_threshold: float
    iou_threshold: float
    device: str
    default_rf_enabled: bool


class DetectRequest(BaseModel):
    image_name: str
    detector_name: str = "__demo__"
    classifier_name: Optional[str] = None
    patch_size: int = Field(default=640, ge=128, le=4096)
    use_classifier: bool = True
    conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    nms_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    device: str = "auto"


class ManualDetectionRequest(BaseModel):
    bbox: List[float]
    class_name: str = Field(default="manual_target", min_length=1, max_length=80)


class DetectionOut(BaseModel):
    id: int
    bbox: List[float]
    confidence: float
    class_id: int
    class_name: str
    source: str
    rf_result: int
    scatter_point_count: int
    scatter_mean_amplitude: float
    hu_moment_1: float
    hu_moment_2: float


class JobSummary(BaseModel):
    job_id: str
    status: str
    progress_current: int
    progress_total: int
    message: str
    error: Optional[str] = None
    image_name: str
    detector_name: str
    classifier_name: Optional[str] = None
    detection_count: int = 0
    created_at: float
    updated_at: float


class ExportResponse(BaseModel):
    output_dir: str
    image_url: str
    csv_url: str


class UploadResponse(BaseModel):
    filename: str
    category: str
    message: str
