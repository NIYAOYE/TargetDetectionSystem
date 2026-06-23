from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from time import time
from typing import Optional
from uuid import uuid4

import cv2

from backend.algorithms.feature_extractor import extract_scatter_features_from_bbox
from backend.config import DEFAULT_CONFIG
from backend.core.base_detector import Detection
from backend.image_io import detections_to_dicts
from backend.pipeline import DetectionPipeline, ModelRegistry
from backend.schemas import DetectRequest, JobSummary


@dataclass
class JobRecord:
    job_id: str
    request: DetectRequest
    image_path: Path
    detector_path: Optional[Path]
    classifier_path: Optional[Path]
    status: str = "queued"
    progress_current: int = 0
    progress_total: int = 100
    message: str = "Queued"
    error: Optional[str] = None
    detections: list[Detection] = field(default_factory=list)
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)


class JobStore:
    def __init__(self, overlap: int, max_jobs: int = 50):
        self._lock = RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pipeline = DetectionPipeline(ModelRegistry())
        self._overlap = overlap
        self._max_jobs = max_jobs

    def _evict_finished_locked(self) -> None:
        """Drop the oldest finished jobs so the in-memory store stays bounded."""
        excess = len(self._jobs) - self._max_jobs
        if excess <= 0:
            return
        finished = sorted(
            (r for r in self._jobs.values() if r.status in {"completed", "error"}),
            key=lambda r: r.updated_at,
        )
        for record in finished[:excess]:
            self._jobs.pop(record.job_id, None)

    def create_job(
        self,
        request: DetectRequest,
        image_path: Path,
        detector_path: Optional[Path],
        classifier_path: Optional[Path],
    ) -> JobSummary:
        job_id = uuid4().hex
        record = JobRecord(
            job_id=job_id,
            request=request,
            image_path=image_path,
            detector_path=detector_path,
            classifier_path=classifier_path,
        )
        with self._lock:
            self._jobs[job_id] = record
            self._evict_finished_locked()
        self._executor.submit(self._run_job, job_id)
        return self.summary(job_id)

    def _run_job(self, job_id: str) -> None:
        try:
            self._update(job_id, status="running", progress_current=0, message="Starting detection...")
            record = self.get_record(job_id)

            def progress(current: int, total: int, message: str) -> None:
                self._update(
                    job_id,
                    progress_current=int(current),
                    progress_total=int(total),
                    message=message,
                )

            detections = self._pipeline.run(
                image_path=record.image_path,
                detector_path=record.detector_path,
                classifier_path=record.classifier_path,
                patch_size=record.request.patch_size,
                overlap=self._overlap,
                conf_threshold=record.request.conf_threshold,
                iou_threshold=record.request.iou_threshold,
                nms_threshold=record.request.nms_threshold,
                use_classifier=record.request.use_classifier and record.classifier_path is not None,
                device=record.request.device,
                progress=progress,
            )
            with self._lock:
                current = self._jobs[job_id]
                current.detections = detections
                current.status = "completed"
                current.progress_current = 100
                current.progress_total = 100
                current.message = f"Detection complete, found {len(detections)} targets"
                current.updated_at = time()
        except Exception as exc:
            self._update(
                job_id,
                status="error",
                error=str(exc),
                message="Detection failed",
                progress_current=0,
                progress_total=100,
            )

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            record = self._jobs[job_id]
            for key, value in changes.items():
                setattr(record, key, value)
            record.updated_at = time()

    def get_record(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def summary(self, job_id: str) -> JobSummary:
        record = self.get_record(job_id)
        with self._lock:
            return JobSummary(
                job_id=record.job_id,
                status=record.status,
                progress_current=record.progress_current,
                progress_total=record.progress_total,
                message=record.message,
                error=record.error,
                image_name=record.request.image_name,
                detector_name=record.request.detector_name,
                classifier_name=record.request.classifier_name,
                detection_count=len(record.detections),
                created_at=record.created_at,
                updated_at=record.updated_at,
            )

    def detection_dicts(self, job_id: str) -> list[dict]:
        record = self.get_record(job_id)
        with self._lock:
            return detections_to_dicts(record.detections)

    def add_manual_detection(self, job_id: str, bbox: list[float], class_name: str) -> list[dict]:
        record = self.get_record(job_id)
        if len(bbox) != 4:
            raise ValueError("bbox must contain x, y, width, height")

        detection = Detection(
            bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            confidence=1.0,
            class_id=-1,
            class_name=class_name.strip() or "manual_target",
            source="manual",
        )

        full_image = cv2.imread(str(record.image_path), cv2.IMREAD_UNCHANGED)
        if full_image is not None:
            scatter_features = extract_scatter_features_from_bbox(full_image, detection.bbox)
            detection.scatter_point_count = int(scatter_features["scatter_point_count"])
            detection.scatter_mean_amplitude = float(scatter_features["scatter_mean_amplitude"])
            detection.hu_moment_1 = float(scatter_features["hu_moment_1"])
            detection.hu_moment_2 = float(scatter_features["hu_moment_2"])

        with self._lock:
            record.detections.append(detection)
            record.updated_at = time()
            return detections_to_dicts(record.detections)

    def delete_detection(self, job_id: str, index: int) -> list[dict]:
        record = self.get_record(job_id)
        with self._lock:
            if index < 0 or index >= len(record.detections):
                raise IndexError(index)
            record.detections.pop(index)
            record.updated_at = time()
            return detections_to_dicts(record.detections)

