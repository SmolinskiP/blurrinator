from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from registry.models import ModelEntry


@dataclass
class RawFace:
    x: int
    y: int
    w: int
    h: int
    score: float
    landmarks: np.ndarray  # shape (5, 2): right eye, left eye, nose, right mouth, left mouth
    raw_row: np.ndarray   # original 15-element YuNet row, for SFace alignCrop


class FaceDetector:
    """Wraps cv2.FaceDetectorYN. Recreate per process — the underlying ONNX
    session is not thread-safe, so workers should hold their own instance."""

    BACKEND_ID = cv2.dnn.DNN_BACKEND_CUDA if hasattr(cv2.dnn, "DNN_BACKEND_CUDA") else 0
    TARGET_ID = cv2.dnn.DNN_TARGET_CUDA if hasattr(cv2.dnn, "DNN_TARGET_CUDA") else 0

    def __init__(
        self,
        weights_path: str | Path,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        self.weights_path = str(weights_path)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._size: tuple[int, int] | None = None
        self._model = cv2.FaceDetectorYN.create(
            model=self.weights_path,
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
            backend_id=self.BACKEND_ID,
            target_id=self.TARGET_ID,
        )

    @classmethod
    def from_registry(cls, **kwargs) -> "FaceDetector":
        entry = (
            ModelEntry.objects
            .filter(family=ModelEntry.Family.FACE_DETECTION, name="YuNet")
            .order_by("-registered_at")
            .first()
        )
        if not entry or not entry.file_path or not Path(entry.file_path).exists():
            raise RuntimeError(
                "YuNet weights not registered. Run: manage.py fetch_models"
            )
        return cls(entry.file_path, **kwargs)

    def detect(self, frame: np.ndarray) -> list[RawFace]:
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self._model.setInputSize((w, h))
            self._size = (w, h)

        _, raw = self._model.detect(frame)
        if raw is None:
            return []

        faces: list[RawFace] = []
        for row in raw:
            x, y, fw, fh = (int(round(v)) for v in row[:4])
            x = max(0, x)
            y = max(0, y)
            fw = max(1, min(w - x, fw))
            fh = max(1, min(h - y, fh))
            landmarks = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
            score = float(row[14])
            faces.append(RawFace(x, y, fw, fh, score, landmarks, row.copy()))
        return faces

    def detect_multiscale(
        self, frame: np.ndarray, scales: tuple[float, ...] = (1.0, 0.5, 0.25)
    ) -> list[RawFace]:
        """Run detection at multiple frame scales and merge with NMS. The
        downscaled passes catch close-up faces that exceed YuNet's largest
        anchor box at native resolution; the native pass keeps small/distant
        faces intact."""
        h, w = frame.shape[:2]
        all_faces: list[RawFace] = []
        for scale in scales:
            if abs(scale - 1.0) < 1e-6:
                all_faces.extend(self.detect(frame))
                continue
            sw = max(64, int(round(w * scale)))
            sh = max(64, int(round(h * scale)))
            small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
            for face in self.detect(small):
                inv = 1.0 / scale
                scaled_row = face.raw_row.copy()
                scaled_row[:14] = scaled_row[:14] * inv
                all_faces.append(
                    RawFace(
                        x=max(0, min(w - 1, int(round(face.x * inv)))),
                        y=max(0, min(h - 1, int(round(face.y * inv)))),
                        w=max(1, min(w - int(round(face.x * inv)), int(round(face.w * inv)))),
                        h=max(1, min(h - int(round(face.y * inv)), int(round(face.h * inv)))),
                        score=face.score,
                        landmarks=face.landmarks * inv,
                        raw_row=scaled_row,
                    )
                )
        return _nms_faces(all_faces, iou_threshold=0.4)


def _nms_faces(faces: list[RawFace], iou_threshold: float = 0.4) -> list[RawFace]:
    sorted_faces = sorted(faces, key=lambda f: f.score, reverse=True)
    kept: list[RawFace] = []
    for f in sorted_faces:
        if any(_face_iou(f, k) >= iou_threshold for k in kept):
            continue
        kept.append(f)
    return kept


def _face_iou(a: RawFace, b: RawFace) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / max(1, union)
