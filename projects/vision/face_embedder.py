from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from registry.models import ModelEntry

from .face_detector import RawFace


class FaceEmbedder:
    """Wraps cv2.FaceRecognizerSF. Produces a 128-D L2-normalised vector per face."""

    MODEL_NAME = "SFace"
    MODEL_VERSION = "2021dec"
    DIM = 128
    BACKEND_ID = cv2.dnn.DNN_BACKEND_CUDA if hasattr(cv2.dnn, "DNN_BACKEND_CUDA") else 0
    TARGET_ID = cv2.dnn.DNN_TARGET_CUDA if hasattr(cv2.dnn, "DNN_TARGET_CUDA") else 0

    def __init__(self, weights_path: str | Path) -> None:
        self.weights_path = str(weights_path)
        self._model = cv2.FaceRecognizerSF.create(
            self.weights_path,
            "",
            self.BACKEND_ID,
            self.TARGET_ID,
        )

    @classmethod
    def from_registry(cls) -> "FaceEmbedder":
        entry = (
            ModelEntry.objects
            .filter(family=ModelEntry.Family.FACE_RECOGNITION, name="SFace")
            .order_by("-registered_at")
            .first()
        )
        if not entry or not entry.file_path or not Path(entry.file_path).exists():
            raise RuntimeError(
                "SFace weights not registered. Run: manage.py fetch_models"
            )
        return cls(entry.file_path)

    def embed(self, frame: np.ndarray, face: RawFace) -> np.ndarray:
        # SFace expects the original YuNet row (first 14 values) for alignment.
        aligned = self._model.alignCrop(frame, face.raw_row[:14])
        feature = self._model.feature(aligned)
        vec = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))
