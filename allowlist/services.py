from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from django.conf import settings

from projects.vision import FaceDetector, FaceEmbedder

from .models import AllowedPerson, EnrollmentImage, FaceEmbedding


MIN_FACE_PIXELS = 60  # short edge of detected face
MIN_DETECTOR_SCORE = 0.7  # stricter than analysis - this is the reference set


@dataclass
class EnrollmentResult:
    image: EnrollmentImage
    accepted: bool
    reason: str = ""


def _ensure_writable(image_path: str) -> np.ndarray:
    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("File could not be decoded as an image.")
    return frame


def enroll_image(person: AllowedPerson, image: EnrollmentImage) -> EnrollmentResult:
    """Detect a single face and persist the SFace embedding. Mutates `image`."""

    detector = FaceDetector.from_registry(score_threshold=settings.FACE_DETECTOR_SCORE)
    embedder = FaceEmbedder.from_registry()

    try:
        frame = _ensure_writable(image.image.path)
    except ValueError as exc:
        return _reject(image, str(exc))

    faces = detector.detect(frame)
    if not faces:
        return _reject(image, "No face detected.")
    if len(faces) > 1:
        return _reject(image, f"Multiple faces ({len(faces)}). Use a single-person crop.")

    face = faces[0]
    if face.score < MIN_DETECTOR_SCORE:
        return _reject(image, f"Detector score {face.score:.2f} below {MIN_DETECTOR_SCORE}.")
    if min(face.w, face.h) < MIN_FACE_PIXELS:
        return _reject(image, f"Face too small ({face.w}×{face.h}). Need ≥ {MIN_FACE_PIXELS}px.")

    vector = embedder.embed(frame, face)

    FaceEmbedding.objects.update_or_create(
        image=image,
        defaults={
            "person": person,
            "model_name": embedder.MODEL_NAME,
            "model_version": embedder.MODEL_VERSION,
            "vector": vector.astype(np.float32).tobytes(),
            "dim": vector.shape[0],
            "detector_score": face.score,
        },
    )

    image.status = EnrollmentImage.Status.ACCEPTED
    image.quality_score = face.score
    image.rejected_reason = ""
    image.save(update_fields=["status", "quality_score", "rejected_reason"])
    return EnrollmentResult(image=image, accepted=True)


def _reject(image: EnrollmentImage, reason: str) -> EnrollmentResult:
    image.status = EnrollmentImage.Status.REJECTED
    image.rejected_reason = reason
    image.save(update_fields=["status", "rejected_reason"])
    FaceEmbedding.objects.filter(image=image).delete()
    return EnrollmentResult(image=image, accepted=False, reason=reason)


def load_allowlist_index() -> tuple[list[AllowedPerson], np.ndarray, list[int]]:
    """Return (people, stacked_vectors, person_id_per_row)."""

    people = list(AllowedPerson.objects.filter(is_active=True).prefetch_related("embeddings"))
    rows: list[np.ndarray] = []
    owners: list[int] = []
    for person in people:
        for emb in person.embeddings.all():
            vec = np.frombuffer(emb.vector, dtype=np.float32)
            rows.append(vec)
            owners.append(person.pk)
    if not rows:
        return people, np.zeros((0, FaceEmbedder.DIM), dtype=np.float32), []
    return people, np.vstack(rows), owners
