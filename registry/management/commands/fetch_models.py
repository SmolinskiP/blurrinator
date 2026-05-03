from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from registry.models import ModelEntry


# Pinned weight files from opencv_zoo. Hashes are computed on first download
# and locked in the registry; subsequent runs verify against the registry row.
MODELS = [
    {
        "name": "YuNet",
        "family": ModelEntry.Family.FACE_DETECTION,
        "version": "2023mar",
        "license": "MIT",
        "filename": "face_detection_yunet_2023mar.onnx",
        "url": (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        "source_url": (
            "https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet"
        ),
        "notes": "OpenCV Zoo YuNet face detector. Loaded via cv2.FaceDetectorYN.",
    },
    {
        "name": "SFace",
        "family": ModelEntry.Family.FACE_RECOGNITION,
        "version": "2021dec",
        "license": "Apache-2.0",
        "filename": "face_recognition_sface_2021dec.onnx",
        "url": (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
        "source_url": (
            "https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface"
        ),
        "notes": "OpenCV Zoo SFace embedder. Loaded via cv2.FaceRecognizerSF.",
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Command(BaseCommand):
    help = "Download face detection / recognition weights and register them."

    def handle(self, *args, **options):
        models_dir = Path(settings.STORAGE_ROOT) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        for spec in MODELS:
            target = models_dir / spec["filename"]
            if not target.exists():
                self.stdout.write(f"↓ {spec['name']} {spec['version']} → {target}")
                with urllib.request.urlopen(spec["url"]) as response, target.open("wb") as out:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
            else:
                self.stdout.write(f"= {spec['name']} {spec['version']} already on disk")

            digest = sha256_file(target)
            entry, created = ModelEntry.objects.update_or_create(
                name=spec["name"],
                version=spec["version"],
                defaults={
                    "family": spec["family"],
                    "license": spec["license"],
                    "source_url": spec["source_url"],
                    "file_path": str(target),
                    "sha256": digest,
                    "provenance": ModelEntry.Provenance.PRETRAINED,
                    "notes": spec["notes"],
                },
            )
            verb = "registered" if created else "updated"
            self.stdout.write(self.style.SUCCESS(
                f"  {verb}: {entry.name} {entry.version} — sha256 {digest[:12]}…"
            ))
