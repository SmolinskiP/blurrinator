from __future__ import annotations

from django.db import models
from django.utils import timezone


class ModelEntry(models.Model):
    class Family(models.TextChoices):
        FACE_DETECTION = "face_detection", "Face detection"
        FACE_RECOGNITION = "face_recognition", "Face recognition"
        PERSON_DETECTION = "person_detection", "Person detection"
        PERSON_SEGMENTATION = "person_segmentation", "Person segmentation"
        POSE_ESTIMATION = "pose_estimation", "Pose estimation"
        TRACKER = "tracker", "Tracker"

    class Provenance(models.TextChoices):
        PRETRAINED = "pretrained", "Pretrained (upstream weights)"
        FINE_TUNED = "fine_tuned", "Fine-tuned"
        TRAINED_LOCAL = "trained_local", "Trained locally"

    name = models.CharField(max_length=200)
    family = models.CharField(max_length=32, choices=Family.choices)
    version = models.CharField(max_length=64)
    license = models.CharField(max_length=64)
    source_url = models.URLField(blank=True)
    file_path = models.CharField(max_length=1024, blank=True)
    sha256 = models.CharField(max_length=64, blank=True)
    provenance = models.CharField(
        max_length=20, choices=Provenance.choices, default=Provenance.PRETRAINED
    )
    notes = models.TextField(blank=True)
    registered_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["family", "name", "-registered_at"]
        unique_together = [("name", "version")]

    def __str__(self) -> str:
        return f"{self.name} {self.version}"
