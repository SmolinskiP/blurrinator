from __future__ import annotations

from django.db import models
from django.utils import timezone


class AllowedPerson(models.Model):
    display_name = models.CharField(max_length=200)
    consent_basis = models.TextField(
        help_text="Where the consent comes from. Required by policy.",
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now)
    enrolled_by = models.CharField(max_length=200, blank=True)

    class Meta:
        ordering = ["display_name"]

    def __str__(self) -> str:
        return self.display_name


class EnrollmentImage(models.Model):
    class Status(models.TextChoices):
        ACCEPTED = "accepted", "Accepted"
        REJECTED = "rejected", "Rejected"

    person = models.ForeignKey(
        AllowedPerson, on_delete=models.CASCADE, related_name="images"
    )
    image = models.ImageField(upload_to="allowlist/")
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.ACCEPTED)
    quality_score = models.FloatField(null=True, blank=True)
    rejected_reason = models.CharField(max_length=200, blank=True)
    uploaded_at = models.DateTimeField(default=timezone.now)


class FaceEmbedding(models.Model):
    person = models.ForeignKey(
        AllowedPerson, on_delete=models.CASCADE, related_name="embeddings"
    )
    image = models.OneToOneField(
        EnrollmentImage, on_delete=models.CASCADE, related_name="embedding"
    )
    model_name = models.CharField(max_length=64)
    model_version = models.CharField(max_length=64)
    vector = models.BinaryField()
    dim = models.PositiveIntegerField()
    detector_score = models.FloatField()
    created_at = models.DateTimeField(default=timezone.now)
