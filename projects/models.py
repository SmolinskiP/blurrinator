from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.db import models
from django.urls import reverse
from django.utils import timezone


class Project(models.Model):
    title = models.CharField(max_length=200)
    slug = models.SlugField(max_length=220, unique=True)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.title

    def get_absolute_url(self) -> str:
        return reverse("projects:detail", args=[self.slug])

    @property
    def project_dir(self) -> Path:
        path = Path(settings.STORAGE_ROOT) / "projects" / self.slug
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def status_label(self) -> str:
        source = getattr(self, "source", None)
        if not source:
            return "no source"
        latest = self.analysis_jobs.order_by("-created_at").first()
        if latest:
            return latest.get_status_display()
        return "ready"


class SourceVideo(models.Model):
    project = models.OneToOneField(
        Project, on_delete=models.CASCADE, related_name="source"
    )
    original_filename = models.CharField(max_length=512)
    stored_path = models.CharField(max_length=1024)
    size_bytes = models.BigIntegerField()
    sha256 = models.CharField(max_length=64, db_index=True)
    container = models.CharField(max_length=64, blank=True)
    video_codec = models.CharField(max_length=64, blank=True)
    audio_codec = models.CharField(max_length=64, blank=True)
    width = models.PositiveIntegerField(null=True, blank=True)
    height = models.PositiveIntegerField(null=True, blank=True)
    fps = models.FloatField(null=True, blank=True)
    duration_seconds = models.FloatField(null=True, blank=True)
    bitrate = models.BigIntegerField(null=True, blank=True)
    probe_raw = models.JSONField(default=dict, blank=True)
    uploaded_at = models.DateTimeField(default=timezone.now)

    @property
    def absolute_path(self) -> Path:
        return Path(self.stored_path)

    @property
    def resolution(self) -> str:
        if self.width and self.height:
            return f"{self.width} × {self.height}"
        return "—"


class AnalysisJob(models.Model):
    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        DONE = "done", "Done"
        FAILED = "failed", "Failed"

    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name="analysis_jobs"
    )
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED)
    progress = models.PositiveSmallIntegerField(default=0)
    message = models.TextField(blank=True)
    log = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    task_id = models.CharField(max_length=64, blank=True)

    class Meta:
        ordering = ["-created_at"]


class ExportJob(models.Model):
    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        DONE = "done", "Done"
        FAILED = "failed", "Failed"

    class Style(models.TextChoices):
        MOSAIC = "mosaic", "Mosaic"
        GAUSSIAN = "gaussian", "Gaussian blur"
        SOLID = "solid", "Solid fill"
        EYE_BAR = "eye_bar", "Eye bar"

    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name="exports"
    )
    style = models.CharField(max_length=16, choices=Style.choices, default=Style.MOSAIC)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED)
    progress = models.PositiveSmallIntegerField(default=0)
    output_path = models.CharField(max_length=1024, blank=True)
    size_bytes = models.BigIntegerField(null=True, blank=True)
    message = models.TextField(blank=True)
    log = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    task_id = models.CharField(max_length=64, blank=True)

    class Meta:
        ordering = ["-created_at"]

    @property
    def is_done(self) -> bool:
        return self.status == self.Status.DONE


class FaceDetection(models.Model):
    class Decision(models.TextChoices):
        ALLOWED = "allowed", "Allowed"
        UNKNOWN = "unknown", "Unknown — blur"
        UNCERTAIN = "uncertain", "Uncertain — blur + flag"
        CONFLICT = "conflict", "Identity conflict — blur + flag"

    job = models.ForeignKey(
        AnalysisJob, on_delete=models.CASCADE, related_name="detections"
    )
    frame_index = models.PositiveIntegerField()
    timestamp_seconds = models.FloatField()
    x = models.PositiveIntegerField()
    y = models.PositiveIntegerField()
    width = models.PositiveIntegerField()
    height = models.PositiveIntegerField()
    score = models.FloatField()
    matched_person = models.ForeignKey(
        "allowlist.AllowedPerson",
        null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name="matches",
    )
    similarity = models.FloatField(null=True, blank=True)
    landmark_implausible = models.BooleanField(default=False)
    decision = models.CharField(max_length=16, choices=Decision.choices, default=Decision.UNKNOWN)

    class Meta:
        indexes = [
            models.Index(fields=["job", "frame_index"]),
        ]


class DetectionOverride(models.Model):
    detection = models.OneToOneField(
        FaceDetection, on_delete=models.CASCADE, related_name="override"
    )
    decision = models.CharField(max_length=16, choices=FaceDetection.Decision.choices)
    x = models.PositiveIntegerField()
    y = models.PositiveIntegerField()
    width = models.PositiveIntegerField()
    height = models.PositiveIntegerField()
    note = models.CharField(max_length=240, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["detection__timestamp_seconds", "detection_id"]


class ManualBlurRegion(models.Model):
    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name="manual_blur_regions"
    )
    start_seconds = models.FloatField()
    end_seconds = models.FloatField()
    x = models.PositiveIntegerField()
    y = models.PositiveIntegerField()
    width = models.PositiveIntegerField()
    height = models.PositiveIntegerField()
    note = models.CharField(max_length=240, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["start_seconds", "id"]
