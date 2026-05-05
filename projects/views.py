from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.db import connection, models
from django.http import FileResponse, Http404, HttpResponse, StreamingHttpResponse
from django.http import JsonResponse
from django.template.loader import render_to_string
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST
from django.views.generic import ListView
from django_q.tasks import async_task

from .forms import ProjectUploadForm
from .models import AnalysisJob, ExportJob, FaceDetection, ManualBlurRegion, Project
from .runtime import gpu_requirement_error
from .services import (
    apply_detection_override,
    attach_source_video,
    delete_project_storage,
    effective_detection_row,
)


def _is_ajax(request) -> bool:
    return request.headers.get("x-requested-with") == "XMLHttpRequest"


def _queue_preflight_error() -> str | None:
    existing_tables = set(connection.introspection.table_names())
    missing = []
    if "django_q_ormq" not in existing_tables:
        missing.append("django_q broker table")
    if "projects_facedetection" not in existing_tables:
        missing.append("projects face-detection table")
    if missing:
        joined = ", ".join(missing)
        return (
            f"Queue prerequisites are missing ({joined}). "
            "Run `venv/bin/python manage.py migrate` and restart `venv/bin/python manage.py qcluster`."
        )

    try:
        from . import services  # noqa: F401
    except Exception as exc:
        return (
            "Worker code is out of sync with the current project state. "
            f"Restart `venv/bin/python manage.py qcluster`. Import error: {exc}"
        )
    return None


def _project_runtime_error(project: Project) -> str | None:
    source = getattr(project, "source", None)
    if not source:
        return None
    return gpu_requirement_error(source.video_codec or "")


def _job_preview_rows(job: AnalysisJob, limit: int = 14) -> list[dict]:
    rows = []
    preview_qs = (
        job.detections.exclude(decision="allowed")
        .select_related("matched_person")
        .order_by("timestamp_seconds", "id")[:limit]
    )
    for detection in preview_qs:
        rows.append(
            {
                "timestamp_seconds": detection.timestamp_seconds,
                "timestamp_label": _format_timestamp(detection.timestamp_seconds),
                "decision": detection.get_decision_display(),
                "score": detection.score,
                "similarity": detection.similarity,
                "matched_person": detection.matched_person.display_name if detection.matched_person else "",
            }
        )
    return rows


def _log_lines(text: str, limit: int = 80) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-limit:]


def _looks_like_satellite_false(row: dict, peers: list[dict]) -> bool:
    if row.get("landmark_implausible"):
        return True
    if row["effective_decision"] == FaceDetection.Decision.ALLOWED:
        return False
    if row["likely_match"]:
        return False
    if row["similarity"] is not None and row["similarity"] >= settings.FACE_MATCH_CANDIDATE_THRESHOLD:
        return False

    row_area = max(1, row["width"] * row["height"])
    row_cx = row["x"] + row["width"] / 2.0
    row_cy = row["y"] + row["height"] / 2.0
    for peer in peers:
        if peer["id"] == row["id"] or peer["effective_decision"] != FaceDetection.Decision.ALLOWED:
            continue
        peer_area = max(1, peer["width"] * peer["height"])
        if peer_area < row_area * 2.4:
            continue
        peer_cx = peer["x"] + peer["width"] / 2.0
        peer_bottom = peer["y"] + peer["height"]
        horizontal_gap = abs(row_cx - peer_cx)
        if horizontal_gap > peer["width"] * 0.65:
            continue
        if row["y"] < peer["y"] + peer["height"] * 0.45:
            continue
        if row_cy > peer_bottom + peer["height"] * 0.8:
            continue
        if row["score"] > 0.9 and row_area > peer_area * 0.45:
            continue
        return True
    return False


def _mark_likely_false_rows(rows: list[dict]) -> int:
    by_frame: dict[int, list[dict]] = {}
    for row in rows:
        by_frame.setdefault(row["frame_index"], []).append(row)

    count = 0
    for peers in by_frame.values():
        for row in peers:
            row["likely_false"] = _looks_like_satellite_false(row, peers)
            if row["likely_false"]:
                count += 1
    return count


def _file_chunker(handle, *, offset: int = 0, length: int | None = None, chunk_size: int = 8192):
    try:
        handle.seek(offset)
        remaining = length
        while True:
            if remaining is not None and remaining <= 0:
                break
            read_size = chunk_size if remaining is None else min(chunk_size, remaining)
            data = handle.read(read_size)
            if not data:
                break
            if remaining is not None:
                remaining -= len(data)
            yield data
    finally:
        handle.close()


def _thumbnail_response(path: Path):
    response = FileResponse(path.open("rb"), as_attachment=False, filename=path.name, content_type="image/jpeg")
    response["Cache-Control"] = "private, max-age=86400"
    response["Content-Length"] = str(path.stat().st_size)
    return response


def _detection_thumbnail_cache_path(project: Project, detection: FaceDetection, effective) -> Path:
    cache_key = hashlib.sha1(
        (
            f"{detection.pk}:{detection.frame_index}:{effective.x}:{effective.y}:"
            f"{effective.width}:{effective.height}:{effective.override_id or 0}"
        ).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:16]
    return project.project_dir / "thumbs" / f"detection-{detection.pk}-{cache_key}.jpg"


def _render_detection_thumbnail(source_path: Path, frame_index: int, x: int, y: int, width: int, height: int) -> bytes:
    import cv2

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise Http404("Source not ready.")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
        grabbed, frame = cap.read()
    finally:
        cap.release()
    if not grabbed or frame is None:
        raise Http404("Frame could not be decoded.")

    frame_h, frame_w = frame.shape[:2]
    pad_x = max(18, int(width * 0.45))
    pad_y = max(18, int(height * 0.45))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(frame_w, x + width + pad_x)
    y1 = min(frame_h, y + height + pad_y)
    crop = frame[y0:y1, x0:x1].copy()
    if crop.size == 0:
        raise Http404("Thumbnail crop is empty.")

    box_x0 = max(0, x - x0)
    box_y0 = max(0, y - y0)
    box_x1 = min(crop.shape[1] - 1, box_x0 + width)
    box_y1 = min(crop.shape[0] - 1, box_y0 + height)
    cv2.rectangle(crop, (box_x0, box_y0), (box_x1, box_y1), (255, 208, 92), 2)

    target_w = 112
    target_h = 88
    scale = min(target_w / max(1, crop.shape[1]), target_h / max(1, crop.shape[0]), 1.0)
    if scale < 1.0:
        resized = cv2.resize(
            crop,
            (max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = crop
    ok, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        raise Http404("Thumbnail encoding failed.")
    return encoded.tobytes()


def _video_response(request, path: Path):
    if not path.exists():
        raise Http404("Video file missing on disk.")

    size = path.stat().st_size
    content_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
    range_header = request.headers.get("Range", "").strip()
    if not range_header or not range_header.startswith("bytes="):
        response = FileResponse(path.open("rb"), as_attachment=False, filename=path.name, content_type=content_type)
        response["Accept-Ranges"] = "bytes"
        response["Content-Length"] = str(size)
        return response

    try:
        start_text, end_text = range_header.removeprefix("bytes=").split("-", 1)
        if start_text == "":
            length = min(size, int(end_text))
            start = max(0, size - length)
            end = size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
    except ValueError:
        return HttpResponse(status=416)

    if start < 0 or end < start or start >= size:
        response = HttpResponse(status=416)
        response["Content-Range"] = f"bytes */{size}"
        return response

    end = min(end, size - 1)
    length = end - start + 1
    response = StreamingHttpResponse(
        _file_chunker(path.open("rb"), offset=start, length=length),
        status=206,
        content_type=content_type,
    )
    response["Accept-Ranges"] = "bytes"
    response["Content-Length"] = str(length)
    response["Content-Range"] = f"bytes {start}-{end}/{size}"
    return response


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes}:{sec:02d}"


def _analysis_job_cards(project: Project) -> list[dict]:
    cards = []
    source = getattr(project, "source", None)
    for job in project.analysis_jobs.all():
        counts = {
            row["decision"]: row["n"]
            for row in job.detections.values("decision").annotate(n=models.Count("id"))
        }
        cards.append(
            {
                "job": job,
                "counts": counts,
                "preview_rows": _job_preview_rows(job) if job.status == AnalysisJob.Status.DONE else [],
                "log_lines": _log_lines(job.log),
                "source_url": reverse("projects:source_video", args=[project.slug]) if source else "",
            }
        )
    return cards


def _latest_done_analysis(project: Project) -> AnalysisJob | None:
    return (
        project.analysis_jobs
        .filter(status=AnalysisJob.Status.DONE)
        .order_by("-finished_at", "-created_at")
        .first()
    )


def _export_cards(project: Project) -> list[dict]:
    cards = []
    for export in project.exports.all():
        cards.append(
            {
                "export": export,
                "log_lines": _log_lines(export.log),
                "preview_url": reverse("projects:stream_export", args=[export.pk]) if export.is_done and export.output_path else "",
            }
        )
    return cards


def _project_status_payload(project: Project) -> dict:
    analysis_cards = _analysis_job_cards(project)
    export_cards = _export_cards(project)
    analysis_html = render_to_string(
        "projects/_analysis_jobs.html",
        {"project": project, "analysis_cards": analysis_cards},
    )
    exports_html = render_to_string(
        "projects/_exports.html",
        {"project": project, "export_cards": export_cards},
    )
    return {
        "analysis_html": analysis_html,
        "exports_html": exports_html,
        "analysis_hash": hashlib.md5(analysis_html.encode("utf-8")).hexdigest(),
        "exports_hash": hashlib.md5(exports_html.encode("utf-8")).hexdigest(),
        "has_active_jobs": any(
            card["job"].status in {AnalysisJob.Status.QUEUED, AnalysisJob.Status.RUNNING}
            for card in analysis_cards
        )
        or any(
            card["export"].status in {ExportJob.Status.QUEUED, ExportJob.Status.RUNNING}
            for card in export_cards
        ),
    }


class ProjectListView(ListView):
    model = Project
    template_name = "projects/list.html"
    context_object_name = "projects"
    paginate_by = 25

    def get_queryset(self):
        return (
            Project.objects.select_related("source")
            .prefetch_related("analysis_jobs", "exports")
            .all()
        )


def upload_project(request):
    if request.method == "POST":
        form = ProjectUploadForm(request.POST, request.FILES)
        if form.is_valid():
            upload = form.cleaned_data["file"]
            project = form.build_project()
            try:
                attach_source_video(project, upload, upload.name)
            except ValueError as exc:
                project.delete()
                form.add_error("file", str(exc))
            else:
                if _is_ajax(request):
                    return JsonResponse({"redirect_url": project.get_absolute_url()})
                messages.success(request, f"Project “{project.title}” uploaded.")
                return redirect(project.get_absolute_url())
        if _is_ajax(request):
            errors = {
                field: [message["message"] for message in messages_]
                for field, messages_ in form.errors.get_json_data(escape_html=False).items()
            }
            return JsonResponse(
                {
                    "errors": errors,
                    "non_field_errors": [str(e) for e in form.non_field_errors()],
                },
                status=400,
            )
    else:
        form = ProjectUploadForm()
    return render(request, "projects/upload.html", {"form": form})


def project_detail(request, slug: str):
    project = get_object_or_404(
        Project.objects.select_related("source").prefetch_related("analysis_jobs", "exports"),
        slug=slug,
    )
    analysis_cards = _analysis_job_cards(project)
    review_job = _latest_done_analysis(project)
    return render(
        request,
        "projects/detail.html",
        {
            "project": project,
            "analysis_cards": analysis_cards,
            "export_cards": _export_cards(project),
            "review_job": review_job,
            "queue_health_error": _queue_preflight_error(),
            "runtime_health_error": _project_runtime_error(project),
        },
    )


@require_POST
def queue_analysis(request, slug: str):
    project = get_object_or_404(Project, slug=slug)
    queue_error = _queue_preflight_error()
    if queue_error:
        messages.error(request, queue_error)
        return redirect(project.get_absolute_url())
    runtime_error = _project_runtime_error(project)
    if runtime_error:
        messages.error(request, runtime_error)
        return redirect(project.get_absolute_url())
    if not getattr(project, "source", None):
        messages.error(request, "Upload a source video first.")
        return redirect(project.get_absolute_url())
    job = AnalysisJob.objects.create(project=project)
    task_id = async_task("projects.services.run_analysis_job", job.pk, group="analysis")
    job.task_id = task_id or ""
    job.save(update_fields=["task_id"])
    messages.success(request, "Analysis queued.")
    return redirect(project.get_absolute_url())


@require_POST
def queue_final_export(request, slug: str):
    project = get_object_or_404(Project, slug=slug)
    queue_error = _queue_preflight_error()
    if queue_error:
        messages.error(request, queue_error)
        return redirect(project.get_absolute_url())
    runtime_error = _project_runtime_error(project)
    if runtime_error:
        messages.error(request, runtime_error)
        return redirect(project.get_absolute_url())
    if not getattr(project, "source", None):
        messages.error(request, "Upload a source video first.")
        return redirect(project.get_absolute_url())
    if not project.analysis_jobs.filter(status=AnalysisJob.Status.DONE).exists():
        messages.error(request, "Run analysis to completion before exporting final.")
        return redirect(project.get_absolute_url())
    style = request.POST.get("style", ExportJob.Style.MOSAIC)
    if style not in ExportJob.Style.values:
        style = ExportJob.Style.MOSAIC
    export = ExportJob.objects.create(project=project, style=style)
    task_id = async_task("projects.services.render_final_export", export.pk, group="export")
    export.task_id = task_id or ""
    export.save(update_fields=["task_id"])
    messages.success(request, "Final export queued.")
    return redirect(project.get_absolute_url())


def draft_review(request, slug: str):
    project = get_object_or_404(
        Project.objects.select_related("source").prefetch_related("analysis_jobs"),
        slug=slug,
    )
    review_job = _latest_done_analysis(project)
    if not review_job:
        messages.error(request, "Run analysis to completion before opening draft review.")
        return redirect(project.get_absolute_url())

    detections_qs = (
        review_job.detections
        .select_related("matched_person", "override")
        .order_by("timestamp_seconds", "id")
    )
    detections = []
    counts = {"allowed": 0, "unknown": 0, "uncertain": 0, "conflict": 0}
    override_count = 0
    likely_match_count = 0
    for detection in detections_qs:
        effective = effective_detection_row(detection)
        counts[effective.decision] = counts.get(effective.decision, 0) + 1
        if effective.override_id:
            override_count += 1
        likely_match = (
            effective.decision != FaceDetection.Decision.ALLOWED
            and detection.similarity is not None
            and (
                detection.matched_person_id is not None
                or detection.decision == FaceDetection.Decision.CONFLICT
                or detection.similarity >= settings.FACE_MATCH_THRESHOLD
            )
        )
        if likely_match:
            likely_match_count += 1
        detections.append(
            {
                "id": detection.pk,
                "frame_index": detection.frame_index,
                "timestamp_seconds": detection.timestamp_seconds,
                "timestamp_label": _format_timestamp(detection.timestamp_seconds),
                "score": detection.score,
                "similarity": detection.similarity,
                "matched_person": detection.matched_person.display_name if detection.matched_person else "",
                "auto_decision": detection.decision,
                "auto_decision_label": detection.get_decision_display(),
                "effective_decision": effective.decision,
                "effective_decision_label": FaceDetection.Decision(effective.decision).label,
                "likely_match": likely_match,
                "likely_false": False,
                "landmark_implausible": detection.landmark_implausible,
                "x": effective.x,
                "y": effective.y,
                "width": effective.width,
                "height": effective.height,
                "has_override": bool(effective.override_id),
                "override_note": detection.override.note if getattr(detection, "override", None) else "",
            }
        )
    likely_false_count = _mark_likely_false_rows(detections)
    manual_regions = [
        {
            "id": region.pk,
            "start_seconds": region.start_seconds,
            "end_seconds": region.end_seconds,
            "start_label": _format_timestamp(region.start_seconds),
            "end_label": _format_timestamp(region.end_seconds),
            "x": region.x,
            "y": region.y,
            "width": region.width,
            "height": region.height,
            "note": region.note,
        }
        for region in project.manual_blur_regions.all()
    ]

    return render(
        request,
        "projects/draft_review.html",
        {
            "project": project,
            "review_job": review_job,
            "detections": detections,
            "counts": counts,
            "override_count": override_count,
            "likely_match_count": likely_match_count,
            "likely_false_count": likely_false_count,
            "manual_regions": manual_regions,
            "source_url": reverse("projects:source_video", args=[project.slug]),
        },
    )


def detection_thumbnail(request, slug: str, detection_id: int):
    project = get_object_or_404(Project.objects.select_related("source"), slug=slug)
    source = getattr(project, "source", None)
    if not source:
        raise Http404("Source not ready.")

    detection = get_object_or_404(
        FaceDetection.objects.select_related("job__project", "override"),
        pk=detection_id,
        job__project=project,
    )
    effective = effective_detection_row(detection)
    cache_path = _detection_thumbnail_cache_path(project, detection, effective)
    if cache_path.exists():
        return _thumbnail_response(cache_path)

    payload = _render_detection_thumbnail(
        Path(source.absolute_path),
        detection.frame_index,
        effective.x,
        effective.y,
        effective.width,
        effective.height,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(payload)
    return _thumbnail_response(cache_path)


@require_POST
def save_detection_override(request, slug: str, detection_id: int):
    project = get_object_or_404(Project, slug=slug)
    detection = get_object_or_404(
        FaceDetection.objects.select_related("job__project"),
        pk=detection_id,
        job__project=project,
    )
    latest_done_job = _latest_done_analysis(project)
    if latest_done_job is None or detection.job_id != latest_done_job.pk:
        messages.error(request, "Draft review accepts edits only for the latest completed analysis run.")
        return redirect(reverse("projects:draft_review", args=[project.slug]))

    decision = request.POST.get("decision", detection.decision)
    if decision not in FaceDetection.Decision.values:
        messages.error(request, "Invalid review decision.")
        return redirect(reverse("projects:draft_review", args=[project.slug]))

    try:
        x = max(0, int(request.POST.get("x", detection.x)))
        y = max(0, int(request.POST.get("y", detection.y)))
        width = max(1, int(request.POST.get("width", detection.width)))
        height = max(1, int(request.POST.get("height", detection.height)))
    except ValueError:
        messages.error(request, "Box coordinates must be integers.")
        return redirect(reverse("projects:draft_review", args=[project.slug]))

    note = request.POST.get("note", "")
    override = apply_detection_override(
        detection,
        decision=decision,
        x=x,
        y=y,
        width=width,
        height=height,
        note=note,
    )
    if override is None:
        messages.success(request, "Review override cleared; export will use automatic detection.")
    else:
        messages.success(request, "Review override saved for draft export.")
    return redirect(f"{reverse('projects:draft_review', args=[project.slug])}#det-{detection.pk}")


@require_POST
def reset_detection_override(request, slug: str, detection_id: int):
    project = get_object_or_404(Project, slug=slug)
    detection = get_object_or_404(
        FaceDetection.objects.select_related("job__project"),
        pk=detection_id,
        job__project=project,
    )
    detection.override.delete() if hasattr(detection, "override") else None
    messages.success(request, "Review override removed.")
    return redirect(f"{reverse('projects:draft_review', args=[project.slug])}#det-{detection.pk}")


@require_POST
def bulk_allow_detections(request, slug: str):
    project = get_object_or_404(Project, slug=slug)
    latest_done_job = _latest_done_analysis(project)
    if latest_done_job is None:
        messages.error(request, "Run analysis to completion before editing draft review.")
        return redirect(project.get_absolute_url())

    detection_ids = []
    for raw_id in request.POST.getlist("detection_ids"):
        try:
            detection_ids.append(int(raw_id))
        except ValueError:
            continue
    if not detection_ids:
        messages.error(request, "Select at least one detection.")
        return redirect(reverse("projects:draft_review", args=[project.slug]))

    detections = (
        latest_done_job.detections
        .filter(pk__in=detection_ids)
        .select_related("job__project")
    )
    updated = 0
    for detection in detections:
        apply_detection_override(
            detection,
            decision=FaceDetection.Decision.ALLOWED,
            x=detection.x,
            y=detection.y,
            width=detection.width,
            height=detection.height,
            note="Bulk allowed in draft review.",
        )
        updated += 1

    messages.success(request, f"Marked {updated} detection(s) as allowed.")
    return redirect(reverse("projects:draft_review", args=[project.slug]))


@require_POST
def add_manual_blur_region(request, slug: str):
    project = get_object_or_404(Project.objects.select_related("source"), slug=slug)
    if not _latest_done_analysis(project):
        messages.error(request, "Run analysis to completion before adding manual blur regions.")
        return redirect(project.get_absolute_url())

    try:
        start_seconds = max(0.0, float(request.POST.get("start_seconds", "0")))
        end_seconds = max(0.0, float(request.POST.get("end_seconds", "0")))
        x = max(0, int(request.POST.get("x", "0")))
        y = max(0, int(request.POST.get("y", "0")))
        width = max(1, int(request.POST.get("width", "1")))
        height = max(1, int(request.POST.get("height", "1")))
    except ValueError:
        messages.error(request, "Manual blur region needs numeric time and box values.")
        return redirect(reverse("projects:draft_review", args=[project.slug]))

    if end_seconds <= start_seconds:
        end_seconds = start_seconds + 1.0

    source = getattr(project, "source", None)
    if source and source.width and source.height:
        x = min(x, source.width - 1)
        y = min(y, source.height - 1)
        width = min(width, source.width - x)
        height = min(height, source.height - y)

    region = ManualBlurRegion.objects.create(
        project=project,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        x=x,
        y=y,
        width=width,
        height=height,
        note=request.POST.get("note", "").strip(),
    )
    messages.success(request, "Manual blur segment added.")
    return redirect(f"{reverse('projects:draft_review', args=[project.slug])}#manual-{region.pk}")


@require_POST
def delete_manual_blur_region(request, slug: str, region_id: int):
    project = get_object_or_404(Project, slug=slug)
    region = get_object_or_404(ManualBlurRegion, pk=region_id, project=project)
    region.delete()
    messages.success(request, "Manual blur segment removed.")
    return redirect(reverse("projects:draft_review", args=[project.slug]))


def project_status(request, slug: str):
    project = get_object_or_404(
        Project.objects.select_related("source").prefetch_related("analysis_jobs", "exports"),
        slug=slug,
    )
    return JsonResponse(_project_status_payload(project))


@require_POST
def delete_project(request, slug: str):
    project = get_object_or_404(Project, slug=slug)
    title = project.title
    delete_project_storage(project)
    project.delete()
    messages.success(request, f"Project “{title}” deleted.")
    return redirect(reverse("projects:list"))


def download_export(request, pk: int):
    export = get_object_or_404(ExportJob, pk=pk)
    if not export.is_done or not export.output_path:
        raise Http404("Export not ready.")
    path = Path(export.output_path)
    if not path.exists():
        raise Http404("File missing on disk.")
    return FileResponse(path.open("rb"), as_attachment=True, filename=path.name)


def stream_source_video(request, slug: str):
    project = get_object_or_404(Project.objects.select_related("source"), slug=slug)
    source = getattr(project, "source", None)
    if not source:
        raise Http404("Source not ready.")
    path = Path(source.absolute_path)
    return _video_response(request, path)


def stream_export(request, pk: int):
    export = get_object_or_404(ExportJob, pk=pk)
    if not export.is_done or not export.output_path:
        raise Http404("Export not ready.")
    path = Path(export.output_path)
    return _video_response(request, path)
