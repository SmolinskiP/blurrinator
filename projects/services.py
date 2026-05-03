from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from django.conf import settings
from django.db import models
from django.utils import timezone

from .models import (
    AnalysisJob,
    DetectionOverride,
    ExportJob,
    FaceDetection,
    ManualBlurRegion,
    Project,
    SourceVideo,
)
from .runtime import detect_runtime_capabilities, gpu_requirement_error


CHUNK = 4 * 1024 * 1024


@dataclass
class ProbeResult:
    container: str
    video_codec: str
    audio_codec: str
    width: int | None
    height: int | None
    fps: float | None
    duration_seconds: float | None
    bitrate: int | None
    raw: dict


@dataclass
class BlurRegion:
    x: int
    y: int
    width: int
    height: int
    score: float


@dataclass
class EffectiveDetection:
    frame_index: int
    timestamp_seconds: float
    x: int
    y: int
    width: int
    height: int
    score: float
    decision: str
    source_detection_id: int
    override_id: int | None


def stream_to_disk(source: IO[bytes], destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    size = 0
    with destination.open("wb") as out:
        while True:
            chunk = source.read(CHUNK)
            if not chunk:
                break
            out.write(chunk)
            hasher.update(chunk)
            size += len(chunk)
    return size, hasher.hexdigest()


def probe_video(path: Path) -> ProbeResult:
    cmd = [
        settings.FFPROBE_BINARY,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(completed.stdout or "{}")
    fmt = data.get("format", {}) or {}
    streams = data.get("streams", []) or []
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})

    fps = None
    rate = video.get("avg_frame_rate") or video.get("r_frame_rate")
    if rate and "/" in rate:
        num, den = rate.split("/", 1)
        try:
            num_f = float(num)
            den_f = float(den)
            if den_f:
                fps = num_f / den_f
        except ValueError:
            fps = None

    duration = fmt.get("duration") or video.get("duration")
    bitrate = fmt.get("bit_rate")
    return ProbeResult(
        container=fmt.get("format_name", "") or "",
        video_codec=video.get("codec_name", "") or "",
        audio_codec=audio.get("codec_name", "") or "",
        width=video.get("width"),
        height=video.get("height"),
        fps=fps,
        duration_seconds=float(duration) if duration else None,
        bitrate=int(bitrate) if bitrate else None,
        raw=data,
    )


def attach_source_video(project: Project, upload, original_filename: str) -> SourceVideo:
    suffix = Path(original_filename).suffix or ".mp4"
    target = project.project_dir / f"source{suffix}"
    size, sha = stream_to_disk(upload, target)
    try:
        probe = probe_video(target)
    except subprocess.CalledProcessError as exc:
        target.unlink(missing_ok=True)
        raise ValueError(f"ffprobe rejected the file: {exc.stderr.strip() or exc}") from exc

    source, _ = SourceVideo.objects.update_or_create(
        project=project,
        defaults={
            "original_filename": original_filename,
            "stored_path": str(target),
            "size_bytes": size,
            "sha256": sha,
            "container": probe.container,
            "video_codec": probe.video_codec,
            "audio_codec": probe.audio_codec,
            "width": probe.width,
            "height": probe.height,
            "fps": probe.fps,
            "duration_seconds": probe.duration_seconds,
            "bitrate": probe.bitrate,
            "probe_raw": probe.raw,
            "uploaded_at": timezone.now(),
        },
    )
    return source


def _ffmpeg_mux_audio(video_only: Path, source: Path, target: Path) -> None:
    # Transcode the OpenCV-written stream to H.264 yuv420p so browsers can
    # actually decode it (mp4v fourcc plays audio-only in Chrome/Firefox).
    cmd = [
        settings.FFMPEG_BINARY, "-y",
        "-i", str(video_only),
        "-i", str(source),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(target),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffmpeg mux failed")


def _update_export_progress(export_id: int, progress: int, message: str = "") -> None:
    update_fields: dict[str, int | str] = {"progress": max(0, min(100, progress))}
    if message:
        update_fields["message"] = message
    ExportJob.objects.filter(pk=export_id).update(**update_fields)


def _timestamped_log_line(message: str) -> str:
    return f"[{timezone.now().strftime('%H:%M:%S')}] {message}"


def _append_analysis_log(job: AnalysisJob, message: str) -> None:
    line = _timestamped_log_line(message)
    AnalysisJob.objects.filter(pk=job.pk).update(log=models.functions.Concat("log", models.Value(line + "\n")))
    job.log = f"{job.log}{line}\n"


def _append_export_log(export: ExportJob, message: str) -> None:
    line = _timestamped_log_line(message)
    ExportJob.objects.filter(pk=export.pk).update(log=models.functions.Concat("log", models.Value(line + "\n")))
    export.log = f"{export.log}{line}\n"


def _set_analysis_state(job: AnalysisJob, *, progress: int | None = None, message: str | None = None) -> None:
    update_fields: dict[str, int | str] = {}
    if progress is not None:
        update_fields["progress"] = max(0, min(100, progress))
        job.progress = update_fields["progress"]
    if message is not None:
        update_fields["message"] = message
        job.message = message
    if update_fields:
        AnalysisJob.objects.filter(pk=job.pk).update(**update_fields)


def _set_export_state(export: ExportJob, *, progress: int | None = None, message: str | None = None) -> None:
    update_fields: dict[str, int | str] = {}
    if progress is not None:
        update_fields["progress"] = max(0, min(100, progress))
        export.progress = update_fields["progress"]
    if message is not None:
        update_fields["message"] = message
        export.message = message
    if update_fields:
        ExportJob.objects.filter(pk=export.pk).update(**update_fields)


def effective_detection_row(detection: FaceDetection) -> EffectiveDetection:
    override = getattr(detection, "override", None)
    if override is None:
        return EffectiveDetection(
            frame_index=detection.frame_index,
            timestamp_seconds=detection.timestamp_seconds,
            x=detection.x,
            y=detection.y,
            width=detection.width,
            height=detection.height,
            score=detection.score,
            decision=detection.decision,
            source_detection_id=detection.pk,
            override_id=None,
        )
    return EffectiveDetection(
        frame_index=detection.frame_index,
        timestamp_seconds=detection.timestamp_seconds,
        x=override.x,
        y=override.y,
        width=override.width,
        height=override.height,
        score=detection.score,
        decision=override.decision,
        source_detection_id=detection.pk,
        override_id=override.pk,
    )


def apply_detection_override(
    detection: FaceDetection,
    *,
    decision: str,
    x: int,
    y: int,
    width: int,
    height: int,
    note: str = "",
) -> DetectionOverride | None:
    payload = {
        "decision": decision,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "note": note.strip(),
    }
    baseline = {
        "decision": detection.decision,
        "x": detection.x,
        "y": detection.y,
        "width": detection.width,
        "height": detection.height,
        "note": "",
    }
    if payload == baseline:
        DetectionOverride.objects.filter(detection=detection).delete()
        return None
    override, _ = DetectionOverride.objects.update_or_create(
        detection=detection,
        defaults=payload,
    )
    return override


def run_analysis_job(job_id: int) -> None:
    """Phase 2: detect faces with YuNet, embed with SFace, match against allowlist."""

    import cv2
    import numpy as np

    from allowlist.services import load_allowlist_index
    from projects.vision import FaceDetector, FaceEmbedder

    job = AnalysisJob.objects.select_related("project__source").get(pk=job_id)
    job.status = AnalysisJob.Status.RUNNING
    job.started_at = timezone.now()
    job.progress = 1
    job.log = ""
    job.detections.all().delete()
    job.save(update_fields=["status", "started_at", "progress", "log"])

    try:
        source = getattr(job.project, "source", None)
        if not source:
            raise RuntimeError("Project has no source video.")
        runtime_error = gpu_requirement_error(source.video_codec or "")
        if runtime_error:
            raise RuntimeError(runtime_error)

        people, vectors, owners = load_allowlist_index()
        _append_analysis_log(job, f"Analysis started for {source.original_filename}.")
        _append_analysis_log(job, f"Allowlist identities: {len(people)}; embeddings: {vectors.shape[0]}.")

        detector = FaceDetector.from_registry(score_threshold=settings.FACE_DETECTOR_SCORE)
        embedder = FaceEmbedder.from_registry()
        _append_analysis_log(job, f"Loaded YuNet + SFace with detector threshold {settings.FACE_DETECTOR_SCORE:.2f}.")

        cap = cv2.VideoCapture(str(source.absolute_path))
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open source video.")
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or source.fps or 30.0
            stride = max(1, settings.ANALYSIS_FRAME_STRIDE)
            _append_analysis_log(job, f"Video opened: {total_frames or 'unknown'} frames at {fps:.3f} fps; stride {stride}.")

            buffered: list[FaceDetection] = []
            BATCH = 500
            last_progress = -1
            last_logged_progress = -1
            total_faces = 0

            frame_index = 0
            while True:
                grabbed, frame = cap.read()
                if not grabbed:
                    break
                if frame_index % stride != 0:
                    frame_index += 1
                    continue

                faces = [
                    f for f in detector.detect_multiscale(frame, scales=settings.FACE_DETECT_SCALES)
                    if _face_is_plausible(f)
                ]
                total_faces += len(faces)
                for face in faces:
                    decision, matched_id, similarity = _decide_identity(
                        embedder, frame, face, vectors, owners
                    )
                    buffered.append(FaceDetection(
                        job=job,
                        frame_index=frame_index,
                        timestamp_seconds=frame_index / fps,
                        x=face.x, y=face.y, width=face.w, height=face.h,
                        score=face.score,
                        matched_person_id=matched_id,
                        similarity=similarity,
                        decision=decision,
                    ))

                if len(buffered) >= BATCH:
                    FaceDetection.objects.bulk_create(buffered)
                    buffered.clear()

                if total_frames:
                    pct = max(1, min(99, int(frame_index * 100 / total_frames)))
                    if pct != last_progress and pct % 2 == 0:
                        _set_analysis_state(job, progress=pct)
                        last_progress = pct
                    if pct >= last_logged_progress + 10:
                        _append_analysis_log(job, f"Progress {pct}% - scanned frame {frame_index}, detected {total_faces} faces so far.")
                        last_logged_progress = pct

                frame_index += 1

            if buffered:
                FaceDetection.objects.bulk_create(buffered)
            _append_analysis_log(job, f"Detection pass finished. Persisted detections for {job.detections.count()} face instances.")

        finally:
            cap.release()

        counts = {row["decision"]: row["n"] for row in
                  job.detections.values("decision").annotate(n=models.Count("id"))}
        message = (
            f"Faces — allowed: {counts.get('allowed', 0)}, "
            f"blurred: {counts.get('unknown', 0)}, "
            f"uncertain: {counts.get('uncertain', 0)}, "
            f"conflict: {counts.get('conflict', 0)}."
        )

        job.status = AnalysisJob.Status.DONE
        job.progress = 100
        job.finished_at = timezone.now()
        job.message = message
        _append_analysis_log(job, message)
        job.save(update_fields=["progress", "status", "finished_at", "message"])
    except Exception as exc:
        job.status = AnalysisJob.Status.FAILED
        job.finished_at = timezone.now()
        job.message = str(exc)
        _append_analysis_log(job, f"Analysis failed: {exc}")
        job.save(update_fields=["status", "finished_at", "message"])
        raise


def _face_is_plausible(face) -> bool:
    """Reject obvious YuNet false positives — usually hands, ear-shells or
    fragments of skin that pass the score threshold but have wildly off
    aspect ratios or are too tiny to embed reliably."""
    if face.w < settings.FACE_MIN_SIZE_PX or face.h < settings.FACE_MIN_SIZE_PX:
        return False
    aspect = face.w / max(1, face.h)
    if aspect < settings.FACE_ASPECT_MIN or aspect > settings.FACE_ASPECT_MAX:
        return False
    return True


def _decide_identity(embedder, frame, face, vectors, owners):
    """Returns (decision, matched_person_id_or_None, similarity_or_None)."""

    import numpy as np

    if vectors.shape[0] == 0:
        return (FaceDetection.Decision.UNKNOWN, None, None)

    vec = embedder.embed(frame, face)
    sims = vectors @ vec  # both L2-normalised → cosine
    best = int(np.argmax(sims))
    best_sim = float(sims[best])

    threshold = settings.FACE_MATCH_THRESHOLD
    margin = settings.FACE_CONFLICT_MARGIN

    if best_sim < threshold:
        return (FaceDetection.Decision.UNKNOWN, None, best_sim)

    best_owner = owners[best]
    # Find best score per other person.
    per_person_best: dict[int, float] = {}
    for owner_id, sim in zip(owners, sims, strict=True):
        if sim > per_person_best.get(owner_id, -1.0):
            per_person_best[owner_id] = float(sim)
    runner_up = max(
        (sim for pid, sim in per_person_best.items() if pid != best_owner),
        default=-1.0,
    )
    if runner_up >= best_sim - margin and runner_up >= threshold:
        return (FaceDetection.Decision.CONFLICT, None, best_sim)

    return (FaceDetection.Decision.ALLOWED, best_owner, best_sim)


def render_final_export(export_id: int) -> None:
    """Render an MP4 with mocnego Gaussian blur on every blurred/uncertain/conflict face.

    Uses the latest completed AnalysisJob. Audio is re-muxed from the source.
    """

    import cv2
    import numpy as np

    export = ExportJob.objects.select_related("project__source").get(pk=export_id)
    export.status = ExportJob.Status.RUNNING
    export.started_at = timezone.now()
    export.progress = 1
    export.message = "Preparing final render."
    export.log = ""
    export.save(update_fields=["status", "started_at", "progress", "message", "log"])

    try:
        project = export.project
        source = project.source
        if not source:
            raise RuntimeError("Project has no source video.")
        runtime_error = gpu_requirement_error(source.video_codec or "")
        if runtime_error:
            raise RuntimeError(runtime_error)

        latest_job = (
            project.analysis_jobs
            .filter(status=AnalysisJob.Status.DONE)
            .order_by("-finished_at")
            .first()
        )
        if not latest_job:
            raise RuntimeError("Run analysis (completed) before exporting final.")
        caps = detect_runtime_capabilities()
        style = export.style or ExportJob.Style.MOSAIC
        gpu_styles = {ExportJob.Style.GAUSSIAN, ExportJob.Style.MOSAIC}
        use_gpu = caps.gpu_blur_ready and style in gpu_styles
        backend = "opencv-cuda" if use_gpu else "cpu"
        _append_export_log(export, f"Final export started for {source.original_filename}.")
        _append_export_log(export, f"Using analysis job #{latest_job.pk} finished at {latest_job.finished_at}.")
        _append_export_log(
            export,
            (
                f"Inference backend: {'OpenCV DNN CUDA' if caps.gpu_inference_ready else 'cpu'}. "
                f"Redaction style: {style} on {backend}."
            ),
        )
        if style in gpu_styles and not caps.gpu_blur_ready and caps.gpu_inference_ready:
            _append_export_log(
                export,
                "OpenCV exposes CUDA for DNN, but not the image operators required for GPU redaction — falling back to CPU.",
            )

        # Group detections by frame index for fast lookup during render.
        from collections import defaultdict
        per_frame: dict[int, list[BlurRegion]] = defaultdict(list)
        blur_decisions = {
            FaceDetection.Decision.UNKNOWN,
            FaceDetection.Decision.UNCERTAIN,
            FaceDetection.Decision.CONFLICT,
        }
        for det in latest_job.detections.select_related("override").iterator():
            effective = effective_detection_row(det)
            if effective.decision in blur_decisions:
                per_frame[det.frame_index].append(
                    BlurRegion(
                        x=effective.x,
                        y=effective.y,
                        width=effective.width,
                        height=effective.height,
                        score=effective.score,
                    )
                )
        fps_for_manual = source.fps or 30.0
        manual_regions = list(project.manual_blur_regions.all())
        for manual in manual_regions:
            start_frame = max(0, int(manual.start_seconds * fps_for_manual))
            end_frame = max(start_frame, int(manual.end_seconds * fps_for_manual))
            region = BlurRegion(
                x=manual.x,
                y=manual.y,
                width=manual.width,
                height=manual.height,
                score=1.0,
            )
            for frame_id in range(start_frame, end_frame + 1):
                per_frame[frame_id].append(region)
        raw_frames = len(per_frame)
        raw_regions = sum(len(v) for v in per_frame.values())
        _append_export_log(export, f"Loaded {raw_regions} blur regions across {raw_frames} frames.")
        if manual_regions:
            _append_export_log(export, f"Included {len(manual_regions)} manual blur segment(s).")
        per_frame = _densify_blur_regions(per_frame)
        dense_frames = len(per_frame)
        dense_regions = sum(len(v) for v in per_frame.values())
        if dense_frames != raw_frames or dense_regions != raw_regions:
            _append_export_log(
                export,
                f"Temporal safety fill expanded coverage to {dense_regions} regions across {dense_frames} frames.",
            )

        out_dir = Path(project.project_dir) / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_video = out_dir / f"final-{export.pk}.video.mp4"
        final_path = out_dir / f"final-{export.pk}.mp4"

        cap = cv2.VideoCapture(str(source.absolute_path))
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open source video.")
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or source.fps or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(tmp_video), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError("OpenCV could not open VideoWriter.")
            _append_export_log(export, f"Rendering {total_frames or 'unknown'} frames at {fps:.3f} fps.")

            try:
                frame_index = 0
                last_progress = -1
                last_logged_progress = -1
                prev_gray = None
                tracked_regions: list[dict] = []
                fallback_frames = 0
                fallback_regions = 0
                while True:
                    grabbed, frame = cap.read()
                    if not grabbed:
                        break
                    detected_regions = per_frame.get(frame_index, [])
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    fallback_only_regions: list[BlurRegion] = []
                    if prev_gray is not None and tracked_regions:
                        next_tracked_regions = []
                        for track in tracked_regions:
                            propagated = _propagate_region(prev_gray, gray, track["region"], width, height)
                            if propagated is None:
                                continue
                            if _region_overlaps(propagated, detected_regions, threshold=0.30):
                                continue
                            remaining_ttl = int(track["ttl"]) - 1
                            if remaining_ttl <= 0:
                                continue
                            next_tracked_regions.append({"region": propagated, "ttl": remaining_ttl})
                            fallback_only_regions.append(propagated)
                        tracked_regions = next_tracked_regions
                    regions = _merge_regions(detected_regions, fallback_only_regions)
                    if regions:
                        frame = _apply_redaction(frame, regions, style, use_gpu=use_gpu)
                    if fallback_only_regions:
                        fallback_frames += 1
                        fallback_regions += len(fallback_only_regions)
                    writer.write(frame)
                    tracked_regions = _refresh_tracked_regions(detected_regions, tracked_regions)
                    prev_gray = gray
                    if total_frames:
                        pct = max(1, min(97, int(frame_index * 97 / total_frames)))
                        if pct > last_progress and pct % 2 == 0:
                            _set_export_state(
                                export,
                                progress=pct,
                                message=f"Rendering redaction ({style})…",
                            )
                            last_progress = pct
                        if pct >= last_logged_progress + 10:
                            _append_export_log(export, f"Render progress {pct}% - frame {frame_index}.")
                            last_logged_progress = pct
                    frame_index += 1
            finally:
                writer.release()
            if fallback_frames:
                _append_export_log(
                    export,
                    f"Optical-flow fallback covered {fallback_regions} regions across {fallback_frames} frames.",
                )
        finally:
            cap.release()

        _set_export_state(export, progress=98, message="Muxing original audio…")
        _append_export_log(export, "Frame render finished. Muxing original audio.")
        _ffmpeg_mux_audio(tmp_video, source.absolute_path, final_path)
        tmp_video.unlink(missing_ok=True)

        export.output_path = str(final_path)
        export.size_bytes = final_path.stat().st_size
        export.status = ExportJob.Status.DONE
        export.progress = 100
        export.finished_at = timezone.now()
        export.message = f"Blurred {sum(len(v) for v in per_frame.values())} face regions across {len(per_frame)} frames."
        _append_export_log(export, export.message)
        _append_export_log(export, f"Final export ready: {final_path.name}.")
        export.save(update_fields=["output_path", "size_bytes", "status", "progress", "finished_at", "message"])
    except Exception as exc:
        export.status = ExportJob.Status.FAILED
        export.finished_at = timezone.now()
        export.message = str(exc)
        _append_export_log(export, f"Final export failed: {exc}")
        export.save(update_fields=["status", "finished_at", "message"])
        raise


def _region_bounds(det: BlurRegion, frame_width: int, frame_height: int) -> tuple[int, int, int, int] | None:
    # Asymmetric padding — top is generous so raised hair/hats are covered,
    # sides moderate for ears, bottom modest because chin is already inside.
    side = int(det.width * settings.BLUR_MARGIN_SIDE_FACTOR)
    top = int(det.height * settings.BLUR_MARGIN_TOP_FACTOR)
    bottom = int(det.height * settings.BLUR_MARGIN_BOTTOM_FACTOR)
    x0 = max(0, det.x - side)
    y0 = max(0, det.y - top)
    x1 = min(frame_width, det.x + det.width + side)
    y1 = min(frame_height, det.y + det.height + bottom)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _redaction_strength(short_edge: int) -> tuple[float, int]:
    """Scale redaction aggressiveness with face size.

    Small distant faces are already anonymised by modest blur/pixelation.
    Large close-ups need much heavier treatment or identity still leaks
    through facial structure.
    """
    normalized = max(0.0, min(1.0, (short_edge - 72) / 220.0))
    severity = 1.0 + normalized * 2.35
    extra_passes = int(round(normalized * 3.0))
    return severity, extra_passes


def _redact_patch(roi, style: str):
    """Returns a fully obfuscated copy of the ROI patch — works only on the
    extracted region so the kernel can never bleed dark pixels from outside."""
    import cv2
    import numpy as np

    h, w = roi.shape[:2]
    short_edge = max(1, min(h, w))
    severity, extra_passes = _redaction_strength(short_edge)

    if style == ExportJob.Style.SOLID:
        avg = roi.reshape(-1, roi.shape[2]).mean(axis=0).astype(np.uint8)
        out = np.empty_like(roi)
        out[:, :] = (avg * 0.25).astype(np.uint8)  # darkened average — opaque cover
        return out

    if style == ExportJob.Style.MOSAIC:
        # Large close-ups need much chunkier blocks than distant faces.
        target_block = max(12, int(round(12 * severity)))
        cells = max(3, short_edge // target_block)
        small = cv2.resize(roi, (cells, cells), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    if style == ExportJob.Style.EYE_BAR:
        out = roi.copy()
        # Eye band roughly 25% – 55% down the face box. Clamp inside the ROI.
        y0 = int(h * 0.22)
        y1 = int(h * 0.58)
        if y1 > y0:
            out[y0:y1, :] = 0
        return out

    # Default: gaussian — very strong, with replicated-border padding so the
    # edges don't get pulled toward the dark frame outside the ROI.
    pad = max(8, int(short_edge * (0.22 + 0.08 * severity)))
    padded = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    k = max(41, int((short_edge / 1.55) * severity) | 1)
    k = min(k, 251)
    blurred = cv2.GaussianBlur(padded, (k, k), 0)
    for _ in range(1 + extra_passes):
        blurred = cv2.GaussianBlur(blurred, (k, k), 0)
    return blurred[pad : pad + h, pad : pad + w]


def _redact_patch_gpu(roi_gpu, style: str, gaussian_filters: dict):
    """GPU equivalent of _redact_patch for gaussian + mosaic styles. Returns a
    new GpuMat. Other styles are not GPU-accelerated and should fall back to
    CPU at the caller."""
    import cv2

    rw, rh = roi_gpu.size()
    short_edge = max(1, min(rh, rw))
    severity, extra_passes = _redaction_strength(short_edge)

    if style == ExportJob.Style.MOSAIC:
        target_block = max(12, int(round(12 * severity)))
        cells = max(3, short_edge // target_block)
        small = cv2.cuda.resize(roi_gpu, (cells, cells), interpolation=cv2.INTER_AREA)
        return cv2.cuda.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)

    # gaussian
    pad = max(8, int(short_edge * (0.22 + 0.08 * severity)))
    padded = cv2.cuda.copyMakeBorder(roi_gpu, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    k = max(41, int((short_edge / 1.55) * severity) | 1)
    k = min(k, 251)
    # cuda gaussian filter caps at 32; chain shorter passes to approximate.
    gpu_kernel = min(k, 31) | 1
    passes = max(2, (k + gpu_kernel - 1) // gpu_kernel) + extra_passes
    filter_key = (int(padded.type()), gpu_kernel)
    flt = gaussian_filters.get(filter_key)
    if flt is None:
        sigma = max(1.0, gpu_kernel / 3.0)
        flt = cv2.cuda.createGaussianFilter(
            int(padded.type()), -1, (gpu_kernel, gpu_kernel), sigma
        )
        gaussian_filters[filter_key] = flt
    current = padded
    for _ in range(passes):
        current = flt.apply(current)
    pw, ph = current.size()
    return current.rowRange(pad, pad + rh).colRange(pad, pad + rw)


def _apply_redaction(frame, regions, style: str, use_gpu: bool = False):
    """Composite redacted patches into the frame. Eye-bar and solid styles get
    a hard rectangular paste; blur/mosaic get a tight elliptical feather so the
    very corners don't show a square halo, but the face core is fully covered.

    When use_gpu=True and the style supports it, the heavy per-patch work runs
    on CUDA and only the elliptical alpha-blend stays on CPU."""
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    out = frame.copy()
    needs_feather = style in {ExportJob.Style.MOSAIC, ExportJob.Style.GAUSSIAN}

    frame_gpu = None
    gaussian_filters: dict = {}
    if use_gpu:
        frame_gpu = cv2.cuda_GpuMat()
        frame_gpu.upload(out)

    for det in regions:
        bounds = _region_bounds(det, w, h)
        if bounds is None:
            continue
        x0, y0, x1, y1 = bounds
        rw_b = x1 - x0
        rh_b = y1 - y0
        if rw_b <= 0 or rh_b <= 0:
            continue

        if use_gpu:
            roi_gpu = frame_gpu.rowRange(y0, y1).colRange(x0, x1)
            redacted_gpu = _redact_patch_gpu(roi_gpu, style, gaussian_filters)
            redacted = redacted_gpu.download()
            roi = out[y0:y1, x0:x1]
        else:
            roi = out[y0:y1, x0:x1]
            redacted = _redact_patch(roi, style)

        if not needs_feather:
            out[y0:y1, x0:x1] = redacted
            continue

        rh, rw = roi.shape[:2]
        mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.ellipse(
            mask,
            (rw // 2, rh // 2),
            (max(1, int(rw * 0.55)), max(1, int(rh * 0.62))),
            0, 0, 360, 255, -1,
        )
        feather = max(5, int(min(rh, rw) * 0.08)) | 1
        soft = cv2.GaussianBlur(mask, (feather, feather), 0).astype(np.float32) / 255.0
        soft = soft[:, :, None]
        out[y0:y1, x0:x1] = (redacted.astype(np.float32) * soft + roi.astype(np.float32) * (1.0 - soft)).astype(np.uint8)

    return out


def _densify_blur_regions(per_frame: dict[int, list[BlurRegion]]) -> dict[int, list[BlurRegion]]:
    dense = {frame_index: list(regions) for frame_index, regions in per_frame.items()}
    gap = max(0, int(settings.BLUR_TEMPORAL_GAP))
    if gap <= 0 or not per_frame:
        return dense

    frame_ids = sorted(per_frame)
    for index, frame_id in enumerate(frame_ids[:-1]):
        next_frame_id = frame_ids[index + 1]
        delta = next_frame_id - frame_id
        if delta <= 1 or delta > gap + 1:
            continue
        previous_regions = per_frame[frame_id]
        next_regions = per_frame[next_frame_id]
        if not previous_regions or not next_regions:
            continue
        for missing_frame in range(frame_id + 1, next_frame_id):
            alpha = (missing_frame - frame_id) / delta
            interpolated_regions: list[BlurRegion] = []
            matched_pairs = _match_regions(previous_regions, next_regions)
            for left, right in matched_pairs:
                interpolated_regions.append(
                    BlurRegion(
                        x=int(round(left.x + (right.x - left.x) * alpha)),
                        y=int(round(left.y + (right.y - left.y) * alpha)),
                        width=int(round(left.width + (right.width - left.width) * alpha)),
                        height=int(round(left.height + (right.height - left.height) * alpha)),
                        score=min(left.score, right.score),
                    )
                )
            if interpolated_regions:
                dense.setdefault(missing_frame, []).extend(interpolated_regions)

    return {frame_id: _dedupe_regions(regions) for frame_id, regions in dense.items()}


def _match_regions(left_regions: list[BlurRegion], right_regions: list[BlurRegion]) -> list[tuple[BlurRegion, BlurRegion]]:
    candidates: list[tuple[float, int, int]] = []
    for left_index, left in enumerate(left_regions):
        for right_index, right in enumerate(right_regions):
            iou = _region_iou(left, right)
            center_distance = _region_center_distance(left, right)
            max_extent = max(left.width, left.height, right.width, right.height, 1)
            if iou <= 0 and center_distance > max_extent * 1.25:
                continue
            score = iou - (center_distance / max_extent) * 0.1
            candidates.append((score, left_index, right_index))
    candidates.sort(reverse=True)

    matched_left: set[int] = set()
    matched_right: set[int] = set()
    matches: list[tuple[BlurRegion, BlurRegion]] = []
    for _, left_index, right_index in candidates:
        if left_index in matched_left or right_index in matched_right:
            continue
        matched_left.add(left_index)
        matched_right.add(right_index)
        matches.append((left_regions[left_index], right_regions[right_index]))
    return matches


def _refresh_tracked_regions(
    detected_regions: list[BlurRegion],
    propagated_tracks: list[dict],
) -> list[dict]:
    refreshed = list(propagated_tracks)
    for region in detected_regions:
        refreshed.append({"region": region, "ttl": max(1, int(settings.BLUR_TRACK_TTL))})
    return _dedupe_tracks(refreshed)


def _dedupe_tracks(tracks: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    for track in sorted(tracks, key=lambda item: int(item["ttl"]), reverse=True):
        region = track["region"]
        if _region_overlaps(region, [item["region"] for item in deduped], threshold=0.55):
            continue
        deduped.append(track)
    return deduped


def _merge_regions(primary: list[BlurRegion], secondary: list[BlurRegion]) -> list[BlurRegion]:
    merged = list(primary)
    for region in secondary:
        if _region_overlaps(region, merged, threshold=0.35):
            continue
        merged.append(region)
    return _dedupe_regions(merged)


def _dedupe_regions(regions: list[BlurRegion]) -> list[BlurRegion]:
    deduped: list[BlurRegion] = []
    for region in sorted(regions, key=lambda item: item.score, reverse=True):
        if _region_overlaps(region, deduped, threshold=0.55):
            continue
        deduped.append(region)
    return deduped


def _region_overlaps(region: BlurRegion, others: list[BlurRegion], threshold: float) -> bool:
    return any(_region_iou(region, other) >= threshold for other in others)


def _region_iou(left: BlurRegion, right: BlurRegion) -> float:
    left_x1 = left.x + left.width
    left_y1 = left.y + left.height
    right_x1 = right.x + right.width
    right_y1 = right.y + right.height

    inter_x0 = max(left.x, right.x)
    inter_y0 = max(left.y, right.y)
    inter_x1 = min(left_x1, right_x1)
    inter_y1 = min(left_y1, right_y1)
    inter_w = max(0, inter_x1 - inter_x0)
    inter_h = max(0, inter_y1 - inter_y0)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0
    left_area = max(1, left.width * left.height)
    right_area = max(1, right.width * right.height)
    union = left_area + right_area - intersection
    return intersection / max(1, union)


def _region_center_distance(left: BlurRegion, right: BlurRegion) -> float:
    left_cx = left.x + left.width / 2
    left_cy = left.y + left.height / 2
    right_cx = right.x + right.width / 2
    right_cy = right.y + right.height / 2
    dx = left_cx - right_cx
    dy = left_cy - right_cy
    return float((dx * dx + dy * dy) ** 0.5)


def _propagate_region(prev_gray, gray, region: BlurRegion, frame_width: int, frame_height: int) -> BlurRegion | None:
    import cv2

    x0 = max(0, region.x)
    y0 = max(0, region.y)
    x1 = min(frame_width, region.x + region.width)
    y1 = min(frame_height, region.y + region.height)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None

    prev_roi = prev_gray[y0:y1, x0:x1]
    if prev_roi.size == 0:
        return None

    # Wider search radius so the tracker survives fast head whips between
    # detection hits.
    search_factor = settings.BLUR_TRACK_SEARCH_FACTOR
    search_margin_x = max(6, int(region.width * search_factor))
    search_margin_y = max(6, int(region.height * search_factor))
    sx0 = max(0, x0 - search_margin_x)
    sy0 = max(0, y0 - search_margin_y)
    sx1 = min(frame_width, x1 + search_margin_x)
    sy1 = min(frame_height, y1 + search_margin_y)
    search_roi = gray[sy0:sy1, sx0:sx1]

    if (
        search_roi.shape[0] < prev_roi.shape[0]
        or search_roi.shape[1] < prev_roi.shape[1]
    ):
        return None

    result = cv2.matchTemplate(search_roi, prev_roi, cv2.TM_CCOEFF_NORMED)
    _, max_score, _, max_loc = cv2.minMaxLoc(result)
    if max_score < settings.BLUR_TRACK_MATCH_THRESHOLD:
        return None

    new_x = sx0 + int(max_loc[0])
    new_y = sy0 + int(max_loc[1])
    new_x = max(0, min(new_x, max(0, frame_width - region.width)))
    new_y = max(0, min(new_y, max(0, frame_height - region.height)))
    return BlurRegion(
        x=int(new_x),
        y=int(new_y),
        width=region.width,
        height=region.height,
        score=min(region.score, float(max_score)),
    )


def delete_project_storage(project: Project) -> None:
    path = Path(settings.STORAGE_ROOT) / "projects" / project.slug
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
