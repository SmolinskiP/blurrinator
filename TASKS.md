# Blurrinator - tasklist

Living checklist. Updated as work progresses. Source of truth for scope is [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## Phase 1 — Foundation

- [x] Strip FastAPI/Docker scaffolding from previous attempt
- [x] Django 6 project + apps `projects`, `registry`, `allowlist`
- [x] SQLite + Django-Q2 in-DB queue (no Redis)
- [x] Models: `Project`, `SourceVideo`, `AnalysisJob`, `ExportJob`, `ModelEntry`, `AllowedPerson`, `EnrollmentImage`
- [x] Upload form: streaming SHA-256, FFprobe metadata, stored under `storage/projects/<slug>/`
- [x] Draft review workflow after analysis with manual blur overrides
- [x] Polished dark UI (one hand-rolled CSS file, no build step)
- [x] Project list, project detail, allowlist index, registry index pages
- [x] Worker stub for analysis (records "phase 1 stub")
- [x] README + `pyproject.toml`

## Phase 2 — Faces and allowlist

### Models and infrastructure
- [x] `opencv-python` + `numpy` dependencies
- [x] Management command `fetch_models` — download YuNet + SFace ONNX into `storage/models/`, hash, register in `ModelEntry`
- [x] `FaceDetection` model: per-frame face box, score, matched person, similarity, decision
- [x] `FaceEmbedding` model on allowlist: vector blob + model version + detector score
- [x] `EnrollmentImage.status` (accepted / rejected) + rejection reason

### Detection + recognition pipeline
- [x] `face_detector.py` wrapper around YuNet (load ONNX once, infer on a frame, return boxes + landmarks + scores)
- [x] `face_embedder.py` wrapper around SFace (align by landmarks, return 128-D L2-normalised embedding)
- [x] Frame iterator using OpenCV `VideoCapture` with frame index + timestamp mapping
- [x] `run_analysis_job` — iterate frames, detect faces, embed, match against allowlist, persist `FaceDetection` rows
- [x] Cosine-similarity matching with conservative threshold + conflict detection across allowlist

### Allowlist enrollment
- [x] Enrollment view (person create + multi-image upload) running YuNet+SFace
- [x] Reject samples with: zero faces, multiple faces, low score, low resolution
- [x] Per-person embedding count and accepted/rejected gallery

### Render
- [x] Final-blur export: stream frames through OpenCV, apply mocnego Gaussian blur on unknown-face regions, mux original audio
- [x] Margin + soft elliptical feathering on blur masks
- [x] Temporal fallback: interpolate short gaps + optical-flow tracking to keep blur on missed frames
- [ ] Manifest JSON next to export with model versions and decision summary

### UI
- [x] Project detail: per-job decision summary in job message
- [x] Project detail: live polling status, expandable visual logs, source/final video preview
- [x] Draft review screen: inspect source video, seek detections, override blur decisions and boxes
- [x] Upload form: visual progress, transfer speed, ETA, submit lock during upload
- [x] Allowlist: image gallery with quality + accepted/rejected status
- [ ] Registry: visual grouping with download status (✓ if file present + hash matches)

### End-to-end check
- [x] Lena enrolled → analysis matches at sim ≈ 0.99 → 0 frames blurred
- [x] Lena removed → all 30 frames blurred → output std drops in face ROI

## Phase 3 — Person + head fallback

- [ ] Select permissive-license person detector or explicitly revisit Ultralytics licensing
- [ ] Person tracking
- [ ] Pose/head-region model for head estimation when no face is visible
- [ ] Track association face ↔ person
- [ ] Review flags for fallback-head regions

## Phase 4+ — see IMPLEMENTATION_PLAN.md sections "Faza 4..6"
