# Blurrinator

Local Django web app for privacy redaction in YouTube videos. Originals never leave the workstation. Licensed AGPL-3.0 because the planned detection stack uses Ultralytics YOLO.

Current slice: project upload, SHA-256 + FFprobe metadata, model registry, allowlist scaffold, in-DB job queue, completed face analysis, manual draft review overrides, and final redaction export.

## Setup

Requires Python 3.12+ and `ffmpeg` / `ffprobe` on `PATH`. GPU (NVIDIA + CUDA) is required for YuNet/SFace inference. `cv2` must come from a CUDA-enabled OpenCV build prepared on the host; installing stock `opencv-python` is not sufficient.

```bash
python -m venv venv
# Make sure a CUDA-enabled OpenCV Python build is already installed in this venv.
venv/bin/pip install -e .
venv/bin/python manage.py migrate
venv/bin/python manage.py createsuperuser
```

## Run

Two processes — web and worker. Open a terminal each:

```bash
venv/bin/python manage.py runserver 0.0.0.0:8000
venv/bin/python manage.py qcluster
```

Open <http://localhost:8000/>.

## Configuration

Environment variables (all optional):

| Variable | Default | Purpose |
| --- | --- | --- |
| `BLURRINATOR_SECRET_KEY` | dev placeholder | Set before exposing publicly. |
| `BLURRINATOR_DEBUG` | `1` | Set to `0` for production-style serving. |
| `BLURRINATOR_ALLOWED_HOSTS` | `*` | Comma-separated host list. |
| `BLURRINATOR_STORAGE_ROOT` | `./storage` | Originals, exports and analysis artifacts. |
| `BLURRINATOR_MAX_UPLOAD_BYTES` | `85899345920` (80 GiB) | Hard upload ceiling. |
| `BLURRINATOR_FFPROBE_BINARY` | `ffprobe` | Override if not on PATH. |
| `BLURRINATOR_FFMPEG_BINARY` | `ffmpeg` | Override if not on PATH. |
| `BLURRINATOR_REQUIRE_GPU_PIPELINE` | `1` | Refuse analysis/export unless runtime has CUDA-ready OpenCV for GPU inference. |

## GPU requirement

This project now fails fast when the runtime is not actually ready for GPU inference.
Having an NVIDIA card alone is not enough:

- OpenCV must expose CUDA devices (`cv2.cuda.getCudaEnabledDeviceCount() > 0`)

FFmpeg hardware decode is optional in the current pipeline. Video decode can stay on CPU; the critical part is that YuNet/SFace inference runs through CUDA-enabled OpenCV.

Current limitation:

- Face analysis runs on OpenCV DNN CUDA when available.
- Final blur compositing uses GPU only when the OpenCV Python binding exposes `cv2.cuda.resize` and `cv2.cuda.createGaussianFilter`.
- The export job log now reports which blur backend is actually used.

## Rebuild OpenCV For GPU Blur

The local issue was caused by an OpenCV `BUILD_LIST` that included `dnn` and `cudev`, but omitted the CUDA image modules needed for blur. Rebuild with:

```bash
./scripts/rebuild_opencv_cuda.sh
```

That script configures OpenCV from the checked-out sources under `.build/` with:

- `BUILD_LIST=core,imgproc,imgcodecs,videoio,dnn,objdetect,python3,cudev,cudaarithm,cudafilters,cudaimgproc,cudawarping`
- `WITH_CUDA=ON`
- `WITH_CUDNN=ON`
- `OPENCV_DNN_CUDA=ON`
- `WITH_GTK=OFF`, `WITH_GSTREAMER=OFF`, `WITH_QT=OFF`, `WITH_OPENGL=OFF` to avoid GTK/GIO conflicts with Homebrew Python
- `CUDA_ARCH_BIN=9.0` and `CUDA_ARCH_PTX=9.0` by default, so CUDA 12.0 can JIT PTX on newer Blackwell GPUs even though `sm_120` is not supported by that toolkit yet

The script also stages a repo-local [cv2](/home/patryk/blurrinator/cv2) package. This is intentional: the venv already contains a root-owned `site-packages/cv2`, so the project shadows it from the repo root instead of trying to overwrite it.

To validate the result without rebuilding:

```bash
./venv/bin/python scripts/validate_opencv_cuda.py
```

## Layout

```
blurrinator/        Django project (settings, urls, context)
projects/           Project, SourceVideo, AnalysisJob, ExportJob + pipeline services
registry/           Model registry (license + hash bookkeeping)
allowlist/          AllowedPerson + EnrollmentImage
templates/          Hand-rolled UI templates
static/css/app.css  Single CSS file, no build step
storage/            Local artifact tree (gitignored)
```
