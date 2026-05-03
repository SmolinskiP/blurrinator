from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache

from django.conf import settings


@dataclass(frozen=True)
class RuntimeCapabilities:
    nvidia_smi_available: bool
    ffmpeg_hwaccels: tuple[str, ...]
    ffmpeg_decoders: tuple[str, ...]
    opencv_cuda_devices: int
    opencv_cuda_image_ops: bool

    @property
    def ffmpeg_has_cuda_hwaccel(self) -> bool:
        return any(name in {"cuda", "nvdec"} for name in self.ffmpeg_hwaccels)

    @property
    def ffmpeg_has_av1_gpu_decoder(self) -> bool:
        return any(
            name in {"av1_cuvid", "av1_nvdec"}
            for name in self.ffmpeg_decoders
        )

    @property
    def opencv_has_cuda(self) -> bool:
        return self.opencv_cuda_devices > 0

    @property
    def gpu_inference_ready(self) -> bool:
        return self.nvidia_smi_available and self.opencv_has_cuda

    @property
    def gpu_blur_ready(self) -> bool:
        return self.gpu_inference_ready and self.opencv_cuda_image_ops

    @property
    def blur_backend(self) -> str:
        return "opencv-cuda" if self.gpu_blur_ready else "cpu"


@lru_cache(maxsize=1)
def detect_runtime_capabilities() -> RuntimeCapabilities:
    return RuntimeCapabilities(
        nvidia_smi_available=shutil.which("nvidia-smi") is not None,
        ffmpeg_hwaccels=_read_ffmpeg_hwaccels(),
        ffmpeg_decoders=_read_ffmpeg_decoders(),
        opencv_cuda_devices=_read_opencv_cuda_devices(),
        opencv_cuda_image_ops=_has_opencv_cuda_image_ops(),
    )


def gpu_requirement_error(video_codec: str = "") -> str | None:
    if not settings.REQUIRE_GPU_PIPELINE:
        return None

    caps = detect_runtime_capabilities()
    if caps.gpu_inference_ready:
        return None

    reasons: list[str] = []
    if not caps.nvidia_smi_available:
        reasons.append("`nvidia-smi` is not available")
    if not caps.opencv_has_cuda:
        reasons.append("OpenCV was built without CUDA support")

    if not reasons:
        reasons.append("GPU inference requirements are not satisfied")

    return (
        "GPU inference is required, but the current runtime is not ready: "
        + "; ".join(reasons)
        + ". Install a CUDA-enabled OpenCV build, then restart the web app and `qcluster`."
    )


def _read_ffmpeg_hwaccels() -> tuple[str, ...]:
    try:
        completed = subprocess.run(
            [settings.FFMPEG_BINARY, "-hide_banner", "-hwaccels"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return ()

    names: list[str] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line or line.endswith(":"):
            continue
        names.append(line)
    return tuple(names)


def _read_ffmpeg_decoders() -> tuple[str, ...]:
    try:
        completed = subprocess.run(
            [settings.FFMPEG_BINARY, "-hide_banner", "-decoders"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return ()

    names: list[str] = []
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            names.append(parts[1])
    return tuple(names)


def _read_opencv_cuda_devices() -> int:
    try:
        import cv2
    except Exception:
        return 0

    cuda = getattr(cv2, "cuda", None)
    if cuda is None:
        return 0
    try:
        return int(cuda.getCudaEnabledDeviceCount())
    except Exception:
        return 0


def _has_opencv_cuda_image_ops() -> bool:
    try:
        import cv2
    except Exception:
        return False

    cuda = getattr(cv2, "cuda", None)
    if cuda is None:
        return False

    required_symbols = (
        "GpuMat",
        "resize",
        "cvtColor",
        "createGaussianFilter",
    )
    return all(hasattr(cuda, symbol) for symbol in required_symbols)
