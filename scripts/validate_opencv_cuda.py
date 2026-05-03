#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2


REQUIRED_CUDA_FUNCS = (
    "resize",
    "createGaussianFilter",
)
REQUIRED_CUDA_MODULES = (
    "cudaarithm",
    "cudafilters",
    "cudaimgproc",
    "cudawarping",
)


def _build_list_from_cache(cache_path: Path) -> str | None:
    if not cache_path.exists():
        return None
    for line in cache_path.read_text().splitlines():
        if line.startswith("BUILD_LIST:STRING="):
            return line.split("=", 1)[1].strip()
    return None


def _missing_disabled_modules(build_info: str) -> list[str]:
    match = re.search(r"Disabled by dependency:\s+([^\n]+)", build_info)
    if not match:
        return []
    disabled = set(match.group(1).split())
    return [name for name in REQUIRED_CUDA_MODULES if name in disabled]


def main() -> int:
    cuda = getattr(cv2, "cuda", None)
    if cuda is None:
        print("FAIL: cv2.cuda namespace is missing")
        return 1

    errors: list[str] = []
    device_count = 0
    try:
        device_count = int(cuda.getCudaEnabledDeviceCount())
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"cv2.cuda.getCudaEnabledDeviceCount() failed: {exc}")

    if device_count <= 0:
        errors.append("no CUDA device detected by OpenCV")

    for name in REQUIRED_CUDA_FUNCS:
        if not hasattr(cuda, name):
            errors.append(f"cv2.cuda.{name} is missing")

    if not hasattr(cv2, "cuda_GpuMat"):
        errors.append("cv2.cuda_GpuMat is missing")

    build_info = cv2.getBuildInformation()
    if "NVIDIA CUDA:                   YES" not in build_info:
        errors.append("OpenCV build info does not report NVIDIA CUDA support")

    disabled_modules = _missing_disabled_modules(build_info)
    if disabled_modules:
        errors.append(
            "OpenCV disabled required CUDA modules by dependency: "
            + ", ".join(disabled_modules)
        )

    cache_candidates = [
        ROOT / ".build" / "opencv-build-gpublur" / "CMakeCache.txt",
        ROOT / ".build" / "opencv-build" / "CMakeCache.txt",
        ROOT / ".build" / "opencv-build-headless" / "CMakeCache.txt",
    ]
    build_list = next(
        (value for path in cache_candidates if (value := _build_list_from_cache(path))),
        None,
    )
    if build_list:
        missing_from_build_list = [
            name for name in REQUIRED_CUDA_MODULES if name not in build_list.split(",")
        ]
        if missing_from_build_list:
            errors.append(
                "BUILD_LIST omits required CUDA modules: "
                + ", ".join(missing_from_build_list)
            )

    print(f"cv2 {cv2.__version__}")
    print(f"cv2 file: {cv2.__file__}")
    print(f"cuda devices: {device_count}")
    print(f"cuda resize: {hasattr(cuda, 'resize')}")
    print(f"cuda gaussian: {hasattr(cuda, 'createGaussianFilter')}")
    if build_list:
        print(f"BUILD_LIST={build_list}")

    if errors:
        print("FAIL:")
        for item in errors:
            print(f"- {item}")
        return 1

    print("OK: OpenCV CUDA runtime is ready for GPU blur")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
