#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/venv/bin/python}"
BUILD_DIR="${BUILD_DIR:-$ROOT/.build/opencv-build-gpublur}"
OPENCV_SRC="${OPENCV_SRC:-$ROOT/.build/opencv}"
OPENCV_CONTRIB_SRC="${OPENCV_CONTRIB_SRC:-$ROOT/.build/opencv_contrib}"
VENV_PREFIX="${VENV_PREFIX:-$ROOT/venv}"
REPO_CV2_DIR="${REPO_CV2_DIR:-$ROOT/cv2}"
CC_BIN="${CC_BIN:-/usr/bin/gcc-12}"
CXX_BIN="${CXX_BIN:-/usr/bin/g++-12}"
CUDA_ARCH_BIN="${CUDA_ARCH_BIN:-9.0}"
CUDA_ARCH_PTX="${CUDA_ARCH_PTX:-9.0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$OPENCV_SRC" || ! -d "$OPENCV_CONTRIB_SRC" ]]; then
  echo "OpenCV sources not found under .build/" >&2
  exit 1
fi

if [[ ! -x "$CC_BIN" || ! -x "$CXX_BIN" ]]; then
  echo "CUDA 12.0 needs a supported host compiler. Missing: $CC_BIN or $CXX_BIN" >&2
  exit 1
fi

PYTHON_VERSION="$("$PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
SITE_PACKAGES="$("$PYTHON_BIN" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["platlib"])
PY
)"
NUMPY_INCLUDE="$("$PYTHON_BIN" - <<'PY'
import numpy as np
print(np.get_include())
PY
)"
PYTHON_INCLUDE="$("$PYTHON_BIN" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("INCLUDEPY"))
PY
)"
PYTHON_LIBRARY="$("$PYTHON_BIN" - <<'PY'
import sysconfig
from pathlib import Path
libdir = Path(sysconfig.get_config_var("LIBDIR"))
libname = sysconfig.get_config_var("LDLIBRARY")
print(libdir / libname)
PY
)"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

CC="$CC_BIN" CXX="$CXX_BIN" CUDAHOSTCXX="$CXX_BIN" \
cmake -S "$OPENCV_SRC" -B "$BUILD_DIR" -G Ninja \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX="$VENV_PREFIX" \
  -D CMAKE_IGNORE_PREFIX_PATH=/home/linuxbrew/.linuxbrew \
  -D CMAKE_C_COMPILER="$CC_BIN" \
  -D CMAKE_CXX_COMPILER="$CXX_BIN" \
  -D CMAKE_CUDA_HOST_COMPILER="$CXX_BIN" \
  -D OPENCV_EXTRA_MODULES_PATH="$OPENCV_CONTRIB_SRC/modules" \
  -D WITH_CUDA=ON \
  -D WITH_CUDNN=ON \
  -D OPENCV_DNN_CUDA=ON \
  -D WITH_CUBLAS=ON \
  -D ENABLE_FAST_MATH=ON \
  -D CUDA_FAST_MATH=ON \
  -D WITH_NVCUVID=ON \
  -D WITH_NVCUVENC=ON \
  -D WITH_GTK=OFF \
  -D WITH_QT=OFF \
  -D WITH_GSTREAMER=OFF \
  -D WITH_OPENGL=OFF \
  -D CUDA_ARCH_BIN="$CUDA_ARCH_BIN" \
  -D CUDA_ARCH_PTX="$CUDA_ARCH_PTX" \
  -D BUILD_LIST=core,imgproc,imgcodecs,videoio,dnn,objdetect,python3,cudev,cudaarithm,cudafilters,cudaimgproc,cudawarping \
  -D BUILD_opencv_python3=ON \
  -D BUILD_opencv_python_bindings_generator=ON \
  -D BUILD_opencv_python2=OFF \
  -D BUILD_opencv_world=OFF \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D BUILD_opencv_apps=OFF \
  -D BUILD_JAVA=OFF \
  -D BUILD_opencv_java=OFF \
  -D WITH_LAPACK=OFF \
  -D WITH_EIGEN=OFF \
  -D PYTHON_DEFAULT_EXECUTABLE="$PYTHON_BIN" \
  -D PYTHON3_EXECUTABLE="$PYTHON_BIN" \
  -D PYTHON3_INCLUDE_DIR="$PYTHON_INCLUDE" \
  -D PYTHON3_LIBRARY="$PYTHON_LIBRARY" \
  -D PYTHON3_PACKAGES_PATH="$SITE_PACKAGES" \
  -D PYTHON3_NUMPY_INCLUDE_DIRS="$NUMPY_INCLUDE"

cmake --build "$BUILD_DIR" --parallel
install_status=0
cmake --install "$BUILD_DIR" || install_status=$?
if [[ "$install_status" -ne 0 ]]; then
  echo "cmake --install hit the root-owned cv2 package in site-packages; continuing with repo-local cv2 overlay" >&2
fi

rm -rf "$REPO_CV2_DIR"
mkdir -p "$REPO_CV2_DIR/python-$PYTHON_VERSION"
cp -R "$BUILD_DIR/python_loader/cv2/." "$REPO_CV2_DIR/"
cp -R "$BUILD_DIR/modules/python_bindings_generator/cv2/." "$REPO_CV2_DIR/"
cp "$BUILD_DIR/lib/python3/cv2.cpython-"*"-x86_64-linux-gnu.so" "$REPO_CV2_DIR/python-$PYTHON_VERSION/"

PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/scripts/validate_opencv_cuda.py"
