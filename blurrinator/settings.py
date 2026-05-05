from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get(
    "BLURRINATOR_SECRET_KEY",
    "dev-insecure-change-me-before-exposing-publicly",
)
DEBUG = os.environ.get("BLURRINATOR_DEBUG", "1") == "1"
ALLOWED_HOSTS = os.environ.get("BLURRINATOR_ALLOWED_HOSTS", "*").split(",")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.humanize",
    "django_q",
    "users",
    "projects",
    "registry",
    "allowlist",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "users.middleware.RequireLoginMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "blurrinator.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "blurrinator.context.navigation",
            ],
        },
    },
]

WSGI_APPLICATION = "blurrinator.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
        "OPTIONS": {"timeout": 30},
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Europe/Warsaw"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

STORAGE_ROOT = Path(os.environ.get("BLURRINATOR_STORAGE_ROOT", BASE_DIR / "storage"))
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

MEDIA_ROOT = STORAGE_ROOT / "media"
MEDIA_URL = "/media/"

# Cosine-similarity threshold for SFace matches. SFace cosine of ~0.363 is the
# OpenCV recommended threshold; 0.40 keeps a privacy margin without causing as
# many frame-by-frame identity drops on real footage.
FACE_MATCH_THRESHOLD = float(os.environ.get("BLURRINATOR_FACE_MATCH_THRESHOLD", "0.40"))
FACE_MATCH_CANDIDATE_THRESHOLD = float(os.environ.get("BLURRINATOR_FACE_MATCH_CANDIDATE_THRESHOLD", "0.34"))
FACE_CONFLICT_MARGIN = float(os.environ.get("BLURRINATOR_FACE_CONFLICT_MARGIN", "0.05"))
FACE_DETECTOR_SCORE = float(os.environ.get("BLURRINATOR_FACE_DETECTOR_SCORE", "0.62"))
FACE_MIN_SIZE_PX = int(os.environ.get("BLURRINATOR_FACE_MIN_SIZE_PX", "24"))
FACE_ASPECT_MIN = float(os.environ.get("BLURRINATOR_FACE_ASPECT_MIN", "0.65"))
FACE_ASPECT_MAX = float(os.environ.get("BLURRINATOR_FACE_ASPECT_MAX", "1.55"))
FACE_LANDMARK_BOX_PADDING = float(os.environ.get("BLURRINATOR_FACE_LANDMARK_BOX_PADDING", "0.08"))
FACE_LANDMARK_MIN_VERTICAL_SPAN = float(os.environ.get("BLURRINATOR_FACE_LANDMARK_MIN_VERTICAL_SPAN", "0.18"))
FACE_LANDMARK_MIN_EYE_SPAN = float(os.environ.get("BLURRINATOR_FACE_LANDMARK_MIN_EYE_SPAN", "0.16"))
FACE_LANDMARK_MIN_MOUTH_SPAN = float(os.environ.get("BLURRINATOR_FACE_LANDMARK_MIN_MOUTH_SPAN", "0.12"))
FACE_LANDMARK_NOSE_MOUTH_TOLERANCE = float(os.environ.get("BLURRINATOR_FACE_LANDMARK_NOSE_MOUTH_TOLERANCE", "0.12"))
ANALYSIS_FRAME_STRIDE = int(os.environ.get("BLURRINATOR_FRAME_STRIDE", "1"))
FACE_TRACK_MAX_GAP = int(os.environ.get("BLURRINATOR_FACE_TRACK_MAX_GAP", "8"))
FACE_TRACK_MIN_ALLOWED_VOTES = int(os.environ.get("BLURRINATOR_FACE_TRACK_MIN_ALLOWED_VOTES", "2"))
FACE_TRACK_MIN_ALLOWED_RATIO = float(os.environ.get("BLURRINATOR_FACE_TRACK_MIN_ALLOWED_RATIO", "0.35"))
FACE_TRACK_MIN_CANDIDATE_RATIO = float(os.environ.get("BLURRINATOR_FACE_TRACK_MIN_CANDIDATE_RATIO", "0.60"))
BLUR_TEMPORAL_GAP = int(os.environ.get("BLURRINATOR_BLUR_TEMPORAL_GAP", "5"))
BLUR_MARGIN_FACTOR = float(os.environ.get("BLURRINATOR_BLUR_MARGIN_FACTOR", "0.45"))
# Asymmetric padding so the bbox actually covers hair (top), ears (sides) and chin (bottom).
BLUR_MARGIN_TOP_FACTOR = float(os.environ.get("BLURRINATOR_BLUR_MARGIN_TOP_FACTOR", "1.10"))
BLUR_MARGIN_SIDE_FACTOR = float(os.environ.get("BLURRINATOR_BLUR_MARGIN_SIDE_FACTOR", "0.55"))
BLUR_MARGIN_BOTTOM_FACTOR = float(os.environ.get("BLURRINATOR_BLUR_MARGIN_BOTTOM_FACTOR", "0.45"))
BLUR_TRACK_TTL = int(os.environ.get("BLURRINATOR_BLUR_TRACK_TTL", "12"))
BLUR_TRACK_MATCH_THRESHOLD = float(os.environ.get("BLURRINATOR_BLUR_TRACK_MATCH", "0.40"))
BLUR_TRACK_SEARCH_FACTOR = float(os.environ.get("BLURRINATOR_BLUR_TRACK_SEARCH", "1.50"))
BLUR_IGNORE_INSIDE_ALLOWED_OVERLAP = float(os.environ.get("BLURRINATOR_BLUR_IGNORE_INSIDE_ALLOWED_OVERLAP", "0.85"))
BLUR_IGNORE_INSIDE_ALLOWED_AREA_RATIO = float(os.environ.get("BLURRINATOR_BLUR_IGNORE_INSIDE_ALLOWED_AREA_RATIO", "1.8"))
# Multi-scale face detection — extra downsampled passes catch huge close-up faces
# that YuNet's anchor pyramid misses at native resolution.
FACE_DETECT_SCALES = tuple(
    float(s) for s in os.environ.get("BLURRINATOR_FACE_DETECT_SCALES", "1.0,0.5,0.25").split(",")
)
FACE_MULTISCALE_MIN_SIZE_PX = int(os.environ.get("BLURRINATOR_FACE_MULTISCALE_MIN_SIZE_PX", "160"))

MAX_UPLOAD_BYTES = int(os.environ.get("BLURRINATOR_MAX_UPLOAD_BYTES", 80 * 1024**3))
DATA_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024

FFPROBE_BINARY = os.environ.get("BLURRINATOR_FFPROBE_BINARY", "ffprobe")
FFMPEG_BINARY = os.environ.get("BLURRINATOR_FFMPEG_BINARY", "ffmpeg")
REQUIRE_GPU_PIPELINE = os.environ.get("BLURRINATOR_REQUIRE_GPU_PIPELINE", "1") == "1"

LOGIN_URL = "/users/login/"
LOGIN_REDIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/users/login/"

Q_CLUSTER = {
    "name": "blurrinator",
    "workers": 1,
    "recycle": 50,
    "timeout": 60 * 60 * 6,
    "retry": 60 * 60 * 12,
    "queue_limit": 10,
    "bulk": 1,
    "orm": "default",
    "sync": False,
    "catch_up": False,
}

MESSAGE_TAGS = {
    10: "debug",
    20: "info",
    25: "success",
    30: "warning",
    40: "error",
}
