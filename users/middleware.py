from django.conf import settings
from django.contrib.auth import get_user_model
from django.shortcuts import redirect
from django.urls import reverse


class RequireLoginMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if self._requires_login(request) and not request.user.is_authenticated:
            login_url = reverse("users:login")
            return redirect(f"{login_url}?next={request.get_full_path()}")
        return self.get_response(request)

    def _requires_login(self, request) -> bool:
        path = request.path
        public_prefixes = (
            reverse("users:login"),
            reverse("users:setup") if not get_user_model().objects.exists() else "",
            self._url_path(settings.STATIC_URL),
        )
        return not any(prefix and path.startswith(prefix) for prefix in public_prefixes)

    def _url_path(self, value: str) -> str:
        return value if value.startswith("/") else f"/{value}"
