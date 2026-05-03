from __future__ import annotations

from django.contrib import messages
from django.contrib.auth import get_user_model, login, logout
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods, require_POST

from .forms import AppPasswordResetForm, AppUserCreationForm, InitialAdminCreationForm, LoginForm


User = get_user_model()


def _is_admin(user) -> bool:
    return user.is_authenticated and user.is_staff


def admin_required(view_func):
    @login_required
    def wrapped(request, *args, **kwargs):
        if not _is_admin(request.user):
            raise PermissionDenied
        return view_func(request, *args, **kwargs)

    return wrapped


@require_http_methods(["GET", "POST"])
def login_view(request):
    if not User.objects.exists():
        return redirect("users:setup")
    if request.user.is_authenticated:
        return redirect("projects:list")
    form = LoginForm(request, data=request.POST or None)
    if request.method == "POST" and form.is_valid():
        login(request, form.get_user())
        return redirect(request.GET.get("next") or reverse("projects:list"))
    return render(request, "users/login.html", {"form": form})


@require_http_methods(["GET", "POST"])
def setup_view(request):
    if User.objects.exists():
        return redirect("users:login")
    form = InitialAdminCreationForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        login(request, user)
        messages.success(request, "Initial admin account created.")
        return redirect("projects:list")
    return render(request, "users/form.html", {"form": form, "title": "Create initial admin"})


@require_POST
@login_required
def logout_view(request):
    logout(request)
    return redirect("users:login")


@login_required
def user_list(request):
    users = User.objects.order_by("username")
    return render(request, "users/list.html", {"users": users})


@require_http_methods(["GET", "POST"])
@admin_required
def user_create(request):
    form = AppUserCreationForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        messages.success(request, f"User {user.username} created.")
        return redirect("users:list")
    return render(request, "users/form.html", {"form": form, "title": "Create user"})


@require_http_methods(["GET", "POST"])
@admin_required
def user_password_reset(request, pk: int):
    user = get_object_or_404(User, pk=pk)
    form = AppPasswordResetForm(user, request.POST or None)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, f"Password reset for {user.username}.")
        return redirect("users:list")
    return render(
        request,
        "users/form.html",
        {"form": form, "title": f"Reset password · {user.username}"},
    )


@require_POST
@admin_required
def user_delete(request, pk: int):
    user = get_object_or_404(User, pk=pk)
    if user.pk == request.user.pk:
        messages.error(request, "You cannot delete your own account.")
        return redirect("users:list")
    if user.is_staff and User.objects.filter(is_staff=True).count() <= 1:
        messages.error(request, "You cannot delete the last admin account.")
        return redirect("users:list")
    username = user.username
    user.delete()
    messages.success(request, f"User {username} deleted.")
    return redirect("users:list")
