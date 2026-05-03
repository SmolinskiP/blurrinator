from __future__ import annotations

from django import forms
from django.conf import settings
from django.utils.text import slugify

from .models import Project


ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}


class ProjectUploadForm(forms.Form):
    title = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={
            "placeholder": "e.g. Stream highlights — week 18",
            "autocomplete": "off",
        }),
    )
    notes = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 3, "placeholder": "Optional context for review"}),
    )
    file = forms.FileField(
        widget=forms.ClearableFileInput(attrs={"accept": "video/*"}),
        help_text="MP4, MOV, MKV, WebM, M4V or AVI.",
    )

    def clean_file(self):
        upload = self.cleaned_data["file"]
        suffix = "." + upload.name.rsplit(".", 1)[-1].lower() if "." in upload.name else ""
        if suffix not in ALLOWED_EXTENSIONS:
            raise forms.ValidationError(
                f"Unsupported extension {suffix or '(none)'}. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            )
        if upload.size > settings.MAX_UPLOAD_BYTES:
            raise forms.ValidationError("File exceeds the configured upload limit.")
        return upload

    def clean_title(self):
        title = self.cleaned_data["title"].strip()
        slug = slugify(title)
        if not slug:
            raise forms.ValidationError("Title must contain at least one alphanumeric character.")
        return title

    def build_project(self) -> Project:
        title = self.cleaned_data["title"]
        base = slugify(title)
        slug = base
        suffix = 2
        while Project.objects.filter(slug=slug).exists():
            slug = f"{base}-{suffix}"
            suffix += 1
        return Project.objects.create(
            title=title,
            slug=slug,
            notes=self.cleaned_data.get("notes", ""),
        )
