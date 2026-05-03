from django.contrib import admin

from .models import ModelEntry


@admin.register(ModelEntry)
class ModelEntryAdmin(admin.ModelAdmin):
    list_display = ("name", "version", "family", "license", "provenance", "registered_at")
    list_filter = ("family", "provenance", "license")
    search_fields = ("name", "version", "sha256")
