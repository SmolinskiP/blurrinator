from django.contrib import admin

from .models import AnalysisJob, ExportJob, Project, SourceVideo


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ("title", "slug", "created_at")
    prepopulated_fields = {"slug": ("title",)}


@admin.register(SourceVideo)
class SourceVideoAdmin(admin.ModelAdmin):
    list_display = ("project", "original_filename", "size_bytes", "uploaded_at")


@admin.register(AnalysisJob)
class AnalysisJobAdmin(admin.ModelAdmin):
    list_display = ("project", "status", "progress", "created_at")
    list_filter = ("status",)


@admin.register(ExportJob)
class ExportJobAdmin(admin.ModelAdmin):
    list_display = ("project", "style", "status", "created_at")
    list_filter = ("status", "style")
