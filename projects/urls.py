from django.urls import path

from . import views

app_name = "projects"

urlpatterns = [
    path("", views.ProjectListView.as_view(), name="list"),
    path("new/", views.upload_project, name="new"),
    path("p/<slug:slug>/", views.project_detail, name="detail"),
    path("p/<slug:slug>/status/", views.project_status, name="status"),
    path("p/<slug:slug>/source/", views.stream_source_video, name="source_video"),
    path("p/<slug:slug>/analyze/", views.queue_analysis, name="analyze"),
    path("p/<slug:slug>/draft/", views.draft_review, name="draft_review"),
    path(
        "p/<slug:slug>/draft/detections/<int:detection_id>/save/",
        views.save_detection_override,
        name="save_detection_override",
    ),
    path(
        "p/<slug:slug>/draft/detections/<int:detection_id>/reset/",
        views.reset_detection_override,
        name="reset_detection_override",
    ),
    path(
        "p/<slug:slug>/draft/detections/bulk-allow/",
        views.bulk_allow_detections,
        name="bulk_allow_detections",
    ),
    path(
        "p/<slug:slug>/draft/manual-blurs/add/",
        views.add_manual_blur_region,
        name="add_manual_blur_region",
    ),
    path(
        "p/<slug:slug>/draft/manual-blurs/<int:region_id>/delete/",
        views.delete_manual_blur_region,
        name="delete_manual_blur_region",
    ),
    path("p/<slug:slug>/export/final/", views.queue_final_export, name="export_final"),
    path("p/<slug:slug>/delete/", views.delete_project, name="delete"),
    path("exports/<int:pk>/download/", views.download_export, name="download_export"),
    path("exports/<int:pk>/stream/", views.stream_export, name="stream_export"),
]
