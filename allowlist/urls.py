from django.urls import path

from . import views

app_name = "allowlist"

urlpatterns = [
    path("", views.allowlist_index, name="index"),
    path("new/", views.person_create, name="new"),
    path("p/<int:pk>/", views.person_detail, name="detail"),
    path("p/<int:pk>/upload/", views.person_upload, name="upload"),
    path("p/<int:pk>/toggle-active/", views.person_toggle_active, name="toggle_active"),
    path("p/<int:pk>/delete/", views.person_delete, name="delete"),
    path("img/<int:pk>/delete/", views.image_delete, name="image_delete"),
]
