from django.urls import path

from . import views

app_name = "users"

urlpatterns = [
    path("", views.user_list, name="list"),
    path("new/", views.user_create, name="new"),
    path("setup/", views.setup_view, name="setup"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("<int:pk>/password/", views.user_password_reset, name="password_reset"),
    path("<int:pk>/delete/", views.user_delete, name="delete"),
]
