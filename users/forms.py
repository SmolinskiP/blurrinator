from __future__ import annotations

from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import AuthenticationForm, SetPasswordForm, UserCreationForm


User = get_user_model()


class LoginForm(AuthenticationForm):
    username = forms.CharField(widget=forms.TextInput(attrs={"autofocus": True}))


class AppUserCreationForm(UserCreationForm):
    ROLE_CHOICES = (
        ("user", "User"),
        ("admin", "Admin"),
    )

    role = forms.ChoiceField(choices=ROLE_CHOICES, initial="user")

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username", "email", "first_name", "last_name", "role")

    def save(self, commit=True):
        user = super().save(commit=False)
        role = self.cleaned_data["role"]
        user.is_staff = role == "admin"
        user.is_superuser = False
        if commit:
            user.save()
        return user


class InitialAdminCreationForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username", "email", "first_name", "last_name")

    def save(self, commit=True):
        user = super().save(commit=False)
        user.is_staff = True
        user.is_superuser = False
        if commit:
            user.save()
        return user


class AppPasswordResetForm(SetPasswordForm):
    pass
