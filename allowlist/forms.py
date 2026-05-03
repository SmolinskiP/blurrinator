from __future__ import annotations

from django import forms

from .models import AllowedPerson, EnrollmentImage


class PersonForm(forms.ModelForm):
    class Meta:
        model = AllowedPerson
        fields = ["display_name", "consent_basis", "enrolled_by", "is_active"]
        widgets = {
            "display_name": forms.TextInput(attrs={"placeholder": "Screen name"}),
            "consent_basis": forms.Textarea(attrs={
                "rows": 3,
                "placeholder": "Where the consent comes from. Required by policy.",
            }),
            "enrolled_by": forms.TextInput(attrs={"placeholder": "Your name"}),
        }


class EnrollmentImageForm(forms.Form):
    """File handling is done directly from request.FILES in the view because
    Django refuses ``multiple`` on its standard file widget."""

    def clean(self):
        cleaned = super().clean()
        files = self.files.getlist("images")
        if not files:
            raise forms.ValidationError("Pick at least one file.")
        cleaned["files"] = files
        return cleaned
