from __future__ import annotations

from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from .forms import EnrollmentImageForm, PersonForm
from .models import AllowedPerson, EnrollmentImage
from .services import enroll_image


def allowlist_index(request):
    people = AllowedPerson.objects.prefetch_related("images", "embeddings").all()
    return render(request, "allowlist/index.html", {"people": people})


def person_create(request):
    if request.method == "POST":
        form = PersonForm(request.POST)
        if form.is_valid():
            person = form.save()
            messages.success(request, f"Enrolled {person.display_name}. Now add face images.")
            return redirect(reverse("allowlist:detail", args=[person.pk]))
    else:
        form = PersonForm()
    return render(request, "allowlist/form.html", {"form": form, "person": None})


def person_detail(request, pk: int):
    person = get_object_or_404(
        AllowedPerson.objects.prefetch_related("images__embedding"), pk=pk
    )
    upload_form = EnrollmentImageForm()
    return render(request, "allowlist/detail.html", {
        "person": person,
        "upload_form": upload_form,
    })


@require_POST
def person_upload(request, pk: int):
    person = get_object_or_404(AllowedPerson, pk=pk)
    form = EnrollmentImageForm(request.POST, request.FILES)
    if not form.is_valid():
        for err in form.errors.values():
            messages.error(request, "; ".join(err))
        return redirect(reverse("allowlist:detail", args=[person.pk]))

    accepted = 0
    rejected = 0
    for upload in form.cleaned_data["files"]:
        image = EnrollmentImage.objects.create(person=person, image=upload)
        result = enroll_image(person, image)
        if result.accepted:
            accepted += 1
        else:
            rejected += 1

    if accepted:
        messages.success(request, f"Accepted {accepted} image(s).")
    if rejected:
        messages.warning(request, f"Rejected {rejected} image(s) — see gallery for reasons.")
    return redirect(reverse("allowlist:detail", args=[person.pk]))


@require_POST
def person_toggle_active(request, pk: int):
    person = get_object_or_404(AllowedPerson, pk=pk)
    person.is_active = not person.is_active
    person.save(update_fields=["is_active"])
    state = "activated" if person.is_active else "deactivated"
    messages.success(request, f"{person.display_name} {state}.")
    return redirect(request.POST.get("next") or reverse("allowlist:detail", args=[person.pk]))


@require_POST
def person_delete(request, pk: int):
    person = get_object_or_404(AllowedPerson, pk=pk)
    name = person.display_name
    person.delete()
    messages.success(request, f"Removed {name} from the allowlist.")
    return redirect(reverse("allowlist:index"))


@require_POST
def image_delete(request, pk: int):
    image = get_object_or_404(EnrollmentImage, pk=pk)
    person_id = image.person_id
    image.image.delete(save=False)
    image.delete()
    return redirect(reverse("allowlist:detail", args=[person_id]))
