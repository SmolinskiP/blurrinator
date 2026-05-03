from collections import defaultdict

from django.shortcuts import render

from .models import ModelEntry


def registry_index(request):
    grouped: dict[str, list[ModelEntry]] = defaultdict(list)
    for entry in ModelEntry.objects.all():
        grouped[entry.get_family_display()].append(entry)
    return render(request, "registry/index.html", {"groups": dict(grouped)})
