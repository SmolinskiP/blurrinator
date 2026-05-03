from django.contrib import admin

from .models import AllowedPerson, EnrollmentImage


class EnrollmentImageInline(admin.TabularInline):
    model = EnrollmentImage
    extra = 0


@admin.register(AllowedPerson)
class AllowedPersonAdmin(admin.ModelAdmin):
    list_display = ("display_name", "is_active", "created_at", "enrolled_by")
    inlines = [EnrollmentImageInline]
