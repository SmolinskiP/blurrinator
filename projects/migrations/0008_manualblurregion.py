from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ("projects", "0007_detectionoverride"),
    ]

    operations = [
        migrations.CreateModel(
            name="ManualBlurRegion",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("start_seconds", models.FloatField()),
                ("end_seconds", models.FloatField()),
                ("x", models.PositiveIntegerField()),
                ("y", models.PositiveIntegerField()),
                ("width", models.PositiveIntegerField()),
                ("height", models.PositiveIntegerField()),
                ("note", models.CharField(blank=True, max_length=240)),
                ("created_at", models.DateTimeField(default=django.utils.timezone.now)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("project", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="manual_blur_regions", to="projects.project")),
            ],
            options={
                "ordering": ["start_seconds", "id"],
            },
        ),
    ]
