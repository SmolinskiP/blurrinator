from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ("projects", "0006_remove_exportjob_kind"),
    ]

    operations = [
        migrations.CreateModel(
            name="DetectionOverride",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("decision", models.CharField(choices=[("allowed", "Allowed"), ("unknown", "Unknown — blur"), ("uncertain", "Uncertain — blur + flag"), ("conflict", "Identity conflict — blur + flag")], max_length=16)),
                ("x", models.PositiveIntegerField()),
                ("y", models.PositiveIntegerField()),
                ("width", models.PositiveIntegerField()),
                ("height", models.PositiveIntegerField()),
                ("note", models.CharField(blank=True, max_length=240)),
                ("created_at", models.DateTimeField(default=django.utils.timezone.now)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("detection", models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name="override", to="projects.facedetection")),
            ],
            options={
                "ordering": ["detection__timestamp_seconds", "detection_id"],
            },
        ),
    ]
