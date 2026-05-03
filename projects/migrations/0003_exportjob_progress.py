from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("projects", "0002_facedetection"),
    ]

    operations = [
        migrations.AddField(
            model_name="exportjob",
            name="progress",
            field=models.PositiveSmallIntegerField(default=0),
        ),
    ]
