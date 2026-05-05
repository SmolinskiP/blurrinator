from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("projects", "0008_manualblurregion"),
    ]

    operations = [
        migrations.AddField(
            model_name="facedetection",
            name="landmark_implausible",
            field=models.BooleanField(default=False),
        ),
    ]
