from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("projects", "0003_exportjob_progress"),
    ]

    operations = [
        migrations.AddField(
            model_name="analysisjob",
            name="log",
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name="exportjob",
            name="log",
            field=models.TextField(blank=True),
        ),
    ]
