# Generated by Django 3.1.7 on 2021-04-14 17:33

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('detection', '0005_auto_20210409_1701'),
    ]

    operations = [
        migrations.AddField(
            model_name='detection',
            name='detection_date',
            field=models.DateField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='detection',
            name='detection_time',
            field=models.TimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
    ]
