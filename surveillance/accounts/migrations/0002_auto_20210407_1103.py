# Generated by Django 3.1.7 on 2021-04-07 05:03

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='order',
            name='customer',
        ),
        migrations.RemoveField(
            model_name='order',
            name='product',
        ),
        migrations.RemoveField(
            model_name='product',
            name='tags',
        ),
        migrations.DeleteModel(
            name='Customer',
        ),
        migrations.DeleteModel(
            name='Order',
        ),
        migrations.DeleteModel(
            name='Product',
        ),
        migrations.DeleteModel(
            name='Tag',
        ),
    ]
