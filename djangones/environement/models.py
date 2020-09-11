from django.db import models

class NintendoEntertainementSystem(models.Model):
    rom = models.TextField(blank=True, null=True)

    class Meta:
        verbose_name = 'Nintendo Entertainement System'
        verbose_name_plural = 'Nintendo Entertainement System'
