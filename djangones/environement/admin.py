from django.contrib import admin
from . import models

@admin.register(models.NintendoEntertainementSystem)
class NintendoEntertainementSystemAdmin(admin.ModelAdmin):
    list_select_related = True
    list_display = [field.name for field in models.NintendoEntertainementSystem._meta.get_fields()
    				if field.many_to_one is None
                    and field.related_model is None]
