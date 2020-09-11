from rest_framework.serializers import ModelSerializer
from . import models


class NintendoEntertainementSystemSerializer(ModelSerializer):
    class Meta:
        model = models.NintendoEntertainementSystem
        fields = '__all__'
        depth = 0