from rest_framework import views, viewsets
from rest_framework import mixins, generics
from . import serializers
from . import models
from rest_framework.response import Response
from django.http import HttpResponse


class NintendoEntertainementSystemViewSet(mixins.ListModelMixin,
                                          viewsets.GenericViewSet):
    serializer_class = serializers.NintendoEntertainementSystemSerializer
    model = models.NintendoEntertainementSystem
    queryset = model.objects.all()
    search_fields = [field.name for field in model._meta.get_fields()
                     if field.get_internal_type() == 'TextField']


class DisplayView(views.APIView):
    def get_extra_actions(cls):
        # TODO: patch this workaround later
        return []

    def get(self, *args, **kwargs):
        from PIL import Image
        import requests
        from io import BytesIO
        url = 'https://images-na.ssl-images-amazon.com/images/I/71vcyxrIZ6L._AC_SX450_.jpg'
        r = requests.get(url)
        return HttpResponse(r.content, content_type="image/jpeg")
