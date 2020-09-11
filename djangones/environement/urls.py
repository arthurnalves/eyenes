from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views


router = DefaultRouter()
router.register('nes', views.NintendoEntertainementSystemViewSet, basename='nes')

urlpatterns = [
    path('', include(router.urls)),
    path('display/', views.DisplayView.as_view(), name='display'),
]