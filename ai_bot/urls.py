from django.urls import path
from .views import TranscribeAudio

urlpatterns = [
    path('transcribe/', TranscribeAudio.as_view(), name='transcribe_audio'),
]