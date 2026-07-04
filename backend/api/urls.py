from django.urls import path

from api.views import FaceEnrollView, FaceHealthView, FaceScanView, FaceVerifyView

urlpatterns = [
    path('face/health/', FaceHealthView.as_view(), name='face-health'),
    path('face/scan/', FaceScanView.as_view(), name='face-scan'),
    path('face/enroll/', FaceEnrollView.as_view(), name='face-enroll'),
    path('face/verify/', FaceVerifyView.as_view(), name='face-verify'),
]
