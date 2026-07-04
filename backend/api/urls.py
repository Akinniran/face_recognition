from django.urls import path

from authentication.views import FaceEnrollView, FaceAPICheckView, FaceScanView, FaceVerifyView

urlpatterns = [
    path('face/health/', FaceAPICheckView.as_view(), name='api-check'),
    path('face/scan/', FaceScanView.as_view(), name='face-scan'),
    path('face/enroll/', FaceEnrollView.as_view(), name='face-enroll'),
    path('face/verify/', FaceVerifyView.as_view(), name='face-verify'),
]
