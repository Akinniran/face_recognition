from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from authentication.serializers import FaceEnrollSerializer, FaceScanSerializer, FaceVerifySerializer
from face_recognition.face_service.service import FaceRecognitionService, FaceServiceError


service = FaceRecognitionService()

class FaceAPICheckView(APIView):
	permission_classes = [AllowAny]

	def get(self, request):
		return Response({"status": "ok"})


class FaceScanView(APIView):
	permission_classes = [AllowAny]

	def post(self, request):
		serializer = FaceScanSerializer(data=request.data)
		serializer.is_valid(raise_exception=True)

		try:
			image = service.image_file_to_bgr(serializer.validated_data["image"])
			data = service.scan(image)
		except FaceServiceError as exc:
			return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

		return Response(data)


class FaceEnrollView(APIView):
	permission_classes = [AllowAny]

	def post(self, request):
		serializer = FaceEnrollSerializer(data=request.data)
		serializer.is_valid(raise_exception=True)

		try:
			image = service.image_file_to_bgr(serializer.validated_data["image"])
			data = service.enroll(serializer.validated_data["name"], image)
		except FaceServiceError as exc:
			return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

		return Response(data, status=status.HTTP_201_CREATED)


class FaceVerifyView(APIView):
	permission_classes = [AllowAny]

	def post(self, request):
		serializer = FaceVerifySerializer(data=request.data)
		serializer.is_valid(raise_exception=True)

		try:
			image = service.image_file_to_bgr(serializer.validated_data["image"])
			matches = service.verify(image)
		except FaceServiceError as exc:
			return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

		best_match = matches[0]
		expected_name = serializer.validated_data.get("expected_name")
		verified = best_match.label != "Unknown"
		if expected_name:
			verified = best_match.label == expected_name

		return Response(
			{
				"verified": verified,
				"expected_name": expected_name,
				"match_name": best_match.label,
				"distance": best_match.distance,
				"face_box": best_match.box,
				"matches": [
					{
						"name": match.label,
						"distance": match.distance,
						"box": match.box,
					}
					for match in matches
				],
			}
		)
