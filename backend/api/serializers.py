from rest_framework import serializers


class FaceEnrollSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=255)
    image = serializers.ImageField()


class FaceVerifySerializer(serializers.Serializer):
    image = serializers.ImageField()
    expected_name = serializers.CharField(max_length=255, required=False, allow_blank=False)


class FaceScanSerializer(serializers.Serializer):
    image = serializers.ImageField()