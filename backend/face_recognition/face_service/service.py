from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from collections import Counter

import cv2
import numpy as np
from django.conf import settings


class FaceServiceError(Exception):
    pass


@dataclass
class FaceMatch:
    label: str
    distance: float
    box: list[int]


class FaceRecognitionService:
    def __init__(self, dataset_dir: Path | None = None, threshold: float = 5000.0):
        self.dataset_dir = Path(dataset_dir or settings.BASE_DIR / "face_recognition" / "face_dataset")
        self.cascade_path = settings.BASE_DIR / "face_recognition" / "haarcascade_frontalface_alt.xml"
        self.threshold = threshold
        self._lock = Lock()

        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.face_cascade = cv2.CascadeClassifier(str(self.cascade_path))
        if self.face_cascade.empty():
            raise FaceServiceError(f"Could not load face cascade from {self.cascade_path}")

    @staticmethod
    def image_file_to_bgr(uploaded_file):
        image_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise FaceServiceError("Could not decode the uploaded image")
        return image

    @staticmethod
    def _distance(vector_one, vector_two):
        return float(np.sqrt(((vector_one - vector_two) ** 2).sum()))

    def _knn(self, trainset, test_vector, k: int = 5):
        # trainset here is expected to be (n_samples, n_features) and train_labels a list
        raise NotImplementedError("_knn should be called with train_vectors and train_labels")

    @staticmethod
    def _sanitize_name(name: str) -> str:
        cleaned = name.replace("/", " ").replace("\\", " ").strip()
        if not cleaned:
            raise FaceServiceError("A valid name is required")
        return cleaned

    def _face_box(self, image, face):
        x, y, w, h = face
        offset = 5
        height, width = image.shape[:2]
        x1 = max(x - offset, 0)
        y1 = max(y - offset, 0)
        x2 = min(x + w + offset, width)
        y2 = min(y + h + offset, height)
        face_region = image[y1:y2, x1:x2]
        if face_region.size == 0:
            raise FaceServiceError("Could not crop face from the image")
        face_region = cv2.resize(face_region, (100, 100))
        return face_region, [int(x), int(y), int(w), int(h)]

    def detect_faces(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_image, 1.3, 5)
        return list(faces)

    def _load_trainset(self):
        vectors = []
        labels = []

        for dataset_file in sorted(self.dataset_dir.glob("*.npy")):
            name = dataset_file.stem
            data_item = np.load(dataset_file)
            if data_item.ndim == 1:
                data_item = data_item.reshape(1, -1)
            if data_item.size == 0:
                continue

            for row in data_item:
                vectors.append(row)
                labels.append(name)

        if not vectors:
            return None, None

        train_vectors = np.vstack(vectors)
        return train_vectors, labels

    def enroll(self, name: str, image):
        name = self._sanitize_name(name)
        faces = self.detect_faces(image)
        if not faces:
            raise FaceServiceError("No face detected in the uploaded image")

        face = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)[0]
        face_region, _ = self._face_box(image, face)
        face_vector = face_region.flatten().reshape(1, -1)

        dataset_file = self.dataset_dir / f"{name}.npy"
        with self._lock:
            if dataset_file.exists():
                current_samples = np.load(dataset_file)
                if current_samples.ndim == 1:
                    current_samples = current_samples.reshape(1, -1)
                updated_samples = np.concatenate((current_samples, face_vector), axis=0)
            else:
                updated_samples = face_vector

            np.save(dataset_file, updated_samples)

        return {
            "name": name,
            "sample_count": int(updated_samples.shape[0]),
        }

    def scan(self, image):
        faces = self.detect_faces(image)
        return {
            "face_count": len(faces),
            "faces": [[int(x), int(y), int(w), int(h)] for x, y, w, h in faces],
        }

    def verify(self, image):
        faces = self.detect_faces(image)
        if not faces:
            raise FaceServiceError("No face detected in the uploaded image")
        train_vectors, train_labels = self._load_trainset()
        if train_vectors is None:
            raise FaceServiceError("No enrolled faces are available yet")

        results = []
        for face in sorted(faces, key=lambda box: box[2] * box[3], reverse=True):
            face_region, box = self._face_box(image, face)
            test_vec = face_region.flatten().astype(np.float32)

            # compute distances vectorized
            diffs = train_vectors.astype(np.float32) - test_vec
            dists = np.linalg.norm(diffs, axis=1)
            idx = np.argsort(dists)[:5]
            nearest_labels = [train_labels[i] for i in idx]
            counts = Counter(nearest_labels)
            predicted_name, _ = counts.most_common(1)[0]
            distance = float(dists[idx[0]])
            predicted_name = predicted_name if distance <= self.threshold else "Unknown"

            results.append(FaceMatch(label=predicted_name, distance=distance, box=box))

        return results