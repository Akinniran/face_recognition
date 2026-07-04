## Real-time-Face-Recognition-Project

Steps to run 
1. First run video read.py to check that your webcam is running or not.
2. Run face_detection.py to check that whether camera is able to capture your face or not, with the help of haar cascase classifier
3. Now, run face_data.py --> this will open up the camera and extract your face from video frames multiple time. Test and store faces of multiple people.
4. At last, run face_recognition.py --> this will detect your face from the dataset made and form a bounding box with you name writtern around your face.

## Django service mode

This folder now also powers the Django face-recognition API used by the app.

Available endpoints:
1. `GET /api/face/health/` - service health check
2. `POST /api/face/scan/` - detect faces in an uploaded image
3. `POST /api/face/enroll/` - store a new face sample under a name
4. `POST /api/face/verify/` - compare an uploaded face against enrolled samples

Send images as `multipart/form-data`. For enrollment, send `name` and `image`. For verification, send `image` and optionally `expected_name`.
