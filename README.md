# REAL-TIME DRIVER DROWSINESS DETECTION SYSTEM
An AI-powered real-time driver drowsiness detection system built with Python,
OpenCV, and Google MediaPipe. The system tracks the driver's eyes using the
Eye Aspect Ratio (EAR) algorithm, features an automatic calibration phase to
seamlessly adapt to wearers with glasses, and triggers audio-visual warnings
when prolonged eye-closure is detected.

FEATURES
Automatic Calibration: Features a 3-second initial calibration window to
measure the baseline open-eye EAR specific to each user, making it fully
compatible with people wearing glasses.

High Performance & Optimization: Implements frame skipping (SKIP_FRAME)
and resolution downscaling to maintain a high and stable Frames Per Second
(FPS).

Real-Time EAR Calculation: Computes the Eye Aspect Ratio (EAR) separately
for both eyes using precise MediaPipe facial landmark coordinates.

Visual Status Feedback: Displays live metrics (EAR, current threshold, and
status) directly on the video feed with color-coded eye landmarks.

TECH STACK & REQUIREMENTS
Python 3.x

OpenCV (cv2) - Video capture and image processing

MediaPipe (mediapipe) - Facial landmark detection (FaceLandmarker)

Pyttsx3 (pyttsx3) - Offline text-to-speech audio alerts

NumPy - Mathematical operations

INSTALLATION & SETUP
Clone the repository:
git clone https://github.com/your-username/driver-drowsiness-detection.git
cd driver-drowsiness-detection

Install the required Python libraries:
pip install opencv-python mediapipe pyttsx3 numpy

Download the MediaPipe Model:

Download the 'face_landmarker.task' model file from the official Google
MediaPipe repository.

Place 'face_landmarker.task' in the root directory of the project
alongside your main Python script.

HOW TO RUN
Connect your webcam.

Run the main Python script:
python main.py

Calibration Phase:
During the first 3 seconds after startup, look straight ahead into the
camera with your eyes open normally to let the system calibrate your
baseline EAR.

Exit:
Press the 'q' key on your keyboard at any time to close the application and
release the camera.

HOW IT WORKS
Frame Capture & Downscaling:
The webcam captures frames at a set resolution (640x360), downscaling them
to 320x180 for AI processing to boost overall performance.

Landmark Extraction:
MediaPipe's FaceLandmarker extracts facial keypoints, focusing specifically
on indices mapped to the left and right eyes.

EAR Mathematical Model:
Calculates eye openness using vertical and horizontal euclidean distances
between specific eyelid landmarks:
EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

Drowsiness Trigger:
If the calculated cached_ear stays below the calibrated EAR_THRESHOLD for
more than 2.0 seconds, the system triggers visual red highlights and an
audio warning.
