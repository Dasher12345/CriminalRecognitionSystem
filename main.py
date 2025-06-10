import cv2
import numpy as np
import face_recognition
import pickle
import os
import time
from datetime import datetime
import mediapipe as mp

# Load face encodings
with open('models/encodings.pkl', 'rb') as f:
    data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

os.makedirs("results", exist_ok=True)
ALERT_THRESHOLD = 0.45

def detect_faces_mediapipe(frame):
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes = []
    if results.detections:
        h, w, _ = frame.shape
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            top = max(y1, 0)
            right = min(x1 + width, w)
            bottom = min(y1 + height, h)
            left = max(x1, 0)
            boxes.append((top, right, bottom, left))
    return boxes

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam...")

prev_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = detect_faces_mediapipe(frame)
    face_encs = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_enc in zip(face_locations, face_encs):
        dists = face_recognition.face_distance(known_encodings, face_enc)
        idx = np.argmin(dists)
        name = "Unknown"
        if dists[idx] < ALERT_THRESHOLD:
            name = known_names[idx]
            print(f"[ALERT] MATCH FOUND: {name}")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"results/match_{name}_{ts}.jpg", frame)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # === FPS Measurement ===
    frame_count += 1
    curr_time = time.time()
    elapsed = curr_time - prev_time
    if True:
        fps = frame_count / elapsed
        prev_time = curr_time
        frame_count = 0
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.imshow("Criminal Recognition (MediaPipe)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
