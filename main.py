import cv2
import numpy as np
import face_recognition
import pickle
import os
from datetime import datetime
import mediapipe as mp
import time

# ========== CONFIG ==========
ALERT_THRESHOLD = 0.45
DETECTION_INTERVAL = 10  # Run detection every N frames
RESIZE_SCALE = 0.5       # Resize input frame for performance

# ========== LOAD ENCODINGS ==========
with open('models/encodings.pkl', 'rb') as f:
    data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]

# ========== INITIALIZE MEDIAPIPE ==========
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# ========== INIT ==========
os.makedirs("results", exist_ok=True)
cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam...")

# Variables for tracking
trackers = []
frame_count = 0

# FPS tracking
prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_frame = frame.copy()
    small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
    frame_count += 1

    # ========== RUN DETECTION ==========
    if frame_count % DETECTION_INTERVAL == 0:
        trackers = []  # Reset trackers
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_small)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = small_frame.shape
                x1 = int(bboxC.xmin * w)
                y1 = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)

                # Convert to original scale
                scale = 1 / RESIZE_SCALE
                x1_full = int(x1 * scale)
                y1_full = int(y1 * scale)
                width_full = int(width * scale)
                height_full = int(height * scale)

                tracker = cv2.TrackerCSRT_create()
                tracker.init(orig_frame, (x1_full, y1_full, width_full, height_full))
                trackers.append(tracker)

    # ========== UPDATE TRACKERS ==========
    boxes = []
    for tracker in trackers:
        success, box = tracker.update(orig_frame)
        if success:
            x, y, w, h = [int(v) for v in box]
            boxes.append((y, x + w, y + h, x))  # (top, right, bottom, left)

    # ========== FACE RECOGNITION ==========
    rgb_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
    face_encs = face_recognition.face_encodings(rgb_frame, boxes)

    for (top, right, bottom, left), face_enc in zip(boxes, face_encs):
        dists = face_recognition.face_distance(known_encodings, face_enc)
        idx = np.argmin(dists)
        name = "Unknown"

        if dists[idx] < ALERT_THRESHOLD:
            name = known_names[idx]
            print(f"[ALERT] MATCH FOUND: {name}")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"results/match_{name}_{ts}.jpg", orig_frame)

        # Draw box & label
        cv2.rectangle(orig_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(orig_frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ========== FPS COUNTER ==========
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(orig_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # ========== SHOW FRAME ==========
    cv2.imshow("Criminal Recognition (MediaPipe + CSRT)", orig_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
