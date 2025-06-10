import cv2
import numpy as np
import face_recognition
import pickle
import os
from datetime import datetime
import mediapipe as mp
import time
import csv

# === CONFIGURATION ===
ALERT_THRESHOLD = 0.45
DETECTION_INTERVAL = 30
RESIZE_SCALE = 0.5

# === Load face encodings ===
with open('models/encodings.pkl', 'rb') as f:
    data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]

# === MediaPipe Face Detector ===
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# === Create output directory ===
os.makedirs("results", exist_ok=True)

# === Setup CSV Logging ===
log_file_path = os.path.join("results", "results_log.csv")
csv_exists = os.path.isfile(log_file_path)
log_file = open(log_file_path, mode='a', newline='')
csv_writer = csv.writer(log_file)
if not csv_exists:
    csv_writer.writerow(["timestamp", "name", "distance", "image", "result"])

# === Face Detection using MediaPipe ===
def detect_faces_mediapipe(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
    results = face_detection.process(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
    boxes = []

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = small_frame.shape
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            top = int(y1 / RESIZE_SCALE)
            right = int((x1 + width) / RESIZE_SCALE)
            bottom = int((y1 + height) / RESIZE_SCALE)
            left = int(x1 / RESIZE_SCALE)

            boxes.append((top, right, bottom, left))
    return boxes

# === Initialize Webcam ===
cap = cv2.VideoCapture(0)
frame_count = 0
trackers = []
names = []

# Recognition Accuracy Counters
correct_matches = 0
wrong_matches = 0
missed_matches = 0

# FPS init
fps = 0
prev_time = time.time()

print("[INFO] Starting webcam...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame_count % DETECTION_INTERVAL == 0:
        face_locations = detect_faces_mediapipe(frame)
        face_encs = face_recognition.face_encodings(rgb_frame, face_locations)

        trackers = []
        names = []

        for (top, right, bottom, left), face_enc in zip(face_locations, face_encs):
            dists = face_recognition.face_distance(known_encodings, face_enc)
            idx = np.argmin(dists)
            name = "Unknown"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            if dists[idx] < ALERT_THRESHOLD:
                name = known_names[idx]
                print(f"[ALERT] MATCH FOUND: {name}")
                image_filename = f"match_{name}_{ts}.jpg"
                cv2.imwrite(os.path.join("results", image_filename), frame)

                # Accuracy Tracking
                if known_names[idx] == name:
                    correct_matches += 1
                else:
                    wrong_matches += 1

                # Log match
                csv_writer.writerow([ts, name, round(dists[idx], 4), image_filename, "Match"])
            else:
                missed_matches += 1
                # Log unknown
                csv_writer.writerow([ts, "Unknown", "N/A", "N/A", "Unknown"])

            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, (left, top, right - left, bottom - top))
            trackers.append(tracker)
            names.append(name)

        # === Show Stats ===
        total_attempts = correct_matches + wrong_matches + missed_matches
        precision = correct_matches / (correct_matches + wrong_matches) if (correct_matches + wrong_matches) > 0 else 0
        recall = correct_matches / (correct_matches + missed_matches) if (correct_matches + missed_matches) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    else:
        new_trackers = []
        new_names = []
        for tracker, name in zip(trackers, names):
            success, box = tracker.update(frame)
            if success:
                left, top, w, h = [int(v) for v in box]
                right = left + w
                bottom = top + h

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                new_trackers.append(tracker)
                new_names.append(name)
        trackers = new_trackers
        names = new_names

    # === FPS Calculation ===
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    frame_count += 1
    cv2.imshow("Criminal Recognition (MediaPipe + Tracking)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Final Stats ===
print("\n=== FINAL STATS ===")
print(f"Correct Matches: {correct_matches}")
print(f"Wrong Matches:   {wrong_matches}")
print(f"Missed Matches:  {missed_matches}")
total_attempts = correct_matches + wrong_matches + missed_matches
if total_attempts > 0:
    precision = correct_matches / (correct_matches + wrong_matches) if (correct_matches + wrong_matches) > 0 else 0
    recall = correct_matches / (correct_matches + missed_matches) if (correct_matches + missed_matches) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Precision:       {precision:.2f}")
    print(f"Recall:          {recall:.2f}")
    print(f"F1 Score:        {f1:.2f}")
else:
    print("No recognition attempts recorded.")

# Clean up
log_file.close()
cap.release()
cv2.destroyAllWindows()
