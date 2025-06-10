import cv2
import numpy as np
import face_recognition
import pickle
import os
import time
from datetime import datetime

# Load face encodings
with open('models/encodings.pkl', 'rb') as f:
    data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]

# Load OpenCV DNN face detector
modelFile = "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/face_detector/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

os.makedirs("results", exist_ok=True)
ALERT_THRESHOLD = 0.45

def detect_faces_dnn(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), False, False)
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            boxes.append((y1, x2, y2, x1))  # top, right, bottom, left
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
    face_locations = detect_faces_dnn(rgb_frame)
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

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # === FPS Measurement ===
    frame_count += 1
    curr_time = time.time()
    elapsed = curr_time - prev_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        prev_time = curr_time
        frame_count = 0
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.imshow("Criminal Recognition (OpenCV DNN)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
