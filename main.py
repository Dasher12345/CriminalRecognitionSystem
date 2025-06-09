import cv2
import numpy as np
import face_recognition
import pickle
import os
from datetime import datetime

ALERT_THRESHOLD = 0.45  # Lower = stricter match

# 1. Load face encodings
with open('models/encodings.pkl', 'rb') as f:
    data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]

# 2. Load OpenCV DNN face detector
modelFile = "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/face_detector/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Helper function: detect faces using OpenCV DNN
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
            boxes.append((y1, x2, y2, x1))  # top, right, bottom, left (face_recognition format)
    return boxes

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 3. Detect faces using DNN
    face_locations = detect_faces_dnn(rgb_frame)
    face_encs = face_recognition.face_encodings(rgb_frame, face_locations)

    # 4. Check for matches and trigger alerts
    for (top, right, bottom, left), face_enc in zip(face_locations, face_encs):
        dists = face_recognition.face_distance(known_encodings, face_enc)
        idx = np.argmin(dists)
        name = "Unknown"

        if dists[idx] < ALERT_THRESHOLD:
            name = known_names[idx]
            print(f"[ALERT] MATCH FOUND: {name}")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"results/match_{name}_{ts}.jpg", frame)

        # Draw bounding box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Criminal Recognition (DNN)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
