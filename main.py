import cv2
import face_recognition
import pickle
import numpy as np
import time
import os
from datetime import datetime

ALERT_THRESHOLD = 0.5  # Lower = stricter match

# Load saved face encodings
with open('models/encodings.pkl', 'rb') as f:
    data = pickle.load(f)
    known_encodings = data['encodings']
    known_names = data['names']

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

# Initialize webcam
video_capture = cv2.VideoCapture(0)
print("[INFO] Starting webcam...")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and compute encodings
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    # Loop through faces found
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(distances)

        name = "Unknown"
        if distances[best_match_index] < ALERT_THRESHOLD:
            name = known_names[best_match_index]
            print(f"[ALERT] MATCH FOUND: {name}")

            # Save snapshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            alert_path = os.path.join("results", f"match_{name}_{timestamp}.jpg")
            cv2.imwrite(alert_path, frame)

        # Draw rectangle and name
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Criminal Recognition System', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
