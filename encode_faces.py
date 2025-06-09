import os
import cv2
import face_recognition
import pickle

dataset_path = "dataset/criminals"
encoding_file = "models/encodings.pkl"

known_encodings = []
known_names = []

print("[INFO] Processing dataset...")

# Loop through each person's folder
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    print(f"[INFO] Encoding faces for: {person_name}")

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"[WARNING] Skipping unreadable file: {img_path}")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

print(f"[INFO] Encoded {len(known_encodings)} face(s) total.")
os.makedirs("models", exist_ok=True)
with open(encoding_file, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)
print(f"[INFO] Encodings saved to {encoding_file}")
