import os
import cv2
import face_recognition
import pickle

dataset_dir = 'dataset/criminals'
encoding_file = 'models/encodings.pkl'

known_encodings = []
known_names = []

print("[INFO] Processing images...")

for person_name in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_path):
        continue  # skip files, only process folders

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

print(f"[INFO] Encoded {len(known_encodings)} faces and saving...")

# Save to pickle
with open(encoding_file, 'wb') as f:
    pickle.dump({'encodings': known_encodings, 'names': known_names}, f)

print(f"[INFO] Saved encodings to {encoding_file}")
