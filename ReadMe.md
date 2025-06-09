# Criminal Recognition System

A real-time facial recognition system to detect and alert when a known individual (e.g., a criminal or watchlisted person) appears on a camera feed.

---

## 📁 Project Structure

```
CriminalRecognitionSystem/
├── dataset/
│   ├── [criminal folders]/       # e.g., "Test1", "Abdullah (test2)"
├── models/
│   ├── face_detector/
│   │   ├── deploy.prototxt
│   │   └── res10_300x300_ssd_iter_140000.caffemodel
│   └── encodings.pkl            # Generated after encoding
├── results/                     # Match alert screenshots
├── encode_faces.py              # Script to generate encodings
├── main.py                      # Main recognition script
├── requirements.txt
```

---

## ⚙️ Setup Instructions

### 1. Clone or copy the project folder

Ensure you have the complete folder structure as shown above.

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# OR
source venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add known individuals to dataset

Inside the `dataset/` folder, create subfolders with images of known individuals. Example:

```
dataset/
└── Test1/
    ├── img1.jpg
    └── img2.jpg
```

The folder name (e.g., `Test1`) will be used as the person's name.

---

## 🧠 Step 1: Encode Faces

```bash
python encode_faces.py
```

This will process the images and save the encodings to `models/encodings.pkl`.

---

## 📹 Step 2: Run the Main Recognition Script

```bash
python main.py
```

This opens your webcam and begins scanning for matches in real-time. If a match is found:

* An alert will be printed in the console.
* A screenshot will be saved in the `results/` folder.

Press `Q` to quit.

---

## 📝 Notes

* The face detector uses OpenCV's DNN module with a pre-trained SSD model.
* Matching is controlled by the `ALERT_THRESHOLD` value (lower = stricter).
* Works best in well-lit conditions.

---

## ✅ Requirements

* Python 3.7+
* OpenCV
* face\_recognition
* numpy

All dependencies are listed in `requirements.txt`.

---

## 📧 Contact

For questions or contributions, contact the project team.
