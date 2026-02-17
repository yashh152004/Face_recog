import cv2
import os
import numpy as np
from deepface import DeepFace

# ==========================================
# Configuration
# ==========================================
# List the exact names of your folders here
TARGET_FOLDERS = ["Harini", "Thrisha", "Trupti"]

# Model Selection: FaceNet512 is high accuracy (99.65%)
MODEL_NAME = "Facenet512"

# SENSITIVITY THRESHOLD
# 0.5 is the balance point.
# If it says "Error" for the real girls -> INCREASE to 0.6
# If it confuses a stranger for the girls -> DECREASE to 0.4
THRESHOLD = 0.5

# ==========================================
# 1. Training Phase (Load & Encode)
# ==========================================
print("[INFO] Initializing TensorFlow/DeepFace... (Please wait)")
# Pre-build model to avoid lag on first frame
DeepFace.build_model(MODEL_NAME)

known_face_encodings = []
known_face_names = []

print("[INFO] Loading images from folders...")

# Loop through the specific list of names you gave me
for name in TARGET_FOLDERS:
    # Look for the folder in the current directory
    person_dir = os.path.join(os.getcwd(), name)

    # Check if the folder actually exists
    if not os.path.exists(person_dir):
        print(f"[WARNING] Folder '{name}' not found. Skipping.")
        continue

    print(f"[INFO] Processing folder: {name}")

    # Loop through images inside that folder
    for filename in os.listdir(person_dir):
        # Check for valid image files
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(person_dir, filename)

            try:
                # Get the embedding (the "digital signature" of the face)
                # enforce_detection=False allows it to load even if the face is slightly turned
                embedding_objs = DeepFace.represent(img_path=filepath, model_name=MODEL_NAME, enforce_detection=False)

                if embedding_objs:
                    # Store the embedding and the label (name)
                    known_face_encodings.append(embedding_objs[0]["embedding"])
                    known_face_names.append(name)
                    print(f"  -> Learned: {filename}")

            except Exception as e:
                print(f"  [Skipped] Could not process {filename}: {e}")

print(f"[INFO] Training Complete. Learned {len(known_face_encodings)} faces.")

if len(known_face_encodings) == 0:
    print("[ERROR] No faces were learned. Check your folder names and image files.")
    exit()

# ==========================================
# 2. Real-time Detection Logic (Webcam)
# ==========================================
print("[INFO] Starting Webcam...")
cap = cv2.VideoCapture(0)

# Load OpenCV's fast face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cant read from webcam.")
        break

    # Convert to grayscale for detection (faster)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    # Loop through every face detected in the frame
    for (x, y, w, h) in faces:
        # 1. Crop the face from the frame
        face_roi = frame[y:y + h, x:x + w]

        # Default settings (Assumed Unknown/Error)
        identity = "Error"
        best_distance = 100.0  # Start with a high number (no match)
        color = (0, 0, 255)  # Red for Error

        try:
            # 2. Get embedding for the live face
            # We enforce_detection=False because OpenCV already found the face for us
            results = DeepFace.represent(img_path=face_roi, model_name=MODEL_NAME, enforce_detection=False)

            if results:
                live_embedding = results[0]["embedding"]

                # 3. Compare live face with ALL known faces
                for i, known_encoding in enumerate(known_face_encodings):

                    # Calculate Cosine Distance
                    # Distance = 0 means identical, Distance = 1 means very different
                    a = np.array(live_embedding)
                    b = np.array(known_encoding)

                    # Math: Cosine Distance = 1 - Cosine Similarity
                    dot_product = np.dot(a, b)
                    norm_a = np.linalg.norm(a)
                    norm_b = np.linalg.norm(b)

                    cosine_distance = 1 - (dot_product / (norm_a * norm_b))

                    # Check if this is the closest match so far
                    if cosine_distance < best_distance:
                        best_distance = cosine_distance

                        # Only accept the name if the distance is BELOW the threshold
                        if cosine_distance < THRESHOLD:
                            identity = known_face_names[i]
                            color = (0, 255, 0)  # Green for Match

        except Exception as e:
            # If DeepFace fails on a blurry frame, just ignore this frame
            pass

        # 4. Draw the box and the name
        # Box around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Label background
        cv2.rectangle(frame, (x, y - 35), (x + w, y), color, cv2.FILLED)

        # Text (Name or Error)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, identity, (x + 6, y - 6), font, 0.8, (255, 255, 255), 1)

    # Show the video
    cv2.imshow('Face Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()