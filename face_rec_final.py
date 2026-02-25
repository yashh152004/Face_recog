import cv2
import os
import numpy as np
import csv
import json
import uuid
from datetime import datetime
from deepface import DeepFace
from antispoof import is_live_face  # MiniFASNetV2 anti-spoof

# Google Sheets
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ----------------- CONFIG -----------------
TARGET_FOLDERS = ["Harini", "Thrisha", "Trupti","Yash"]
MODEL_NAME = "Facenet512"
THRESHOLD = 0.35
FRAME_SKIP = 4
SESSION = "Morning"

CSV_FILE = "attendance.csv"
JSON_FILE = "attendance.json"
UNKNOWN_DIR = "Unknown_log"
CREDS_FILE = "credentials.json"

os.makedirs(UNKNOWN_DIR, exist_ok=True)

# ----------------- GOOGLE SHEETS -----------------
SCOPE = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]

CREDS = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
client = gspread.authorize(CREDS)
spreadsheet = client.open("Attendence")
known_sheet = spreadsheet.worksheet("Attendence")
unknown_sheet = spreadsheet.worksheet("Unknown_Faces")

# ----------------- INITIALIZE DEEPFACE -----------------
print("[INFO] Initializing DeepFace...")
DeepFace.build_model(MODEL_NAME)

# ----------------- TRAINING PHASE -----------------
known_face_encodings = []
known_face_names = []

print("[INFO] Loading training images...")
for name in TARGET_FOLDERS:
    person_dir = os.path.join(os.getcwd(), name)
    if not os.path.exists(person_dir):
        continue
    for file in os.listdir(person_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(person_dir, file)
            try:
                rep = DeepFace.represent(img_path=path, model_name=MODEL_NAME, enforce_detection=True)
                known_face_encodings.append(np.array(rep[0]["embedding"]))
                known_face_names.append(name)
                print(f"✔ Learned {name} - {file}")
            except:
                print(f"[SKIPPED] {file}")

if not known_face_encodings:
    print("[ERROR] No training faces found.")
    exit()

# ----------------- JSON HANDLING -----------------
def load_json():
    if not os.path.exists(JSON_FILE):
        return []
    try:
        with open(JSON_FILE, "r") as f:
            content = f.read().strip()
            return json.loads(content) if content else []
    except json.JSONDecodeError:
        return []

def save_json(data):
    with open(JSON_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ----------------- HELPERS -----------------
marked_today = set()
unknown_embeddings = []

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def save_unknown_image(face):
    name = f"Unknown_{uuid.uuid4().hex}.jpg"
    path = os.path.join(UNKNOWN_DIR, name)
    cv2.imwrite(path, face)
    return name

def is_duplicate_unknown(embedding, threshold=0.25):
    for e in unknown_embeddings:
        if cosine_distance(embedding, e) < threshold:
            return True
    return False

# ----------------- LOGGING -----------------
def log_known(name, confidence):
    today = datetime.now().strftime("%Y-%m-%d")
    key = f"{name}_{today}"
    if key in marked_today:
        return
    now = datetime.now()
    file_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Date", "Time", "Session", "Status", "Confidence"])
        writer.writerow([name, today, now.strftime("%H:%M:%S"), SESSION, "Present", f"{confidence:.2f}"])
    data = load_json()
    data.append({"type": "known", "name": name, "date": today, "time": now.strftime("%H:%M:%S"),
                 "session": SESSION, "status": "Present", "confidence": round(confidence, 2)})
    save_json(data)
    known_sheet.append_row([name, today, now.strftime("%H:%M:%S"), SESSION, "Present", f"{confidence:.2f}"])
    marked_today.add(key)
    print(f"[ATTENDANCE] {name} marked")

def log_unknown(image_name):
    now = datetime.now()
    data = load_json()
    data.append({"type": "unknown", "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S"),
                 "session": SESSION, "reason": "Face not recognized", "image_name": image_name})
    save_json(data)
    unknown_sheet.append_row([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), SESSION,
                              "Face not recognized", image_name])
    print("[UNKNOWN] Logged once")

# ----------------- WEBCAM LOOP -----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("[INFO] Press Q to quit")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(80, 80))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # ----------------- ANTI-SPOOF -----------------
        try:
            live_prob = is_live_face(face, threshold=0.50)  # lower threshold
            print("[ANTI-SPOOF] Live probability:", live_prob)
        except:
            live_prob = True  # fallback if anti-spoof fails

        if not live_prob:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "SPOOF", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            continue

        # ----------------- FACE RECOGNITION -----------------
        try:
            rep = DeepFace.represent(face, model_name=MODEL_NAME, enforce_detection=True)
            live_embedding = np.array(rep[0]["embedding"])
        except:
            continue

        best_dist = 1.0
        best_name = "Unknown"

        for i, known in enumerate(known_face_encodings):
            dist = cosine_distance(live_embedding, known)
            if dist < best_dist:
                best_dist = dist
                best_name = known_face_names[i]

        confidence = max(0, (1 - best_dist)) * 100

        if best_dist < THRESHOLD:
            log_known(best_name, confidence)
            label = f"{best_name}"
            color = (0, 255, 0)
        else:
            if not is_duplicate_unknown(live_embedding):
                unknown_embeddings.append(live_embedding)
                img_name = save_unknown_image(face)
                log_unknown(img_name)
            label = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Program exited successfully")