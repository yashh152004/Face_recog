# antispoof.py
import torch
import cv2
from torchvision import transforms
from models.mini_fasnet import MiniFASNetV2  # Make sure checkpoint matches

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"d:\New folder (21)\face_recog\2.7_80x80_MiniFASNetV2.pth"

# ----------------- LOAD MODEL -----------------
model = MiniFASNetV2()
model.to(DEVICE)
model.eval()

# Load checkpoint safely
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
new_state_dict = {}
for k, v in state_dict.items():
    new_state_dict[k.replace("module.", "")] = v
model.load_state_dict(new_state_dict, strict=False)

# ----------------- PREPROCESS -----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ----------------- ANTI-SPOOF FUNCTION -----------------
def is_live_face(face_img, threshold=0.50, return_prob=False):
    """
    Returns True if LIVE, False if SPOOF by default.
    Can also return live probability with return_prob=True.
    
    Args:
        face_img: BGR image from webcam
        threshold: live probability threshold
        return_prob: if True, returns probability instead of True/False
    """
    try:
        # resize & convert
        face = cv2.resize(face_img, (80, 80))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = transform(face).unsqueeze(0).to(DEVICE)

        # predict
        with torch.no_grad():
            output = model(face)
            prob = torch.softmax(output, dim=1)[0][1].item()  # LIVE probability

        if return_prob:
            return prob
        return prob > threshold

    except Exception as e:
        print("[ANTI-SPOOF ERROR]", e)
        return False if not return_prob else 0.0