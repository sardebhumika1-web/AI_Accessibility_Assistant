import os
import time
import threading
from collections import Counter

import cv2
import easyocr
import numpy as np
from PIL import Image
from ultralytics import YOLO

from config import (
    MODEL_PATH,
    OCR_LANGUAGES,
    OUTPUT_FOLDER,
    VOICE_RATE,
)

# ==========================================================
# Load YOLO Model
# ==========================================================

model = YOLO(MODEL_PATH)

# ==========================================================
# Load EasyOCR
# ==========================================================

reader = easyocr.Reader(
    OCR_LANGUAGES,
    gpu=False
)

# ==========================================================
# Voice Assistant
# ==========================================================
# ======================================================
# Text To Speech
# ======================================================

engine = pyttsx3.init()

engine.setProperty("rate", VOICE_RATE)
engine.setProperty("volume", VOICE_VOLUME)

speech_lock = threading.Lock()

_last_spoken = ""


def speak(text):
    """
    Speak text safely in Streamlit.
    """

    global _last_spoken

    if not text:
        return

    if not st.session_state.get("voice", True):
        return

    if text == _last_spoken:
        return

    _last_spoken = text

    def worker():

        with speech_lock:

            try:

                engine.stop()

                engine.say(text)

                engine.runAndWait()

            except Exception as e:

                print("Speech Error:", e)

    threading.Thread(
        target=worker,
        daemon=True,
    ).start()

# ==========================================================
# Object Detection
# ==========================================================

def detect_objects(frame, confidence=0.45):

    try:

        results = model.predict(
            frame,
            conf=confidence,
            verbose=False
        )[0]

        annotated = results.plot()

        detections = []

        height, width = frame.shape[:2]

        for box in results.boxes:

            cls = int(box.cls.item())

            conf = float(box.conf.item())

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = model.names[cls]

            # Direction

            center_x = (x1 + x2) / 2

            if center_x < width / 3:
                direction = "left"

            elif center_x > (2 * width / 3):
                direction = "right"

            else:
                direction = "center"

            # Distance

            box_height = y2 - y1

            if box_height > 300:
                distance = "Very Close"

            elif box_height > 200:
                distance = "Close"

            elif box_height > 100:
                distance = "Medium"

            else:
                distance = "Far"

            detections.append({

                "label": label,

                "confidence": round(conf, 2),

                "direction": direction,

                "distance": distance,

                "box": (x1, y1, x2, y2)

            })

        return annotated, detections

    except Exception as e:

        print("Detection Error:", e)

        return frame, []


# ==========================================================
# OCR
# ==========================================================

def read_text(image):

    try:

        result = reader.readtext(image, detail=0)

        if not result:
            return ""

        text = "\n".join(result)

        return text

    except Exception as e:

        print("OCR Error:", e)

        return ""


# ==========================================================
# Upload Image
# ==========================================================

def load_uploaded_image(file):

    image = Image.open(file).convert("RGB")

    return np.array(image)


# ==========================================================
# Detection Summary
# ==========================================================

def detection_summary(detections):

    labels = [d["label"] for d in detections]

    return dict(Counter(labels))


# ==========================================================
# Save Image
# ==========================================================

def save_image(image):

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    filename = f"detection_{int(time.time())}.jpg"

    path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    cv2.imwrite(path, image)

    return path


# ==========================================================
# Detection History
# ==========================================================

history = []

_last_history = None


def add_history(detections):

    global _last_history

    labels = sorted([d["label"] for d in detections])

    if labels == _last_history:
        return

    _last_history = labels

    history.append({

        "time": time.strftime("%H:%M:%S"),

        "detections": detections

    })


def get_history():

    return history