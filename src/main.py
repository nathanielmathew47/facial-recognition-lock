from picamera2 import Picamera2
from pathlib import Path
import cv2
import numpy as np
import time

OUT_DIR = Path("data") 
OUT_DIR.mkdir(parents=True, exist_ok=True)

CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def capture_frame():
    cam = Picamera2()
    # 640x480 is snappy on a Pi; you can bump later
    cam.configure(cam.create_preview_configuration(main={"size": (640, 480)}))
    cam.start()
    time.sleep(0.5)  # let AE/AWB settle
    frame = cam.capture_array()  # ndarray, RGB
    cam.stop()
    return frame

def detect_and_save():
    frame_rgb = capture_frame()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

    out_path = OUT_DIR / "faces_detected.jpg"
    cv2.imwrite(str(out_path), frame_bgr)
    print(f"Detected {len(faces)} face(s). Saved annotated image to: {out_path.resolve()}")

if __name__ == "__main__":
    detect_and_save()