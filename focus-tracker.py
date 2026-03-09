import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import random
import pygame
import os
import sys
from collections import deque

# Platformdan bagimsiz olarak, sadece uygulama execution dosyasini kullanma imkani saglar.
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- Ses Sistemi Başlatma ---
pygame.mixer.init()

def play_sound():
    # Ses dosyasını yükle ve döngüye al (-1 değeri sonsuz döngü demektir)
    pygame.mixer.music.load(resource_path("utkucann.mp3"))
    pygame.mixer.music.play(loops=-1)

def stop_sound():
    pygame.mixer.music.stop()

# --- Modern Tasks API Kurulumu ---
base_options = python.BaseOptions(model_asset_path=resource_path('face_landmarker.task'))
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow("Focus Tracker", cv2.WINDOW_NORMAL)

LEFT_IRIS = [474,475,476,477]
RIGHT_IRIS = [469,470,471,472]
LEFT_EYE = [33,133]
RIGHT_EYE = [362,263]
LEFT_EYE_BOX = [33,160,158,133,153,144]
RIGHT_EYE_BOX = [362,385,387,263,373,380]

history = deque(maxlen=10)
playing = False
not_looking_start = None

def get_center(points, landmarks, w, h):
    coords = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in points]
    x = int(np.mean([p[0] for p in coords]))
    y = int(np.mean([p[1] for p in coords]))
    return (x, y)

def get_point(index, landmarks, w, h):
    return (int(landmarks[index].x * w), int(landmarks[index].y * h))

def draw_eye_box(indices, landmarks, frame, w, h):
    points = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in indices]
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    padding = 10

    cv2.rectangle(
        frame,
        (x_min-padding, y_min-padding),
        (x_max+padding, y_max+padding),
        (0,255,0),
        2
    )

def draw_gaze_line(iris, eye_left, eye_right, frame, color):
    eye_center = (
        int((eye_left[0] + eye_right[0]) / 2),
        int((eye_left[1] + eye_right[1]) / 2)
    )

    dx = iris[0] - eye_center[0]
    dy = iris[1] - eye_center[1]

    scale = 10

    end_point = (
        int(iris[0] + dx*scale),
        int(iris[1] + dy*scale)
    )

    cv2.line(frame, iris, end_point, color, 6)

    return end_point

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(mp_image)

    looking = False

    if results.face_landmarks:
        landmarks = results.face_landmarks[0]
        
        draw_eye_box(LEFT_EYE_BOX, landmarks, frame, w, h)
        draw_eye_box(RIGHT_EYE_BOX, landmarks, frame, w, h)

        left_iris = get_center(LEFT_IRIS, landmarks, w, h)
        right_iris = get_center(RIGHT_IRIS, landmarks, w, h)
        left_eye_l = get_point(LEFT_EYE[0], landmarks, w, h)
        left_eye_r = get_point(LEFT_EYE[1], landmarks, w, h)
        right_eye_l = get_point(RIGHT_EYE[0], landmarks, w, h)
        right_eye_r = get_point(RIGHT_EYE[1], landmarks, w, h)

        cv2.circle(frame, left_iris, 4, (255,0,0), -1)
        cv2.circle(frame, right_iris, 4, (255,0,0), -1)

        line_color = (0,0,255) if playing else (255,0,0)

        left_end = draw_gaze_line(left_iris, left_eye_l, left_eye_r, frame, line_color)
        right_end = draw_gaze_line(right_iris, right_eye_l, right_eye_r, frame, line_color)

        left_ratio = (left_iris[0] - left_eye_l[0]) / (left_eye_r[0] - left_eye_l[0])
        right_ratio = (right_iris[0] - right_eye_l[0]) / (right_eye_r[0] - right_eye_l[0])

        gaze_ratio = (left_ratio + right_ratio)/2

        nose = get_point(1, landmarks, w, h)
        left_face = get_point(234, landmarks, w, h)
        right_face = get_point(454, landmarks, w, h)
        face_width = right_face[0] - left_face[0]
        nose_offset = (nose[0] - (left_face[0] + face_width/2)) / face_width

        score = abs(gaze_ratio - 0.5) + abs(nose_offset)
        history.append(score)
        if np.mean(history) < 0.4: looking = True

    if not looking:
        if not_looking_start is None:
            not_looking_start = time.time()
        elapsed = time.time() - not_looking_start

        if elapsed > 1.5:
            if not playing:
                play_sound()
                playing = True

    else:
        not_looking_start = None
        if playing:
            stop_sound()
            playing = False

    if playing:
        text = "YINE DDDEE ASK BOYUN EEEGGMEEEZZZ"
        mid_x = int((left_iris[0] + right_iris[0]) / 2)
        mid_y = int((left_iris[1] + right_iris[1]) / 2) - 30
        shake_x = random.randint(-10, 10)
        shake_y = random.randint(-10, 10)

        cv2.putText(
            frame,
            text,
            (mid_x - 250 + shake_x, mid_y + shake_y),
            cv2.FONT_HERSHEY_DUPLEX,
            1.2,
            (0,0,255),
            4
        )

    else:
        cv2.putText(frame,"OK",(30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) ,2)

    cv2.imshow("Focus Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    # Çıkmak için ESC veya Q'ya bas.
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()