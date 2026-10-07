#!/usr/bin/env python3
"""
MediaPipe Tasks API GestureRecognizer 웹캠 실시간 제스처 인식
모델: models/gesture_recognizer.task (공식 모델, 사전 다운로드)
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gesture_recognizer.task"
CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


def draw_hand(frame, landmarks):
    """손 랜드마크 21개와 연결선 그리기"""
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for c in CONNECTIONS:
        cv2.line(frame, points[c.start], points[c.end], (0, 255, 0), 2)
    for p in points:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)
    return points


def run_webcam():
    if not MODEL_PATH.exists():
        print(f"✗ 모델 파일이 없습니다: {MODEL_PATH}")
        return

    options = vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ 웹캠을 열 수 없습니다.")
        return

    print("✓ 웹캠 실행 중입니다. 'q' 또는 ESC를 눌러 종료")

    start = time.monotonic()
    last_ts = -1

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ts = max(int((time.monotonic() - start) * 1000), last_ts + 1)
            last_ts = ts
            result = recognizer.recognize_for_video(mp_image, ts)

            if result.hand_landmarks:
                for i, landmarks in enumerate(result.hand_landmarks):
                    points = draw_hand(frame, landmarks)

                    hand = result.handedness[i][0].category_name if result.handedness else "?"
                    if result.gestures and result.gestures[i]:
                        g = result.gestures[i][0]
                        label = f"{hand}: {g.category_name} ({g.score:.2f})"
                    else:
                        label = f"{hand}: None"

                    x = min(p[0] for p in points)
                    y = max(min(p[1] for p in points) - 10, 20)
                    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "No hands detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Gesture Recognition - Press Q or ESC to Exit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("✓ 종료")


if __name__ == "__main__":
    run_webcam()
