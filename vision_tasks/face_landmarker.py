#!/usr/bin/env python3
"""
MediaPipe Tasks API FaceLandmarker 웹캠 실시간 얼굴 랜드마크
모델: models/face_landmarker.task (공식 모델)
출력: 478 랜드마크, blendshape(52), facial transformation matrix
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
C = vision.FaceLandmarksConnections

# (연결선, BGR 색상) - 영역별 구분
REGIONS = [
    (C.FACE_LANDMARKS_TESSELATION, (90, 90, 90), 1),
    (C.FACE_LANDMARKS_FACE_OVAL, (255, 255, 255), 2),
    (C.FACE_LANDMARKS_LEFT_EYEBROW, (0, 165, 255), 2),
    (C.FACE_LANDMARKS_RIGHT_EYEBROW, (0, 165, 255), 2),
    (C.FACE_LANDMARKS_LEFT_EYE, (0, 255, 0), 2),
    (C.FACE_LANDMARKS_RIGHT_EYE, (0, 255, 0), 2),
    (C.FACE_LANDMARKS_LEFT_IRIS, (255, 255, 0), 2),
    (C.FACE_LANDMARKS_RIGHT_IRIS, (255, 255, 0), 2),
    (C.FACE_LANDMARKS_LIPS, (0, 0, 255), 2),
]
TOP_N = 8


def draw_face(frame, landmarks):
    """영역별 색상으로 얼굴 랜드마크 그리기"""
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for connections, color, thickness in REGIONS:
        for c in connections:
            cv2.line(frame, points[c.start], points[c.end], color, thickness)
    return points


def draw_blendshapes(frame, blendshapes):
    """score 상위 TOP_N개 blendshape를 왼쪽 위에 표시"""
    top = sorted(blendshapes, key=lambda b: b.score, reverse=True)[:TOP_N]
    cv2.rectangle(frame, (5, 5), (300, 15 + 24 * (len(top) + 1)), (0, 0, 0), -1)
    cv2.putText(frame, "Top blendshapes", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for i, b in enumerate(top, start=1):
        y = 25 + 24 * i
        cv2.putText(frame, f"{b.category_name}: {b.score:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (220, y - 10), (220 + int(b.score * 70), y), (0, 255, 0), -1)


def run_webcam():
    if not MODEL_PATH.exists():
        print(f"✗ 모델 파일이 없습니다: {MODEL_PATH}")
        return

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ 웹캠을 열 수 없습니다.")
        return

    print("✓ 웹캠 실행 중입니다. 'q' 또는 ESC를 눌러 종료")

    start = time.monotonic()
    last_ts = -1

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ts = max(int((time.monotonic() - start) * 1000), last_ts + 1)
            last_ts = ts
            result = landmarker.detect_for_video(mp_image, ts)

            if result.face_landmarks:
                for i, landmarks in enumerate(result.face_landmarks):
                    draw_face(frame, landmarks)
                    if result.face_blendshapes:
                        draw_blendshapes(frame, result.face_blendshapes[i])
                    # 추가 출력: 4x4 변환 행렬 (필요 시 사용)
                    # matrix = result.facial_transformation_matrixes[i]
            else:
                cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Face Landmarker - Press Q or ESC to Exit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("✓ 종료")


if __name__ == "__main__":
    run_webcam()
