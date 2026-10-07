#!/usr/bin/env python3
"""
OpenCV 기반 간단한 손 탐지
(MediaPipe 모델이 없을 때 사용하는 대체 버전)
"""

import cv2
import numpy as np
from pathlib import Path
import os

def detect_hand_edges(image):
    """피부색 검출을 통한 손 탐지"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 피부색 범위 (HSV)
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])

    mask1 = cv2.inRange(hsv, lower_skin, upper_skin)

    lower_skin = np.array([170, 20, 70])
    upper_skin = np.array([180, 255, 255])

    mask2 = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = mask1 | mask2

    # 모폴로지 연산
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 컨투어 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return mask, contours

def run_webcam():
    """웹캠에서 손 탐지"""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("✗ 웹캠을 열 수 없습니다.")
        return

    print("✓ 웹캠 실행 중입니다.")
    print("  - 'q'를 눌러 종료")
    print("  - 주의: 이것은 OpenCV 기반의 간단한 구현입니다.\n")

    frame_count = 0
    min_area = 5000

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # 손 탐지
        mask, contours = detect_hand_edges(frame)

        # 프레임 정보
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        hands_detected = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > min_area:
                hands_detected += 1

                # 바운딩 박스
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

                # 컨투어 그리기
                cv2.drawContours(frame, [cnt], 0, (255, 0, 0), 2)

                # 라벨
                cv2.putText(frame, "Hand", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if hands_detected > 0:
            cv2.putText(frame, f"Hands: {hands_detected}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hands detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Hand Detection - Press Q to Exit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ 종료. 총 {frame_count}개 프레임 처리됨.")

def process_image(image_path):
    """이미지 파일 처리"""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"✗ 이미지를 열 수 없습니다: {image_path}")
        return

    mask, contours = detect_hand_edges(image)

    hands_detected = 0
    min_area = 5000

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > min_area:
            hands_detected += 1

            # 바운딩 박스
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 컨투어 그리기
            cv2.drawContours(image, [cnt], 0, (255, 0, 0), 2)

    if hands_detected > 0:
        print(f"✓ 손 {hands_detected}개 감지됨")
    else:
        print("✗ 손이 감지되지 않았습니다.")

    output_path = Path(image_path).stem + "_hands.jpg"
    cv2.imwrite(output_path, image)
    print(f"\n✓ 결과 저장: {output_path}")

    cv2.imshow('Hand Detection', image)
    print("  아무 키나 눌러 닫기...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print("=" * 50)
    print("  Hand Detection (OpenCV 기반)")
    print("=" * 50)
    print("  (MediaPipe 모델 없이 사용 가능)")
    print("=" * 50)
    print()

    print("선택 메뉴:")
    print("  1. 웹캠 실행 (실시간 손 탐지)")
    print("  2. 이미지 파일 처리")
    print("  0. 종료")
    print()

    choice = input("선택 (0-2): ").strip()
    print()

    if choice == '1':
        run_webcam()
    elif choice == '2':
        image_path = input("이미지 파일 경로를 입력하세요: ").strip()
        if image_path:
            process_image(image_path)
        else:
            print("✗ 경로가 입력되지 않았습니다.")
    elif choice == '0':
        print("✓ 프로그램을 종료합니다.")
    else:
        print("✗ 유효하지 않은 선택입니다.")

if __name__ == "__main__":
    main()
