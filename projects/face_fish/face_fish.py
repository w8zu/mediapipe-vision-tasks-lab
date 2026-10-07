#!/usr/bin/env python3
"""
Face Fish (face_fish.py) - 미니게임: pinch로 물고기 집기/드래그/놓기 + 입 추적 + 먹기 판정 + 물고기별 얼굴 효과(local morph + soft tint + particles)
모델: models/hand_landmarker.task, models/face_landmarker.task (공식 모델)
"""

import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "hand_landmarker.task"
FACE_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "face_landmarker.task"
CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
LIPS_CONNECTIONS = vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS

# 얼굴 랜드마크 인덱스
INNER_LIP_TOP, INNER_LIP_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
FOREHEAD, CHIN = 10, 152
# 입 벌림 = 안쪽 윗입술~아랫입술 거리 / 얼굴 높이(이마~턱). 이 값 이상이면 MOUTH OPEN
MOUTH_OPEN_THRESHOLD = 0.06
# 먹기 판정: 물고기 중심~입 중심 거리 / 얼굴 높이 가 이 값 이하이고 입이 열려 있으면 먹은 것으로 처리
EAT_DISTANCE_RATIO = 0.25
YUM_DURATION = 0.9  # 먹었을 때 "+점수" 표시 시간(초)

# 미니게임 설정
GAME_DURATION = 45.0   # 게임 시간(초)
FISH_MIN_SPEED = 70    # 물고기 좌우 이동 최소 속도(px/초)
FISH_MAX_SPEED = 140   # 물고기 좌우 이동 최대 속도(px/초)
SWIM_BOB_AMPLITUDE = 6     # 헤엄치며 위아래로 흔들리는 폭(px, 작게 유지)
SWIM_BOB_SPEED = 3.0       # 위아래 흔들림 속도(rad/초)
FISH_AREA_HEIGHT = 170  # 물고기가 움직이는 화면 하단 영역 높이(px, 물고기 지름보다 커야 함)
RESPAWN_DELAY = 1.5    # 먹은 물고기가 다시 나타나기까지 시간(초)
# 상호작용 피드백 설정 (DEBUG와 무관하게 항상 표시되는 게임 UI)
INTERACTION_DISTANCE = 110        # pinch 중심이 물고기 중심에서 이 거리(px) 안이면 "잡을 수 있음" 물결 표시
FISH_HOVER_PULSE_STRENGTH = 0.5   # 손이 가까울 때 물결의 진하기 (0~1)
GRAB_PULSE_DURATION = 0.35        # 잡는 순간 pulse 시간(초)
MOUTH_RIPPLE_DURATION = 0.9       # 물고기를 든 채 입을 열었을 때 입 주변 물결 시간(초)
MOUTH_RIPPLE_SIZE = 0.12          # 입 물결 크기 (얼굴 높이 × 값)
MOUTH_ATTRACTION_STRENGTH = 0.30  # 입 쪽으로 끌리는 최대 비율 (입에 닿을 때 손-입 거리의 몇 %까지 당길지)
MOUTH_ATTRACTION_RANGE = 2.5      # 끌림이 시작되는 거리 (먹기 거리 × 값)
EAT_POP_DURATION = 0.45           # 먹는 순간 물고기가 커졌다 사라지는 시간(초)
# 상호작용 marker (손 pinch 중심과 입에 같은 디자인으로 표시, DEBUG와 무관하게 항상 표시)
INTERACTION_MARKER_RADIUS = 12        # marker 바깥 링 반지름(px)
INTERACTION_MARKER_ALPHA = 0.28       # idle 상태 불투명도 (매우 은은하게)
INTERACTION_MARKER_ACTIVE_SCALE = 1.3    # active 상태에서 커지는 배율
INTERACTION_MARKER_ACTIVE_ALPHA = 0.65   # active 상태 불투명도
INTERACTION_MARKER_LINE_WIDTH = 1     # 링 두께(px)
INTERACTION_MARKER_COLOR = (250, 245, 225)  # BGR, 손/입/물결 공통 (pale cyan-white)
DEBUG = False          # True면 손/입 추적 정보를 화면에 표시
EFFECT_DURATION = 10.0  # 물고기를 먹은 뒤 얼굴 효과가 유지되는 시간(초)
MAX_EFFECT_LEVEL = 3        # 효과 레벨 상한: 같은 물고기를 이보다 많이 먹어도 효과는 여기서 멈춤

# 얼굴 효과 공통 규칙: local morph + soft tint + small particles + (LV3) subtle pulse. 레벨은 같은 효과의 강도만 키움 (LV1, LV2, LV3 순)
MORPH_STRENGTH = {"Red Fish": (0.35, 0.55, 0.75),      # 볼 bulge   (0~1: 클수록 많이 부풀어 보임, 중심이 약 1/(1-값)배 확대)
                  "Yellow Fish": (0.12, 0.20, 0.28),   # 눈 주변 bulge
                  "Green Fish": (0.10, 0.16, 0.22)}    # 이마/윗얼굴 bulge
MORPH_STRENGTH_MAX = 0.80       # 안전 상한 (얼굴이 망가지지 않도록 위 값이 커도 여기서 제한)
CHEEK_WARP_RADIUS_RATIO = 0.30  # 볼 morph 반경 = 얼굴 너비 × 이 값
CHEEK_WARP_OUTWARD = 0.06       # 볼 morph 중심을 얼굴 바깥쪽으로 옮기는 정도 (얼굴 너비 × 이 값)
EYE_WARP_RADIUS_RATIO = 0.11    # 눈 morph 반경
FOREHEAD_WARP_RADIUS_RATIO = 0.26  # 이마 morph 반경
EFFECT_TINT_ALPHA = (0.20, 0.28, 0.36)  # 레벨별 soft tint 진하기 (0~1)
EFFECT_PARTICLES = (4, 9, 15)           # 레벨별 파티클 개수
EFFECT_PARTICLE_SIZE = 0.014            # 파티클 반지름 (얼굴 너비 × 값, 개별 ±30% 차이)
EFFECT_PARTICLE_ALPHA = 0.75            # 파티클 최대 불투명도 (0~1)
EFFECT_PULSE_LV3 = 0.18                 # LV3 tint가 천천히 숨 쉬는 정도 (0이면 pulse 없음)
EFFECT_FADE_IN, EFFECT_FADE_OUT = 0.8, 1.0  # 효과가 서서히 나타나고 사라지는 시간(초)
DEBUG_WARP = False              # True면 morph 중심/반경을 원으로 표시

# 물고기 그래픽 (모양은 모두 같고 색만 다름)
FISH_ALPHA = 0.60               # 몸통 불투명도 (반투명)
FISH_GLOW_ALPHA = 0.28          # 몸통 주위 soft glow 진하기
RARITY_GLOW = {"Common": 1.0, "Uncommon": 1.25, "Rare": 1.6, "Epic": 2.0}  # 등급이 높을수록 glow만 조금 더 큼 (링/문구 없음)

# UI
UI_FONT = cv2.FONT_HERSHEY_DUPLEX
UI_TEXT_COLOR = (245, 245, 240)  # BGR
UI_TEXT_ALPHA = 0.9
UI_WARN_COLOR = (120, 128, 240)  # 남은 시간 10초 이하 (coral)

# 효과용 얼굴 랜드마크 인덱스
LEFT_CHEEK, RIGHT_CHEEK = 205, 425
FACE_LEFT, FACE_RIGHT = 234, 454
IRIS_CENTERS = (468, 473)
GLABELLA = 168  # 미간 (이마 morph/tint 기준점)
FACE_OVAL = vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL

THUMB_TIP, INDEX_TIP, WRIST, MIDDLE_MCP = 4, 8, 0, 9
PINCH_ON = 0.30   # 손 크기 대비 엄지-검지 거리 비율: 이하면 pinch 시작
PINCH_OFF = 0.40  # 이상이면 pinch 해제 (떨림 방지용 히스테리시스)

# 물고기 카탈로그 (이름, BGR 색상, rarity). 세 색은 채도/밝기를 맞춘 한 세트: coral / muted gold / mint-sage
# Epic 물고기는 아직 없음 - 추가하면 자동으로 등장 풀에 포함됨
FISH_TYPES = [("Red Fish", (120, 128, 240), "Common"), ("Yellow Fish", (110, 196, 232), "Uncommon"),
              ("Green Fish", (170, 208, 140), "Rare")]
FISH_COLORS = {name: color for name, color, _ in FISH_TYPES}
FISH_RADIUS = 40

# rarity 설정: 재등장할 때 등급을 먼저 이 가중치로 뽑고, 같은 등급 안에서는 균등하게 뽑음
FISH_SPAWN_WEIGHTS = {"Common": 60, "Uncommon": 28, "Rare": 10, "Epic": 2}
RARITY_SCORE = {"Common": 1, "Uncommon": 2, "Rare": 5, "Epic": 10}  # 먹었을 때 점수
HIGH_RARITIES = ("Rare", "Epic")  # 희소성 유지를 위해 화면 동시 등장 수를 제한하는 등급
MAX_HIGH_RARITY_ON_SCREEN = 1     # Rare/Epic 물고기가 화면에 동시에 있을 수 있는 최대 개수
FISH_SLOTS = 5                    # 화면에 동시에 있는 물고기 수 (이전 3마리의 약 1.5배)
RARE_FLASH_DURATION = 1.2         # Rare/Epic 등장 시 퍼지는 물결 시간(초)


def fish_band(h):
    """물고기 중심 y가 움직일 수 있는 범위 (화면 하단 영역)"""
    return h - FISH_AREA_HEIGHT + FISH_RADIUS, h - FISH_RADIUS


def random_velocity():
    return random.choice((-1, 1)) * random.uniform(FISH_MIN_SPEED, FISH_MAX_SPEED)


def pick_fish(fishes, skip=None):
    """rarity 가중치로 물고기 하나를 뽑음. Rare/Epic이 이미 화면에 최대치면 그 등급은 제외"""
    on_screen = sum(1 for j, f in enumerate(fishes) if j != skip and f["state"] != "hidden" and f["rarity"] in HIGH_RARITIES)
    rarities = [r for r in FISH_SPAWN_WEIGHTS if any(fd[2] == r for fd in FISH_TYPES)]
    if on_screen >= MAX_HIGH_RARITY_ON_SCREEN:
        rarities = [r for r in rarities if r not in HIGH_RARITIES] or rarities
    rarity = random.choices(rarities, weights=[FISH_SPAWN_WEIGHTS[r] for r in rarities])[0]
    return random.choice([fd for fd in FISH_TYPES if fd[2] == rarity])


def apply_fish(fish, spec, now):
    """슬롯(fish)에 뽑힌 물고기(spec) 정보를 적용. Rare/Epic이면 등장 강조 시간을 설정"""
    fish["name"], fish["color"], fish["rarity"] = spec
    fish["flash_until"] = now + RARE_FLASH_DURATION if spec[2] in HIGH_RARITIES else 0.0


def make_fishes(w, h, now=None):
    """물고기 FISH_SLOTS개를 하단 영역에 균등 간격으로 배치. state: moving / held / hidden"""
    now = time.monotonic() if now is None else now
    y_min, y_max = fish_band(h)
    fishes = []
    for i in range(FISH_SLOTS):
        vx = random_velocity()
        fish = {"state": "moving", "respawn_at": 0.0, "vx": vx, "rarity": "Common",
                "facing": 1 if vx > 0 else -1, "phase": random.uniform(0, 2 * math.pi),
                "grab_at": -1e9, "eaten_at": -1e9,
                "pos": (int(w * (i + 1) / (FISH_SLOTS + 1)), y_min + (i % 3) / 2 * (y_max - y_min))}  # 3개 레인에 번갈아 배치해 겹침 최소화
        apply_fish(fish, pick_fish(fishes), now)
        fishes.append(fish)
    return fishes


def respawn_fish(fishes, i, w, h, now):
    """rarity 가중치로 새 물고기를 뽑고, 다른 물고기와 가장 멀리 떨어진 위치(후보 12개 중)에 랜덤 방향/속도로 재등장"""
    y_min, y_max = fish_band(h)
    others = [f["pos"] for j, f in enumerate(fishes) if j != i and f["state"] != "hidden"]
    candidates = [(random.uniform(FISH_RADIUS, w - FISH_RADIUS), random.uniform(y_min, y_max)) for _ in range(12)]
    pos = max(candidates, key=lambda c: min((math.dist(c, o) for o in others), default=0))  # 다른 물고기와 가장 먼 후보
    apply_fish(fishes[i], pick_fish(fishes, skip=i), now)
    vx = random_velocity()
    fishes[i].update(pos=pos, vx=vx, facing=1 if vx > 0 else -1, state="moving")


def update_fishes(fishes, w, h, dt, now):
    """moving 물고기는 좌우로 이동(벽에서 반사), hidden 물고기는 시간이 되면 재등장, held는 건드리지 않음"""
    y_min, y_max = fish_band(h)
    for i, fish in enumerate(fishes):
        if fish["state"] == "hidden":
            if now >= fish["respawn_at"]:
                respawn_fish(fishes, i, w, h, now)
        elif fish["state"] == "moving":
            x, y = fish["pos"]
            x += fish["vx"] * dt
            fish["phase"] += SWIM_BOB_SPEED * dt
            y += SWIM_BOB_AMPLITUDE * SWIM_BOB_SPEED * math.cos(fish["phase"]) * dt  # 사인파 위아래 흔들림
            if x < FISH_RADIUS:
                x, fish["vx"] = FISH_RADIUS, abs(fish["vx"])
            elif x > w - FISH_RADIUS:
                x, fish["vx"] = w - FISH_RADIUS, -abs(fish["vx"])
            y += (min(max(y, y_min), y_max) - y) * 0.2  # 놓은 위치가 하단 영역 밖이면 부드럽게 복귀
            fish["facing"] = 1 if fish["vx"] > 0 else -1  # 이동 방향으로 머리 향함
            fish["pos"] = (x, y)


def draw_fish_shape(frame, x, y, color, facing, wag, scale=1.0, glow=1.0, alpha=1.0):
    """부드러운 타원 몸통 + 단순한 꼬리, 반투명 + soft glow. 물고기 주변 작은 영역만 처리 (facing: 1=머리 오른쪽, -1=왼쪽)"""
    r, f = int(FISH_RADIUS * scale), facing
    if r < 4:
        return
    h, w = frame.shape[:2]
    pad = int(r * 2.0) + 4
    x0, x1, y0, y1 = max(x - pad, 0), min(x + pad + 1, w), max(y - pad, 0), min(y + pad + 1, h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    ox, oy = x - x0, y - y0
    shape = np.zeros(roi.shape[:2], np.float32)
    cv2.ellipse(shape, (ox, oy), (r, int(r * 0.55)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    tail = np.array([(ox - f * 0.7 * r, oy), (ox - f * 1.2 * r, oy - 0.38 * r + wag), (ox - f * 1.55 * r, oy - 0.5 * r + wag),
                     (ox - f * 1.3 * r, oy + wag * 0.5), (ox - f * 1.55 * r, oy + 0.5 * r + wag),
                     (ox - f * 1.2 * r, oy + 0.38 * r + wag)], dtype=np.int32)
    cv2.fillPoly(shape, [tail], 1.0, cv2.LINE_AA)
    body = cv2.GaussianBlur(shape, (0, 0), max(r * 0.05, 1.0))   # 가장자리를 부드럽게 (물고기 영역만 blur)
    small = cv2.resize(shape, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)  # 큰 blur는 축소본에서 (속도)
    halo = cv2.resize(cv2.GaussianBlur(small, (0, 0), r * 0.1), (shape.shape[1], shape.shape[0]), interpolation=cv2.INTER_LINEAR)
    blend_mask(roi, halo, color, FISH_GLOW_ALPHA * glow * alpha)
    blend_mask(roi, body, color, FISH_ALPHA * alpha)
    light = np.zeros_like(shape)  # 등 쪽 은은한 하이라이트
    cv2.ellipse(light, (int(ox + f * 0.15 * r), int(oy - 0.2 * r)), (int(r * 0.55), max(int(r * 0.16), 1)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    blend_mask(roi, cv2.GaussianBlur(light, (0, 0), r * 0.12) * body, (255, 255, 255), 0.30 * alpha)
    eye = np.zeros_like(shape)  # 눈은 아주 작은 점
    cv2.circle(eye, (int(ox + f * 0.58 * r), int(oy - 0.1 * r)), max(int(r * 0.06), 1), 1.0, -1, cv2.LINE_AA)
    blend_mask(roi, eye, (70, 60, 55), 0.6 * alpha)


def draw_ripple(frame, center, radius, alpha, color=INTERACTION_MARKER_COLOR, thickness=1):
    """center를 중심으로 한 반투명 얇은 물결 링 하나 (작은 영역만 블렌딩, 배경은 그대로)"""
    if alpha <= 0.01 or radius < 2:
        return
    h, w = frame.shape[:2]
    x, y, r = int(center[0]), int(center[1]), int(radius) + thickness + 2
    x0, x1, y0, y1 = max(x - r, 0), min(x + r + 1, w), max(y - r, 0), min(y + r + 1, h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    layer = roi.copy()
    cv2.circle(layer, (x - x0, y - y0), int(radius), color, thickness, cv2.LINE_AA)
    cv2.addWeighted(layer, min(alpha, 1.0), roi, 1 - min(alpha, 1.0), 0, roi)


def draw_fishes(frame, fishes, held, hover=None):
    now = time.monotonic()
    for i, fish in enumerate(fishes):
        x, y = int(fish["pos"][0]), int(fish["pos"][1])
        wag = math.sin(now * 10 + fish["phase"]) * FISH_RADIUS * 0.2  # 꼬리 흔들기
        glow = RARITY_GLOW.get(fish["rarity"], 1.0)
        if fish["state"] == "hidden":
            u = (now - fish["eaten_at"]) / EAT_POP_DURATION
            if 0 <= u < 1:  # 먹힌 직후: 살짝 커졌다가 작아지며 사라짐 + 물결 한 번
                scale = 1 + 0.2 * u / 0.3 if u < 0.3 else 1.2 * (1 - (u - 0.3) / 0.7)
                draw_fish_shape(frame, x, y, fish["color"], fish["facing"], wag, scale, glow, 1 - u)
                draw_ripple(frame, (x, y), FISH_RADIUS * (0.6 + u), 0.5 * (1 - u))
            continue
        grab_u = (now - fish["grab_at"]) / GRAB_PULSE_DURATION
        scale = 1 + 0.15 * math.sin(math.pi * grab_u) if 0 <= grab_u < 1 else 1.0  # 잡는 순간 pop
        draw_fish_shape(frame, x, y, fish["color"], fish["facing"], wag, scale, glow)
        if 0 <= grab_u < 1:
            draw_ripple(frame, (x, y), FISH_RADIUS * (1.0 + 0.9 * grab_u), 0.6 * (1 - grab_u))
        elif i == held:
            draw_ripple(frame, (x, y), FISH_RADIUS + 6, 0.25)  # 잡고 있는 동안은 아주 은은하게
        if i == hover and held is None:  # 손이 가장 가까운 물고기만: 중심에서 퍼지는 물결 두 겹
            for offset in (0.0, 0.5):
                t = (now * 1.1 + i * 0.3 + offset) % 1.0
                draw_ripple(frame, (x, y), FISH_RADIUS * (0.8 + 0.9 * t), FISH_HOVER_PULSE_STRENGTH * (1 - t))
        remain = fish["flash_until"] - now
        if remain > 0:  # Rare/Epic 등장: 물고기 색 물결이 한 번 퍼짐 (링/문구 없음)
            t = 1.0 - remain / RARE_FLASH_DURATION
            draw_ripple(frame, (x, y), FISH_RADIUS + 10 + t * 40, 0.4 * (1 - t), fish["color"])


def attracted_pos(hand_pos, mouth_center, face_h):
    """손 위치를 기준으로, 입에 가까울수록 물고기를 입 쪽으로 아주 약간 당긴 위치 (누적되지 않아 조작감 유지)"""
    if mouth_center is None or face_h <= 0:
        return hand_pos
    reach = MOUTH_ATTRACTION_RANGE * EAT_DISTANCE_RATIO * face_h
    d = math.dist(hand_pos, mouth_center)
    if d >= reach:
        return hand_pos
    k = MOUTH_ATTRACTION_STRENGTH * (1 - d / reach) ** 2
    return (hand_pos[0] + (mouth_center[0] - hand_pos[0]) * k, hand_pos[1] + (mouth_center[1] - hand_pos[1]) * k)


def draw_marker(frame, center, level, pulse=1.0, boost=0.0):
    """손/입 공통 상호작용 marker: 작은 반투명 원 + 얇은 외곽 링 (◎).
    level 0=idle(은은) ~ 1=active(밝고 약간 큼), pulse=크기 배율(순간 pulse), boost=engaged 정도(0~1, 더 밝고 큼)"""
    r = INTERACTION_MARKER_RADIUS * (1 + (INTERACTION_MARKER_ACTIVE_SCALE - 1) * level) * pulse * (1 + 0.25 * boost)
    alpha = min(INTERACTION_MARKER_ALPHA + (INTERACTION_MARKER_ACTIVE_ALPHA - INTERACTION_MARKER_ALPHA) * level + 0.2 * boost, 1.0)
    h, w = frame.shape[:2]
    x, y, pad = int(center[0]), int(center[1]), int(r) + INTERACTION_MARKER_LINE_WIDTH + 2
    x0, x1, y0, y1 = max(x - pad, 0), min(x + pad + 1, w), max(y - pad, 0), min(y + pad + 1, h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    c = (x - x0, y - y0)
    ring = roi.copy()
    cv2.circle(ring, c, int(r), INTERACTION_MARKER_COLOR, INTERACTION_MARKER_LINE_WIDTH, cv2.LINE_AA)
    cv2.addWeighted(ring, alpha, roi, 1 - alpha, 0, roi)
    dot = roi.copy()
    cv2.circle(dot, c, max(int(r * 0.4), 2), INTERACTION_MARKER_COLOR, -1, cv2.LINE_AA)
    cv2.addWeighted(dot, alpha * 0.6, roi, 1 - alpha * 0.6, 0, roi)


def draw_mouth_ripple(frame, mouth_center, face_h, age):
    """입 주변으로 빨려 들어가듯 안쪽으로 줄어드는 얇은 물결 두 겹 (age: 시작 후 경과 시간)"""
    base = face_h * MOUTH_RIPPLE_SIZE
    for delay in (0.0, 0.25):
        u = (age - delay * MOUTH_RIPPLE_DURATION) / (0.6 * MOUTH_RIPPLE_DURATION)
        if 0 <= u < 1:
            draw_ripple(frame, mouth_center, base * (1.8 - 1.0 * u), 0.5 * math.sin(math.pi * u))


def find_fish_at(fishes, point, radius=FISH_RADIUS):
    """point가 radius 안에 있는 moving 물고기 중 가장 가까운 것의 인덱스 (없으면 None)"""
    best, best_dist = None, radius
    for i, fish in enumerate(fishes):
        if fish["state"] != "moving":
            continue
        d = math.dist(fish["pos"], point)
        if d <= best_dist:
            best, best_dist = i, d
    return best


def put_text(frame, text, org, scale, color=UI_TEXT_COLOR, thickness=1, alpha=UI_TEXT_ALPHA):
    """얇은 반투명 텍스트 (아주 옅은 그림자만, 두꺼운 외곽선 없음). org = 글자 왼쪽 아래"""
    (tw, th), base = cv2.getTextSize(text, UI_FONT, scale, thickness)
    h, w = frame.shape[:2]
    x, y, pad = int(org[0]), int(org[1]), 4
    x0, x1, y0, y1 = max(x - pad, 0), min(x + tw + pad, w), max(y - th - pad, 0), min(y + base + pad, h)
    if x1 <= x0 or y1 <= y0 or alpha <= 0.01:
        return
    roi = frame[y0:y1, x0:x1]
    shadow = roi.copy()
    cv2.putText(shadow, text, (x - x0 + 1, y - y0 + 1), UI_FONT, scale, (20, 20, 20), thickness + 1, cv2.LINE_AA)
    cv2.addWeighted(shadow, 0.3 * alpha, roi, 1 - 0.3 * alpha, 0, roi)
    layer = roi.copy()
    cv2.putText(layer, text, (x - x0, y - y0), UI_FONT, scale, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, roi)


def put_text_center(frame, text, y, scale, color=UI_TEXT_COLOR, thickness=1, alpha=UI_TEXT_ALPHA, x=None):
    """가로 중앙(또는 x 중심)에 텍스트"""
    size = cv2.getTextSize(text, UI_FONT, scale, thickness)[0]
    cx = frame.shape[1] // 2 if x is None else int(x)
    put_text(frame, text, (cx - size[0] // 2, y), scale, color, thickness, alpha)


def draw_hud(frame, phase, score, time_left):
    """SCORE / TIME만 표시. 시작·종료 화면은 짧은 문구만"""
    h, w = frame.shape[:2]
    put_text(frame, f"SCORE  {score}", (20, 36), 0.7)
    timer = f"TIME  {int(math.ceil(time_left))}"
    size = cv2.getTextSize(timer, UI_FONT, 0.7, 1)[0]
    put_text(frame, timer, (w - size[0] - 20, 36), 0.7, UI_WARN_COLOR if phase == "playing" and time_left <= 10 else UI_TEXT_COLOR)
    if phase == "ready":
        put_text_center(frame, "FACE FISH", h // 2 - 20, 1.6, thickness=2)
        put_text_center(frame, "Pinch a fish and open wide", h // 2 + 20, 0.6, alpha=0.75)
        put_text_center(frame, "SPACE to start", h // 2 + 56, 0.7)
    elif phase == "over":
        put_text_center(frame, "TIME UP", h // 2 - 30, 1.4, thickness=2)
        put_text_center(frame, f"Score  {score}", h // 2 + 14, 0.9)
        put_text_center(frame, "SPACE to restart", h // 2 + 56, 0.7, alpha=0.75)


def draw_hand(frame, points):
    for c in CONNECTIONS:
        cv2.line(frame, points[c.start], points[c.end], (0, 255, 0), 2)
    for p in points:
        cv2.circle(frame, p, 3, (0, 0, 255), -1)


def pinch_ratio(landmarks):
    """엄지 끝-검지 끝 거리를 손 크기(손목~중지 뿌리)로 나눈 값 (손 거리와 무관)"""
    dist = math.dist((landmarks[THUMB_TIP].x, landmarks[THUMB_TIP].y),
                     (landmarks[INDEX_TIP].x, landmarks[INDEX_TIP].y))
    size = math.dist((landmarks[WRIST].x, landmarks[WRIST].y),
                     (landmarks[MIDDLE_MCP].x, landmarks[MIDDLE_MCP].y))
    return dist / size if size > 0 else 1.0


def mouth_info(landmarks, w, h):
    """입 중심 좌표(px), 정규화된 입 벌림 값, 얼굴 높이(px) 반환"""
    pts = [(lm.x * w, lm.y * h) for lm in landmarks]
    corners = (MOUTH_LEFT, MOUTH_RIGHT, INNER_LIP_TOP, INNER_LIP_BOTTOM)
    cx = sum(pts[i][0] for i in corners) / len(corners)
    cy = sum(pts[i][1] for i in corners) / len(corners)
    gap = math.dist(pts[INNER_LIP_TOP], pts[INNER_LIP_BOTTOM])
    face_h = math.dist(pts[FOREHEAD], pts[CHIN])
    openness = gap / face_h if face_h > 0 else 0.0
    return (int(cx), int(cy)), openness, face_h


def draw_mouth(frame, landmarks, center, is_open):
    h, w = frame.shape[:2]
    color = (0, 0, 255) if is_open else (255, 200, 0)
    for c in LIPS_CONNECTIONS:
        a, b = landmarks[c.start], landmarks[c.end]
        cv2.line(frame, (int(a.x * w), int(a.y * h)), (int(b.x * w), int(b.y * h)), color, 2)
    cv2.circle(frame, center, 8, color, -1)
    cv2.circle(frame, center, 14, (255, 255, 255), 2)


def blend_mask(frame, mask, color, alpha):
    """마스크(0~1, 단일 채널)가 있는 곳에만 color를 alpha만큼 합성. 마스크가 0인 곳(배경)은 원본 픽셀 그대로.
    부드러운 가장자리가 필요하면 frame이 아니라 mask만 blur할 것 (frame을 blur하면 화면 전체가 뿌옇게 됨)"""
    m = mask[..., None] * alpha
    frame[:] = (frame * (1 - m) + np.array(color, np.float32) * m).astype(np.uint8)


def effect_level(streak):
    """같은 물고기를 연속으로 먹은 횟수(streak) → 효과 레벨 (1~MAX_EFFECT_LEVEL)"""
    return max(1, min(streak, MAX_EFFECT_LEVEL))


def by_level(level, values):
    """레벨(1~3)에 맞는 값 선택"""
    return values[max(1, min(level, MAX_EFFECT_LEVEL)) - 1]


def effect_envelope(age):
    """효과 시작/끝에서 서서히 나타나고 사라지는 0~1 값 (morph/tint/particle 공통)"""
    return max(0.0, min(age / EFFECT_FADE_IN, 1.0, (EFFECT_DURATION - age) / EFFECT_FADE_OUT))


def axes(pts):
    """얼굴 중심, 위쪽/오른쪽 단위벡터 (얼굴 기울기 반영)"""
    top, chin = np.array(pts[FOREHEAD]), np.array(pts[CHIN])
    up = (top - chin) / (np.linalg.norm(top - chin) + 1e-6)
    right = np.array(pts[FACE_RIGHT]) - np.array(pts[FACE_LEFT])
    right = right / (np.linalg.norm(right) + 1e-6)
    return (top + chin) / 2, up, right


def ipt(v):
    return (int(v[0]), int(v[1]))


def forehead_point(pts):
    """이마 중앙~미간 사이 기준점"""
    return np.array(pts[FOREHEAD]) * 0.5 + np.array(pts[GLABELLA]) * 0.5


def tint_mask(shape, pts, face_w, face_h, name):
    """물고기별 tint 위치 (Red=볼, Yellow=눈, Green=이마/윗얼굴). 마스크만 blur"""
    mask = np.zeros(shape, np.float32)
    if name == "Red Fish":
        for i in (LEFT_CHEEK, RIGHT_CHEEK):
            cv2.circle(mask, ipt(pts[i]), int(face_w * 0.14), 1.0, -1)
        sigma = face_w * 0.05
    elif name == "Yellow Fish":
        for i in IRIS_CENTERS:
            cv2.circle(mask, ipt(pts[i]), int(face_w * 0.09), 1.0, -1)
        sigma = face_w * 0.05
    else:
        _, _, right = axes(pts)
        c = np.array(pts[FOREHEAD]) * 0.45 + np.array(pts[GLABELLA]) * 0.55
        cv2.ellipse(mask, ipt(c), (int(face_w * 0.30), int(face_h * 0.11)),
                    math.degrees(math.atan2(right[1], right[0])), 0, 360, 1.0, -1)
        sigma = face_w * 0.06
    return cv2.GaussianBlur(mask, (0, 0), sigma)


_PARTICLE_RAND = np.random.RandomState(7).rand(EFFECT_PARTICLES[-1], 4)  # 파티클마다 고정 난수(위치, 위상, 속도, 크기)


def soft_dot(frame, pos, r, color, alpha):
    """가장자리가 부드러운 반투명 점 하나 (작은 영역만 처리)"""
    if alpha <= 0.01:
        return
    h, w = frame.shape[:2]
    x, y, pad = int(pos[0]), int(pos[1]), int(r * 3) + 2
    x0, x1, y0, y1 = max(x - pad, 0), min(x + pad + 1, w), max(y - pad, 0), min(y + pad + 1, h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    m = np.zeros(roi.shape[:2], np.float32)
    cv2.circle(m, (x - x0, y - y0), max(int(r), 1), 1.0, -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), max(r * 0.6, 0.8))
    if m.max() > 0:
        blend_mask(roi, m / m.max(), color, min(alpha, 1.0))


def effect_particles(frame, pts, face_w, face_h, age, level, name, color, env):
    """작은 soft 점들. 물고기별 움직임만 다름: Red=볼에서 위로 떠오름 / Yellow=얼굴 둘레에서 반짝 / Green=이마 위에서 천천히 내려옴"""
    center, up, right = axes(pts)
    for k in range(by_level(level, EFFECT_PARTICLES)):
        u, phase, speed, size = _PARTICLE_RAND[k]
        life = (age * (0.12 + 0.1 * speed) + phase) % 1.0
        fade = math.sin(math.pi * life)  # 생성 → 소멸
        if name == "Red Fish":
            side = -1 if k % 2 == 0 else 1
            pos = (np.array(pts[LEFT_CHEEK if side < 0 else RIGHT_CHEEK]) + right * side * face_w * 0.10 * u
                   + up * face_h * 0.30 * life + right * math.sin(age * 1.5 + k) * face_w * 0.02)
            a = fade
        elif name == "Yellow Fish":
            ang = u * 2 * math.pi + age * 0.15
            pos = center + right * math.cos(ang) * face_w * 0.62 + up * math.sin(ang) * face_h * 0.52
            a = 0.5 + 0.5 * math.sin(age * 3 + phase * 6.28)  # 반짝임
        else:
            pos = (np.array(pts[FOREHEAD]) + up * face_h * (0.22 - 0.35 * life) + right * (u - 0.5) * face_w * 0.9
                   + right * math.sin(age * 1.2 + k) * face_w * 0.03)
            a = fade
        soft_dot(frame, pos, face_w * EFFECT_PARTICLE_SIZE * (0.7 + 0.6 * size), color, EFFECT_PARTICLE_ALPHA * a * env)


def effect_face(frame, pts, face_w, face_h, age, level, name):
    """모든 물고기 공통 그래픽: soft tint + small particles (+ LV3 약한 pulse). morph는 warp_face가 그리기 전에 따로 적용"""
    env = effect_envelope(age)
    color = FISH_COLORS[name]
    pulse = 1.0 + EFFECT_PULSE_LV3 * math.sin(age * 3.0) if level >= 3 else 1.0
    blend_mask(frame, tint_mask(frame.shape[:2], pts, face_w, face_h, name), color,
               min(by_level(level, EFFECT_TINT_ALPHA) * pulse * env, 0.9))
    effect_particles(frame, pts, face_w, face_h, age, level, name, color, env)


def bulge(frame, center, radius, strength):
    """center 주변 radius 안의 픽셀을 바깥쪽으로 밀어 부풀어 보이게 하는 국소 warp (in-place)"""
    h, w = frame.shape[:2]
    cx, cy = center
    x0, x1 = max(int(cx - radius), 0), min(int(cx + radius) + 1, w)
    y0, y1 = max(int(cy - radius), 0), min(int(cy + radius) + 1, h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    dx, dy = xs - cx, ys - cy
    u2 = np.clip((dx * dx + dy * dy) / (radius * radius), 0.0, 1.0)  # (거리/반경)^2
    # 샘플링 위치를 중심 쪽으로 d*strength*(1-u²)² 만큼 당김 → 중심 확대, 반경 끝에서는 변위 0(경계 매끄러움)
    scale = 1.0 - strength * (1.0 - u2) ** 2
    frame[y0:y1, x0:x1] = cv2.remap(frame, (cx + dx * scale).astype(np.float32), (cy + dy * scale).astype(np.float32),
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def warp_face(frame, landmarks, name, level=1, amount=1.0):
    """물고기별 국소 morph (Red=양 볼, Yellow=양 눈, Green=이마). 반경은 얼굴 너비 기준이라 얼굴 크기를 따라감.
    amount(0~1)는 효과 시작/끝 fade용"""
    h, w = frame.shape[:2]
    pts = [(lm.x * w, lm.y * h) for lm in landmarks]
    face_w = math.dist(pts[FACE_LEFT], pts[FACE_RIGHT])
    if name == "Red Fish":
        mid = ((pts[FACE_LEFT][0] + pts[FACE_RIGHT][0]) / 2, (pts[FACE_LEFT][1] + pts[FACE_RIGHT][1]) / 2)
        spots = []
        for i in (LEFT_CHEEK, RIGHT_CHEEK):
            ox, oy = pts[i][0] - mid[0], pts[i][1] - mid[1]  # 볼 랜드마크를 얼굴 중심선에서 먼 방향(바깥쪽)으로 살짝 이동
            n = math.hypot(ox, oy) or 1.0
            spots.append(((pts[i][0] + ox / n * face_w * CHEEK_WARP_OUTWARD, pts[i][1] + oy / n * face_w * CHEEK_WARP_OUTWARD),
                          face_w * CHEEK_WARP_RADIUS_RATIO))
    elif name == "Yellow Fish":
        spots = [(pts[i], face_w * EYE_WARP_RADIUS_RATIO) for i in IRIS_CENTERS]
    else:
        spots = [(tuple(forehead_point(pts)), face_w * FOREHEAD_WARP_RADIUS_RATIO)]
    strength = min(by_level(level, MORPH_STRENGTH[name]) * amount, MORPH_STRENGTH_MAX)
    for center, radius in spots:
        bulge(frame, center, radius, strength)
        if DEBUG_WARP:
            cv2.circle(frame, (int(center[0]), int(center[1])), int(radius), (255, 0, 255), 1)


WARPS = {n: (lambda frame, lm, level=1, amount=1.0, n=n: warp_face(frame, lm, n, level, amount)) for n in FISH_COLORS}
EFFECTS = {n: (lambda frame, pts, fw, fh, age, level=1, n=n: effect_face(frame, pts, fw, fh, age, level, n)) for n in FISH_COLORS}


def draw_effect(frame, name, landmarks, age, level=1):
    """마지막으로 먹은 물고기(name)에 맞는 얼굴 효과 그리기"""
    h, w = frame.shape[:2]
    pts = [(lm.x * w, lm.y * h) for lm in landmarks]
    face_w = math.dist(pts[FACE_LEFT], pts[FACE_RIGHT])
    face_h = math.dist(pts[FOREHEAD], pts[CHIN])
    EFFECTS[name](frame, pts, face_w, face_h, age, level)


def run_webcam():
    for path in (MODEL_PATH, FACE_MODEL_PATH):
        if not path.exists():
            print(f"✗ 모델 파일이 없습니다: {path}")
            return

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )

    face_options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=FACE_MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ 웹캠을 열 수 없습니다.")
        return

    print("✓ 웹캠 실행 중입니다. 'q' 또는 ESC를 눌러 종료")

    start = time.monotonic()
    last_ts = -1
    pinching = False
    fishes = None
    held = None  # 현재 집고 있는 물고기 인덱스 (동시에 하나만)
    yum_text, yum_until, yum_pos = "", 0.0, (0, 0)  # 먹기 성공 피드백 문구, 표시 종료 시각, 표시 위치(입)
    last_eaten, effect_until = None, 0.0  # 마지막으로 먹은 물고기 이름과 효과 종료 시각
    current_streak = 0  # last_eaten 물고기를 연속으로 먹은 횟수 (효과 레벨 계산용)
    prev_feeding, mouth_ripple_start = False, -1e9  # 물고기를 든 채 입을 연 순간 감지 / 입 물결 시작 시각
    hand_marker, mouth_marker = 0.0, 0.0  # marker 상태 값 (0=idle, 1=active), 부드럽게 변함
    phase, score, game_start = "ready", 0, 0.0  # phase: ready(SPACE 대기) / playing / over
    last_t = time.monotonic()

    with vision.HandLandmarker.create_from_options(options) as landmarker, \
            vision.FaceLandmarker.create_from_options(face_options) as face_landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.monotonic()
            dt = min(now - last_t, 0.1)  # 프레임 간 시간(초), 멈춤 후 튀는 것 방지
            last_t = now

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ts = max(int((time.monotonic() - start) * 1000), last_ts + 1)
            last_ts = ts
            result = landmarker.detect_for_video(mp_image, ts)
            face_result = face_landmarker.detect_for_video(mp_image, ts)

            # 효과 중에는 아무것도 그리기 전에 원본 영상을 물고기별로 warp (얼굴 없으면 생략)
            if face_result.face_landmarks and last_eaten in WARPS and time.monotonic() < effect_until:
                WARPS[last_eaten](frame, face_result.face_landmarks[0], effect_level(current_streak),
                                  effect_envelope(time.monotonic() - (effect_until - EFFECT_DURATION)))

            mouth_center, mouth_is_open, face_h = None, False, 0.0  # 얼굴 없으면 먹기 불가
            face = None
            if face_result.face_landmarks:
                face = face_result.face_landmarks[0]
                mouth_center, openness, face_h = mouth_info(face, w, h)
                mouth_is_open = openness >= MOUTH_OPEN_THRESHOLD
                if DEBUG:
                    draw_mouth(frame, face, mouth_center, mouth_is_open)
                    text = "MOUTH OPEN" if mouth_is_open else "Mouth closed"
                    cv2.putText(frame, f"{text} ({openness:.3f})", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255) if mouth_is_open else (255, 200, 0), 2)
            elif DEBUG:
                cv2.putText(frame, "No face detected", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if fishes is None:
                fishes = make_fishes(w, h)

            if phase == "playing":
                if now - game_start >= GAME_DURATION:
                    phase, held = "over", None  # 시간 종료: 이동/상호작용 정지
                else:
                    update_fishes(fishes, w, h, dt, now)
            playing = phase == "playing"

            hand_center = None  # 손 marker를 그릴 pinch 중심 (손이 없으면 None)
            hover = None  # 손이 가까워 "잡을 수 있음" 물결을 보여줄 물고기 인덱스
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                if DEBUG:
                    draw_hand(frame, points)

                ratio = pinch_ratio(landmarks)
                was_pinching = pinching
                pinching = ratio < (PINCH_OFF if pinching else PINCH_ON)

                thumb, index = points[THUMB_TIP], points[INDEX_TIP]
                center = ((thumb[0] + index[0]) // 2, (thumb[1] + index[1]) // 2)
                hand_center = center

                if playing and pinching and not was_pinching and held is None:
                    held = find_fish_at(fishes, center)  # pinch 시작 순간에만 집기
                    if held is not None:
                        fishes[held]["grab_at"] = now  # 잡는 순간 pop 효과 시작
                elif not pinching:
                    held = None  # pinch를 풀면 그 자리에 놓기
                if held is not None:
                    fishes[held]["pos"] = attracted_pos(center, mouth_center, face_h)  # 입 근처에서 아주 약하게 끌림
                elif playing:
                    hover = find_fish_at(fishes, center, INTERACTION_DISTANCE)  # 손에 가장 가까운 물고기 하나
                color = (0, 255, 255) if pinching else (255, 255, 255)
                if DEBUG:  # 손 위의 상시 표시는 DEBUG에서만
                    cv2.line(frame, thumb, index, color, 3)
                    if pinching:
                        cv2.circle(frame, center, 12, (0, 255, 255), -1)

                if DEBUG:
                    label = f"PINCH ({ratio:.2f})" if pinching else f"Open ({ratio:.2f})"
                    if held is not None:
                        label += f" - Holding: {fishes[held]['name']}"
                    cv2.putText(frame, label, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            else:
                pinching = False
                held = None
                if DEBUG:
                    cv2.putText(frame, "No hand detected", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 먹기 판정: 집고 있음 + 입 근처(얼굴 크기 기준) + 입 열림. 먹으면 held가 None이 되어 한 번만 처리됨
            if playing and held is not None and mouth_center is not None and mouth_is_open:
                if math.dist(fishes[held]["pos"], mouth_center) <= EAT_DISTANCE_RATIO * face_h:
                    gained = RARITY_SCORE[fishes[held]["rarity"]]
                    yum_text, yum_pos = f"+{gained}", mouth_center
                    yum_until = time.monotonic() + YUM_DURATION
                    eaten = fishes[held]["name"]
                    # 같은 물고기를 연속으로 먹으면 streak 증가, 다른 물고기면 그 물고기로 바꾸고 Level 1부터 시작
                    current_streak = current_streak + 1 if eaten == last_eaten else 1
                    last_eaten = eaten
                    effect_until = time.monotonic() + EFFECT_DURATION
                    fishes[held]["eaten_at"] = now  # 먹힘 pop 효과 시작
                    fishes[held]["state"] = "hidden"  # RESPAWN_DELAY 뒤 update_fishes가 재등장시킴
                    fishes[held]["respawn_at"] = now + RESPAWN_DELAY
                    score += gained
                    held = None

            # 물고기 상태 동기화: 집고 있는 물고기는 held, 놓은 물고기는 다시 moving
            for i, fish in enumerate(fishes):
                if i == held:
                    fish["state"] = "held"
                elif fish["state"] == "held":
                    fish["state"] = "moving"

            if last_eaten and face is not None and time.monotonic() < effect_until:
                draw_effect(frame, last_eaten, face, time.monotonic() - (effect_until - EFFECT_DURATION),
                            effect_level(current_streak))

            draw_fishes(frame, fishes, held, hover)

            # 물고기를 든 채 입을 연 순간에만 입 주변 물결 시작 (입이 닫혀 있으면 아무것도 안 그림)
            feeding = playing and held is not None and mouth_center is not None and mouth_is_open
            if feeding and not prev_feeding:
                mouth_ripple_start = now
            prev_feeding = feeding
            if mouth_center is not None and 0 <= now - mouth_ripple_start < MOUTH_RIPPLE_DURATION:
                draw_mouth_ripple(frame, mouth_center, face_h, now - mouth_ripple_start)

            # 상호작용 marker: 손(pinch 중심)과 입에 같은 디자인으로 "여기서 상호작용"을 안내 (반응 효과와 역할 분리)
            rate = min(dt * 10, 1.0)
            hand_active = hand_center is not None and (hover is not None or held is not None)
            hand_marker += ((1.0 if hand_active else 0.0) - hand_marker) * rate
            if hand_center is not None:
                hand_pulse = 1.0
                if held is not None:  # 잡는 순간 짧은 pulse, 이후 잡고 있는 동안은 active 유지
                    u = (now - fishes[held]["grab_at"]) / GRAB_PULSE_DURATION
                    if 0 <= u < 1:
                        hand_pulse = 1 + 0.5 * math.sin(math.pi * u)
                draw_marker(frame, hand_center, hand_marker, hand_pulse)

            mouth_marker += ((1.0 if mouth_center is not None and mouth_is_open else 0.0) - mouth_marker) * rate
            if mouth_center is not None:
                near = 0.0  # 잡은 물고기가 입에 가까운 정도 (0~1)
                if playing and held is not None:
                    reach = MOUTH_ATTRACTION_RANGE * EAT_DISTANCE_RATIO * face_h
                    near = max(0.0, 1 - math.dist(fishes[held]["pos"], mouth_center) / reach)
                mouth_pulse = 1 + 0.12 * near * math.sin(now * 12)  # engaged: 가까울수록 숨쉬듯 수축/확장
                draw_marker(frame, mouth_center, mouth_marker, mouth_pulse, near)

            if time.monotonic() < yum_until:  # 입 위로 살짝 떠오르며 사라지는 짧은 점수 표시
                u = 1 - (yum_until - time.monotonic()) / YUM_DURATION
                put_text_center(frame, yum_text, int(yum_pos[1] - face_h * 0.3 - 30 * u), 0.9, alpha=1 - u, x=yum_pos[0])

            time_left = GAME_DURATION if phase == "ready" else max(GAME_DURATION - (now - game_start), 0.0)
            if phase == "over":
                time_left = 0.0
            draw_hud(frame, phase, score, time_left)
            if DEBUG:
                if last_eaten:
                    put_text(frame, f"{last_eaten} x{current_streak} / Effect Lv.{effect_level(current_streak)}",
                             (15, 150), 0.7, (255, 255, 255))

            cv2.imshow("Face Fish - Press Q or ESC to Exit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and phase != "playing":  # SPACE: 시작 / 재시작
                phase, score, game_start = "playing", 0, time.monotonic()
                fishes, held = make_fishes(w, h), None
                last_eaten, effect_until, yum_until = None, 0.0, 0.0
                current_streak = 0  # 재시작 시 streak 초기화 (last_eaten도 None)
                mouth_ripple_start = -1e9

    cap.release()
    cv2.destroyAllWindows()
    print("✓ 종료")


if __name__ == "__main__":
    run_webcam()
