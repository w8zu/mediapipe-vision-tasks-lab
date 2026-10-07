# FACE FISH

웹캠 앞에서 즐기는 미니게임입니다. 화면 하단을 헤엄치는 반투명 물고기를 손으로 집어 입으로 가져가 먹으면, 물고기 색에 따라 얼굴에 서로 다른 효과가 나타납니다. 이 프로젝트 전체의 학습 배경은 [루트 README](../../README.md)에 있고, 이 문서는 FACE FISH 자체를 다룹니다.

## Game concept

- 세 가지 색 물고기(Red / Yellow / Green)가 화면 하단을 헤엄칩니다.
- 먹은 물고기에 따라 얼굴의 다른 부위가 살짝 변형되고 은은한 색이 입혀집니다.
- 같은 색을 연속으로 먹을수록 같은 효과가 더 강해집니다.
- 화면에는 얇고 부드러운 반투명 AR layer만 얹히도록 비주얼을 통일했습니다.

## Interaction flow (프레임마다)

```
Webcam frame (좌우 반전)
  → 같은 프레임을 Hand Landmarker → Face Landmarker 순서로 추론하고 결과를 조합
  → (효과 중이면) 얼굴 morph를 원본 프레임에 먼저 적용
  → 물고기 이동 → pinch 판정 → grab / drag
  → mouth openness + 물고기-입 거리 → eat event → 점수 / streak
  → 얼굴 tint·파티클 → 물고기·marker·물결 → SCORE / TIME
```

## 사용한 MediaPipe 기능

| 기능 | 사용 목적 |
|---|---|
| Hand Landmarker | 엄지 끝(4)·검지 끝(8)·손목(0)·중지 뿌리(9) landmark로 pinch 판정, pinch 중심 좌표를 물고기 위치로 사용 |
| Face Landmarker | 입 중심·안쪽 윗/아랫입술 간격(입 벌림), 이마~턱 거리(얼굴 크기), 볼·홍채·이마 위치(얼굴 효과) |

Gesture Recognizer와 OpenCV 손 detection(`vision_tasks/`의 실습)은 사용하지 않습니다. 잡기는 제스처 분류가 아니라 landmark 거리 기반 pinch 판정입니다.

## Gameplay

1. `SPACE`로 시작합니다 (제한 시간 45초).
2. 엄지·검지를 모아(pinch) 물고기를 잡고 드래그합니다.
3. 입을 벌린 채 물고기를 입에 가까이 가져가면 먹습니다.
4. 먹으면 점수를 얻고 얼굴 효과가 10초간 나타납니다. 같은 물고기를 연속으로 먹으면 Level이 올라가고(최대 3), 다른 물고기를 먹으면 Level 1부터 다시 시작합니다.

| 물고기 | 등급 | 점수 | 얼굴 효과 위치 |
|---|---|---|---|
| Red Fish (coral) | Common | 1 | 볼 |
| Yellow Fish (gold) | Uncommon | 2 | 눈 주변 |
| Green Fish (mint) | Rare | 5 | 이마·윗얼굴 |

`SPACE` 시작/재시작, `q` 또는 `ESC` 종료.

## 주요 구현 로직

- **Pinch:** `엄지-검지 거리 / 손목-중지뿌리 거리`, `PINCH_ON=0.30` / `PINCH_OFF=0.40` 히스테리시스. grab은 pinch가 시작되는 순간에만 발생해, 빈손으로 pinch한 채 물고기 위를 지나가도 잡히지 않습니다.
- **Mouth open:** `안쪽 입술 간격 / 얼굴 높이 ≥ MOUTH_OPEN_THRESHOLD(0.06)`. face landmark 좌표(안쪽 입술 13·14번, 이마·턱 10·152번)로 계산하며 blendshape는 사용하지 않습니다.
- **Eat event:** 잡고 있음 + 입 열림 + `물고기-입 거리 ≤ EAT_DISTANCE_RATIO(0.25) × 얼굴 높이`가 동시에 맞을 때 한 번만 처리됩니다.
- **Attraction:** 입 근처에서는 손 위치 기준으로 매 프레임 새로 계산한 약한 보간으로 물고기가 입 쪽으로 살짝 끌립니다 (누적하지 않아 조작감 유지).
- **Rarity:** 등급 가중치로 재등장 물고기를 뽑고, Rare/Epic은 화면 동시 등장을 1마리로 제한합니다.
- **Face effect:** 물고기별 국소 pixel warp(`bulge`)를 그리기 전에 원본 프레임에 적용하고, 마스크만 blur해 tint를 합성합니다. 전체 프레임 blur는 사용하지 않습니다. 모든 효과는 morph + tint + 작은 파티클 + (Level 3) 약한 pulse의 같은 규칙을 따릅니다.
- **Feedback:** 손 pinch 중심과 입에 같은 디자인의 marker, 물고기 주변 물결, 먹는 순간의 pop과 `+점수` 표시.

판정 상수는 파일 상단에 모여 있으며 모두 추정값입니다. `DEBUG = True`로 바꾸면 landmark와 내부 값을 볼 수 있습니다.

## 실행 방법

저장소 루트에서 (PowerShell):

```powershell
python -m pip install -r requirements.txt
python projects/face_fish/face_fish.py
```

웹캠이 필요하고, 밝은 곳에서 얼굴과 손이 모두 보이게 앉아야 합니다. 모델은 저장소 루트의 `models/`에서 읽습니다.

## 현재 한계

- 판정 상수(입 벌림, 먹기 거리, 물고기 속도 등)는 실측 없이 정한 값이라 환경에 따라 조정이 필요합니다.
- 손 1개, 얼굴 1개만 처리합니다.
- 효과 streak는 시간이 지나도 끊기지 않고, 다른 물고기를 먹을 때만 초기화됩니다.
- Epic 등급은 슬롯만 있고 물고기는 없습니다.
- 사운드와 최고 점수 저장은 없습니다.

## 향후 web version 계획

별도 폴더에서 MediaPipe Tasks Vision(Web)으로 같은 인터랙션을 구현하고, 얼굴 효과를 Canvas/WebGL로 옮겨 보는 것이 목표입니다. 아직 구현하지 않았습니다.
