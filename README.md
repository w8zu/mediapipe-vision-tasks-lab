# MediaPipe Vision Tasks Lab

MediaPipe Vision Tasks를 이용해 **웹캠으로 손과 얼굴을 인식하고, 그 결과를 실제 인터랙션으로 연결해 보는 학습 프로젝트**입니다.

이 저장소의 핵심은 손이나 얼굴에 점을 찍어 보는 데 있지 않습니다. 카메라가 찾아낸 손·얼굴의 위치 정보를 이용해 **"잡기", "입 벌리기", "먹기"처럼 사람이 이해할 수 있는 행동으로 바꾸는 과정**을 직접 구현하는 것이 목적입니다. 개별 기능을 하나씩 실습한 뒤, 마지막에 그 기능들을 합친 응용 예제로 미니게임 **FACE FISH**를 만들었습니다.

## 한눈에 보는 전체 흐름

```text
카메라 영상
   ↓
손 / 얼굴의 위치 정보를 추출          ← MediaPipe가 해 주는 일
   ↓
좌표 사이의 거리와 상태를 계산         ← 직접 구현한 부분
   ↓
"손으로 잡기", "입을 벌리기" 같은 행동으로 해석
   ↓
게임 이벤트로 연결 (FACE FISH)
```

기술적인 표현으로 줄이면 이렇습니다.

```text
Vision Task Output → Interaction Logic → Real-time Experience
```

MediaPipe는 "엄지 끝이 화면의 어디에 있다"까지만 알려 줍니다. "지금 물고기를 집었다"는 판단은 그 좌표를 가지고 직접 만든 규칙입니다. 이 저장소는 그 사이의 단계를 다룹니다.

---

## 먼저 알아 두면 좋은 용어

처음 읽을 때 필요한 만큼만 정리했습니다. 이 프로젝트에서 어떻게 쓰였는지를 함께 적었습니다.

| 용어 | 쉬운 설명 | 이 프로젝트에서는 |
|---|---|---|
| **MediaPipe** | Google이 만든 컴퓨터 비전 도구 모음. 카메라 영상에서 손, 얼굴, 자세 등을 실시간으로 분석합니다. | 손/얼굴 인식 전부를 MediaPipe의 **Tasks API**로 구현했습니다. |
| **Vision Task** | 영상에서 특정 정보를 찾아 주는 기능 하나하나. | Gesture Recognizer, Face Landmarker, Hand Landmarker를 사용했습니다. |
| **landmark** | 손가락 끝, 손목, 입술, 턱처럼 신체의 특정 위치를 나타내는 좌표점. | 손은 21개, 얼굴은 478개의 점으로 추적합니다. 이 좌표가 모든 계산의 재료입니다. |
| **classification** | 입력을 보고 "어떤 종류인지" 분류하는 것. | Gesture Recognizer가 손 모양을 `Open_Palm`, `Victory` 같은 이름으로 분류합니다. |
| **blendshape** | 얼굴 표정을 "눈을 감은 정도", "입꼬리가 올라간 정도"처럼 여러 수치로 나타낸 값. | Face Landmarker 독립 예제에서 상위 8개를 화면에 표시합니다. FACE FISH는 사용하지 않습니다. |
| **pinch** | 엄지와 검지를 오므려 집는 동작. | MediaPipe가 알려 주는 값이 아니라, 엄지·검지 landmark 사이 거리로 **직접 정의한 상태**입니다. |
| **threshold(임계값)** | "이 값 이상이면 A, 아니면 B"를 가르는 기준선. | "거리 0.30 이하면 집었다", "입 벌림 0.06 이상이면 입을 열었다" 같은 기준을 코드 상단 상수로 두었습니다. |
| **normalization(정규화)** | 값을 어떤 기준 크기로 나눠서, 카메라와의 거리에 상관없이 비교할 수 있게 만드는 것. | 손가락 거리는 손 크기로, 입술 간격은 얼굴 크기로 나눕니다. |
| **hysteresis** | 켜지는 기준과 꺼지는 기준을 다르게 두어 경계에서 상태가 깜빡이는 것을 막는 방법. | pinch가 경계값 근처에서 떨리지 않도록 사용했습니다. |

---

## Project Goal

- MediaPipe Vision Tasks의 주요 기능을 직접 실행해 보기
- landmark와 classification 결과가 무엇을 의미하는지 이해하기
- 모델의 출력값을 거리, 상태, 조건 같은 규칙으로 해석해 **행동 판정**으로 바꾸기
- 손과 얼굴 정보를 하나의 인터랙션으로 연결하기
- 응용 예제 FACE FISH로 전체를 검증해 보기

학습 흐름은 다음과 같습니다.

```text
MediaPipe / Vision 개념
        ↓
Vision Tasks Practice (개별 실습)
├── Hand Detection (OpenCV 기반 대체 구현)
├── Gesture Recognizer (MediaPipe)
└── Face Landmarker (MediaPipe)
        ↓
Interaction Logic (거리·상태·조건 규칙)
        ↓
Applied Example: FACE FISH (Hand Landmarker + Face Landmarker)
```

---

## Vision Tasks Practice

MediaPipe 실습은 **Tasks API**(`mediapipe.tasks.python.vision`)를 사용합니다. (구버전 `mp.solutions.*`는 사용하지 않습니다.)

| Practice | 구현 방식 | 독립 실행 파일 | FACE FISH에서 사용 |
|---|---|:---:|:---:|
| Hand Detection | OpenCV HSV | O | X |
| Gesture Recognizer | MediaPipe Tasks API | O | X |
| Face Landmarker | MediaPipe Tasks API | O | O |
| Hand Landmarker | MediaPipe Tasks API | X | O |

### 1. Hand Detection

- 파일: `vision_tasks/hand_detection.py`
- **OpenCV HSV 기반 손 영역 탐지 / 대체 구현**입니다. 영상에서 피부색에 가까운 영역을 찾아 손처럼 보이는 부분을 표시합니다.
- MediaPipe Hand Landmarker와는 **별개**입니다. 손가락 위치(landmark)는 알 수 없고, 손 영역 정도만 추정합니다.
- 이후 FACE FISH에서는 이 방식 대신 MediaPipe Hand Landmarker를 사용했습니다.

```text
카메라 영상 → 피부색에 가까운 영역 찾기 → 손처럼 보이는 영역 표시
```

### 2. Gesture Recognizer

- 파일: `vision_tasks/gesture_recognition.py`
- MediaPipe Tasks API의 **Gesture Recognizer**로 웹캠 속 손동작을 **classification(분류)** 합니다.
- 분류하는 제스처: `Closed_Fist`(주먹), `Open_Palm`(손바닥), `Pointing_Up`(검지 들기), `Thumb_Up`, `Thumb_Down`, `Victory`(V 표시), `ILoveYou`
- 이 실습은 모델이 **"어떤 제스처인지" 이름을 붙여 주는 과정**을 확인하는 독립 실습입니다.
- **FACE FISH의 물고기 잡기에는 사용하지 않았습니다.** 잡기는 제스처 분류가 아니라 엄지·검지 landmark 거리 기반 pinch 판정입니다. (Hand Landmarker 항목 참고)

### 3. Face Landmarker

- 파일: `vision_tasks/face_landmarker.py`
- MediaPipe Tasks API의 **Face Landmarker**로 얼굴을 추적합니다. 독립 예제에서는 다음을 확인합니다.
  - 478개 facial landmark 추적 (영역별 색으로 표시)
  - blendshape 상위 8개 표시
- **FACE FISH에서의 활용(응용 로직):** mouth center(입 중심) 계산, mouth openness(입 벌림) 계산, 얼굴 크기 계산, 얼굴 효과 위치 계산. 이 계산들은 `face_landmarker.py`의 기능이 아니라 `face_fish.py`에서 **landmark 좌표로 직접 구현**한 부분입니다.
- FACE FISH는 blendshape로 입 벌림을 계산하지 않고, 입술 landmark 좌표 사이의 거리로 계산합니다.

### 4. Hand Landmarker

- **독립 실행 예제는 현재 없습니다.**
- 대신 FACE FISH 내부에서 MediaPipe Hand Landmarker를 직접 사용합니다. 손가락과 손목의 위치를 landmark 좌표로 알려 주며, FACE FISH는 엄지 끝, 검지 끝, 손목, 중지 뿌리 좌표를 사용합니다.
- 이 좌표로 "지금 손이 물고기를 집고 있는가?"를 계산합니다.

---

## Interaction Logic: Key Implementation Points

MediaPipe의 출력은 좌표와 분류 결과일 뿐이어서, 그것을 "행동"으로 바꾸려면 규칙이 필요합니다. 아래는 FACE FISH에서 실제로 구현한 해석 방식입니다. 각 항목은 **쉬운 설명 → 실제 기술** 순서로 적었습니다.

### 1. 엄지와 검지가 가까워졌는지로 "집기" 판정

> 엄지와 검지가 가까워졌는지를 계산해, 물고기를 잡는 동작으로 사용했습니다.

MediaPipe가 "지금 pinch 중이다"라고 알려 주지는 않습니다. Hand Landmarker가 주는 **엄지 끝 landmark와 검지 끝 landmark의 좌표**로 두 점 사이 거리를 계산하고, 충분히 가까우면 pinch 상태로 봅니다.

```text
엄지 끝 ●────● 검지 끝      → 거리를 계산 → 기준(threshold)보다 작으면 pinch
```

### 2. 손 크기로 거리 정규화 (normalization)

> 손이 카메라에 가까워도 멀어도 같은 기준으로 "집었다"를 판단하려고 했습니다.

엄지-검지 거리만 쓰면 손이 카메라에 가까울 때는 실제로는 벌어진 손가락도 크게, 멀 때는 작게 보여서 기준이 흔들립니다. 그래서 **손 자체의 크기(손목~중지 뿌리 거리)로 나눈 비율**을 사용합니다.

```text
엄지-검지 거리
──────────────  =  pinch 비율
손목-중지 뿌리 거리
```

### 3. hysteresis로 손떨림 줄이기

> 손가락이 기준선 근처에 있을 때 "잡혔다/놓였다"가 깜빡이지 않게 했습니다.

기준값 하나만 쓰면 경계에서 `잡힘 → 안 잡힘 → 잡힘 → …`이 프레임마다 반복될 수 있습니다. 그래서 **집기 시작하는 기준(비율 0.30)과 놓는 기준(0.40)을 다르게** 두었습니다. 한 번 잡으면 조금 벌어져도 바로 놓지 않고, 충분히 벌어졌을 때만 놓습니다.

또한 물고기는 **pinch가 시작되는 순간에만** 잡힙니다. 빈손으로 pinch한 채 물고기 위를 지나가도 잡히지 않습니다.

### 4. 입 벌림 정도 계산 (mouth openness)

> 입술 사이 간격을 얼굴 크기에 맞춰 환산해, 입을 벌렸는지 판단했습니다.

Face Landmarker의 **안쪽 윗입술·안쪽 아랫입술 landmark** 사이 거리를 계산하고, 얼굴이 카메라에 가까울수록 간격도 커 보이므로 **얼굴 높이(이마~턱)로 나눠 정규화**합니다.

```text
윗입술-아랫입술 거리
───────────────  ≥ 0.06 이면 "입을 벌렸다"
   얼굴 높이
```

blendshape가 아니라 **facial landmark 좌표로 직접 계산**합니다.

### 5. 여러 조건을 조합한 Eat Event

> 물고기가 입 근처에 있다고 바로 먹은 것으로 처리하지 않고, 세 가지가 모두 맞을 때만 먹었다고 판정했습니다.

```text
물고기를 잡고 있음  +  입을 벌리고 있음  +  물고기가 입과 충분히 가까움  →  EAT
```

"입과 충분히 가까움"의 기준도 물고기-입 거리를 얼굴 높이로 나눈 비율(얼굴 크기 기준)이라서, 얼굴이 크든 작든 같은 감각으로 동작합니다. 한 번 먹으면 잡고 있던 상태가 풀려 같은 프레임에 중복 처리되지 않습니다.

### 6. 상태 기반 물고기 관리

> 물고기가 "지금 무엇을 하는 중인지"를 이름 붙은 상태로 관리했습니다.

- `moving`: 화면에서 헤엄치는 중
- `held`: 손에 잡힌 상태
- `hidden`: 먹힌 뒤 잠시 사라진 상태 (일정 시간 뒤 다시 등장)

### 7. 연속 섭취와 effect level

> 같은 색을 이어서 먹으면 같은 효과가 점점 강해지고, 다른 색을 먹으면 처음부터 다시 시작합니다.

```text
Red → Red → Red   :  Level 1 → 2 → 3
Red → Red → Yellow:  Yellow Level 1 (연속 기록 초기화)
```

단순히 먹은 횟수를 세는 것이 아니라 **직전에 먹은 종류를 기억하고 다음 이벤트와 비교**합니다.

### 8. 같은 프레임에서 Hand + Face 결과 조합

> 손 정보와 얼굴 정보를 같은 순간의 영상에서 얻어 함께 사용했습니다.

FACE FISH는 같은 루프에서 **같은 웹캠 프레임을 Hand Landmarker와 Face Landmarker에 차례로 넣어** 결과를 얻습니다. 병렬 추론은 아니며, 같은 프레임에서 얻은 손/얼굴 결과를 같은 좌표계에서 조합해 사용합니다.

---

## Applied Example: FACE FISH

경로: `projects/face_fish/` (세부 구현은 [`projects/face_fish/README.md`](projects/face_fish/README.md)를 참고하세요.)

FACE FISH는 위에서 정리한 판정 규칙들을 한곳에 모아 본 **응용 예제**입니다. 웹캠 화면 아래쪽에 물고기가 헤엄치고, 손가락으로 물고기를 집어 입으로 가져가 먹으면 물고기 색에 따라 얼굴 효과가 나타납니다.

```text
손을 움직인다 → 엄지와 검지를 오므린다 → 물고기를 잡는다
→ 입까지 가져간다 → 입을 벌린다 → 먹기 판정 → 점수 + 얼굴 효과
```

Interaction Flow (기술 관점):

```text
Webcam
  ↓
Hand Landmarker → 엄지/검지 landmark → Pinch Detection → Fish Grab / Drag
  ↓
Face Landmarker → 입술 landmark → Mouth Open Detection → Eat Event
  ↓
Score / Face Effect
```

주요 기능:
- 실시간 Hand + Face tracking
- pinch 기반 물고기 grab / drag
- 물고기 이동, 등급(rarity)별 등장 확률과 점수
- 입 벌림 판정, 먹기 판정
- 같은 색 연속 섭취 시 effect level 증가, 얼굴 효과
- score / timer
- interaction marker, 물결 같은 시각 피드백

### FACE FISH Controls

- 엄지와 검지를 오므려 **pinch**하면 물고기를 잡습니다.
- pinch를 유지한 채 손을 움직이면 물고기가 손을 따라옵니다. pinch를 풀면 그 자리에 놓습니다.
- 입을 벌리고 물고기를 입 가까이 가져가면 먹습니다.
- 같은 색 물고기를 연속으로 먹으면 effect level이 최대 3까지 올라가고, 다른 색을 먹으면 Level 1부터 다시 시작합니다.
- `SPACE`: 시작 / 재시작 (제한 시간 45초)
- `q` 또는 `ESC`: 종료

---

## Project Structure

```text
mediapipe/
├── vision_tasks/
│   ├── hand_detection.py
│   ├── gesture_recognition.py
│   └── face_landmarker.py
│
├── projects/
│   └── face_fish/
│       ├── face_fish.py
│       └── README.md
│
├── models/
│   ├── hand_landmarker.task
│   ├── face_landmarker.task
│   └── gesture_recognizer.task
│
├── requirements.txt
├── README.md
└── .gitignore
```

- `vision_tasks/`: 개별 비전 기능을 연습한 코드
- `projects/face_fish/`: 여러 비전 기능을 하나의 인터랙션으로 통합한 응용 예제
- `models/`: MediaPipe Tasks가 사용하는 사전학습 모델 파일

---

## Installation

PowerShell에서 프로젝트 폴더로 이동한 뒤 필요한 패키지를 설치합니다.

```powershell
python -m pip install -r requirements.txt
```

개발·확인 환경: Windows 11, Python 3.14, mediapipe 0.10.31, opencv-python 5.0.0. 웹캠 기반 실습이므로 웹캠이 필요합니다. (`hand_detection.py`는 이미지 파일 모드도 있습니다.)

모델 파일(`models/*.task`)은 저장소에 포함되어 있습니다. 공식 모델을 다시 내려받아야 하는 경우 아래 주소를 사용할 수 있습니다.

```text
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
```

## Run

프로젝트 루트 폴더에서 실행합니다.

```powershell
python vision_tasks/hand_detection.py
python vision_tasks/gesture_recognition.py
python vision_tasks/face_landmarker.py
python projects/face_fish/face_fish.py
```

- `hand_detection.py`: 실행 후 메뉴에서 선택 (`1` 웹캠, `2` 이미지 파일), 웹캠에서는 `q`로 종료
- `gesture_recognition.py`, `face_landmarker.py`: `q` 또는 `ESC`로 종료
- `face_fish.py`: 조작법은 위 FACE FISH Controls 참고

---

## Tech Stack

- **Python**: 전체 로직 구현
- **OpenCV**: 웹캠 입력, 화면 렌더링, 손 영역 탐지(HSV)
- **MediaPipe Tasks API**: 손 / 얼굴 / 제스처 인식
- **NumPy**: 좌표와 수치 계산

---

## What I Learned

처음에는 MediaPipe가 손이나 얼굴을 인식해 주면 바로 인터랙션을 만들 수 있을 것처럼 보였습니다. 하지만 모델이 주는 결과와 사용자가 느끼는 "행동" 사이에는 한 단계가 더 필요했습니다.

MediaPipe는 엄지와 검지의 위치를 알려 주지만 "사용자가 지금 물고기를 집고 있다"는 상태를 만들어 주지는 않습니다. 그래서 landmark 좌표로 거리를 계산하고, 손·얼굴 크기로 정규화하고, 임계값을 정하고, hysteresis로 떨림을 줄이고, 상태를 저장하는 로직을 직접 만들어야 했습니다. "먹기"도 하나의 모델이 알려 주는 결과가 아니라, 잡고 있음 + 입이 열림 + 입과 가까움이라는 여러 조건을 조합해 만든 이벤트입니다.

이 프로젝트를 통해 컴퓨터 비전 모델의 출력값을 실제 사용자 경험으로 연결하는 과정을 직접 구현하며 학습했습니다.

```text
Vision Task Output → Interaction Logic → Real-time Experience
```

---

## Next Steps

아래는 모두 **아직 구현하지 않은 Future Work**입니다.

- Standalone Hand Landmarker example
- Web version
- Pose Landmarker
- Image Segmenter
- Object Detector
- WebGL / Canvas face effects

---

## License

저장소 코드에는 아직 별도의 라이선스가 지정되어 있지 않습니다. `models/*.task`는 Google MediaPipe 공식 배포 모델이며, 사용 조건은 각 모델 카드와 MediaPipe의 라이선스 조건(Apache 2.0)을 확인하세요.
