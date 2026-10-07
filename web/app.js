// FACE FISH 웹 버전: 메인 메뉴 → (How to Play) → Play(카메라 허용) → 게임. MediaPipe Hand/Face Landmarker + 물고기 게임 + 얼굴 morph/효과
import { FilesetResolver, HandLandmarker, FaceLandmarker, DrawingUtils } from "@mediapipe/tasks-vision";

// index.html import map의 버전과 맞춰야 함
const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm";
const HAND_MODEL = "./models/hand_landmarker.task";
const FACE_MODEL = "./models/face_landmarker.task";

const startBtn = document.getElementById("startBtn");
const statusEl = document.getElementById("status");
const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const drawing = new DrawingUtils(ctx);

// ---- 물고기 설정 (projects/face_fish/face_fish.py 값을 480px 높이 기준으로 참고, 실제 canvas 높이에 맞춰 비례 환산) ----
const REF_HEIGHT = 480;
const FISH_COUNT = 5;
const FISH_RADIUS = 40;          // 몸통 반지름(px @480)
const FISH_AREA_HEIGHT = 170;    // 물고기가 움직이는 화면 하단 영역 높이(px @480)
const FISH_MIN_SPEED = 70;       // 좌우 이동 속도(px/초 @480), 물고기마다 이 범위에서 랜덤
const FISH_MAX_SPEED = 140;
const SWIM_BOB_AMPLITUDE = 6;    // 위아래 흔들림 폭(px @480)
const SWIM_BOB_SPEED = 3.0;      // 흔들림 속도(rad/초)
const TAIL_WAG_SPEED = 10;       // 꼬리 흔들기 속도(rad/초)
const FISH_ALPHA = 0.6;          // 몸통 불투명도
const FISH_GLOW = 0.8;           // soft glow 크기(반지름 배율)
// ---- 게임 / rarity 설정 (projects/face_fish/face_fish.py와 동일) ----
const GAME_DURATION = 45;            // 게임 시간(초)
const MAX_EFFECT_LEVEL = 3;          // 효과 레벨 상한 (같은 물고기를 이보다 많이 먹어도 여기서 멈춤)
const EFFECT_DURATION = 10;          // 먹은 뒤 효과 유지 시간(초). 이번 단계에서는 값만 관리 (face effect 미구현)
// 물고기 카탈로그: coral / muted gold / mint-sage 팔레트 (Python의 BGR 값을 RGB로 옮긴 것). Epic은 슬롯만 있고 물고기는 없음
const FISH_TYPES = [
  { name: "Red Fish", color: "#f08078", rarity: "Common" },
  { name: "Yellow Fish", color: "#e8c46e", rarity: "Uncommon" },
  { name: "Green Fish", color: "#8cd0aa", rarity: "Rare" },
];
// 재등장할 때 등급을 먼저 이 가중치로 뽑고, 같은 등급 안에서는 균등하게 뽑음
const FISH_SPAWN_WEIGHTS = { Common: 60, Uncommon: 28, Rare: 10, Epic: 2 };
const RARITY_SCORE = { Common: 1, Uncommon: 2, Rare: 5, Epic: 10 };   // 먹었을 때 점수
const HIGH_RARITIES = ["Rare", "Epic"];   // 화면 동시 등장 수를 제한하는 등급
const MAX_HIGH_RARITY_ON_SCREEN = 1;
const RARITY_GLOW = { Common: 1.0, Uncommon: 1.25, Rare: 1.6, Epic: 2.0 };   // 등급이 높을수록 glow만 조금 더 큼
const RARE_FLASH_DURATION = 1.2;     // Rare/Epic 등장 시 퍼지는 물결 시간(초)
// ---- 얼굴 효과 설정. 규칙: morph(주인공) + soft tint + small particles(보조) + (LV3) subtle pulse. Python보다 웹에서 더 눈에 띄게 강화한 값 ----
// morph는 WebGL fragment shader에서 국소 bulge(+ 선택적 lift)로 영상을 변형
// Python 값(Red 0.35/0.55/0.75, Yellow 0.12/0.20/0.28, Green 0.10/0.16/0.22)을 웹 체감용으로 강화
const MORPH_STRENGTH = {                        // 레벨별 bulge 강도 (0~1, 클수록 많이 부풀어 보임. 중심 확대 ≈ 1/(1-값))
  "Red Fish": [0.42, 0.60, 0.78],               // 양 볼: 통통하게 부풂 (가장 강한 morph)
  "Yellow Fish": [0.38, 0.52, 0.66],            // 양 눈 주변: 눈이 커지고 또렷해짐
  "Green Fish": [0.32, 0.44, 0.56],             // 이마 / 윗얼굴: 이마가 부풂
};
// "위로 들어올림": 변형 영역의 내용을 얼굴 위쪽 방향으로 밀어 올림 (반경 대비 비율). Green=이마/헤어라인이 위로 확장.
// Yellow는 0: 눈 안쪽 내용(홍채)까지 같이 올라가 눈꺼풀에 가려져 보이므로 둥근 확대(bulge)만 사용
const MORPH_LIFT = {
  "Red Fish": [0, 0, 0],
  "Yellow Fish": [0, 0, 0],
  "Green Fish": [0.20, 0.30, 0.40],
};
const MORPH_STRENGTH_MAX = 0.80;                // 안전 상한
const CHEEK_WARP_RADIUS_RATIO = 0.30;           // 반경 = 얼굴 너비 × 값 (고정 픽셀 없음)
const CHEEK_WARP_OUTWARD = 0.06;                // 볼 중심을 얼굴 바깥쪽으로 옮기는 정도 (얼굴 너비 × 값)
const EYE_WARP_RADIUS_RATIO = 0.15;             // Python 0.11 → 눈 + 눈꺼풀까지 덮도록 확대 (두 눈 사이 거리 ≈ 0.4 × 얼굴 너비라 겹치지 않음)
const FOREHEAD_WARP_RADIUS_RATIO = 0.28;        // Python 0.26
const EFFECT_TINT_ALPHA = [0.28, 0.40, 0.54];   // 레벨별 soft tint 진하기 (0~1)
const EFFECT_PARTICLES = [6, 12, 20];            // 레벨별 파티클 개수
const EFFECT_PARTICLE_SIZE = 0.018;             // 파티클 반지름 (얼굴 너비 × 값, 개별 ±30% 차이)
const EFFECT_PARTICLE_ALPHA = 0.9;             // 파티클 최대 불투명도
const EFFECT_PULSE_LV3 = 0.22;                  // LV3 tint가 천천히 숨 쉬는 정도
const EFFECT_FADE_IN = 0.8, EFFECT_FADE_OUT = 1.0;   // 효과가 서서히 나타나고 사라지는 시간(초)
const FACE_LEFT = 234, FACE_RIGHT = 454;         // 얼굴 너비 기준
const LEFT_CHEEK = 205, RIGHT_CHEEK = 425;
const IRIS_CENTERS = [468, 473];
const GLABELLA = 168;                            // 미간
const DEBUG_GAME = true;            // true면 마지막으로 먹은 물고기 / streak / effect level을 작게 표시

// ---- pinch 설정 (projects/face_fish/face_fish.py와 동일) ----
const PINCH_ON = 0.30;           // 엄지-검지 거리 / 손 크기가 이 값 이하면 pinch 시작
const PINCH_OFF = 0.40;          // 이미 pinch 중이면 이 값 이상이 될 때까지 유지 (히스테리시스)
const WRIST = 0, THUMB_TIP = 4, INDEX_TIP = 8, MIDDLE_MCP = 9;
// ---- mouth / eat 설정 (projects/face_fish/face_fish.py와 동일) ----
const INNER_LIP_TOP = 13, INNER_LIP_BOTTOM = 14;   // 안쪽 윗/아랫입술
const MOUTH_LEFT = 61, MOUTH_RIGHT = 291;          // 입꼬리
const FOREHEAD = 10, CHIN = 152;                   // 얼굴 높이 기준
const MOUTH_OPEN_THRESHOLD = 0.06;   // 안쪽 입술 간격 / 얼굴 높이가 이 값 이상이면 입 열림
const EAT_DISTANCE_RATIO = 0.25;     // 물고기-입 중심 거리 / 얼굴 높이가 이 값 이하면 먹을 수 있는 거리
const RESPAWN_DELAY = 1.5;           // 먹힌 물고기가 다시 나타나기까지 시간(초)
const EAT_POP_DURATION = 0.45;       // 먹히는 순간 커졌다 사라지는 시간(초). RESPAWN_DELAY보다 짧아야 함
// ---- interaction marker (손 pinch 위치 표시, Python 버전과 같은 pale cyan-white) ----
const MARKER_COLOR = "#e1f5fa";
const MARKER_RADIUS = 14;        // px @480
const MARKER_ALPHA_IDLE = 0.3;   // 손이 보일 때 (은은하게)
const MARKER_ALPHA_PINCH = 0.7;  // pinch 중
const MARKER_SCALE_PINCH = 1.3;
// 성능 측정: 켜면 구간별 평균 시간(ms)을 1초에 한 번 console에 출력. 기본 꺼짐. 주소 끝에 ?perf 를 붙여도 켜짐 (코드를 고치지 않고 측정용)
const DEBUG_PERFORMANCE = typeof location !== "undefined" && new URLSearchParams(location.search).has("perf");
// Hand/Face inference 최대 횟수/초. 0 = 새 video 프레임마다 (기본). 측정에서 MediaPipe가 병목일 때만 24~30 같은 값으로 제한 (화면 렌더링은 영향 없음)
const INFERENCE_MAX_FPS = 0;
const DEBUG_LANDMARKS = false;   // true로 바꾸면 손/얼굴 landmark 디버그 표시 (face mesh, hand skeleton, 점). 기본 꺼짐: 켜면 매 프레임 수천 개 선을 그려 느려짐

let handLandmarker = null;
let faceLandmarker = null;
let lastVideoTime = -1;
let frames = 0;
let lastReport = performance.now();
let counts = { hands: 0, face: 0 };
let lastHandResult = null;   // 추론은 새 프레임에서만 하지만 물고기는 매 화면 갱신마다 그리므로 마지막 결과를 보관
let lastFaceResult = null;
let lastTick = performance.now();
let lastInferenceAt = -Infinity;   // 마지막 inference 시각(ms), INFERENCE_MAX_FPS 용
let fishes = [];
let pinching = false;        // 이전 프레임의 pinch 상태 (히스테리시스용)
let pinchPoint = null;       // 엄지-검지 중간점 (canvas 좌표). 손이 안 보이면 null
let heldFish = null;         // 지금 잡고 있는 물고기 (동시에 하나만)
let mouth = null;            // { center, openness, isOpen, faceH } (canvas 좌표). 얼굴이 안 보이면 null → 먹기 불가
let markerLevel = 0;         // 0 = idle ~ 1 = pinch, 부드럽게 보간
// 게임 상태 (Python의 phase / score / game_start / last_eaten / current_streak / effect_until)
let phase = "ready";         // ready(SPACE 대기) / playing / over
let score = 0;
let gameStart = 0;           // 게임 시작 시각(초)
let lastEaten = null;        // 마지막으로 먹은 물고기 이름
let streak = 0;              // lastEaten 물고기를 연속으로 먹은 횟수
let effectUntil = 0;         // 효과 종료 시각(초). face effect 단계에서 사용
let rng = Math.random;       // 테스트에서 바꿀 수 있게 분리
// 웹페이지 UI 화면 (게임 phase ready/playing/over와 별개): menu / howto / loading / game
let uiScreen = "menu";
let cameraReady = false;     // 카메라+게임 loop가 이미 시작되었는지 (다시 요청하지 않기 위해)
let modelsPromise = null;
let modelsReady = false;

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

// GPU delegate를 먼저 시도하고, 실패하면 CPU로 다시 만든다
async function createWithFallback(create) {
  try {
    return await create("GPU");
  } catch (err) {
    console.warn("GPU delegate failed, falling back to CPU:", err);
    return await create("CPU");
  }
}

async function loadModels() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  [handLandmarker, faceLandmarker] = await Promise.all([
    createWithFallback((delegate) =>
      HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: HAND_MODEL, delegate },
        runningMode: "VIDEO",
        numHands: 1,
      })),
    createWithFallback((delegate) =>
      FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: FACE_MODEL, delegate },
        runningMode: "VIDEO",
        numFaces: 1,
      })),
  ]);
  // 워밍업: 첫 detectForVideo는 GPU 그래프/shader 준비 때문에 약 0.8초씩(손+얼굴 합쳐 ~1.6초) 걸려 게임 시작 직후 화면이 멈춘 것처럼 보임.
  // 작은 빈 canvas로 미리 한 번씩 실행해 그 비용을 로딩 단계로 옮김 (입력 크기와 무관하게 준비됨, 결과/게임 로직에는 영향 없음)
  setStatus("Warming up models...");
  await new Promise((r) => setTimeout(r, 30));   // 상태 문구가 먼저 그려지도록
  const warm = document.createElement("canvas");
  warm.width = warm.height = 64;
  handLandmarker.detectForVideo(warm, 1);        // 실제 inference 타임스탬프(performance.now())보다 항상 작은 값
  faceLandmarker.detectForVideo(warm, 1);
}

// 카메라를 켜고 게임 loop를 시작한다. 성공하면 null, 실패하면 화면에 보여줄 { message, hint }를 반환
async function startCamera() {
  startBtn.disabled = true;
  setStatus("Requesting camera permission...");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
  } catch (err) {
    console.error(err);
    setStatus(`Camera error: ${err.name} - ${err.message}`, true);
    startBtn.disabled = false;
    return describeCameraError(err);
  }
  if (!modelsReady) showLoading("모델을 불러오고 있어요...");   // 카메라 권한은 받았지만 모델이 아직 준비 중
  try {
    await modelsPromise;
  } catch (err) {
    video.srcObject.getTracks().forEach((t) => t.stop());   // 카메라를 놓아줌
    startBtn.disabled = false;
    return { message: "모델을 불러오지 못했어요.", hint: "인터넷 연결을 확인하고 다시 시도해 주세요.", reloadModels: true };
  }
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  initWarp(video.videoWidth, video.videoHeight);
  document.getElementById("stage").style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
  startBtn.textContent = "CAMERA ON";
  setStatus("Camera on");
  fishes = makeFishes(canvas.width, canvas.height, performance.now() / 1000);
  phase = "ready";
  updateHud();
  lastTick = performance.now();
  requestAnimationFrame(loop);
  return null;
}

// ---- 성능 측정 (DEBUG_PERFORMANCE일 때만 동작, 꺼져 있으면 시간 측정 호출 자체를 하지 않음) ----
const perf = { sum: {}, n: {}, raf: 0, last: performance.now() };
function perfAdd(key, ms) {
  perf.sum[key] = (perf.sum[key] || 0) + ms;
  perf.n[key] = (perf.n[key] || 0) + 1;
}
const perfNow = () => (DEBUG_PERFORMANCE ? performance.now() : 0);
// 1초마다 평균 ms 한 줄 출력 (프레임마다 출력하지 않음)
function perfReport(now) {
  perf.raf++;
  if (now - perf.last < 1000) return;
  const secs = (now - perf.last) / 1000;
  const avg = (k) => (perf.n[k] ? (perf.sum[k] / perf.n[k]).toFixed(2) : "-");
  const line = `[perf] render ${(perf.raf / secs).toFixed(0)}fps | inference ${((perf.n.hand || 0) / secs).toFixed(0)}/s | ` +
    `hand ${avg("hand")}ms face ${avg("face")}ms | warp upload ${avg("upload")}ms morph ${avg("morph")}ms | ` +
    `overlay ${avg("overlay")}ms | update ${avg("update")}ms | frame ${avg("frame")}ms`;
  console.log(line);
  window.__perfLine = line;
  perf.sum = {}; perf.n = {}; perf.raf = 0; perf.last = now;
}

function loop() {
  const tick = performance.now();
  const frameT0 = perfNow();
  const dt = Math.min((tick - lastTick) / 1000, 0.1);  // 멈춤 후 튀는 것 방지
  lastTick = tick;

  // 새 프레임이 있을 때만 추론 (같은 프레임에 두 번 넣지 않음)
  if (video.currentTime !== lastVideoTime && video.readyState >= 2 &&
      (!INFERENCE_MAX_FPS || tick - lastInferenceAt >= 1000 / INFERENCE_MAX_FPS - 1)) {
    lastVideoTime = video.currentTime;
    lastInferenceAt = tick;
    // 같은 프레임을 같은 루프에서 Hand → Face 순서로 차례로 처리 (병렬 아님)
    const t0 = perfNow();
    lastHandResult = handLandmarker.detectForVideo(video, tick);
    const t1 = perfNow();
    lastFaceResult = faceLandmarker.detectForVideo(video, tick);
    const t2 = perfNow();
    counts = { hands: lastHandResult.landmarks.length, face: lastFaceResult.faceLandmarks.length };
    updatePinch(lastHandResult);
    updateMouth(lastFaceResult);
    tryEat(tick / 1000);
    frames++;
    if (DEBUG_PERFORMANCE) { perfAdd("hand", t1 - t0); perfAdd("face", t2 - t1); perfAdd("update", performance.now() - t2); }
  }
  updateGameClock(tick / 1000);
  if (phase === "playing") {          // 물고기 이동/재등장은 playing에서만 (ready / over에서는 멈춤)
    updateFishes(canvas.width, canvas.height, dt);
    respawnFishes(canvas.width, canvas.height, tick / 1000);
  }
  const u0 = perfNow();
  updateHud();
  updateWarp(tick / 1000);
  const r0 = perfNow();
  render(dt);
  if (DEBUG_PERFORMANCE) { perfAdd("update", r0 - u0); perfAdd("overlay", performance.now() - r0); }
  report();
  if (DEBUG_PERFORMANCE) { perfAdd("frame", performance.now() - frameT0); perfReport(performance.now()); }
  requestAnimationFrame(loop);
}

// ---------------- 게임 phase / 점수 / streak ----------------
// ready --SPACE--> playing --45초--> over --SPACE--> playing ... (Python과 동일: playing이 아닐 때만 SPACE로 시작/재시작)
const hudEl = document.getElementById("hud");
const scoreEl = document.getElementById("score");
const timeEl = document.getElementById("time");
const bannerEl = document.getElementById("banner");
const bannerTitle = document.getElementById("bannerTitle");
const bannerSub = document.getElementById("bannerSub");
const bannerHint = document.getElementById("bannerHint");
const debugEl = document.getElementById("debugGame");

// 같은 물고기를 연속으로 먹은 횟수(streak) → 효과 레벨 (1 ~ MAX_EFFECT_LEVEL)
function effectLevel(s) {
  return Math.max(1, Math.min(s, MAX_EFFECT_LEVEL));
}

function timeLeft(nowSec) {
  if (phase === "ready") return GAME_DURATION;
  if (phase === "over") return 0;
  return Math.max(GAME_DURATION - (nowSec - gameStart), 0);
}

// SPACE: 시작 / 재시작. 점수, 시간, 물고기, streak, 효과 상태를 모두 처음 상태로 되돌림
function startGame(nowSec) {
  phase = "playing";
  score = 0;
  gameStart = nowSec;
  fishes = makeFishes(canvas.width, canvas.height, nowSec);
  heldFish = null;
  lastEaten = null;
  effectUntil = 0;
  streak = 0;
}

// 시간이 다 되면 over: 이동/상호작용 정지, 잡고 있던 물고기는 놓음
function updateGameClock(nowSec) {
  if (phase === "playing" && nowSec - gameStart >= GAME_DURATION) {
    phase = "over";
    releaseFish();
  }
}

window.addEventListener("keydown", (e) => {
  if (uiScreen !== "game") {            // 메뉴 / How to Play / 로딩 화면: 게임 phase를 건드리지 않음 (버튼은 브라우저 기본 동작으로 Enter/Space 활성화)
    if (e.code === "Escape" && uiScreen === "howto") setScreen("menu");
    return;
  }
  if (e.code !== "Space" || !fishes.length) return;   // 카메라를 시작하기 전에는 무시
  e.preventDefault();
  if (phase !== "playing") startGame(performance.now() / 1000);
});

// HUD는 HTML 요소 (canvas는 CSS로 좌우 반전되어 있어 글자를 그리면 뒤집힘). 값이 바뀔 때만 DOM을 갱신
function setText(el, text) {
  if (el.textContent !== text) el.textContent = text;
}

function updateHud() {
  hudEl.hidden = false;
  const left = timeLeft(performance.now() / 1000);
  setText(scoreEl, `SCORE  ${score}`);
  setText(timeEl, `TIME  ${Math.ceil(left)}`);
  timeEl.classList.toggle("warn", phase === "playing" && left <= 10);
  bannerEl.hidden = phase === "playing";
  if (phase === "ready") {
    setText(bannerTitle, "FACE FISH");
    setText(bannerSub, "");
    setText(bannerHint, "SPACE to start");
  } else if (phase === "over") {
    setText(bannerTitle, "TIME UP");
    setText(bannerSub, `Score  ${score}`);
    setText(bannerHint, "SPACE to restart");
  }
  debugEl.hidden = !(DEBUG_GAME && lastEaten);
  if (!debugEl.hidden) setText(debugEl, `${lastEaten} x${streak} / Effect Lv.${effectLevel(streak)}`);
}

// ---------------- Pinch / grab / drag / release ----------------
// 좌표계: landmark(0~1) × canvas 크기 = canvas 내부 좌표. video와 canvas가 같은 크기로 CSS 반전되어 있어서
// 물고기 좌표와 landmark를 따로 뒤집을 필요 없이 같은 좌표계로 바로 비교할 수 있다.

// 엄지 끝-검지 끝 거리를 손목-중지 뿌리 거리(손 크기)로 나눈 값 (Python의 pinch_ratio와 동일, 정규화 좌표 사용)
function pinchRatio(lm) {
  const dist = Math.hypot(lm[THUMB_TIP].x - lm[INDEX_TIP].x, lm[THUMB_TIP].y - lm[INDEX_TIP].y);
  const size = Math.hypot(lm[WRIST].x - lm[MIDDLE_MCP].x, lm[WRIST].y - lm[MIDDLE_MCP].y);
  return size > 0 ? dist / size : 1.0;
}

// 히스테리시스: pinch 중이 아니면 ON 이하일 때 시작, 이미 pinch 중이면 OFF 미만인 동안 유지
function nextPinchState(prevPinching, ratio) {
  return prevPinching ? ratio < PINCH_OFF : ratio <= PINCH_ON;
}

// 잡을 수 있는 거리(물고기 반지름) 안에서 point와 가장 가까운 moving 물고기 하나 (없으면 null)
function findFishAt(point) {
  let best = null;
  let bestDist = Infinity;
  for (const f of fishes) {
    if (f.state !== "moving") continue;
    const d = Math.hypot(f.x - point.x, f.y - point.y);
    if (d <= f.size && d < bestDist) { best = f; bestDist = d; }
  }
  return best;
}

function releaseFish() {
  if (heldFish) heldFish.state = "moving";   // 놓은 자리에서 다시 수영 시작 (vx / direction 유지)
  heldFish = null;
}

// 새 프레임의 손 결과로 pinch 상태 갱신 → grab / drag / release
function updatePinch(handResult) {
  const lm = handResult.landmarks[0];
  if (!lm) {                          // 손 tracking이 사라지면 안전하게 release
    pinching = false;
    pinchPoint = null;
    releaseFish();
    return;
  }
  const wasPinching = pinching;
  pinching = nextPinchState(wasPinching, pinchRatio(lm));
  pinchPoint = {
    x: ((lm[THUMB_TIP].x + lm[INDEX_TIP].x) / 2) * canvas.width,
    y: ((lm[THUMB_TIP].y + lm[INDEX_TIP].y) / 2) * canvas.height,
  };
  if (phase === "playing" && pinching && !wasPinching && !heldFish) {
    // grab은 pinch가 시작되는 순간에만: 빈 곳에서 pinch한 채 물고기 위로 가도 잡히지 않는다
    heldFish = findFishAt(pinchPoint);
    if (heldFish) heldFish.state = "held";
  } else if (!pinching) {
    releaseFish();
  }
  if (heldFish) {                     // drag: pinch 중간점을 따라가되 화면 안으로 clamp
    const r = heldFish.size;
    heldFish.x = Math.min(Math.max(pinchPoint.x, r), canvas.width - r);
    heldFish.y = Math.min(Math.max(pinchPoint.y, r), canvas.height - r);
  }
}

// ---------------- Mouth / eat ----------------
// landmark(0~1) → canvas 좌표(px)로 바꾼 뒤 거리를 잰다 (Python mouth_info와 동일: 픽셀 기준)
// 반환: 입 중심(입꼬리 2개 + 안쪽 입술 2개의 평균), 입 벌림(안쪽 입술 간격 / 얼굴 높이), 얼굴 높이(이마~턱)
function mouthInfo(lm, w, h) {
  const p = (i) => ({ x: lm[i].x * w, y: lm[i].y * h });
  const corners = [MOUTH_LEFT, MOUTH_RIGHT, INNER_LIP_TOP, INNER_LIP_BOTTOM].map(p);
  const center = {
    x: corners.reduce((s, c) => s + c.x, 0) / corners.length,
    y: corners.reduce((s, c) => s + c.y, 0) / corners.length,
  };
  const top = p(INNER_LIP_TOP), bottom = p(INNER_LIP_BOTTOM), f = p(FOREHEAD), c = p(CHIN);
  const faceH = Math.hypot(f.x - c.x, f.y - c.y);
  const gap = Math.hypot(top.x - bottom.x, top.y - bottom.y);
  return { center, openness: faceH > 0 ? gap / faceH : 0, faceH };
}

// 새 프레임의 얼굴 결과로 mouth 상태 갱신 (blendshape가 아니라 landmark 좌표 기반)
function updateMouth(faceResult) {
  const lm = faceResult.faceLandmarks[0];
  if (!lm) { mouth = null; return; }   // 얼굴이 사라지면 먹기 불가
  const info = mouthInfo(lm, canvas.width, canvas.height);
  mouth = { ...info, isOpen: info.openness >= MOUTH_OPEN_THRESHOLD };
}

// 먹기 조건: 물고기를 잡고 있음 + 입 열림 + 물고기-입 거리 <= EAT_DISTANCE_RATIO × 얼굴 높이
function canEat(fish, m) {
  if (!fish || fish.state !== "held" || !m || !m.isOpen) return false;
  return Math.hypot(fish.x - m.center.x, fish.y - m.center.y) <= EAT_DISTANCE_RATIO * m.faceH;
}

// 먹으면 hidden으로 바꾸고 heldFish를 해제 → 같은 프레임/같은 물고기는 한 번만 처리됨. playing일 때만 먹을 수 있음
function tryEat(nowSec) {
  if (phase !== "playing" || !canEat(heldFish, mouth)) return false;
  // 점수 / streak: 같은 물고기를 연속으로 먹으면 +1, 다른 물고기면 그 물고기로 바꾸고 1부터 다시 시작
  score += RARITY_SCORE[heldFish.rarity];
  streak = heldFish.name === lastEaten ? streak + 1 : 1;
  lastEaten = heldFish.name;
  effectUntil = nowSec + EFFECT_DURATION;
  heldFish.state = "hidden";
  heldFish.eatenAt = nowSec;                       // 먹힘 pop 연출 시작
  heldFish.respawnAt = nowSec + RESPAWN_DELAY;
  heldFish = null;
  return true;
}

// hidden 물고기를 시간이 되면 다른 물고기와 가장 먼 위치(후보 12개 중)에 새 방향/속도로 다시 등장시킴
function respawnFishes(w, h, nowSec) {
  const k = h / REF_HEIGHT;
  for (const f of fishes) {
    if (f.state !== "hidden" || nowSec < f.respawnAt) continue;
    const yMin = h - FISH_AREA_HEIGHT * k + f.size;
    const yMax = h - f.size;
    const others = fishes.filter((o) => o !== f && o.state !== "hidden");
    let best = null;
    let bestScore = -1;
    for (let i = 0; i < 12; i++) {
      const c = { x: f.size + Math.random() * (w - 2 * f.size), y: yMin + Math.random() * (yMax - yMin) };
      const score = others.length ? Math.min(...others.map((o) => Math.hypot(c.x - o.x, c.y - o.y))) : 0;
      if (score > bestScore) { best = c; bestScore = score; }
    }
    const dir = Math.random() < 0.5 ? -1 : 1;
    const speed = (FISH_MIN_SPEED + Math.random() * (FISH_MAX_SPEED - FISH_MIN_SPEED)) * k;
    applyFish(f, pickFish(fishes, f), nowSec);   // 등급 가중치로 새 물고기를 다시 뽑음
    f.x = best.x; f.y = best.y;
    f.vx = dir * speed; f.direction = dir;
    f.state = "moving";
  }
}

// 손/pinch 위치 marker: 얇은 반투명 링 + 작은 중심점 (idle → pinch → 물고기를 잡음 순으로 조금씩 밝아짐)
function drawMarker(dt) {
  if (!pinchPoint) { markerLevel = 0; return; }
  markerLevel += ((pinching ? 1 : 0) - markerLevel) * Math.min(dt * 10, 1);
  const k = canvas.height / REF_HEIGHT;
  const r = MARKER_RADIUS * k * (1 + (MARKER_SCALE_PINCH - 1) * markerLevel);
  const alpha = MARKER_ALPHA_IDLE + (MARKER_ALPHA_PINCH - MARKER_ALPHA_IDLE) * markerLevel;
  ctx.save();
  ctx.strokeStyle = MARKER_COLOR;
  ctx.fillStyle = MARKER_COLOR;
  ctx.lineWidth = Math.max(1.5 * k, 1);
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.arc(pinchPoint.x, pinchPoint.y, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.globalAlpha = alpha * 0.6;
  ctx.beginPath();
  ctx.arc(pinchPoint.x, pinchPoint.y, Math.max(r * 0.35, 2), 0, Math.PI * 2);
  ctx.fill();
  if (heldFish) {                     // 물고기를 잡고 있으면 바깥에 옅은 링 하나 더
    ctx.globalAlpha = 0.25;
    ctx.beginPath();
    ctx.arc(pinchPoint.x, pinchPoint.y, r * 1.7, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

// ---------------- 물고기 ----------------
// 등급 가중치로 물고기 하나를 뽑음 (Python pick_fish와 동일). Rare/Epic이 이미 화면에 최대치면 그 등급은 제외.
// skip: 재등장 중인 물고기 자신은 세지 않음
function pickFish(list, skip = null) {
  const onScreen = list.filter((f) => f !== skip && f.state !== "hidden" && HIGH_RARITIES.includes(f.rarity)).length;
  let rarities = Object.keys(FISH_SPAWN_WEIGHTS).filter((r) => FISH_TYPES.some((t) => t.rarity === r));
  if (onScreen >= MAX_HIGH_RARITY_ON_SCREEN) {
    const normal = rarities.filter((r) => !HIGH_RARITIES.includes(r));
    if (normal.length) rarities = normal;
  }
  const total = rarities.reduce((s, r) => s + FISH_SPAWN_WEIGHTS[r], 0);
  let roll = rng() * total;
  let rarity = rarities[rarities.length - 1];
  for (const r of rarities) {
    roll -= FISH_SPAWN_WEIGHTS[r];
    if (roll < 0) { rarity = r; break; }
  }
  const pool = FISH_TYPES.filter((t) => t.rarity === rarity);   // 같은 등급 안에서는 균등
  return pool[Math.floor(rng() * pool.length)];
}

// 슬롯(fish)에 뽑힌 물고기(spec) 정보를 적용. Rare/Epic이면 등장 강조 시간을 설정
function applyFish(fish, spec, nowSec) {
  fish.name = spec.name;
  fish.color = spec.color;
  fish.rarity = spec.rarity;
  fish.flashUntil = HIGH_RARITIES.includes(spec.rarity) ? nowSec + RARE_FLASH_DURATION : 0;
}

// 상태 객체. state: moving / held / hidden
function makeFishes(w, h, nowSec = performance.now() / 1000) {
  const k = h / REF_HEIGHT;            // 화면 크기 비례 환산
  const r = FISH_RADIUS * k;
  const yMin = h - FISH_AREA_HEIGHT * k + r;
  const yMax = h - r;
  const list = [];
  for (let i = 0; i < FISH_COUNT; i++) {
    const speed = (FISH_MIN_SPEED + Math.random() * (FISH_MAX_SPEED - FISH_MIN_SPEED)) * k;
    const dir = Math.random() < 0.5 ? -1 : 1;
    const fish = {
      name: "",                                   // applyFish가 채움 (name / color / rarity / flashUntil)
      color: "",
      rarity: "Common",
      flashUntil: 0,
      state: "moving",                            // moving / held / hidden
      eatenAt: -Infinity,                         // 먹힌 시각(초), pop 연출용
      respawnAt: 0,                               // hidden에서 다시 나타날 시각(초)
      x: (w * (i + 1)) / (FISH_COUNT + 1),
      y: yMin + ((i % 3) / 2) * (yMax - yMin),   // 3개 레인에 번갈아 배치해 겹침 최소화
      vx: dir * speed,                            // 부호가 이동 방향
      direction: dir,                             // 1 = 머리가 오른쪽(canvas 좌표), -1 = 왼쪽
      size: r,
      phase: Math.random() * Math.PI * 2,         // 흔들림/꼬리 위상 (물고기마다 달라 같이 움직이지 않음)
    };
    applyFish(fish, pickFish(list), nowSec);       // 지금까지 만든 물고기를 보고 등급을 뽑음 (Python make_fishes와 동일)
    list.push(fish);
  }
  return list;
}

function updateFishes(w, h, dt) {
  const k = h / REF_HEIGHT;
  for (const f of fishes) {
    if (f.state !== "moving") continue;
    const yMin = h - FISH_AREA_HEIGHT * k + f.size;
    const yMax = h - f.size;
    f.x += f.vx * dt;
    f.phase += SWIM_BOB_SPEED * dt;
    f.y += SWIM_BOB_AMPLITUDE * k * SWIM_BOB_SPEED * Math.cos(f.phase) * dt;   // 사인파 상하 bobbing
    const margin = f.size * 1.2;                                               // 꼬리가 화면 밖으로 덜 나가게
    if (f.x < margin) { f.x = margin; f.vx = Math.abs(f.vx); }
    else if (f.x > w - margin) { f.x = w - margin; f.vx = -Math.abs(f.vx); }
    f.y += (Math.min(Math.max(f.y, yMin), yMax) - f.y) * 0.2;                  // 하단 영역 밖이면 부드럽게 복귀 (resize 대비)
    f.direction = f.vx > 0 ? 1 : -1;
  }
}

// 반투명 타원 몸통 + 단순한 꼬리 + 작은 눈. 이미지 없이 Canvas 2D 도형만 사용 (glow는 이 물고기에만 적용)
function drawFish(f, timeSec, scale = 1, alpha = 1) {
  const r = f.size;
  const wag = Math.sin(timeSec * TAIL_WAG_SPEED + f.phase) * r * 0.2;
  ctx.save();
  ctx.translate(f.x, f.y);
  ctx.scale(f.direction * scale, scale);   // direction이 -1이면 몸과 꼬리가 통째로 뒤집힘
  ctx.globalAlpha = FISH_ALPHA * alpha;
  ctx.fillStyle = f.color;
  ctx.shadowColor = f.color;
  ctx.shadowBlur = r * FISH_GLOW * (RARITY_GLOW[f.rarity] || 1);   // 등급이 높을수록 glow만 조금 더 큼

  // 꼬리: 몸 뒤쪽(-x)으로 둥글게 갈라진 형태
  ctx.beginPath();
  ctx.moveTo(-0.7 * r, 0);
  ctx.bezierCurveTo(-1.1 * r, -0.15 * r + wag * 0.5, -1.3 * r, -0.5 * r + wag, -1.55 * r, -0.5 * r + wag);
  ctx.quadraticCurveTo(-1.3 * r, wag * 0.5, -1.55 * r, 0.5 * r + wag);
  ctx.bezierCurveTo(-1.3 * r, 0.5 * r + wag, -1.1 * r, 0.15 * r + wag * 0.5, -0.7 * r, 0);
  ctx.fill();

  // 몸통
  ctx.beginPath();
  ctx.ellipse(0, 0, r, r * 0.55, 0, 0, Math.PI * 2);
  ctx.fill();

  // 등 쪽 은은한 하이라이트와 아주 작은 눈 (glow 없이)
  ctx.shadowBlur = 0;
  ctx.fillStyle = "#ffffff";
  ctx.globalAlpha = 0.25 * alpha;
  ctx.beginPath();
  ctx.ellipse(0.15 * r, -0.2 * r, 0.55 * r, 0.14 * r, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 0.6 * alpha;
  ctx.fillStyle = "#463c37";
  ctx.beginPath();
  ctx.arc(0.58 * r, -0.1 * r, Math.max(r * 0.06, 1), 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// ---------------- 얼굴 효과 (Canvas 2D) ----------------
// 전부 얼굴 landmark 좌표 기준의 국소 효과(radial gradient)만 사용. 화면/얼굴 전체 blur나 전체 덮는 필터는 쓰지 않는다.

// 파티클마다 고정 난수 (위치, 위상, 속도, 크기). Python은 numpy RandomState(7)이고 웹은 seed 고정 mulberry32
const PARTICLE_RAND = (() => {
  let a = 7;
  const next = () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return Array.from({ length: EFFECT_PARTICLES[EFFECT_PARTICLES.length - 1] }, () => [next(), next(), next(), next()]);
})();

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function fishColor(name) {
  const t = FISH_TYPES.find((x) => x.name === name);
  return t ? t.color : "#ffffff";
}

// 효과 시작/끝에서 서서히 나타나고 사라지는 0~1 값
function effectEnvelope(age) {
  return Math.max(0, Math.min(age / EFFECT_FADE_IN, 1, (EFFECT_DURATION - age) / EFFECT_FADE_OUT));
}

function byLevel(level, values) {
  return values[Math.max(1, Math.min(level, MAX_EFFECT_LEVEL)) - 1];
}

// 얼굴 중심, 위쪽/오른쪽 단위벡터 (얼굴 기울기 반영)
function faceAxes(p) {
  const top = p(FOREHEAD), chin = p(CHIN), l = p(FACE_LEFT), r = p(FACE_RIGHT);
  const un = (x, y) => { const n = Math.hypot(x, y) + 1e-6; return { x: x / n, y: y / n }; };
  return { center: { x: (top.x + chin.x) / 2, y: (top.y + chin.y) / 2 }, up: un(top.x - chin.x, top.y - chin.y), right: un(r.x - l.x, r.y - l.y) };
}

// 가장자리가 부드러운 반투명 타원 blob 하나 (rx, ry: 반축, sigma: 가장자리 번짐, 회전 angle)
function softBlob(cx, cy, rx, ry, angle, sigma, rgb, alpha) {
  if (alpha <= 0.005) return;
  const R = rx + 2 * sigma;
  const col = (a) => `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`;
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(angle);
  ctx.scale(1, ry / rx);
  const g = ctx.createRadialGradient(0, 0, 0, 0, 0, R);
  g.addColorStop(0, col(alpha));
  g.addColorStop(Math.max((rx - sigma) / R, 0), col(alpha));
  g.addColorStop(rx / R, col(alpha * 0.5));
  g.addColorStop(1, col(0));
  ctx.fillStyle = g;
  ctx.fillRect(-R, -R, 2 * R, 2 * R);
  ctx.restore();
}

// 물고기별 tint 위치: Red=볼 / Yellow=눈 / Green=이마·윗얼굴
function drawTint(name, p, axes, faceW, faceH, rgb, alpha) {
  if (name === "Red Fish") {
    for (const i of [LEFT_CHEEK, RIGHT_CHEEK]) softBlob(p(i).x, p(i).y, faceW * 0.14, faceW * 0.14, 0, faceW * 0.05, rgb, alpha);
  } else if (name === "Yellow Fish") {
    for (const i of IRIS_CENTERS) softBlob(p(i).x, p(i).y, faceW * 0.09, faceW * 0.09, 0, faceW * 0.05, rgb, alpha);
  } else {
    const f = p(FOREHEAD), g = p(GLABELLA);
    softBlob(f.x * 0.45 + g.x * 0.55, f.y * 0.45 + g.y * 0.55, faceW * 0.30, faceH * 0.11,
      Math.atan2(axes.right.y, axes.right.x), faceW * 0.06, rgb, alpha);
  }
}

// 작은 soft 점들. 움직임만 물고기별로 다름: Red=볼에서 위로 떠오름 / Yellow=얼굴 둘레에서 반짝 / Green=이마 위에서 천천히 내려옴
function drawParticles(name, p, axes, faceW, faceH, age, level, rgb, env) {
  const { center, up, right } = axes;
  const n = byLevel(level, EFFECT_PARTICLES);
  for (let k = 0; k < n; k++) {
    const [u, phase, speed, size] = PARTICLE_RAND[k];
    const life = (age * (0.12 + 0.1 * speed) + phase) % 1;
    const fade = Math.sin(Math.PI * life);   // 생성 → 소멸
    let x, y, a;
    if (name === "Red Fish") {
      const side = k % 2 === 0 ? -1 : 1;
      const c = p(side < 0 ? LEFT_CHEEK : RIGHT_CHEEK);
      const sway = Math.sin(age * 1.5 + k) * faceW * 0.02;
      x = c.x + right.x * (side * faceW * 0.10 * u + sway) + up.x * faceH * 0.30 * life;
      y = c.y + right.y * (side * faceW * 0.10 * u + sway) + up.y * faceH * 0.30 * life;
      a = fade;
    } else if (name === "Yellow Fish") {
      const ang = u * 2 * Math.PI + age * 0.15;
      x = center.x + right.x * Math.cos(ang) * faceW * 0.62 + up.x * Math.sin(ang) * faceH * 0.52;
      y = center.y + right.y * Math.cos(ang) * faceW * 0.62 + up.y * Math.sin(ang) * faceH * 0.52;
      a = 0.5 + 0.5 * Math.sin(age * 3 + phase * 6.28);   // 반짝임
    } else {
      const f = p(FOREHEAD);
      const dx = (u - 0.5) * faceW * 0.9 + Math.sin(age * 1.2 + k) * faceW * 0.03;
      const dy = faceH * (0.22 - 0.35 * life);
      x = f.x + right.x * dx + up.x * dy;
      y = f.y + right.y * dx + up.y * dy;
      a = fade;
    }
    const r = faceW * EFFECT_PARTICLE_SIZE * (0.7 + 0.6 * size);
    const alpha = Math.min(EFFECT_PARTICLE_ALPHA * a * env, 1);
    if (alpha <= 0.01) continue;
    const g = ctx.createRadialGradient(x, y, 0, x, y, r * 2);
    g.addColorStop(0, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`);
    g.addColorStop(0.5, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha * 0.6})`);
    g.addColorStop(1, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0)`);
    ctx.fillStyle = g;
    ctx.fillRect(x - r * 2, y - r * 2, r * 4, r * 4);
  }
}

// 마지막으로 먹은 물고기의 효과: 효과 시간 안이고 얼굴이 보일 때만 그림 (얼굴이 사라지면 아무것도 남지 않음)
function drawFaceEffect(nowSec) {
  if (!lastEaten || nowSec >= effectUntil || !lastFaceResult) return false;
  const lm = lastFaceResult.faceLandmarks[0];
  if (!lm) return false;
  const w = canvas.width, h = canvas.height;
  const p = (i) => ({ x: lm[i].x * w, y: lm[i].y * h });
  const axes = faceAxes(p);
  const faceW = Math.hypot(p(FACE_LEFT).x - p(FACE_RIGHT).x, p(FACE_LEFT).y - p(FACE_RIGHT).y);
  const faceH = Math.hypot(p(FOREHEAD).x - p(CHIN).x, p(FOREHEAD).y - p(CHIN).y);
  const age = nowSec - (effectUntil - EFFECT_DURATION);
  const level = effectLevel(streak);
  const env = effectEnvelope(age);
  const rgb = hexToRgb(fishColor(lastEaten));
  const pulse = level >= 3 ? 1 + EFFECT_PULSE_LV3 * Math.sin(age * 3.0) : 1;   // Lv3: 아주 약한 숨쉬기
  drawTint(lastEaten, p, axes, faceW, faceH, rgb, Math.min(byLevel(level, EFFECT_TINT_ALPHA) * pulse * env, 0.9));
  drawParticles(lastEaten, p, axes, faceW, faceH, age, level, rgb, env);
  return true;
}

// ---------------- 얼굴 morph (WebGL) ----------------
// 파이프라인: video → WebGL texture → fragment shader에서 bulge 중심 주변만 샘플 위치를 옮김 → #warp canvas →
// (그 위에) 기존 #overlay canvas(landmark / 물고기 / tint / particle) + HTML HUD.
// 좌표계: texture와 landmark는 모두 "영상 원본" 좌표(0~1, y는 위가 0)로 같고, 좌우 반전은 CSS(#video, #warp, #overlay 동일 scaleX(-1))로만
// 처리하므로 shader 안에서는 뒤집지 않는다. y도 shader에서 uv.y = 위가 0이 되도록 만들어 flip이 필요 없다.
const warpCanvas = document.getElementById("warp");
const WARP_VS = `
attribute vec2 aPos;
varying vec2 vUv;
void main() {
  gl_Position = vec4(aPos, 0.0, 1.0);
  vUv = vec2((aPos.x + 1.0) * 0.5, (1.0 - aPos.y) * 0.5);   // 위쪽이 v=0 → video 첫 행과 같은 방향
}`;
// Python bulge와 같은 식: u2 = (거리/반경)^2, scale = 1 - strength * (1 - u2)^2, 샘플 위치 = center + d * scale. 반경 밖은 그대로.
const WARP_FS = `
precision highp float;
uniform sampler2D uTex;
uniform vec2 uSize;        // 영상 크기(px)
uniform vec3 uSpots[2];    // (x, y, radius) px. radius 0이면 사용 안 함
uniform float uStrength;
uniform float uLift;       // 위로 들어올리는 비율 (반경 대비)
uniform vec2 uUp;          // 얼굴 위쪽 단위벡터 (영상 좌표, 기울기 반영)
varying vec2 vUv;
vec2 bulge(vec2 p, vec3 s) {
  if (s.z <= 0.0) return p;
  vec2 d = p - s.xy;
  float u2 = clamp(dot(d, d) / (s.z * s.z), 0.0, 1.0);
  float f = (1.0 - u2) * (1.0 - u2);
  float k = 1.0 - uStrength * f;
  return s.xy + d * k - uUp * (uLift * s.z * f);   // 샘플 위치를 아래로 옮기면 내용이 위로 올라가 보임
}
void main() {
  vec2 p = vUv * uSize;
  p = bulge(p, uSpots[1]);   // Python은 spot을 순서대로 프레임에 적용 → 합성하면 마지막 spot부터 역순으로 적용
  p = bulge(p, uSpots[0]);
  gl_FragColor = texture2D(uTex, p / uSize);
}`;
let warp = null;   // { gl, tex, uniforms, ... }. WebGL을 못 쓰면 null (tint/particle 효과만 동작)
let warpVisible = false;
let lastWarpUpload = -1;

function initWarp(w, h) {
  warpCanvas.width = w;
  warpCanvas.height = h;
  const gl = warpCanvas.getContext("webgl", { alpha: false, antialias: false });
  if (!gl) { console.warn("WebGL not available: face morph disabled"); return; }
  const compile = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  try {
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, WARP_VS));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, WARP_FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    gl.useProgram(prog);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "aPos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);   // 가장자리는 늘려서 채움 (Python의 BORDER_REPLICATE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.viewport(0, 0, w, h);
    warp = {
      gl, tex, w, h,
      uSize: gl.getUniformLocation(prog, "uSize"),
      uSpots: gl.getUniformLocation(prog, "uSpots"),
      uStrength: gl.getUniformLocation(prog, "uStrength"),
      uLift: gl.getUniformLocation(prog, "uLift"),
      uUp: gl.getUniformLocation(prog, "uUp"),
    };
  } catch (err) {
    console.warn("Face morph shader failed, disabled:", err);
    warp = null;
  }
}

// 물고기별 bulge 중심/반경 (Python warp_face와 동일, 픽셀 좌표). 반경은 얼굴 너비에 비례
function morphSpots(name, lm, w, h) {
  const p = (i) => ({ x: lm[i].x * w, y: lm[i].y * h });
  const l = p(FACE_LEFT), r = p(FACE_RIGHT);
  const faceW = Math.hypot(l.x - r.x, l.y - r.y);
  if (name === "Red Fish") {
    const mid = { x: (l.x + r.x) / 2, y: (l.y + r.y) / 2 };
    return [LEFT_CHEEK, RIGHT_CHEEK].map((i) => {
      const c = p(i);
      const ox = c.x - mid.x, oy = c.y - mid.y;
      const n = Math.hypot(ox, oy) || 1;   // 볼 landmark를 얼굴 중심선에서 먼 방향(바깥쪽)으로 살짝 이동
      return { x: c.x + (ox / n) * faceW * CHEEK_WARP_OUTWARD, y: c.y + (oy / n) * faceW * CHEEK_WARP_OUTWARD, r: faceW * CHEEK_WARP_RADIUS_RATIO };
    });
  }
  if (name === "Yellow Fish") {
    return IRIS_CENTERS.map((i) => ({ ...p(i), r: faceW * EYE_WARP_RADIUS_RATIO }));
  }
  const f = p(FOREHEAD), g = p(GLABELLA);   // Green: 이마(10)와 미간(168)의 중간
  return [{ x: (f.x + g.x) / 2, y: (f.y + g.y) / 2, r: faceW * FOREHEAD_WARP_RADIUS_RATIO }];
}

function morphStrength(name, level, amount) {
  return Math.min(byLevel(level, MORPH_STRENGTH[name]) * amount, MORPH_STRENGTH_MAX);
}

function morphLift(name, level, amount) {
  return byLevel(level, MORPH_LIFT[name]) * amount;
}

// 얼굴 위쪽 단위벡터 (이마 - 턱, 영상 좌표). 얼굴을 기울이면 같이 기울어짐
function faceUp(lm, w, h) {
  const x = (lm[FOREHEAD].x - lm[CHIN].x) * w, y = (lm[FOREHEAD].y - lm[CHIN].y) * h;
  const n = Math.hypot(x, y) || 1;
  return { x: x / n, y: y / n };
}

// 한 프레임 그리기. upload=true면 현재 video 프레임을 texture로 올림
function drawWarp(name, lm, level, amount, upload) {
  if (!warp) return false;
  const { gl } = warp;
  const spots = morphSpots(name, lm, warp.w, warp.h);
  const arr = new Float32Array(6);
  spots.slice(0, 2).forEach((s, i) => arr.set([s.x, s.y, s.r], i * 3));
  const w0 = perfNow();
  if (upload) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
  const w1 = perfNow();
  gl.uniform2f(warp.uSize, warp.w, warp.h);
  gl.uniform3fv(warp.uSpots, arr);
  gl.uniform1f(warp.uStrength, morphStrength(name, level, amount));
  gl.uniform1f(warp.uLift, morphLift(name, level, amount));
  const up = faceUp(lm, warp.w, warp.h);
  gl.uniform2f(warp.uUp, up.x, up.y);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  if (DEBUG_PERFORMANCE) {
    gl.finish();   // 측정 모드에서만: GPU 작업이 끝날 때까지 기다려 실제 시간을 잼 (평소에는 호출하지 않음)
    if (upload) perfAdd("upload", w1 - w0);
    perfAdd("morph", performance.now() - w1);
  }
  return true;
}

// 매 화면 갱신마다: 효과 중이고 얼굴이 보이면 warp canvas를 보이고 그림. 그 외(효과 끝, 얼굴 사라짐, 재시작)에는 즉시 숨김 → 원본 video가 보임
function updateWarp(nowSec) {
  const lm = lastFaceResult && lastFaceResult.faceLandmarks[0];
  const active = !!(warp && lastEaten && nowSec < effectUntil && lm && effectEnvelope(nowSec - (effectUntil - EFFECT_DURATION)) > 0);
  if (!active) {
    if (warpVisible) { warpCanvas.style.visibility = "hidden"; warpVisible = false; }
    return false;
  }
  const newFrame = video.currentTime !== lastWarpUpload || !warpVisible;
  lastWarpUpload = video.currentTime;
  const age = nowSec - (effectUntil - EFFECT_DURATION);
  drawWarp(lastEaten, lm, effectLevel(streak), effectEnvelope(age), newFrame);
  if (!warpVisible) { warpCanvas.style.visibility = "visible"; warpVisible = true; }
  return true;
}

function render(dt) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (DEBUG_LANDMARKS && lastHandResult && lastFaceResult) drawLandmarks(lastHandResult, lastFaceResult);
  const t = performance.now() / 1000;
  drawFaceEffect(t);
  for (const f of fishes) {
    if (f.state === "moving" || f.state === "held") {
      drawFish(f, t);
      const remain = f.flashUntil - t;
      if (remain > 0) {                   // Rare/Epic 등장: 물고기 색 물결이 한 번 퍼짐 (링/문구 없음)
        const u = 1 - remain / RARE_FLASH_DURATION;
        const k = canvas.height / REF_HEIGHT;
        ctx.save();
        ctx.globalAlpha = 0.4 * (1 - u);
        ctx.strokeStyle = f.color;
        ctx.lineWidth = Math.max(k, 1);
        ctx.beginPath();
        ctx.arc(f.x, f.y, f.size + (10 + u * 40) * k, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
    } else if (f.state === "hidden") {
      const u = (t - f.eatenAt) / EAT_POP_DURATION;   // 먹힌 직후: 살짝 커졌다가 작아지며 사라짐 (Python과 같은 곡선)
      if (u >= 0 && u < 1) drawFish(f, t, u < 0.3 ? 1 + (0.2 * u) / 0.3 : 1.2 * (1 - (u - 0.3) / 0.7), 1 - u);
    }
  }
  drawMarker(dt);
}

function drawLandmarks(handResult, faceResult) {
  for (const landmarks of faceResult.faceLandmarks) {
    drawing.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION,
      { color: "#ffffff22", lineWidth: 0.5 });
    drawing.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#6cf", lineWidth: 1.5 });
    drawing.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "#f77", lineWidth: 1.5 });
  }
  for (const landmarks of handResult.landmarks) {
    drawing.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: "#7f7", lineWidth: 3 });
    drawing.drawLandmarks(landmarks, { color: "#f55", radius: 3 });
  }
}

// 0.5초마다 상태 텍스트 갱신
function report() {
  const now = performance.now();
  if (now - lastReport < 500) return;
  const fps = (frames * 1000) / (now - lastReport);
  setStatus(`Camera on | FPS ${fps.toFixed(0)} | hands: ${counts.hands} | face: ${counts.face ? "detected" : "none"}`);
  frames = 0;
  lastReport = now;
}

// ---------------- 웹페이지 UI 화면 (메뉴 / How to Play / 로딩 / 게임) ----------------
const screenEls = {
  menu: document.getElementById("menuScreen"),
  howto: document.getElementById("howtoScreen"),
  loading: document.getElementById("loadingScreen"),
};
const stageEl = document.getElementById("stage");
const loadingMsg = document.getElementById("loadingMsg");
const loadingHint = document.getElementById("loadingHint");
const loadingActions = document.getElementById("loadingActions");
const loadingPulse = document.getElementById("loadingPulse");

// 화면 하나만 보이게 함. 보이는 화면의 주 버튼에 focus를 줘서 Enter/Space로도 바로 선택 가능
function setScreen(name) {
  uiScreen = name;
  document.body.dataset.screen = name;
  for (const [key, el] of Object.entries(screenEls)) el.hidden = key !== name;
  stageEl.hidden = name !== "game";
  const focusId = { menu: "playBtn", howto: "startGameBtn" }[name];
  if (focusId) document.getElementById(focusId).focus({ preventScroll: true });
}

function showLoading(msg) {
  loadingMsg.textContent = msg;
  loadingHint.hidden = true;
  loadingActions.hidden = true;
  screenEls.loading.classList.remove("error");
}

let lastError = null;
function showLoadingError(err) {
  lastError = err;
  loadingMsg.textContent = err.message;
  loadingHint.textContent = err.hint || "";
  loadingHint.hidden = !err.hint;
  loadingActions.hidden = false;
  screenEls.loading.classList.add("error");
  document.getElementById("retryBtn").focus({ preventScroll: true });
}

// 카메라 오류를 사용자가 이해할 수 있는 문장으로
function describeCameraError(err) {
  switch (err && err.name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
    case "SecurityError":
      return { message: "FACE FISH를 플레이하려면 카메라 권한이 필요합니다.", hint: "주소창의 카메라 아이콘에서 허용으로 바꾼 뒤 다시 시도해 주세요." };
    case "NotFoundError":
    case "OverconstrainedError":
      return { message: "사용할 수 있는 카메라를 찾지 못했어요.", hint: "웹캠이 연결되어 있는지 확인한 뒤 다시 시도해 주세요." };
    case "NotReadableError":
    case "AbortError":
      return { message: "카메라를 사용할 수 없어요.", hint: "다른 앱이나 탭이 카메라를 쓰고 있지 않은지 확인해 주세요." };
    default:
      return { message: "카메라를 시작하지 못했어요.", hint: `다시 시도해 주세요. (${(err && err.name) || "unknown"})` };
  }
}

function startModelLoad() {
  modelsReady = false;
  modelsPromise = loadModels().then(() => { modelsReady = true; setStatus("Models ready"); });
  modelsPromise.catch((err) => { console.error(err); setStatus(`Model load failed: ${err.message}`, true); });
}

// Play / Start Game / Try Again: 카메라 권한은 이 클릭 이후에 요청하고, 이미 켜져 있으면 다시 요청하지 않음
async function enterGame() {
  if (uiScreen === "loading" && loadingActions.hidden) return;   // 준비 중 중복 클릭 무시 (오류 화면의 Try Again은 허용)
  if (cameraReady) { setScreen("game"); return; }
  setScreen("loading");
  showLoading("카메라를 준비하고 있어요...");
  if (!window.isSecureContext) {
    showLoadingError({ message: "카메라를 쓰려면 보안 주소로 열어야 해요.", hint: "http://localhost 또는 https 주소로 접속해 주세요. (file:// 로는 사용할 수 없어요)" });
    return;
  }
  if (lastError && lastError.reloadModels) startModelLoad();   // 모델 로딩 실패 후 다시 시도하면 모델부터 다시
  lastError = null;
  const err = await startCamera();
  if (err) { showLoadingError(err); return; }
  cameraReady = true;
  setScreen("game");
}

function init() {
  statusEl.hidden = !(DEBUG_PERFORMANCE || DEBUG_LANDMARKS);   // 개발용 상태 줄은 디버그 옵션에서만
  startModelLoad();                                           // 메뉴를 보는 동안 모델을 미리 불러오고 워밍업
  document.getElementById("playBtn").addEventListener("click", enterGame);
  document.getElementById("startGameBtn").addEventListener("click", enterGame);
  document.getElementById("retryBtn").addEventListener("click", enterGame);
  startBtn.addEventListener("click", enterGame);              // 기존 START CAMERA 버튼(숨김)도 같은 흐름
  document.getElementById("howtoBtn").addEventListener("click", () => setScreen("howto"));
  document.getElementById("backBtn").addEventListener("click", () => setScreen("menu"));
  document.getElementById("loadingBackBtn").addEventListener("click", () => setScreen("menu"));
  setScreen("menu");
}

init();
