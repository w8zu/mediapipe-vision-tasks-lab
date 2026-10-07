// 메뉴 장면 효과 (게임 로직과 무관): 배경에서 천천히 떠다니는 물고기/거품, 물고기 버튼의 hover / click 반응.
// DOM 이벤트와 Canvas 2D만 사용. 화면이 게임(game)으로 바뀌면 애니메이션을 멈춰 게임 성능에 영향을 주지 않는다.

const COLORS = { coral: [240, 128, 120], gold: [232, 196, 110], mint: [140, 208, 170] };
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const ambient = document.getElementById("ambient");
const fxCanvas = document.getElementById("fx");
const actx = ambient.getContext("2d");
const fctx = fxCanvas.getContext("2d");
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

let W = 0, H = 0;
const pointer = { x: -999, y: -999 };

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
  W = window.innerWidth;
  H = window.innerHeight;
  for (const [cv, cx] of [[ambient, actx], [fxCanvas, fctx]]) {
    cv.width = Math.round(W * dpr);
    cv.height = Math.round(H * dpr);
    cx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
}

// ---------------- 배경: 천천히 떠다니는 물고기 + 거품 ----------------
// 크기가 작을수록 멀리 있는 것처럼 더 흐리고 느리게 (깊이감)
const FISH_SPECS = [
  { color: "coral", r: 34, speed: 22, alpha: 0.42, y: 0.22 },
  { color: "gold", r: 26, speed: 17, alpha: 0.34, y: 0.72 },
  { color: "mint", r: 30, speed: 19, alpha: 0.38, y: 0.86 },
  { color: "coral", r: 18, speed: 12, alpha: 0.24, y: 0.46 },
  { color: "gold", r: 20, speed: 13, alpha: 0.26, y: 0.1 },
];
let fishes = [];
let bubbles = [];

function initScene() {
  fishes = FISH_SPECS.map((s, i) => {
    const dir = i % 2 === 0 ? 1 : -1;
    return { ...s, x: W * (0.15 + 0.7 * ((i * 0.37) % 1)), baseY: H * s.y, y: H * s.y, cruise: dir * s.speed, vx: dir * s.speed, vy: 0, dir, phase: Math.random() * 6.28 };
  });
  bubbles = Array.from({ length: 16 }, () => newBubble(true));
}

function newBubble(anywhere) {
  return { x: Math.random() * W, y: anywhere ? Math.random() * H : H + 10, r: 1.5 + Math.random() * 3, vy: 8 + Math.random() * 14, sway: Math.random() * 6.28, a: 0.08 + Math.random() * 0.16 };
}

function drawFish(cx, f, t) {
  const r = f.r, wag = Math.sin(t * 5 + f.phase) * r * 0.18;
  cx.save();
  cx.translate(f.x, f.y);
  cx.scale(f.dir, 1);
  cx.globalAlpha = f.alpha;
  const col = COLORS[f.color];
  cx.fillStyle = rgba(col, 1);
  cx.shadowColor = rgba(col, 1);
  cx.shadowBlur = r * 0.8;
  cx.beginPath();                                   // 꼬리
  cx.moveTo(-0.7 * r, 0);
  cx.bezierCurveTo(-1.1 * r, -0.15 * r + wag * 0.5, -1.3 * r, -0.5 * r + wag, -1.55 * r, -0.5 * r + wag);
  cx.quadraticCurveTo(-1.3 * r, wag * 0.5, -1.55 * r, 0.5 * r + wag);
  cx.bezierCurveTo(-1.3 * r, 0.5 * r + wag, -1.1 * r, 0.15 * r + wag * 0.5, -0.7 * r, 0);
  cx.fill();
  cx.beginPath();                                   // 몸통
  cx.ellipse(0, 0, r, r * 0.55, 0, 0, Math.PI * 2);
  cx.fill();
  cx.shadowBlur = 0;
  cx.fillStyle = "#fff";
  cx.globalAlpha = f.alpha * 0.5;
  cx.beginPath();
  cx.ellipse(0.15 * r, -0.2 * r, 0.55 * r, 0.14 * r, 0, 0, Math.PI * 2);
  cx.fill();
  cx.restore();
}

function updateScene(dt) {
  for (const f of fishes) {
    // 마우스가 가까우면 살짝 비켜 헤엄침 (너무 놀라지 않게 부드럽게)
    const dx = f.x - pointer.x, dy = f.y - pointer.y, d = Math.hypot(dx, dy), range = 170 + f.r * 2;
    let pushX = 0, pushY = 0;
    if (d < range && d > 1) { const k = (1 - d / range) * 70; pushX = (dx / d) * k; pushY = (dy / d) * k; }
    f.vx += (f.cruise + pushX - f.vx) * Math.min(dt * 2, 1);
    f.vy += ((f.baseY + Math.sin(f.phase * 0.6) * 14 - f.y) * 0.8 + pushY - f.vy) * Math.min(dt * 2, 1);
    f.x += f.vx * dt;
    f.y += f.vy * dt;
    f.phase += dt * 1.2;
    const margin = f.r * 2;
    if (f.x < margin) { f.x = margin; f.cruise = Math.abs(f.cruise); }
    else if (f.x > W - margin) { f.x = W - margin; f.cruise = -Math.abs(f.cruise); }
    if (Math.abs(f.vx) > 6) f.dir = f.vx > 0 ? 1 : -1;   // 이동 방향으로 머리를 향함
  }
  for (const b of bubbles) {
    b.y -= b.vy * dt;
    b.sway += dt;
    if (b.y < -10) Object.assign(b, newBubble(false));
  }
}

function drawScene(t) {
  actx.clearRect(0, 0, W, H);
  for (const b of bubbles) {
    actx.beginPath();
    actx.arc(b.x + Math.sin(b.sway) * 6, b.y, b.r, 0, Math.PI * 2);
    actx.strokeStyle = `rgba(225,245,250,${b.a})`;
    actx.lineWidth = 1;
    actx.stroke();
  }
  for (const f of fishes) drawFish(actx, f, t);
}

// ---------------- 버튼 hover / click 효과 (물결 + 작은 거품) ----------------
const ripples = [];
const sparks = [];

function addRipple(x, y, colorName, big = 1) {
  ripples.push({ x, y, t: 0, dur: 0.7, max: 46 * big, c: COLORS[colorName] || COLORS.coral });
}
function addBubbles(x, y, colorName, n, spread, burst) {
  const c = COLORS[colorName] || COLORS.coral;
  for (let i = 0; i < n; i++) {
    const a = burst ? Math.random() * Math.PI * 2 : -Math.PI / 2 + (Math.random() - 0.5) * 1.2;
    const sp = burst ? 40 + Math.random() * 70 : 14 + Math.random() * 22;
    sparks.push({ x: x + (Math.random() - 0.5) * spread, y: y + (Math.random() - 0.5) * spread * 0.4, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp - (burst ? 0 : 6), r: 1.5 + Math.random() * 2.5, t: 0, dur: 0.7 + Math.random() * 0.6, c });
  }
}

function updateFx(dt) {
  for (let i = ripples.length - 1; i >= 0; i--) { ripples[i].t += dt; if (ripples[i].t >= ripples[i].dur) ripples.splice(i, 1); }
  for (let i = sparks.length - 1; i >= 0; i--) {
    const s = sparks[i];
    s.t += dt; s.x += s.vx * dt; s.y += s.vy * dt; s.vx *= 1 - Math.min(dt * 2.2, 1); s.vy = s.vy * (1 - Math.min(dt * 1.5, 1)) - 8 * dt;
    if (s.t >= s.dur) sparks.splice(i, 1);
  }
}
function drawFx() {
  fctx.clearRect(0, 0, W, H);
  for (const r of ripples) {
    const u = r.t / r.dur;
    fctx.beginPath();
    fctx.arc(r.x, r.y, r.max * (0.25 + 0.75 * u), 0, Math.PI * 2);
    fctx.strokeStyle = rgba(r.c, 0.5 * (1 - u));
    fctx.lineWidth = 1.5;
    fctx.stroke();
  }
  for (const s of sparks) {
    const u = s.t / s.dur;
    fctx.beginPath();
    fctx.arc(s.x, s.y, s.r * (1 - 0.3 * u), 0, Math.PI * 2);
    fctx.fillStyle = rgba(s.c, 0.6 * (1 - u));
    fctx.fill();
    fctx.strokeStyle = `rgba(225,245,250,${0.5 * (1 - u)})`;
    fctx.lineWidth = 0.8;
    fctx.stroke();
  }
}

// ---------------- 루프: 메뉴 계열 화면에서만 돌고, 게임 화면이거나 탭이 숨겨지면 멈춤 ----------------
let running = false, last = 0;
const inMenu = () => document.body.dataset.screen !== "game" && !document.hidden;

function frame(now) {
  if (!inMenu()) { running = false; return; }
  const dt = Math.min((now - last) / 1000, 0.05);
  last = now;
  if (!reduceMotion) updateScene(dt);
  updateFx(dt);
  drawScene(now / 1000);
  drawFx();
  requestAnimationFrame(frame);
}
function ensureRunning() {
  if (running || !inMenu()) return;
  running = true;
  last = performance.now();
  requestAnimationFrame(frame);
}

new MutationObserver(ensureRunning).observe(document.body, { attributes: true, attributeFilter: ["data-screen"] });
document.addEventListener("visibilitychange", ensureRunning);
window.addEventListener("resize", () => { resize(); initScene(); });
window.addEventListener("pointermove", (e) => { pointer.x = e.clientX; pointer.y = e.clientY; }, { passive: true });
window.addEventListener("pointerleave", () => { pointer.x = pointer.y = -999; });

// 버튼(물고기) 반응: 가까이 가면 작은 거품, 누르면 꾹 눌렸다 튀어오르며 물결 + 거품이 퍼짐
function bind(btn) {
  const color = btn.dataset.fx || "coral";
  btn.addEventListener("pointerenter", (e) => {
    if (e.pointerType === "touch") return;
    const r = btn.getBoundingClientRect();
    addBubbles(r.left + r.width * 0.55, r.top + r.height * 0.3, color, 4, r.width * 0.5, false);
    ensureRunning();
  });
  btn.addEventListener("pointerdown", (e) => {
    squish(btn);
    addRipple(e.clientX, e.clientY, color);
    addBubbles(e.clientX, e.clientY, color, 9, 6, true);
    ensureRunning();
  });
  btn.addEventListener("click", (e) => {
    if (e.detail !== 0) return;                  // 마우스 클릭은 pointerdown에서 이미 처리. 키보드(Enter/Space)로 눌렀을 때만 중앙에서
    const r = btn.getBoundingClientRect();
    squish(btn);
    addRipple(r.left + r.width * 0.55, r.top + r.height * 0.5, color);
    addBubbles(r.left + r.width * 0.55, r.top + r.height * 0.5, color, 9, 6, true);
    ensureRunning();
  });
}
function squish(btn) {
  btn.classList.remove("pressed");
  void btn.offsetWidth;                          // 애니메이션 재시작
  btn.classList.add("pressed");
  setTimeout(() => btn.classList.remove("pressed"), 360);
}
document.querySelectorAll(".fishbtn").forEach(bind);
// 빈 배경을 눌러도 작은 물결 하나
window.addEventListener("pointerdown", (e) => {
  if (e.target.closest && e.target.closest(".fishbtn")) return;
  addRipple(e.clientX, e.clientY, "mint", 0.7);
  ensureRunning();
});

resize();
initScene();
ensureRunning();
