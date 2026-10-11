# -*- coding: utf-8 -*-
"""Gesture Beats — 웹캠 제스처 오디오/영상 믹서.

학습은 GestureBeats.ipynb (YOLOv8n, HaGRID 10클래스) 에서 끝낸 뒤 이 파일을 실행한다.

    python demo.py
    python demo.py --port 8080

브라우저에서 http://127.0.0.1:8080 을 연다.
file:// 로 열면 웹캠·ONNX가 막히므로 반드시 이 서버를 쓴다.

구성 (이 파일 하나):
  1) 아래 INDEX_HTML : 페이지 + DSP + YOLO(ONNX) + 제스처 상태머신
  2) Handler / main : onnx·wav 를 서빙하는 로컬 HTTP

읽는 순서:
  1) main() / Handler
     파이썬이 웹 서버를 열고 HTML·ONNX·샘플 음원을 브라우저에 보낸다.
  2) startMixer() / loop()
     시작 버튼을 누른 뒤 웹캠·모델을 준비하고, 매 화면마다 추론을 반복한다.
  3) createYolo()
     웹캠 이미지를 320x320 숫자 배열로 바꾸고 ONNX 모델의 결과를 박스로 복원한다.
  4) createGestureMachine()
     YOLO가 찾은 클래스와 손 위치를 "재생", "볼륨 변화" 같은 이벤트로 바꾼다.
  5) applyEvents()
     제스처 이벤트를 실제 플레이어와 오디오 필터에 적용한다.
  6) createDsp() / createPlayer() / createUi()
     소리를 가공하고, 파일을 재생하고, 화면의 숫자와 박스를 갱신한다.

제스처 매핑:
  fist → palm     재생/일시정지 (엣지 1회)
  ok + 높이/좌우  볼륨 / 시크 (잡은 점이 원점, 밀어두면 계속. 시크 끝=다음 파일)
  like + 좌우     DJ 로우패스 (동일 조이스틱)
  dislike         다음 파일 (한 번)
  one / peace / three + 높이   EQ Low / Mid / High
"""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

INDEX_HTML = r'''<!DOCTYPE html>
<html lang="ko">
<head>
  <!-- 브라우저가 한글과 모바일 화면 크기를 올바르게 처리하도록 하는 기본 설정 -->
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Gesture Control</title>
  <!-- 화면 글꼴. 모델 동작과 관계없는 디자인 요소이다. -->
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Sora:wght@500;700&display=swap" rel="stylesheet" />
  <!-- Python에서 만든 best.onnx를 브라우저에서 실행해 주는 라이브러리 -->
  <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.min.js"></script>
  <style>
/* ===== 0. 화면 디자인(CSS) =====
   CSS는 모델을 학습하거나 추론하지 않는다.
   HTML 요소의 색, 크기, 위치만 정한다. */
:root {
  /* 여러 곳에서 반복하는 색을 변수로 선언해 한 번에 바꿀 수 있게 했다. */
  --bg: #100e0c;
  --panel: #1a1612;
  --line: #3a3126;
  --ink: #efe6d6;
  --dim: #9a8b74;
  --amber: #e2a34a;
  --mint: #7ee0c6;
  --warn: #e0b36a;
  --err: #e07a5f;
  --ok: #8fce7a;
}

* { box-sizing: border-box; }

html, body {
  margin: 0;
  height: 100%;
  min-height: 100%;
  background:
    radial-gradient(1200px 500px at 10% -10%, #2a2116 0%, transparent 55%),
    var(--bg);
  color: var(--ink);
  font-family: Sora, "Segoe UI", sans-serif;
}

.mono { font-family: "IBM Plex Mono", ui-monospace, monospace; }

.app {
  max-width: 1280px;
  margin: 0 auto;
  padding: 20px 22px 32px;
}

.app.cinema {
  /* 영상 파일을 고르면 붙는 클래스.
     영상은 크게, 웹캠과 조작 패널은 오른쪽에 배치한다. */
  max-width: none;
  width: 100%;
  height: 100vh;
  max-height: 100vh;
  overflow: hidden;
  padding: 8px 10px 10px;
  display: flex;
  flex-direction: column;
}

.app.cinema .top {
  margin-bottom: 8px;
  flex-shrink: 0;
  align-items: center;
}
.app.cinema h1 { font-size: 18px; }
.app.cinema .eyebrow { margin: 0; }
.app.cinema .status { max-width: 280px; font-size: 12px; }

.top {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-end;
  margin-bottom: 18px;
}

.eyebrow {
  margin: 0 0 4px;
  color: var(--amber);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  font-size: 11px;
}

h1 {
  margin: 0;
  font-size: 28px;
  letter-spacing: -0.03em;
}

.top-actions {
  display: flex;
  gap: 12px;
  align-items: center;
}

.status {
  max-width: 420px;
  color: var(--dim);
  font-size: 13px;
  text-align: right;
}

.status[data-kind="ok"] { color: var(--ok); }
.status[data-kind="warn"] { color: var(--warn); }
.status[data-kind="err"] { color: var(--err); }
.status[data-kind="busy"] { color: var(--mint); }

button, .file-btn {
  border: 1px solid var(--amber);
  background: #2a2014;
  color: var(--amber);
  padding: 10px 14px;
  font: 600 13px Sora, sans-serif;
  cursor: pointer;
}

button:hover, .file-btn:hover { background: #3a2b18; }
button:disabled { opacity: 0.45; cursor: default; }

.grid {
  display: grid;
  grid-template-columns: minmax(320px, 1.1fr) minmax(320px, 0.9fr);
  gap: 16px;
}

.app.cinema .grid {
  flex: 1;
  min-height: 0;
  grid-template-columns: minmax(0, 1fr) 280px;
  grid-template-rows: 150px minmax(0, 1fr);
  grid-template-areas:
    "stage cam"
    "stage rack";
  gap: 10px;
}

.stage {
  display: none;
  position: relative;
  background: #000;
  border: 1px solid var(--line);
  overflow: hidden;
}

.app.cinema .stage {
  display: block;
  grid-area: stage;
  min-height: 0;
  height: auto;
}

.cam-wrap {
  position: relative;
  min-height: 420px;
  background: #000;
  border: 1px solid var(--line);
  overflow: hidden;
}

.app.cinema .cam-wrap {
  grid-area: cam;
  min-height: 0;
  height: 100%;
  width: auto;
  border-color: var(--amber);
}

#cam, #overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  transform: scaleX(-1);
}

.mode-osd {
  position: absolute;
  left: 12px;
  bottom: 12px;
  z-index: 2;
  padding: 6px 10px;
  background: #100e0ccc;
  border: 1px solid var(--line);
  font-family: "IBM Plex Mono", monospace;
  font-size: 12px;
  color: var(--mint);
}

.rack {
  background: var(--panel);
  border: 1px solid var(--line);
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.player {
  display: grid;
  grid-template-columns: auto auto auto 1fr auto;
  gap: 8px;
  align-items: center;
}

#seek { width: 100%; accent-color: var(--amber); }

.now-playing {
  margin: 0;
  font-size: 11px;
  color: var(--dim);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-btn input { display: none; }

.queue-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--dim);
}

.queue-head .queue-actions {
  display: flex;
  gap: 6px;
  align-items: center;
}

.queue-head .file-btn,
.queue-head button {
  padding: 6px 10px;
  font-size: 11px;
}

.queue {
  list-style: none;
  margin: 0;
  padding: 0;
  max-height: 160px;
  overflow-y: auto;
  border: 1px solid var(--line);
  background: #141210;
}

.app.cinema .queue { max-height: 120px; }

.queue li {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 8px;
  align-items: center;
  padding: 7px 8px;
  cursor: pointer;
  font-size: 12px;
  color: var(--dim);
  border-bottom: 1px solid var(--line);
}

.queue li:last-child { border-bottom: 0; }
.queue li.empty { cursor: default; color: #6f6454; }
.queue li.active {
  color: var(--ink);
  background: #2a2116;
  box-shadow: inset 2px 0 0 var(--amber);
}

.queue .kind {
  display: inline-block;
  margin-right: 6px;
  color: var(--amber);
  font-size: 10px;
}

.queue li button {
  padding: 2px 8px;
  font-size: 11px;
}

.media-wrap {
  display: none;
  position: absolute;
  inset: 0;
  background: #000;
}

.media-wrap.visible { display: block; }

#track {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #000;
}

.app.cinema .viz,
.app.cinema .viz-wave,
.app.cinema .help,
.app.cinema .debug { display: none; }

.app.cinema .rack {
  grid-area: rack;
  position: relative;
  top: auto;
  right: auto;
  bottom: auto;
  width: auto;
  min-height: 0;
  overflow-y: auto;
  padding: 10px;
  gap: 8px;
}

.app.cinema .meters li { margin-bottom: 4px; }

.meters { list-style: none; margin: 0; padding: 0; }
.meters li {
  display: grid;
  grid-template-columns: 44px 1fr 88px;
  gap: 8px;
  align-items: center;
  margin-bottom: 6px;
  font-size: 12px;
  color: var(--dim);
}

.bar {
  height: 8px;
  background: #2a241c;
  border: 1px solid var(--line);
}

.bar i {
  display: block;
  height: 100%;
  width: 50%;
  background: linear-gradient(90deg, #c47a2a, var(--amber));
}

.meters b { text-align: right; color: var(--ink); font-weight: 500; }

.viz {
  width: 100%;
  height: 88px;
  background: #141210;
  border: 1px solid var(--line);
}

.viz-wave { height: 64px; }

.guide {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
}

.guide li {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 8px;
  border: 1px solid var(--line);
  color: var(--dim);
  font-size: 12px;
}

.guide li.active {
  border-color: var(--mint);
  color: var(--ink);
}

.guide kbd {
  font-family: "IBM Plex Mono", monospace;
  color: var(--amber);
  font-size: 11px;
}

.keys, .debug {
  margin: 0;
  color: var(--dim);
  font-size: 11px;
  white-space: pre-wrap;
}

.debug {
  background: #141210;
  border: 1px solid var(--line);
  padding: 8px;
}

@media (max-width: 900px) {
  .grid { grid-template-columns: 1fr; }
  .top { flex-direction: column; align-items: flex-start; }
  .status { text-align: left; }
  .cam-wrap { min-height: 320px; }
  .app.cinema {
    height: auto;
    max-height: none;
    overflow: auto;
  }
  .app.cinema .grid {
    grid-template-columns: 1fr;
    grid-template-rows: minmax(42vh, 56vh) 140px auto;
    grid-template-areas:
      "stage"
      "cam"
      "rack";
  }
  .app.cinema .cam-wrap { height: 140px; }
}

  </style>
</head>
<body>
  <div id="app" class="app">
    <header class="top">
      <div>
        <p class="eyebrow">HaGRID 10-class · YOLOv8n ONNX</p>
        <h1>Gesture Control</h1>
      </div>
      <div class="top-actions">
        <span id="model-status" class="status">Start Mixer 를 누르세요.</span>
        <button id="btn-start" type="button">Start Mixer</button>
      </div>
    </header>

    <main class="grid">
      <!-- cam: 제스처 인식용 웹캠. overlay 캔버스가 검출 박스를 위에 그린다. -->
      <div class="cam-wrap" id="cam-wrap">
        <video id="cam" playsinline muted></video>
        <canvas id="overlay"></canvas>
        <div id="mode-osd" class="mode-osd">IDLE</div>
      </div>
      <!-- stage: 사용자가 올린 영상이 크게 보이는 영역 -->
      <section class="stage" id="stage">
        <div id="media-wrap" class="media-wrap">
          <video id="track" playsinline src="/assets/audio/demo.wav"></video>
        </div>
      </section>

      <section class="rack">
        <!-- 실제 미디어 재생 버튼·시간 막대 -->
        <p id="now-playing" class="now-playing mono">demo.wav</p>
        <div class="player">
          <button id="btn-play" type="button">Play</button>
          <button id="btn-prev" type="button" title="이전 곡">◀</button>
          <button id="btn-next" type="button" title="다음 곡">▶</button>
          <input id="seek" type="range" min="0" max="1000" value="0" />
          <span id="time" class="mono">0:00 / 0:00</span>
        </div>

        <div class="queue-panel">
          <!-- 여러 오디오/영상 파일을 브라우저 메모리에만 담는 재생목록 -->
          <div class="queue-head">
            <span>재생목록 <b id="queue-count" class="mono">0</b></span>
            <div class="queue-actions">
              <label class="file-btn">추가<input id="file" type="file" accept="audio/*,video/*" multiple /></label>
              <button id="btn-clear-queue" type="button">비우기</button>
            </div>
          </div>
          <ul id="queue" class="queue"><li class="empty">파일을 추가하세요</li></ul>
        </div>

        <ul class="meters">
          <!-- DSP 값을 숫자와 막대로 보여 주는 부분 -->
          <li>
            <span>VOL</span>
            <div class="bar"><i id="vol-bar"></i></div>
            <b id="vol-val" class="mono">0.70</b>
          </li>
          <li>
            <span>LOW</span>
            <div class="bar"><i id="low-bar"></i></div>
            <b id="low-val" class="mono">0.0 dB</b>
          </li>
          <li>
            <span>MID</span>
            <div class="bar"><i id="mid-bar"></i></div>
            <b id="mid-val" class="mono">0.0 dB</b>
          </li>
          <li>
            <span>HIGH</span>
            <div class="bar"><i id="high-bar"></i></div>
            <b id="high-val" class="mono">0.0 dB</b>
          </li>
          <li>
            <span>CUT</span>
            <div class="bar"><i id="cut-bar"></i></div>
            <b id="cut-val" class="mono">16000 Hz</b>
          </li>
        </ul>

        <canvas id="viz-freq" class="viz"></canvas>
        <canvas id="viz-wave" class="viz viz-wave"></canvas>

        <div class="help">
          <ul id="guide" class="guide"></ul>
          <p class="keys mono">
            Space 재생 · ←→ 시크 · Shift+←→ 이전/다음 · ok 높이 볼륨·좌우 시크 · like 로우패스 · dislike 다음 · ↑↓ 볼륨 · 1/2/3 후 ↑↓ EQ · [ ] 컷오프
          </p>
        </div>
        <pre id="debug" class="debug mono">class=-  conf=0.00  mode=IDLE  gain=0.70  cutoff=16000</pre>
      </section>
    </main>
  </div>
  <script>
/* ===== 1. Web Audio DSP =====
   <video id="track"> 하나를 MediaElementSource 로 묶는다. (오디오/영상 공통)
   그래프: source → EQ Low/Mid/High → lowpass(CUT) → master gain → 스피커+analyser
   setTargetAtTime 으로 값을 부드럽게 바꿔 클릭 노이즈를 줄인다. */
function createDsp(audioEl) {
  /* AudioContext는 브라우저의 "소리 작업 공간"이다.
     audioEl은 화면의 <video id="track">이며, 오디오 파일도 이 태그로 재생한다. */
  const ctx = new AudioContext();

  // HTML 미디어 태그의 소리를 Web Audio 그래프로 가져오는 시작 노드
  const src = ctx.createMediaElementSource(audioEl);

  // peaking 필터는 특정 주파수 주변만 키우거나 줄인다.
  // Low: 저음(100Hz), Mid: 중음(1kHz), High: 고음(8kHz)
  const eqLow = ctx.createBiquadFilter();
  eqLow.type = "peaking";
  eqLow.frequency.value = 100;
  eqLow.Q.value = 1;
  eqLow.gain.value = 0;

  const eqMid = ctx.createBiquadFilter();
  eqMid.type = "peaking";
  eqMid.frequency.value = 1000;
  eqMid.Q.value = 1;
  eqMid.gain.value = 0;

  const eqHigh = ctx.createBiquadFilter();
  eqHigh.type = "peaking";
  eqHigh.frequency.value = 8000;
  eqHigh.Q.value = 1;
  eqHigh.gain.value = 0;

  // lowpass는 cutoff보다 높은 주파수를 깎는다.
  // cutoff가 낮아질수록 소리가 먹먹해져 DJ 필터처럼 들린다.
  const lowpass = ctx.createBiquadFilter();
  lowpass.type = "lowpass";
  lowpass.frequency.value = 16000;
  lowpass.Q.value = 0.707;

  // GainNode는 전체 소리 크기를 곱한다. 0은 무음, 1은 원래 크기다.
  const master = ctx.createGain();
  master.gain.value = 0.7;

  // AnalyserNode는 소리를 바꾸지 않고 주파수·파형 숫자만 읽게 해 준다.
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0.75;

  // connect 순서가 실제 신호 흐름이다.
  // 원음 → 저/중/고 EQ → 로우패스 → 전체 볼륨
  src
    .connect(eqLow)
    .connect(eqMid)
    .connect(eqHigh)
    .connect(lowpass)
    .connect(master);
  // analyser는 시각화용, destination은 실제 스피커 출력이다.
  master.connect(analyser);
  master.connect(ctx.destination);

  // Web Audio 노드 값은 직접 읽기 불편하므로 화면 표시용 값을 따로 보관한다.
  const state = {
    volume: 0.7,
    eq: { low: 0, mid: 0, high: 0 },
    cutoff: 16000,
  };

  const TAU = 0.06;

  function setTarget(param, value, tau = TAU) {
    // 값을 즉시 바꾸면 "딱" 하는 잡음이 날 수 있어 약 0.06초에 걸쳐 부드럽게 이동한다.
    param.setTargetAtTime(value, ctx.currentTime, tau);
  }

  const eqNodes = { low: eqLow, mid: eqMid, high: eqHigh };

  return {
    ctx,
    analyser,
    state,

    async resume() {
      // 브라우저는 사용자가 버튼을 누르기 전 자동 소리를 막는다.
      // Start Mixer 클릭 후 suspended 상태를 running으로 바꾼다.
      if (ctx.state === "suspended") await ctx.resume();
    },

    setVolume(v) {
      // 범위를 0.05~1로 제한한다. 완전 0은 데모에서 고장처럼 느껴져 피했다.
      const next = Math.min(1, Math.max(0.05, v));
      state.volume = next;
      setTarget(master.gain, next);
    },

    setEq(band, db) {
      // EQ는 -15dB~+15dB. band는 low/mid/high 중 하나다.
      const next = Math.min(15, Math.max(-15, db));
      state.eq[band] = next;
      setTarget(eqNodes[band].gain, next);
    },

    setCutoff(hz) {
      // 사람이 들을 수 있는 범위 안에서 200~16000Hz로 제한한다.
      const next = Math.min(16000, Math.max(200, hz));
      state.cutoff = next;
      setTarget(lowpass.frequency, next, 0.05);
    },
  };
}

function xToCutoff(nx) {
  // 선형 0~1 → 로그 Hz. 사람 귀가 주파수에 로그로 반응해서 CUT 도 로그 스케일
  const t = Math.min(1, Math.max(0, nx));
  const logMin = Math.log(200);
  const logMax = Math.log(16000);
  return Math.exp(logMin + t * (logMax - logMin));
}

function isVideoFile(file) {
  // 먼저 MIME type을 보고, 브라우저가 type을 못 준 경우 파일 확장자를 다시 본다.
  if (file && file.type && file.type.indexOf("video/") === 0) return true;
  return /\.(mp4|webm|mov|mkv|ogv)$/i.test((file && file.name) || "");
}

function setCinema(on) {
  // 영상일 때만 재생 화면을 거의 전체로, 웹캠은 왼쪽 아래 PIP
  const app = document.getElementById("app");
  const wrap = document.getElementById("media-wrap");
  if (app) app.classList.toggle("cinema", Boolean(on));
  if (wrap) wrap.classList.toggle("visible", Boolean(on));
}

function escapeHtml(s) {
  // 파일명에 < 또는 >가 들어 있어도 HTML 코드로 실행되지 않게 문자로 바꾼다.
  // 보안상 필요한 처리이며, 재생목록을 innerHTML로 만들기 때문에 꼭 필요하다.
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[c]));
}

function createPlayer(mediaEl) {
  // 재생목록 + <video id="track"> 제어. DSP 소스는 이 엘리먼트 하나라 src 만 갈아끼운다.
  // 함수 안의 items와 index를 밖에서 직접 바꾸지 못하게 하고,
  // 아래 return 객체의 메서드로만 조작하는 간단한 모듈 구조다.
  mediaEl.preload = "auto";
  // preload=auto: 가능하면 재생 전에 파일 정보를 미리 읽는다.
  mediaEl.crossOrigin = "anonymous";

  // items: 재생목록, index: 현재 재생 중인 항목 번호(0부터 시작)
  const items = [];
  let index = -1;
  let seq = 0;
  let onQueueChange = null;
  let pendingSeek = null;

  mediaEl.addEventListener("seeked", () => {
    // 영상이 이동 중(seeking)일 때 또 이동 요청이 오면 pendingSeek에 마지막 값만 저장한다.
    // 이전 이동이 끝난 뒤 마지막 요청을 적용해 브라우저가 과부하되는 것을 줄인다.
    if (pendingSeek == null) return;
    const t = pendingSeek;
    pendingSeek = null;
    if (!Number.isFinite(mediaEl.duration) || mediaEl.duration <= 0) return;
    const next = Math.min(mediaEl.duration, Math.max(0, t));
    if (Math.abs(mediaEl.currentTime - next) > 0.08) mediaEl.currentTime = next;
  });

  function emit() {
    // 목록이 바뀌었다고 UI에 알린다.
    // slice()는 원본 배열을 직접 넘기지 않고 복사본을 넘기는 안전장치다.
    if (onQueueChange) onQueueChange(items.slice(), index);
  }

  function applyItem(item) {
    // 선택한 항목의 실제 URL을 mediaEl.src에 넣는다.
    const video = Boolean(item && item.video);
    // 오디오가 하나뿐이면 반복 재생, 영상이나 여러 파일이면 다음 항목으로 넘긴다.
    mediaEl.loop = items.length <= 1 && !video;
    setCinema(video);
    mediaEl.src = item.url;
    mediaEl.load();
  }

  async function playAt(i, autoplay) {
    // i가 배열 범위를 벗어나면 아무것도 하지 않아 오류를 막는다.
    if (i < 0 || i >= items.length) return;
    index = i;
    applyItem(items[i]);
    emit();
    if (autoplay) {
      try {
        await mediaEl.play();
      } catch {
        /* 제스처 없이 로드만 한 경우 브라우저가 play를 막을 수 있음 */
      }
    }
  }

  mediaEl.addEventListener("ended", () => {
    // 현재 파일이 끝났고 목록이 2개 이상이면 다음 파일을 자동 재생한다.
    // 마지막 파일 다음에는 나머지 연산자(%)로 0번 항목으로 돌아간다.
    if (items.length > 1) {
      playAt((index + 1) % items.length, true);
    }
  });

  return {
    el: mediaEl,

    setOnQueueChange(fn) {
      // createUi의 renderQueue 함수를 콜백으로 받아 목록 변경 때 호출한다.
      onQueueChange = fn;
      emit();
    },

    hasCustomFile() {
      // 사용자가 파일을 하나라도 추가했는지 검사한다.
      return items.length > 0;
    },

    canSkipNext() {
      // 시크를 끝까지 밀었을 때 다음 파일로 갈 수 있는지 검사한다.
      return items.length > 1;
    },

    currentName() {
      if (index >= 0 && items[index]) return items[index].name;
      return "demo.wav";
    },

    async unlock() {
      // 음소거로 아주 짧게 play/pause하여 브라우저의 자동재생 잠금을 푼다.
      // 사용자가 Start Mixer를 직접 클릭한 시점에 호출하므로 정책상 허용된다.
      const wasMuted = mediaEl.muted;
      mediaEl.muted = true;
      try {
        await mediaEl.play();
        mediaEl.pause();
        mediaEl.currentTime = 0;
      } catch {
        /* 파일 없으면 여기 실패해도 Start Mixer 클릭 자체로 컨텍스트는 풀림 */
      }
      mediaEl.muted = wasMuted;
    },

    loadUrl(url) {
      // 기본 demo.wav로 돌아갈 때 사용한다.
      // createObjectURL로 만든 주소는 revoke해야 브라우저 메모리가 해제된다.
      items.forEach((it) => URL.revokeObjectURL(it.url));
      items.length = 0;
      index = -1;
      mediaEl.loop = true;
      setCinema(false);
      mediaEl.src = url;
      mediaEl.load();
      emit();
    },

    addFiles(fileList) {
      // <input multiple>의 FileList는 배열이 아니므로 Array.from으로 바꾼다.
      const files = Array.from(fileList || []).filter(Boolean);
      if (!files.length) return;
      const startEmpty = items.length === 0;
      for (const file of files) {
        // 실제 파일을 서버에 업로드하지 않는다.
        // createObjectURL은 브라우저 메모리의 파일을 가리키는 임시 주소다.
        items.push({
          id: ++seq,
          name: file.name,
          video: isVideoFile(file),
          url: URL.createObjectURL(file),
        });
      }
      emit();
      if (startEmpty) playAt(0, false);
    },

    playAt(i, autoplay) {
      return playAt(i, autoplay);
    },

    next(autoplay) {
      if (!items.length) return;
      // 마지막 항목이면 % 연산으로 첫 항목(0)으로 순환한다.
      return playAt((index + 1) % items.length, autoplay !== false);
    },

    prev(autoplay) {
      if (!items.length) return;
      // 첫 항목에서 이전을 누르면 마지막 항목으로 간다.
      const i = index <= 0 ? items.length - 1 : index - 1;
      return playAt(i, autoplay !== false);
    },

    removeAt(i) {
      // 삭제하는 항목이 현재 곡인지, 현재 곡보다 앞인지에 따라 index를 보정한다.
      if (i < 0 || i >= items.length) return;
      const wasCurrent = i === index;
      URL.revokeObjectURL(items[i].url);
      items.splice(i, 1);
      if (!items.length) {
        index = -1;
        this.loadUrl("./assets/audio/demo.wav");
        return;
      }
      if (i < index) index -= 1;
      else if (wasCurrent) {
        if (index >= items.length) index = items.length - 1;
        playAt(index, !mediaEl.paused);
        return;
      }
      emit();
    },

    clear() {
      this.loadUrl("./assets/audio/demo.wav");
    },

    async play() {
      await mediaEl.play();
    },

    pause() {
      mediaEl.pause();
    },

    async toggle() {
      // paused는 브라우저 미디어 태그가 제공하는 현재 정지 여부다.
      if (mediaEl.paused) await this.play();
      else this.pause();
    },

    isPlaying() {
      return !mediaEl.paused && !mediaEl.ended;
    },

    seek(ratio) {
      // 화면 슬라이더는 0~1 비율, currentTime은 초 단위라 duration을 곱한다.
      if (!Number.isFinite(mediaEl.duration) || mediaEl.duration <= 0) return;
      this.seekTo(Math.min(1, Math.max(0, ratio)) * mediaEl.duration);
    },

    nudge(seconds) {
      this.seekTo(mediaEl.currentTime + seconds);
    },

    seekTo(t) {
      // currentTime 을 매 프레임 바꾸면 브라우저가 시크를 버린다. 진행 중이면 pending 에 쌓음
      if (!Number.isFinite(mediaEl.duration) || mediaEl.duration <= 0) return;
      const next = Math.min(mediaEl.duration, Math.max(0, t));
      if (mediaEl.seeking) {
        pendingSeek = next;
        return;
      }
      if (Math.abs(mediaEl.currentTime - next) < 0.04) return;
      mediaEl.currentTime = next;
    },
  };
}

function formatTime(sec) {
  // 초 단위 숫자를 화면에 보기 좋은 "분:초" 문자열로 바꾼다.
  if (!Number.isFinite(sec)) return "0:00";
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${String(r).padStart(2, "0")}`;
}

async function startCamera(videoEl) {
  // getUserMedia는 사용자의 실제 웹캠 권한을 요청한다.
  // audio:false이므로 마이크 음성은 수집하지 않는다.
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: { ideal: 640 },
      height: { ideal: 480 },
    },
  });
  videoEl.srcObject = stream;
  videoEl.playsInline = true;
  videoEl.muted = true;
  await videoEl.play();
  return stream;
}

function stopCamera(videoEl) {
  // 페이지를 닫거나 카메라를 멈출 때 모든 MediaStreamTrack을 종료해야
  // 브라우저의 카메라 사용 표시등도 꺼진다.
  const stream = videoEl.srcObject;
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
  }
  videoEl.srcObject = null;
}

function createVisualizer(analyser, freqCanvas, waveCanvas) {
  // 같은 소리 데이터를 위쪽은 주파수 막대, 아래쪽은 시간 파형으로 그린다.
  const freqCtx = freqCanvas.getContext("2d");
  const waveCtx = waveCanvas.getContext("2d");
  const freqBuf = new Uint8Array(analyser.frequencyBinCount);
  const waveBuf = new Uint8Array(analyser.fftSize);

  function sizeToDisplay(canvas) {
    // CSS 화면 크기와 canvas 내부 픽셀 크기가 다르면 그림이 흐려진다.
    // 매 프레임 실제 표시 크기에 맞춰 캔버스 해상도를 보정한다.
    const w = canvas.clientWidth || 320;
    const h = canvas.clientHeight || 80;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    return { w, h };
  }

  function draw() {
    // 0~255 정수 배열로 현재 주파수 세기와 파형을 읽는다.
    analyser.getByteFrequencyData(freqBuf);
    analyser.getByteTimeDomainData(waveBuf);

    const { w: fw, h: fh } = sizeToDisplay(freqCanvas);
    freqCtx.fillStyle = "#141210";
    freqCtx.fillRect(0, 0, fw, fh);
    const bars = 48;
    // FFT 전체 칸을 48개 막대로 줄여 너무 촘촘하지 않게 표시한다.
    const step = Math.floor(freqBuf.length / bars);
    const gap = 2;
    const bw = fw / bars;
    for (let i = 0; i < bars; i++) {
      const v = freqBuf[i * step] / 255;
      const bh = Math.max(2, v * fh);
      const hue = 28 + v * 18;
      freqCtx.fillStyle = `hsl(${hue} 72% ${42 + v * 28}%)`;
      freqCtx.fillRect(i * bw + gap * 0.5, fh - bh, bw - gap, bh);
    }

    const { w: ww, h: wh } = sizeToDisplay(waveCanvas);
    waveCtx.fillStyle = "#141210";
    waveCtx.fillRect(0, 0, ww, wh);
    waveCtx.strokeStyle = "#7ee0c6";
    waveCtx.lineWidth = 1.5;
    waveCtx.beginPath();
    const slice = ww / waveBuf.length;
    for (let i = 0; i < waveBuf.length; i++) {
      // 128 근처가 파형의 가운데, 0과 255가 위/아래 끝이다.
      const y = (waveBuf[i] / 255) * wh;
      const x = i * slice;
      if (i === 0) waveCtx.moveTo(x, y);
      else waveCtx.lineTo(x, y);
    }
    waveCtx.stroke();
  }

  return { draw };
}

const DEFAULT_NAMES = ["fist", "palm", "peace", "one", "two_up", "three", "ok", "like", "dislike", "call"];
// labels.json을 읽지 못했을 때 쓰는 기본 순서다.
// 이 순서는 노트북 학습 때 CLASS_NAMES의 번호 순서와 반드시 같아야 한다.

function nms(boxes, iouTh = 0.45) {
  // 겹치는 박스 중 conf 높은 것만. 한 손에 박스가 여러 개 나오는 것 방지
  // IoU = 두 박스가 겹친 넓이 / 두 박스를 합친 넓이.
  // IoU가 0.45보다 크면 같은 손을 중복 검출했다고 보고 낮은 점수 박스를 버린다.
  const sorted = [...boxes].sort((a, b) => b.conf - a.conf);
  const keep = [];
  for (const b of sorted) {
    let overlap = false;
    for (const k of keep) {
      const xx1 = Math.max(b.x1, k.x1);
      const yy1 = Math.max(b.y1, k.y1);
      const xx2 = Math.min(b.x2, k.x2);
      const yy2 = Math.min(b.y2, k.y2);
      const w = Math.max(0, xx2 - xx1);
      const h = Math.max(0, yy2 - yy1);
      const inter = w * h;
      // 합집합 = 첫 박스 + 둘째 박스 - 두 번 센 겹친 부분
      const union = b.area + k.area - inter;
      if (union > 0 && inter / union > iouTh) {
        overlap = true;
        break;
      }
    }
    if (!overlap) keep.push(b);
  }
  return keep;
}

function createYolo(opts = {}) {
  // 노트북이 만든 best.onnx 를 wasm 으로 돌린다. 입력은 1x3x320x320 letterbox RGB
  // "학습"은 노트북에서 이미 끝났다. 여기서는 가중치를 바꾸지 않고 추론만 한다.
  const modelUrl = opts.modelUrl || "./assets/models/best.onnx";
  const labelsUrl = opts.labelsUrl || "./assets/models/labels.json";
  const capture = document.createElement("canvas");
  const letter = document.createElement("canvas");
  // capture: 웹캠 원본 크기, letter: 모델 입력 크기(기본 320x320)
  const capCtx = capture.getContext("2d", { willReadFrequently: true });
  const letCtx = letter.getContext("2d", { willReadFrequently: true });

  let session = null;
  let names = DEFAULT_NAMES;
  let imgsz = 320;
  let confTh = 0.55;
  let inputName = null;

  async function load() {
    // labels.json에는 클래스 이름, 입력 크기, confidence 기준값이 들어 있다.
    // 모델과 메타데이터를 함께 내보내면 데모의 설정 실수를 줄일 수 있다.
    const labelRes = await fetch(labelsUrl);
    if (labelRes.ok) {
      const meta = await labelRes.json();
      if (Array.isArray(meta.names)) names = meta.names;
      if (meta.imgsz) imgsz = meta.imgsz;
      if (meta.conf) confTh = meta.conf;
    }

    let probe = await fetch(modelUrl, { method: "HEAD" });
    // 큰 모델을 받기 전에 파일 존재 여부만 먼저 확인한다.
    if (probe.status === 405) probe = await fetch(modelUrl);
    if (!probe.ok) {
      const err = new Error("NO_MODEL");
      err.code = "NO_MODEL";
      throw err;
    }

    const ort = window.ort;
    if (!ort) throw new Error("onnxruntime-web가 로드되지 않았습니다.");
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/";
    // WASM은 별도 GPU 설치 없이 대부분의 브라우저에서 실행된다.
    // 데모 호환성을 위해 스레드는 1개, SIMD 벡터 연산은 사용한다.
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    try {
      session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ["wasm"],
      });
    } catch (e) {
      const detail = e && (e.message || e.stack || String(e));
      throw new Error(detail || "ONNX 세션 생성 실패");
    }
    // ONNX 입력 이름은 export 버전에 따라 달라질 수 있어 모델에서 직접 읽는다.
    inputName = session.inputNames[0];
    return { names, imgsz };
  }

  function letterbox(video) {
    // YOLO 학습과 같이 정사각 320에 여백을 넣고 비율을 유지. 회색(#727272)은 ultralytics 기본
    // 단순히 320x320으로 찌그러뜨리면 손 모양이 변해 정확도가 떨어질 수 있다.
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return null;
    if (capture.width !== vw || capture.height !== vh) {
      capture.width = vw;
      capture.height = vh;
    }
    capCtx.drawImage(video, 0, 0, vw, vh);

    letter.width = imgsz;
    letter.height = imgsz;
    letCtx.fillStyle = "#727272";
    letCtx.fillRect(0, 0, imgsz, imgsz);
    const scale = Math.min(imgsz / vw, imgsz / vh);
    // 가로·세로 중 더 빡빡한 비율을 선택해야 이미지 전체가 정사각형 안에 들어간다.
    const nw = vw * scale;
    const nh = vh * scale;
    const dx = (imgsz - nw) / 2;
    const dy = (imgsz - nh) / 2;
    letCtx.drawImage(capture, 0, 0, vw, vh, dx, dy, nw, nh);
    return { scale, dx, dy, vw, vh };
  }

  function toTensor() {
    // HWC uint8 → NCHW float32 0~1. YOLO 학습 입력과 같은 레이아웃
    // 브라우저 픽셀: [높이][너비][R,G,B,A], 값은 0~255
    // YOLO 입력: [배치][R,G,B][높이][너비], 값은 0.0~1.0
    const { data } = letCtx.getImageData(0, 0, imgsz, imgsz);
    const float = new Float32Array(3 * imgsz * imgsz);
    let p = 0;
    for (let c = 0; c < 3; c++) {
      for (let y = 0; y < imgsz; y++) {
        for (let x = 0; x < imgsz; x++) {
          float[p++] = data[(y * imgsz + x) * 4 + c] / 255;
        }
      }
    }
    return new window.ort.Tensor("float32", float, [1, 3, imgsz, imgsz]);
  }

  function decode(outTensor, geom) {
    // 모델 출력의 많은 후보 박스를 사람이 이해할 수 있는 객체 배열로 되돌린다.
    const dims = outTensor.dims;
    const data = outTensor.data;
    let n;
    let nc4;
    let transposed;
    if (dims.length === 3) {
      // Ultralytics/ONNX 버전에 따라 [1, 4+클래스수, 후보수] 또는
      // [1, 후보수, 4+클래스수]가 올 수 있어 둘 다 처리한다.
      const a = dims[1];
      const b = dims[2];
      if (a < b) {
        nc4 = a;
        n = b;
        transposed = true;
      } else {
        n = a;
        nc4 = b;
        transposed = false;
      }
    } else {
      return { dets: [], maxConf: 0 };
    }

    // 앞의 4개 값은 중심 x/y와 너비/높이, 나머지가 클래스별 점수다.
    const nc = nc4 - 4;
    const { scale, dx, dy, vw, vh } = geom;
    const raw = [];
    let maxConf = 0;

    for (let i = 0; i < n; i++) {
      // 각 후보에서 점수가 가장 높은 클래스 하나를 찾는다.
      let cx;
      let cy;
      let w;
      let h;
      let best = 0;
      let bestCls = 0;
      if (transposed) {
        cx = data[0 * n + i];
        cy = data[1 * n + i];
        w = data[2 * n + i];
        h = data[3 * n + i];
        for (let c = 0; c < nc; c++) {
          const s = data[(4 + c) * n + i];
          if (s > best) {
            best = s;
            bestCls = c;
          }
        }
      } else {
        const base = i * nc4;
        cx = data[base];
        cy = data[base + 1];
        w = data[base + 2];
        h = data[base + 3];
        for (let c = 0; c < nc; c++) {
          const s = data[base + 4 + c];
          if (s > best) {
            best = s;
            bestCls = c;
          }
        }
      }
      if (best > maxConf) maxConf = best;
      // 기준보다 낮은 후보는 여기서 일찍 버려 NMS 계산량을 줄인다.
      if (best < confTh) continue;

      // 모델은 중심점(cx, cy), 너비(w), 높이(h)를 출력한다.
      // 화면에 그리기 쉬운 왼쪽 위/오른쪽 아래 좌표로 바꾼다.
      const x1p = cx - w / 2;
      const y1p = cy - h / 2;
      const x2p = cx + w / 2;
      const y2p = cy + h / 2;
      // letterbox 때 추가한 여백(dx,dy)을 빼고 scale로 나눠 웹캠 원본 좌표로 복원한다.
      const x1 = (x1p - dx) / scale;
      const y1 = (y1p - dy) / scale;
      const x2 = (x2p - dx) / scale;
      const y2 = (y2p - dy) / scale;
      const bx1 = Math.min(vw, Math.max(0, x1));
      const by1 = Math.min(vh, Math.max(0, y1));
      const bx2 = Math.min(vw, Math.max(0, x2));
      const by2 = Math.min(vh, Math.max(0, y2));
      const bw = bx2 - bx1;
      const bh = by2 - by1;
      if (bw < 2 || bh < 2) continue;
      raw.push({
        x1: bx1,
        y1: by1,
        x2: bx2,
        y2: by2,
        area: bw * bh,
        conf: best,
        cls: bestCls,
        name: names[bestCls] || String(bestCls),
      });
    }

    const dets = nms(raw).slice(0, 5).map((b) => {
      // 검출 박스 중심을 0~1 상대좌표로 만들어 화면 크기와 무관하게 제스처 이동을 계산한다.
      const cx = (b.x1 + b.x2) / 2;
      const cy = (b.y1 + b.y2) / 2;
      return {
        ...b,
        // 웹캠은 거울처럼 좌우 반전해서 보여 주므로 x도 반전한다.
        nx: 1 - cx / geom.vw,  // 사용자가 보는 화면 오른쪽 = nx 증가
        ny: cy / geom.vh,
      };
    });
    return { dets, maxConf };
  }

  async function detect(video) {
    // 한 번의 추론 순서: 전처리(letterbox) → Tensor → ONNX 실행 → 후처리(decode)
    if (!session) return { dets: [], maxConf: 0 };
    const geom = letterbox(video);
    if (!geom) return { dets: [], maxConf: 0 };
    const tensor = toTensor();
    const feeds = { [inputName]: tensor };
    const out = await session.run(feeds);
    const first = out[session.outputNames[0]];
    return decode(first, geom);
  }

  return {
    load,
    detect,
    get confTh() {
      return confTh;
    },
    get names() {
      return names;
    },
    get ready() {
      return Boolean(session);
    },
  };
}

const HYST_FRAMES = 3;     // 같은 클래스가 N프레임 연속이어야 확정 (깜빡임 방지)
const LOST_FRAMES = 2;     // 손이 N프레임 없으면 모드 해제
const FIST_MS = 150;       // 주먹을 이만큼 유지해야 ARMED
const PALM_WINDOW_MS = 800;  // 그 다음 이 시간 안에 손바닥이어야 토글
const COOLDOWN_MS = 600;   // 토글 직후 재입력 막기
const OK_DEAD = 0.10;      // ok 볼륨·시크 데드존 (화면 비율). 조금 넓혀 손떨림에 안 움직임
const CUT_DEAD = 0.05;
const EQ_DEAD = 0.04;
const SEEK_RATE = 42;      // mag=1 일 때 초당 미디어 초. 볼륨과 같이 "밀어두면 계속"
// 위 숫자들은 모델 가중치가 아니라 데모 조작 감도다.
// 실제 발표에서는 "오검출을 줄이기 위해 연속 프레임과 데드존을 사용했다"고 설명하면 된다.

function createGestureMachine() {
  // 클래스 하나가 여러 기능에 쓰이지 않게 모드를 하나만 연다.
  // 재생은 엣지(fist→palm 순간 1회), 볼륨/EQ/시크는 홀드 중 연속값.
  //
  // 왜 상태머신이 필요한가?
  // YOLO는 매 프레임 "지금 보이는 손은 ok일 확률 0.82"처럼 결과만 준다.
  // "ok를 처음 잡은 위치", "fist 다음 palm", "한 번만 실행" 같은 시간 흐름은
  // 모델이 모르므로 이 코드가 mode라는 상태로 기억한다.
  let mode = "IDLE";
  // stableName: 여러 프레임 확인을 통과한 클래스, sameCount: 연속 등장 횟수
  let stableName = null;
  let sameCount = 0;
  let lostCount = 0;
  let fistStart = 0;
  let armedAt = 0;
  let cooldownUntil = 0;
  // x0/y0는 연속 제어를 시작한 순간의 손 위치다.
  // 이 위치를 기준점으로 사용하므로 화면 어느 곳에서 제스처를 시작해도 된다.
  let y0 = null;
  let x0 = null;
  let eqBand = null;
  let lastLog = { class: "-", conf: 0, mode: "IDLE" };

  function toIdle() {
    // 제스처가 끝났을 때 이전 기준점과 EQ 선택을 모두 지운다.
    mode = "IDLE";
    fistStart = 0;
    armedAt = 0;
    y0 = null;
    x0 = null;
    eqBand = null;
  }

  function enterContinuous(next, det) {
    // 진입 순간의 손 위치 = 원점. 이후 상대 이동량만 씀 (절대 위치로 0에 스냅하지 않음)
    mode = next;
    y0 = det.ny;
    x0 = det.nx;
    eqBand = null;
    if (next === "EQ_LOW") eqBand = "low";
    if (next === "EQ_MID") eqBand = "mid";
    if (next === "EQ_HIGH") eqBand = "high";
  }

  function tick(det, nowMs, dt) {
    // loop()가 매 화면마다 호출한다.
    // det: 가장 확실한 손 검출 1개, nowMs: 현재 시각(ms), dt: 직전 프레임과의 간격(초)
    // 반환 events는 "무엇을 해야 하는지"만 담고 실제 오디오 변경은 applyEvents가 한다.
    const events = [];
    const name = det && det.conf >= 0.55 ? det.name : null;  // labels.json conf 와 동일

    if (name && name === stableName) {
      // 같은 이름이 계속 나오면 신뢰 횟수를 올린다.
      sameCount += 1;
      lostCount = 0;
    } else if (name) {
      // 다른 제스처가 보이면 새 후보로 바꾸고 1부터 다시 센다.
      sameCount = 1;
      stableName = name;
      lostCount = 0;
    } else {
      // 손을 잠깐 놓쳐도 바로 종료하지 않고 LOST_FRAMES만큼 기다린다.
      lostCount += 1;
      sameCount = 0;
      if (lostCount >= LOST_FRAMES) {
        stableName = null;
        if (mode !== "IDLE" && mode !== "COOLDOWN") toIdle();
      }
    }

    // 3프레임 연속 확인 전에는 locked=null이라 실제 기능을 실행하지 않는다.
    const locked = sameCount >= HYST_FRAMES ? stableName : null;

    if (nowMs < cooldownUntil) {
      // 토글 직후 같은 palm이 남아 있어 또 토글되는 문제를 막는 잠금 시간이다.
      mode = "COOLDOWN";
      lastLog = { class: name || "-", conf: det ? det.conf : 0, mode };
      return { events, mode, log: lastLog };
    }
    if (mode === "COOLDOWN") mode = "IDLE";

    if (mode === "WAIT_SKIP") {
      // dislike 를 들고 있는 동안은 한 번만 다음 곡. 손을 떼야 다시 IDLE.
      // 이런 처리를 "엣지 이벤트"라고 한다.
      if (locked === "dislike") {
        lastLog = { class: name || "-", conf: det ? det.conf : 0, mode: "SKIP_NEXT" };
        return { events, mode: "SKIP_NEXT", log: lastLog };
      }
      toIdle();
    }

    if (mode === "ARMED_FIST") {
      // fist 유지 후 palm 이 뜨는 "전환 순간"에만 play/pause. palm 홀드는 토글 안 함
      if (locked === "palm") {
        events.push({ type: "toggle_play" });
        cooldownUntil = nowMs + COOLDOWN_MS;
        toIdle();
        mode = "COOLDOWN";
      } else if (nowMs - armedAt > PALM_WINDOW_MS) {
        toIdle();
      }
      lastLog = { class: name || "-", conf: det ? det.conf : 0, mode };
      return { events, mode, log: lastLog };
    }

    if (mode === "VOLUME_JOYSTICK" && locked === "ok" && det) {
      // ny 는 아래가 큼. y0 - ny > 0 이면 손을 위로 = 볼륨 증가
      const dy = y0 - det.ny;
      let magY = 0;
      // 데드존 안의 작은 흔들림은 0으로 취급한다.
      // 데드존 밖에서는 초과한 거리만 사용해 경계에서 값이 갑자기 튀지 않게 한다.
      if (Math.abs(dy) > OK_DEAD) magY = dy - Math.sign(dy) * OK_DEAD;
      // dt를 곱하면 컴퓨터 프레임 속도가 달라도 "초당 변화량"이 거의 같다.
      events.push({ type: "volume_delta", value: magY * 1.6 * dt });
      const dx = det.nx - x0;  // 화면 오른쪽 = 앞으로
      let magX = 0;
      if (Math.abs(dx) > OK_DEAD) magX = dx - Math.sign(dx) * OK_DEAD;
      events.push({ type: "seek_delta", value: magX * SEEK_RATE * dt, nx: det.nx });
    } else if (mode === "FILTER_SWEEP" && locked === "like" && det) {
      // cutoff는 주파수라 선형 덧셈보다 비율(지수) 변화가 자연스럽다.
      const dx = det.nx - x0;  // 화면 오른쪽 = CUT 증가
      let magX = 0;
      if (Math.abs(dx) > CUT_DEAD) magX = dx - Math.sign(dx) * CUT_DEAD;
      events.push({ type: "cutoff_delta", value: magX * 7.5 * dt });
    } else if (mode.startsWith("EQ_") && det && (
      (mode === "EQ_LOW" && locked === "one") ||
      (mode === "EQ_MID" && locked === "peace") ||
      (mode === "EQ_HIGH" && locked === "three")
    )) {
      // one/peace/three를 각각 Low/Mid/High에 고정해 기능 충돌을 줄였다.
      const dy = y0 - det.ny;
      let mag = 0;
      if (Math.abs(dy) > EQ_DEAD) mag = dy - Math.sign(dy) * EQ_DEAD;
      events.push({ type: "eq_delta", band: eqBand, value: mag * 50 * dt });
    } else if (mode !== "IDLE" && locked && (
      (mode === "VOLUME_JOYSTICK" && locked !== "ok") ||
      (mode === "FILTER_SWEEP" && locked !== "like") ||
      (mode === "EQ_LOW" && locked !== "one") ||
      (mode === "EQ_MID" && locked !== "peace") ||
      (mode === "EQ_HIGH" && locked !== "three")
    )) {
      // 연속 제어 중 다른 제스처가 확정되면 기존 모드를 종료한다.
      // 예: ok로 볼륨을 조절하다 손이 palm으로 바뀌면 더 이상 볼륨을 움직이지 않는다.
      toIdle();
    }

    if (mode === "IDLE") {
      // IDLE은 어떤 기능도 잡고 있지 않은 대기 상태다.
      if (locked === "fist") {
        // 짧은 주먹만으로 바로 토글하지 않음. 실수로 쥐는 것 방지
        if (!fistStart) fistStart = nowMs;
        if (nowMs - fistStart >= FIST_MS) {
          mode = "ARMED_FIST";
          armedAt = nowMs;
          fistStart = 0;
        }
      } else {
        fistStart = 0;
      }

      if (mode === "IDLE" && locked === "dislike") {
        // 먼저 이벤트를 한 번 만들고 WAIT_SKIP으로 들어가 반복 실행을 막는다.
        events.push({ type: "next_track" });
        mode = "WAIT_SKIP";
        lastLog = { class: name || "-", conf: det ? det.conf : 0, mode: "SKIP_NEXT" };
        return { events, mode: "SKIP_NEXT", log: lastLog };
      }
      // 연속형 제스처는 확정되는 순간의 좌표를 x0/y0로 저장한다.
      if (mode === "IDLE" && locked === "ok") enterContinuous("VOLUME_JOYSTICK", det);
      else if (mode === "IDLE" && locked === "like") enterContinuous("FILTER_SWEEP", det);
      else if (mode === "IDLE" && locked === "one") enterContinuous("EQ_LOW", det);
      else if (mode === "IDLE" && locked === "peace") enterContinuous("EQ_MID", det);
      else if (mode === "IDLE" && locked === "three") enterContinuous("EQ_HIGH", det);
    }

    lastLog = { class: name || "-", conf: det ? det.conf : 0, mode };
    return { events, mode, log: lastLog };
  }

  return {
    tick,
    // 화면 표시나 외부 보정에 필요한 최소 기능만 공개한다.
    getMode: () => mode,
    regrabSeek(nx) {
      // 시크로 다음 곡에 넘어간 직후 현재 손 위치를 새 기준점으로 잡는다.
      // 그렇지 않으면 다음 곡도 같은 프레임에 계속 앞으로 밀릴 수 있다.
      if (typeof nx === "number") x0 = nx;
    },
  };
}


const GUIDE = [
  // OSD 하이라이트용. 실제 매핑은 createGestureMachine IDLE 분기와 같아야 함
  { mode: "ARMED_FIST", keys: "fist → palm", label: "재생/일시정지" },
  { mode: "VOLUME_JOYSTICK", keys: "ok + 높이", label: "볼륨 조이스틱" },
  { mode: "VOLUME_JOYSTICK", keys: "ok + 좌우", label: "시크 (밀어두면 계속)" },
  { mode: "FILTER_SWEEP", keys: "like + 좌우", label: "DJ 로우패스" },
  { mode: "SKIP_NEXT", keys: "dislike", label: "다음 파일" },
  { mode: "EQ_LOW", keys: "one + 높이", label: "EQ Low" },
  { mode: "EQ_MID", keys: "peace + 높이", label: "EQ Mid" },
  { mode: "EQ_HIGH", keys: "three + 높이", label: "EQ High" },
];

function createUi(el) {
  // UI 함수는 모델이나 소리를 바꾸지 않고 "보여 주는 일"만 담당한다.
  // 역할을 분리하면 추론 문제와 화면 문제를 따로 찾기 쉽다.
  const overlay = el.overlay;
  const octx = overlay.getContext("2d");

  function fitOverlay(video) {
    // 박스 좌표는 웹캠 원본 픽셀 기준이므로 overlay도 같은 크기로 맞춘다.
    const w = video.videoWidth || 640;
    const h = video.videoHeight || 480;
    if (overlay.width !== w || overlay.height !== h) {
      overlay.width = w;
      overlay.height = h;
    }
  }

  function drawOverlay(video, dets, mode) {
    // 이전 프레임 그림을 지운 뒤 이번 프레임 검출 박스와 라벨을 다시 그린다.
    fitOverlay(video);
    octx.clearRect(0, 0, overlay.width, overlay.height);
    for (const d of dets) {
      const w = d.x2 - d.x1;
      const h = d.y2 - d.y1;
      octx.strokeStyle = "#7ee0c6";
      octx.lineWidth = 3;
      octx.strokeRect(d.x1, d.y1, w, h);
      // confidence를 소수 둘째 자리까지 보여 모델이 얼마나 확신하는지 확인한다.
      const label = `${d.name} ${d.conf.toFixed(2)}`;
      octx.font = "16px ui-sans-serif, sans-serif";
      const tw = octx.measureText(label).width + 10;
      octx.fillStyle = "#7ee0c6";
      octx.fillRect(d.x1, Math.max(0, d.y1 - 22), tw, 22);
      octx.fillStyle = "#12100e";
      octx.fillText(label, d.x1 + 5, Math.max(16, d.y1 - 6));
    }
    el.modeOsd.textContent = mode;
    el.modeOsd.dataset.mode = mode;
  }

  function updateMeters(dsp, log) {
    // DSP가 보관한 실제 값을 텍스트와 0~100% 막대 길이로 변환한다.
    const { volume, eq, cutoff } = dsp.state;
    el.volVal.textContent = volume.toFixed(2);
    el.lowVal.textContent = `${eq.low.toFixed(1)} dB`;
    el.midVal.textContent = `${eq.mid.toFixed(1)} dB`;
    el.highVal.textContent = `${eq.high.toFixed(1)} dB`;
    el.cutVal.textContent = `${Math.round(cutoff)} Hz`;
    el.volBar.style.width = `${((volume - 0.05) / 0.95) * 100}%`;
    el.lowBar.style.width = `${((eq.low + 15) / 30) * 100}%`;
    el.midBar.style.width = `${((eq.mid + 15) / 30) * 100}%`;
    el.highBar.style.width = `${((eq.high + 15) / 30) * 100}%`;
    // cutoff는 귀의 감각에 맞춰 로그 비율로 표시한다.
    const t = (Math.log(cutoff) - Math.log(200)) / (Math.log(16000) - Math.log(200));
    el.cutBar.style.width = `${t * 100}%`;

    if (log) {
      // 발표 시 모델 상태를 확인할 수 있는 디버그 정보:
      // 검출 수, 가장 높은 점수, 확정 클래스, 현재 상태, DSP 값
      el.debug.textContent = `n=${log.nDets ?? 0}  max=${(log.maxConf || 0).toFixed(3)}  class=${log.class}  conf=${(log.conf || 0).toFixed(2)}  mode=${log.mode}  gain=${volume.toFixed(2)}  cutoff=${Math.round(cutoff)}`;
    }

    for (const row of el.guide.querySelectorAll("[data-mode]")) {
      row.classList.toggle("active", row.dataset.mode === (log && log.mode));
    }
  }

  function updatePlayer(audioEl, playing) {
    // 재생 버튼 글자, 현재/전체 시간, 진행 슬라이더를 실제 미디어 상태와 동기화한다.
    el.btnPlay.textContent = playing ? "Pause" : "Play";
    const dur = audioEl.duration;
    const cur = audioEl.currentTime;
    el.time.textContent = `${formatTime(cur)} / ${formatTime(dur)}`;
    if (!el.seek.dataset.dragging) {
      // 사용자가 슬라이더를 잡은 동안 자동 갱신하면 손잡이가 튀므로 그때는 건너뛴다.
      el.seek.value = Number.isFinite(dur) && dur > 0 ? String((cur / dur) * 1000) : "0";
    }
  }

  function renderQueue(items, index) {
    // 배열을 <li> 문자열로 바꿔 재생목록을 다시 그린다.
    if (el.queueCount) el.queueCount.textContent = String(items.length);
    if (el.nowPlaying) {
      el.nowPlaying.textContent =
        index >= 0 && items[index] ? items[index].name : "demo.wav";
    }
    if (!el.queue) return;
    if (!items.length) {
      el.queue.innerHTML = '<li class="empty">파일을 추가하세요</li>';
      return;
    }
    // 파일명에는 escapeHtml을 적용하지만, 영상/오디오 표시와 버튼은 프로그램이 만든 HTML이다.
    el.queue.innerHTML = items
      .map(
        (it, i) =>
          `<li data-i="${i}" class="${i === index ? "active" : ""}"><span><span class="kind">${
            it.video ? "영상" : "오디오"
          }</span>${escapeHtml(it.name)}</span><button type="button" data-remove="${i}" aria-label="삭제">×</button></li>`
      )
      .join("");
  }

  function setModelStatus(text, kind) {
    // kind 값에 따라 CSS가 준비/경고/오류 색을 다르게 표시한다.
    el.modelStatus.textContent = text;
    el.modelStatus.dataset.kind = kind || "";
  }

  function setGuide() {
    el.guide.innerHTML = GUIDE.map(
      (g) =>
        `<li data-mode="${g.mode}"><kbd>${g.keys}</kbd><span>${g.label}</span></li>`
    ).join("");
  }

  setGuide();

  return { drawOverlay, updateMeters, updatePlayer, renderQueue, setModelStatus };
}


const $ = (id) => document.getElementById(id);
// 같은 document.getElementById를 여러 번 쓰지 않기 위한 짧은 도우미다.

const els = {
  // 자주 갱신할 HTML 요소를 한 번만 찾아 객체에 모아 둔다.
  overlay: $("overlay"),
  modeOsd: $("mode-osd"),
  volVal: $("vol-val"),
  lowVal: $("low-val"),
  midVal: $("mid-val"),
  highVal: $("high-val"),
  cutVal: $("cut-val"),
  volBar: $("vol-bar"),
  lowBar: $("low-bar"),
  midBar: $("mid-bar"),
  highBar: $("high-bar"),
  cutBar: $("cut-bar"),
  debug: $("debug"),
  guide: $("guide"),
  btnPlay: $("btn-play"),
  time: $("time"),
  seek: $("seek"),
  modelStatus: $("model-status"),
  queue: $("queue"),
  queueCount: $("queue-count"),
  nowPlaying: $("now-playing"),
};

const audioEl = $("track");  // 재생용 오디오/영상 (DSP 입력)
const videoEl = $("cam");    // 웹캠 (YOLO 입력, 거울)
// 아래 객체들은 각각 한 역할만 담당한다.
// player=재생, dsp=소리 가공, viz=파형, yolo=손 검출, machine=동작 해석, ui=화면 표시
const player = createPlayer(audioEl);
const dsp = createDsp(audioEl);
const viz = createVisualizer(dsp.analyser, $("viz-freq"), $("viz-wave"));
const yolo = createYolo();
const machine = createGestureMachine();
const ui = createUi(els);
player.setOnQueueChange((items, index) => ui.renderQueue(items, index));

let running = false;
// 루프와 비동기 추론 사이에서 공유하는 실행 상태
let lastTs = 0;
let lastDets = [];
let lastMaxConf = 0;
let frame = 0;
let eqFocus = null;
let modelReady = false;
let inferring = false;

// seekHead는 제스처가 원하는 가상 재생 위치다.
// 실제 video.currentTime과 분리해야 빠른 연속 입력을 잃지 않는다.
let seekHead = null;
let seekSkipLock = false;

function applyEvents(events, mode) {
  // 상태머신은 이벤트만 만들고, 여기서 플레이어/DSP 에 적용
  for (const ev of events) {
    // 이벤트 이름별로 대응하는 실제 기능 하나를 호출한다.
    if (ev.type === "toggle_play") player.toggle();
    if (ev.type === "volume_delta") dsp.setVolume(dsp.state.volume + ev.value);
    if (ev.type === "eq_delta") dsp.setEq(ev.band, dsp.state.eq[ev.band] + ev.value);
    if (ev.type === "cutoff_delta") {
      // 주파수는 곱셈 비율로 바꿔 저음 구간과 고음 구간의 체감 속도를 맞춘다.
      dsp.setCutoff(dsp.state.cutoff * Math.exp(ev.value));
    }
    if (ev.type === "seek_delta") {
      // 가상 재생헤드를 쌓아 두고 seekTo. 브라우저 시크 드롭을 피함
      if (seekHead == null || !Number.isFinite(seekHead)) {
        seekHead = audioEl.currentTime || 0;
        seekSkipLock = false;
      }
      seekHead += ev.value;
      const dur = audioEl.duration;
      if (
        !seekSkipLock &&
        player.canSkipNext() &&
        Number.isFinite(dur) &&
        dur > 0 &&
        seekHead >= dur - 0.2
      ) {
        // 시크가 곡 끝에 닿으면 다음 파일을 재생한다.
        // seekSkipLock은 같은 손동작으로 여러 곡이 연속 통과하는 것을 막는다.
        seekSkipLock = true;
        seekHead = 0;
        player.next(true);
        machine.regrabSeek(ev.nx);  // 다음 곡에서 지금 손 위치를 새 원점으로
        continue;
      }
      if (Number.isFinite(dur) && dur > 0) {
        seekHead = Math.min(dur, Math.max(0, seekHead));
      } else {
        seekHead = Math.max(0, seekHead);
      }
      player.seekTo(seekHead);
    }
    if (ev.type === "next_track") player.next(true);
  }
  if (mode !== "VOLUME_JOYSTICK") {
    seekHead = null;
    seekSkipLock = false;
  }
}

function onKey(e) {
  // 모델이 없어도 모든 핵심 기능을 시험하고 발표 중 비상 조작할 수 있는 키보드 대안이다.
  if (!running) return;
  if (e.code === "Space") {
    e.preventDefault();
    player.toggle();
    return;
  }
  // 숫자 1/2/3으로 방향키가 조절할 EQ 대역을 선택한다.
  if (e.key === "1") eqFocus = "low";
  if (e.key === "2") eqFocus = "mid";
  if (e.key === "3") eqFocus = "high";
  if (e.key === "0" || e.key === "`") eqFocus = null;

  if (e.key === "ArrowLeft") {
    e.preventDefault();
    if (e.shiftKey) player.prev(true);
    else player.nudge(-5);
    return;
  }
  if (e.key === "ArrowRight") {
    e.preventDefault();
    if (e.shiftKey) player.next(true);
    else player.nudge(5);
    return;
  }
  if (e.key === "ArrowUp") {
    e.preventDefault();
    if (eqFocus) dsp.setEq(eqFocus, dsp.state.eq[eqFocus] + 1);
    else dsp.setVolume(dsp.state.volume + 0.04);
  }
  if (e.key === "ArrowDown") {
    e.preventDefault();
    if (eqFocus) dsp.setEq(eqFocus, dsp.state.eq[eqFocus] - 1);
    else dsp.setVolume(dsp.state.volume - 0.04);
  }
  if (e.key === "[" || e.key === ",") {
    dsp.setCutoff(dsp.state.cutoff * 0.9);
  }
  if (e.key === "]" || e.key === ".") {
    dsp.setCutoff(dsp.state.cutoff / 0.9);
  }
}

function loop(ts) {
  // 매 프레임: 미터 갱신 + 3프레임마다 YOLO + 상태머신 tick
  // requestAnimationFrame은 브라우저 화면 새로고침 속도에 맞춰 다시 loop를 부른다.
  requestAnimationFrame(loop);
  if (!running) return;
  // 탭이 잠시 멈췄다가 돌아와도 dt를 최대 0.05초로 제한해 값이 갑자기 튀지 않게 한다.
  const dt = Math.min(0.05, lastTs ? (ts - lastTs) / 1000 : 0.016);
  lastTs = ts;
  frame += 1;

  viz.draw();
  ui.updatePlayer(audioEl, player.isPlaying());

  if (modelReady && !inferring && videoEl.readyState >= 2 && frame % 3 === 0) {
    // YOLO는 화면 그리기보다 무거워 3프레임에 한 번만 실행한다.
    // inferring 잠금은 이전 비동기 추론이 끝나기 전에 다음 추론이 겹치는 것을 막는다.
    inferring = true;
    yolo
      .detect(videoEl)
      .then((result) => {
        lastDets = result.dets || [];
        lastMaxConf = result.maxConf || 0;
      })
      .catch((err) => console.warn(err))
      .finally(() => {
        inferring = false;
      });
  }

  // NMS 결과는 confidence 순서이므로 가장 확실한 손 하나로 상태머신을 움직인다.
  const best = lastDets[0] || null;
  const { events, mode, log } = machine.tick(best, ts, dt);
  log.nDets = lastDets.length;
  log.maxConf = lastMaxConf;
  applyEvents(events, mode);
  ui.drawOverlay(videoEl, lastDets, mode);
  ui.updateMeters(dsp, log);
}

async function startMixer() {
  // 브라우저 정책: AudioContext / 웹캠은 클릭 제스처 뒤에만 열림
  $("btn-start").disabled = true;
  ui.setModelStatus("오디오 언락 중…", "busy");
  await dsp.resume();
  if (!player.hasCustomFile()) {
    // 사용자가 파일을 올리지 않았다면 발표 확인용 demo.wav를 사용한다.
    player.loadUrl("./assets/audio/demo.wav");
  }
  await player.unlock();

  ui.setModelStatus("웹캠 요청 중…", "busy");
  try {
    await startCamera(videoEl);
  } catch (err) {
    ui.setModelStatus("웹캠 권한이 필요합니다.", "err");
    $("btn-start").disabled = false;
    throw err;
  }

  ui.setModelStatus("YOLO 모델 로딩…", "busy");
  try {
    // ONNX 로드는 시간이 걸릴 수 있으므로 await로 끝날 때까지 기다린다.
    await yolo.load();
    modelReady = true;
    ui.setModelStatus("모델 준비됨 · ok 볼륨·시크 / like 로우패스 / dislike 다음", "ok");
  } catch (err) {
    modelReady = false;
    if (err && err.code === "NO_MODEL") {
      ui.setModelStatus("best.onnx 없음 · 키보드로 오디오 테스트 가능 (GestureBeats.ipynb export 후 재실행)", "warn");
    } else {
      const detail = (err && (err.message || err.stack)) || String(err);
      ui.setModelStatus(`모델 로드 실패: ${detail}`, "err");
    }
  }

  running = true;
  $("app").classList.add("live");
  requestAnimationFrame(loop);
}

function bind() {
  // HTML 버튼·슬라이더에서 발생한 사용자 행동을 함수와 연결한다.
  // 이런 연결을 이벤트 바인딩이라고 한다.
  $("btn-start").addEventListener("click", () => {
    startMixer().catch((err) => console.error(err));
  });
  $("btn-play").addEventListener("click", () => {
    dsp.resume().then(() => player.toggle());
  });
  $("btn-prev").addEventListener("click", () => {
    dsp.resume().then(() => player.prev(true));
  });
  $("btn-next").addEventListener("click", () => {
    dsp.resume().then(() => player.next(true));
  });
  $("btn-clear-queue").addEventListener("click", () => {
    player.clear();
  });
  $("file").addEventListener("change", (e) => {
    // multiple input이므로 한 번에 고른 파일 전체를 재생목록에 추가한다.
    const files = e.target.files;
    if (!files || !files.length) return;
    player.addFiles(files);
    e.target.value = "";
    if (running) player.play().catch(() => {});
  });
  $("queue").addEventListener("click", (e) => {
    // 각 li마다 별도 리스너를 만들지 않고 부모 하나에서 클릭 위치를 판별한다.
    // 목록을 다시 그려도 계속 작동하는 이벤트 위임 방식이다.
    const rm = e.target.closest("[data-remove]");
    if (rm) {
      player.removeAt(Number(rm.dataset.remove));
      return;
    }
    const row = e.target.closest("li[data-i]");
    if (row) dsp.resume().then(() => player.playAt(Number(row.dataset.i), true));
  });
  ["timeupdate", "loadedmetadata", "play", "pause", "ended"].forEach((ev) => {
    // 미디어 상태가 바뀔 수 있는 사건마다 시간과 버튼을 즉시 갱신한다.
    audioEl.addEventListener(ev, () => ui.updatePlayer(audioEl, player.isPlaying()));
  });
  els.seek.addEventListener("pointerdown", () => {
    els.seek.dataset.dragging = "1";
  });
  els.seek.addEventListener("input", () => {
    if (els.seek.dataset.dragging) player.seek(Number(els.seek.value) / 1000);
  });
  els.seek.addEventListener("pointerup", () => {
    delete els.seek.dataset.dragging;
    player.seek(Number(els.seek.value) / 1000);
  });
  window.addEventListener("keydown", onKey);

  if (location.protocol === "file:") {
    // 웹캠과 fetch는 보안상 file://에서 제한되므로 실행 방법을 안내한다.
    ui.setModelStatus("file:// 로는 웹캠/ONNX가 막힙니다. python demo.py 로 여세요.", "err");
  } else {
    ui.setModelStatus("Start Mixer 를 누르세요.", "");
  }
}

bind();
  </script>
</body>
</html>
'''

ROOT = Path(__file__).resolve().parent
# ROOT는 demo.py가 있는 폴더다. 터미널의 현재 위치가 달라도 파일을 정확히 찾게 한다.
MODEL_DIR = ROOT / "models"
AUDIO_CANDIDATES = [
    ROOT / "assets" / "audio" / "demo.wav",
    ROOT / "web" / "assets" / "audio" / "demo.wav",
]


def find_audio() -> Path | None:
    """가능한 샘플 음원 위치를 순서대로 확인해 처음 발견한 경로를 반환한다."""
    for p in AUDIO_CANDIDATES:
        # Path.is_file()은 경로가 실제 파일일 때만 True다.
        if p.is_file():
            return p
    return None


def send_bytes(handler: BaseHTTPRequestHandler, data: bytes, content_type: str, extra=None):
    """메모리의 바이트를 일반 응답(200) 또는 부분 응답(206)으로 전송한다.

    브라우저는 영상의 중간으로 이동할 때 파일 전체를 다시 받지 않고
    필요한 구간만 ``Range: bytes=시작-끝`` 형태로 요청한다.
    서버가 이를 지원해야 큰 영상의 시크가 빠르고 정상적으로 동작한다.
    """
    rng = handler.headers.get("Range")
    total = len(data)
    extra = extra or {}
    if rng and rng.startswith("bytes="):
        # 예: "bytes=1000-1999"에서 시작과 끝 문자열을 분리한다.
        spec = rng.split("=", 1)[1].split(",")[0].strip()
        start_s, _, end_s = spec.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else total - 1
        start = max(0, min(start, total))
        end = max(start, min(end, total - 1))
        # 파이썬 슬라이스의 끝 번호는 포함되지 않으므로 end + 1을 사용한다.
        chunk = data[start : end + 1]
        # 206 Partial Content는 "파일 일부만 보낸다"는 HTTP 상태 코드다.
        handler.send_response(206)
        handler.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        handler.send_header("Content-Length", str(len(chunk)))
        handler.send_header("Accept-Ranges", "bytes")
        handler.send_header("Content-Type", content_type)
        for k, v in extra.items():
            handler.send_header(k, v)
        handler.end_headers()
        handler.wfile.write(chunk)
        return
    # Range 요청이 없으면 파일 전체를 200 OK로 보낸다.
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(total))
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Cache-Control", "no-cache")
    for k, v in extra.items():
        handler.send_header(k, v)
    handler.end_headers()
    handler.wfile.write(data)


class Handler(BaseHTTPRequestHandler):
    """브라우저 URL을 실제 HTML·모델·음원 응답으로 연결하는 작은 HTTP 서버.

    GET은 내용을 요청하고, HEAD는 내용 없이 크기와 형식만 확인한다.
    """
    def log_message(self, fmt, *args):
        # 브라우저 요청 기록을 터미널에 출력해 문제를 찾기 쉽게 한다.
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_HEAD(self):
        # 모델 존재 확인 요청. 내용(body)은 보내지 않는다.
        self._handle(head_only=True)

    def do_GET(self):
        # 실제 HTML, ONNX, JSON, WAV 바이트 요청
        self._handle(head_only=False)

    def _handle(self, head_only: bool):
        # 쿼리는 제외하고 URL 인코딩된 글자를 원래 문자로 되돌린다.
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            # 루트 주소에는 위의 긴 INDEX_HTML 문자열을 UTF-8로 보낸다.
            data = INDEX_HTML.encode("utf-8")
            if head_only:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return
            send_bytes(self, data, "text/html; charset=utf-8")
            return

        # 요청 URL을 허용된 파일에만 연결한다.
        # 사용자가 보낸 임의 경로를 열지 않아 경로 탐색 문제도 막는다.
        disk = None
        ctype = "application/octet-stream"
        if path in ("/assets/models/best.onnx", "/assets/models/best.onnx/"):
            disk = MODEL_DIR / "best.onnx"
            ctype = "application/octet-stream"
        elif path in ("/assets/models/labels.json", "/assets/models/labels.json/"):
            disk = MODEL_DIR / "labels.json"
            ctype = "application/json; charset=utf-8"
        elif path.startswith("/assets/audio/"):
            disk = find_audio()
            ctype = "audio/wav"

        if disk is None or not disk.is_file():
            # 등록되지 않았거나 실제 파일이 없으면 HTTP 404를 반환한다.
            msg = b"not found"
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            if not head_only:
                self.wfile.write(msg)
            return

        # 이 발표용 서버는 파일을 메모리의 bytes로 읽은 뒤 전송한다.
        data = disk.read_bytes()
        if head_only:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return
        send_bytes(self, data, ctype)


def main():
    """명령행 옵션을 읽고 로컬 서버를 시작하는 파이썬 진입점."""
    # 1. 실행 옵션
    # 예: python demo.py --port 9000. 옵션을 생략하면 default 값을 쓴다.
    parser = argparse.ArgumentParser(description="Gesture Beats demo server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    # 2. 학습 결과 파일 확인 (GestureBeats.ipynb 에서 만든 것)
    # 서버 시작 전에 필수 파일의 위치와 존재 여부를 터미널에 보여 준다.
    onnx = MODEL_DIR / "best.onnx"
    labels = MODEL_DIR / "labels.json"
    audio = find_audio()
    print("ROOT   :", ROOT, flush=True)
    print("onnx   :", onnx, "OK" if onnx.is_file() else "없음 (노트북 ONNX 셀 실행 후 재실행)", flush=True)
    print("labels :", labels, "OK" if labels.is_file() else "없음", flush=True)
    print("audio  :", audio if audio else "없음 — 페이지에서 오디오/영상 파일 올리기 사용", flush=True)
    print(flush=True)
    print(f"Open http://{args.host}:{args.port}", flush=True)
    print("종료: Ctrl+C", flush=True)
    print(flush=True)

    # 3. 로컬 웹 서버 시작 (웹캠은 localhost 또는 https 에서만 열림)
    # ThreadingHTTPServer는 HTML·모델·음원 요청을 동시에 처리할 수 있다.
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)

    def _open():
        # 기본 브라우저에서 데모 주소를 여는 작은 콜백이다.
        webbrowser.open(f"http://{args.host}:{args.port}")

    if not args.no_browser:
        # 서버가 먼저 준비되도록 0.6초 뒤 별도 스레드에서 브라우저를 연다.
        threading.Timer(0.6, _open).start()

    try:
        # Ctrl+C를 누를 때까지 요청을 계속 처리한다.
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        # 종료할 때 사용 중인 포트를 정상적으로 반납한다.
        httpd.server_close()


if __name__ == "__main__":
    # 다른 파일에서 import할 때는 서버를 자동 시작하지 않고,
    # ``python demo.py``로 직접 실행했을 때만 main()을 호출한다.
    main()
