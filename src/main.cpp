/**
 * Mangosteen Ripeness Classifier with Dual Interface:
 *   1. High-Res (240x240) Wi-Fi SoftAP Dashboard (http://192.168.4.1)
 *   2. USB Serial Stream for Python GUI (view_camera.py)
 * Board: LilyGO T-SIMCAM (ESP32-S3 Dual-Core @ 240MHz, 8MB PSRAM)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include "esp_camera.h"
#include "img_converters.h"

// TensorFlow Lite Micro Headers
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "mangosteen_model_data.h"
#include "color_matrix.h"

// ==============================================================================
// 1. PIN CONFIGURATION FOR LILYGO T-SIMCAM (ESP32-S3)
// ==============================================================================
#define PWR_ON_PIN       1

#define PWDN_GPIO_NUM   -1
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM   14
#define SIOD_GPIO_NUM    4
#define SIOC_GPIO_NUM    5

#define Y9_GPIO_NUM     15
#define Y8_GPIO_NUM     16
#define Y7_GPIO_NUM     17
#define Y6_GPIO_NUM     12
#define Y5_GPIO_NUM     10
#define Y4_GPIO_NUM      8
#define Y3_GPIO_NUM      9
#define Y2_GPIO_NUM     11
#define VSYNC_GPIO_NUM   6
#define HREF_GPIO_NUM    7
#define PCLK_GPIO_NUM   13

// ==============================================================================
// 2. MODEL GLOBALS
// ==============================================================================
static const char *kClassNames[3] = {
    "overripe",  // Class 0
    "ripe",      // Class 1
    "unripe"     // Class 2
};

constexpr size_t kTensorArenaSize = 2048 * 1024;
uint8_t *tensor_arena = nullptr;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// 3x3 Color Correction Matrix state (default ON)
static bool g_enable_ccm = true;

#define MODEL_INPUT_WIDTH    112
#define MODEL_INPUT_HEIGHT   112
#define PREVIEW_BUF_SIZE     (MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 2)

// Preview buffer for Serial Streaming (112x112 uint16 = 25088 bytes)
static uint8_t preview_buf[PREVIEW_BUF_SIZE];

// ==============================================================================
// 3. WEB SERVER & SOFTAP CONFIGURATION
// ==============================================================================
const char *ap_ssid = "Mangosonteen";
const char *ap_pass = "12345678";
WebServer server(80);

// Embedded Web Dashboard (HTML5 + CSS3 + Vanilla JS)
static const char PROGMEM INDEX_HTML[] = R"rawliteral(
<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Teachable Machine | Mangosteen Edge AI</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent-blue: #38bdf8;
      --btn-blue: #2563eb;
      --btn-blue-hover: #1d4ed8;
      --c-overripe: #f43f5e;
      --c-ripe: #a855f7;
      --c-unripe: #10b981;
      --track-bg: #0f172a;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
    body { background: var(--bg); color: var(--text-main); min-height: 100vh; padding: 16px; display: flex; flex-direction: column; align-items: center; }
    
    /* Top Header */
    .app-header { text-align: center; margin-bottom: 20px; max-width: 860px; width: 100%; }
    .tm-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 20px; font-size: 0.78rem; font-weight: 700; color: var(--accent-blue); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
    .app-header h1 { font-size: 1.5rem; font-weight: 800; color: #fff; letter-spacing: -0.02em; display: flex; align-items: center; justify-content: center; gap: 8px; }
    .app-header p { font-size: 0.85rem; color: var(--text-muted); margin-top: 4px; }

    /* Main Grid Layout */
    .main-grid { width: 100%; max-width: 880px; display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 740px) {
      .main-grid { grid-template-columns: 1fr; }
    }

    /* Teachable Machine Cards */
    .tm-card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 14px; overflow: hidden; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25); display: flex; flex-direction: column; }
    .tm-card-header { padding: 14px 18px; border-bottom: 1px solid var(--card-border); display: flex; justify-content: space-between; align-items: center; background: rgba(255, 255, 255, 0.02); }
    .tm-card-title { font-size: 0.95rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }
    .chip { padding: 3px 10px; border-radius: 12px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em; }
    .chip-webcam { background: #334155; color: #cbd5e1; }
    .chip-live { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }

    .tm-card-body { padding: 16px 18px; display: flex; flex-direction: column; gap: 14px; flex: 1; }

    /* Input Card: Viewfinder */
    .viewfinder-wrapper { position: relative; width: 100%; max-width: 280px; aspect-ratio: 1/1; background: #000; border-radius: 12px; overflow: hidden; margin: 0 auto; border: 2px solid var(--card-border); box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
    #cam-canvas { width: 100%; height: 100%; object-fit: cover; display: block; }
    
    /* Reticle Corner Brackets */
    .reticle-box { position: absolute; inset: 16px; pointer-events: none; }
    .corner { position: absolute; width: 22px; height: 22px; border-color: var(--accent-blue); border-style: solid; }
    .corner.tl { top: 0; left: 0; border-width: 3px 0 0 3px; border-top-left-radius: 6px; }
    .corner.tr { top: 0; right: 0; border-width: 3px 3px 0 0; border-top-right-radius: 6px; }
    .corner.bl { bottom: 0; left: 0; border-width: 0 0 3px 3px; border-bottom-left-radius: 6px; }
    .corner.br { bottom: 0; right: 0; border-width: 0 3px 3px 0; border-bottom-right-radius: 6px; }
    .reticle-target { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 130px; height: 130px; border: 1px dashed rgba(56, 189, 248, 0.5); border-radius: 50%; }
    .reticle-label { position: absolute; bottom: 8px; width: 100%; text-align: center; font-size: 0.72rem; color: var(--accent-blue); font-weight: 600; text-shadow: 0 1px 3px rgba(0,0,0,0.9); }

    /* Button & Toggle Controls */
    .btn-row { display: flex; gap: 10px; width: 100%; }
    .btn-tm { flex: 1; padding: 12px 8px; border-radius: 10px; border: none; font-size: 0.92rem; font-weight: 700; cursor: pointer; transition: all 0.2s ease; display: flex; align-items: center; justify-content: center; gap: 6px; }
    .btn-primary { background: var(--btn-blue); color: #fff; box-shadow: 0 2px 6px rgba(37, 99, 235, 0.3); }
    .btn-primary:active { background: var(--btn-blue-hover); transform: scale(0.98); }
    .btn-save { background: #059669; color: #fff; box-shadow: 0 2px 6px rgba(5, 150, 105, 0.3); }
    .btn-save:hover { background: #047857; }
    .btn-save:active { transform: scale(0.98); }
    .btn-secondary { background: #334155; color: var(--text-main); }
    .toast { position: fixed; bottom: 24px; background: #22c55e; color: #0f172a; padding: 10px 20px; border-radius: 24px; font-weight: 700; font-size: 0.85rem; box-shadow: 0 4px 16px rgba(0,0,0,0.5); opacity: 0; pointer-events: none; transition: opacity 0.3s ease; z-index: 1000; }
    .toast.show { opacity: 1; }
    
    .toggle-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: rgba(15, 23, 42, 0.6); border-radius: 10px; border: 1px solid var(--card-border); font-size: 0.88rem; font-weight: 600; }
    .switch { position: relative; display: inline-block; width: 44px; height: 24px; }
    .switch input { opacity: 0; width: 0; height: 0; }
    .slider { position: absolute; cursor: pointer; inset: 0; background-color: #475569; transition: .3s; border-radius: 24px; }
    .slider:before { position: absolute; content: ""; height: 18px; width: 18px; left: 3px; bottom: 3px; background-color: white; transition: .3s; border-radius: 50%; }
    input:checked + .slider { background-color: var(--accent-blue); }
    .telemetry-badges {
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      gap: 5px;
      margin-top: 10px;
    }
    .telemetry-pill {
      background: #e9ecef;
      color: #212529;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 0.83rem;
      font-weight: 500;
      padding: 4px 10px;
      border-radius: 6px;
      display: inline-flex;
      align-items: center;
      box-shadow: 0 1px 2px rgba(0,0,0,0.12);
      border: 1px solid rgba(0,0,0,0.06);
    }
    .telemetry-pill span {
      font-weight: 700;
      color: #0f172a;
    }

    /* Output Card: Teachable Machine Verdict & Multi-Class Progress Bars */
    .verdict-banner { padding: 14px; border-radius: 12px; background: rgba(15, 23, 42, 0.8); border: 1px solid var(--card-border); display: flex; align-items: center; justify-content: space-between; transition: all 0.25s ease; }
    .verdict-banner.overripe { border-color: var(--c-overripe); background: rgba(244, 63, 94, 0.12); }
    .verdict-banner.ripe { border-color: var(--c-ripe); background: rgba(168, 85, 247, 0.12); }
    .verdict-banner.unripe { border-color: var(--c-unripe); background: rgba(16, 185, 129, 0.12); }

    .verdict-title { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); font-weight: 700; margin-bottom: 2px; }
    .verdict-val { font-size: 1.35rem; font-weight: 800; color: #fff; }
    .verdict-conf { font-size: 1.35rem; font-weight: 800; color: var(--accent-blue); }

    /* Signature Teachable Machine Class Bars */
    .classes-container { display: flex; flex-direction: column; gap: 14px; margin-top: 4px; }
    .tm-class-item { display: flex; flex-direction: column; gap: 6px; padding: 10px 12px; border-radius: 10px; background: rgba(15, 23, 42, 0.5); border: 1px solid transparent; transition: all 0.2s ease; }
    .tm-class-item.winner { border-color: rgba(255, 255, 255, 0.25); background: rgba(255, 255, 255, 0.03); }

    .class-header { display: flex; justify-content: space-between; align-items: center; font-size: 0.88rem; font-weight: 700; }
    .class-name { display: flex; align-items: center; gap: 8px; }
    .class-dot { width: 10px; height: 10px; border-radius: 50%; }
    .class-pct { font-family: monospace; font-size: 0.95rem; font-weight: 700; }

    .meter-track { width: 100%; height: 26px; background: var(--track-bg); border-radius: 8px; overflow: hidden; position: relative; border: 1px solid rgba(255,255,255,0.06); }
    .meter-fill { height: 100%; width: 0%; border-radius: 7px; transition: width 0.22s cubic-bezier(0.4, 0, 0.2, 1); }
    
    .fill-overripe { background: linear-gradient(90deg, #f43f5e, #fb7185); }
    .fill-ripe { background: linear-gradient(90deg, #9333ea, #a855f7); }
    .fill-unripe { background: linear-gradient(90deg, #059669, #10b981); }

    /* Bottom Panel: Camera Tuner */
    .full-width { grid-column: 1 / -1; margin-top: 4px; }
    .clickable { cursor: pointer; user-select: none; }
    .clickable:hover { background: rgba(255, 255, 255, 0.04); }
    .tuning-body { display: none; padding: 16px 18px; flex-direction: column; gap: 14px; border-top: 1px solid var(--card-border); }
    .tuning-body.open { display: flex; }
    
    .tuner-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    @media (max-width: 600px) { .tuner-grid { grid-template-columns: 1fr; } }
    
    .control-group { display: flex; flex-direction: column; gap: 4px; }
    .control-label { display: flex; justify-content: space-between; font-size: 0.82rem; color: var(--text-muted); font-weight: 600; }
    .control-group input[type=range] { width: 100%; accent-color: var(--accent-blue); cursor: pointer; height: 6px; }
    .control-group select { width: 100%; padding: 8px 10px; background: #0f172a; border: 1px solid var(--card-border); border-radius: 8px; color: var(--text-main); font-size: 0.88rem; outline: none; }
    .btn-reset { padding: 10px; background: #334155; color: var(--text-main); border: none; border-radius: 8px; font-size: 0.85rem; font-weight: 700; cursor: pointer; transition: background 0.2s; align-self: flex-start; }
    .btn-reset:hover { background: #475569; }

    .app-footer { margin-top: 24px; font-size: 0.75rem; color: #64748b; text-align: center; }
  </style>
</head>
<body>
  <div class="app-header">
    <div class="tm-badge">Teachable Machine Edition</div>
    <h1>🥭 Mangosteen Ripeness Classifier</h1>
    <p>LilyGO T-SIMCAM (ESP32-S3) | MobileNetV2 INT8 Edge AI</p>
  </div>

  <div class="main-grid">
    <!-- PANEL 1: INPUT (WEBCAM) -->
    <div class="tm-card">
      <div class="tm-card-header">
        <div class="tm-card-title">📹 Input</div>
        <span class="chip chip-webcam">Webcam (240x240)</span>
      </div>
      <div class="tm-card-body">
        <div class="viewfinder-wrapper">
          <canvas id="cam-canvas" width="240" height="240"></canvas>
          <div class="reticle-box" id="reticle">
            <div class="corner tl"></div>
            <div class="corner tr"></div>
            <div class="corner bl"></div>
            <div class="corner br"></div>
            <div class="reticle-target"></div>
            <div class="reticle-label">จัดตำแหน่งมังคุดตรงกลาง</div>
          </div>
        </div>

        <div class="btn-row">
          <button class="btn-tm btn-primary" id="btn-action" onclick="onActionClick()">📸 Snap & Predict</button>
          <button class="btn-tm btn-save" id="btn-download" onclick="downloadSnapshot()">💾 บันทึกรูปภาพ</button>
        </div>

        <div class="toggle-row">
          <span>⚡ Continuous AI (Live)</span>
          <label class="switch">
            <input type="checkbox" id="chk-auto" onchange="onAutoToggle()">
            <span class="slider"></span>
          </label>
        </div>

        <div class="telemetry-badges">
          <div class="telemetry-pill">Latency: <span id="val-latency">...</span> ms</div>
          <div class="telemetry-pill">P95: <span id="val-p95">...</span> ms</div>
          <div class="telemetry-pill">Tensor Arena: <span id="val-arena">...</span> KB</div>
          <div class="telemetry-pill">FPS: <span id="val-fps">...</span></div>
        </div>
      </div>
    </div>

    <!-- PANEL 2: OUTPUT (TEACHABLE MACHINE BARS) -->
    <div class="tm-card">
      <div class="tm-card-header">
        <div class="tm-card-title">📊 Output</div>
        <span class="chip chip-live" id="chip-status">Waiting</span>
      </div>
      <div class="tm-card-body">
        <!-- Top Verdict Banner -->
        <div class="verdict-banner" id="verdict-banner">
          <div>
            <div class="verdict-title">Top Prediction</div>
            <div class="verdict-val" id="verdict-class">[ Ready ]</div>
          </div>
          <div class="verdict-conf" id="verdict-conf">--%</div>
        </div>

        <!-- Per-Class Meters -->
        <div class="classes-container">
          <!-- Class 0: Overripe -->
          <div class="tm-class-item" id="item-overripe">
            <div class="class-header">
              <span class="class-name">
                <span class="class-dot" style="background:var(--c-overripe);"></span>
                Overripe (สุกเกิน / งอม)
              </span>
              <span class="class-pct" id="val-overripe" style="color:var(--c-overripe);">0%</span>
            </div>
            <div class="meter-track">
              <div class="meter-fill fill-overripe" id="bar-overripe"></div>
            </div>
          </div>

          <!-- Class 1: Ripe -->
          <div class="tm-class-item" id="item-ripe">
            <div class="class-header">
              <span class="class-name">
                <span class="class-dot" style="background:var(--c-ripe);"></span>
                Ripe (สุกพร้อมทาน)
              </span>
              <span class="class-pct" id="val-ripe" style="color:var(--c-ripe);">0%</span>
            </div>
            <div class="meter-track">
              <div class="meter-fill fill-ripe" id="bar-ripe"></div>
            </div>
          </div>

          <!-- Class 2: Unripe -->
          <div class="tm-class-item" id="item-unripe">
            <div class="class-header">
              <span class="class-name">
                <span class="class-dot" style="background:var(--c-unripe);"></span>
                Unripe (ดิบ / เปลือกเขียว)
              </span>
              <span class="class-pct" id="val-unripe" style="color:var(--c-unripe);">0%</span>
            </div>
            <div class="meter-track">
              <div class="meter-fill fill-unripe" id="bar-unripe"></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- PANEL 3: CAMERA TUNER (COLLAPSIBLE) -->
    <div class="tm-card full-width">
      <div class="tm-card-header clickable" onclick="toggleTuning()">
        <div class="tm-card-title">⚙️ Camera Settings & DSP Tuner</div>
        <span id="tuning-arrow">▼</span>
      </div>
      <div class="tuning-body" id="tuning-panel">
        <div class="tuner-grid">
          <div class="control-group">
            <div class="control-label">
              <span>ความสว่าง (Brightness)</span>
              <span id="val-brightness">0</span>
            </div>
            <input type="range" id="rng-brightness" min="-2" max="2" value="0" step="1" oninput="onParamChange('brightness', this.value)">
          </div>

          <div class="control-group">
            <div class="control-label">
              <span>ความต่างระดับสี (Contrast)</span>
              <span id="val-contrast">+1</span>
            </div>
            <input type="range" id="rng-contrast" min="-2" max="2" value="1" step="1" oninput="onParamChange('contrast', this.value)">
          </div>

          <div class="control-group">
            <div class="control-label">
              <span>ความสดของสี (Saturation)</span>
              <span id="val-saturation">+1</span>
            </div>
            <input type="range" id="rng-saturation" min="-2" max="2" value="1" step="1" oninput="onParamChange('saturation', this.value)">
          </div>

          <div class="control-group">
            <div class="control-label">
              <span>ความคมชัด (Sharpness)</span>
              <span id="val-sharpness">+1</span>
            </div>
            <input type="range" id="rng-sharpness" min="-2" max="2" value="1" step="1" oninput="onParamChange('sharpness', this.value)">
          </div>

          <div class="control-group">
            <div class="control-label">
              <span>ชดเชยแสง (Exposure Bias)</span>
              <span id="val-ae_level">0</span>
            </div>
            <input type="range" id="rng-ae_level" min="-2" max="2" value="0" step="1" oninput="onParamChange('ae_level', this.value)">
          </div>

          <div class="control-group">
            <div class="control-label">
              <span>สมดุลแสงขาว (White Balance)</span>
            </div>
            <select id="sel-wb" onchange="onParamChange('wb_mode', this.value)">
              <option value="0" selected>0 - อัตโนมัติ (Auto AWB)</option>
              <option value="1">1 - กลางแจ้ง / แดดจัด (Sunny)</option>
              <option value="2">2 - มีเมฆ / ครึ้ม (Cloudy)</option>
              <option value="3">3 - แสงไฟนีออน / หลอด LED ออฟฟิศ (Office)</option>
              <option value="4">4 - แสงไฟโทนอุ่น / หลอดไส้ (Home Warm)</option>
            </select>
          </div>

          <div class="control-group">
            <div class="control-label">
              <span>🎨 ปรับเทียบสีจริง (Color Matrix CCM)</span>
              <span id="val-ccm" style="color:#10b981;font-weight:700">เปิด (ON)</span>
            </div>
            <select id="sel-ccm" onchange="onCcmChange(this.value)">
              <option value="1" selected>เปิดใช้งานสอบเทียบสี (Calibrated sRGB)</option>
              <option value="0">ปิด (ใช้สีดิบ OV2640 Raw)</option>
            </select>
          </div>
        </div>

        <button class="btn-reset" onclick="resetDefaults()">↺ คืนค่าเริ่มต้น (Reset Defaults)</button>
      </div>
    </div>
  </div>

  <div class="app-footer">
    Mangosonteen SoftAP (192.168.4.1) | หมุนเกลียวรอบเลนส์เพื่อปรับระยะโฟกัส
  </div>

  <script>
    const canvas = document.getElementById('cam-canvas');
    const ctx = canvas.getContext('2d');
    const reticle = document.getElementById('reticle');
    const btnAction = document.getElementById('btn-action');
    const chkAuto = document.getElementById('chk-auto');
    const lblTelemetry = document.getElementById('lbl-telemetry');
    const chipStatus = document.getElementById('chip-status');

    const verdictClass = document.getElementById('verdict-class');
    const verdictConf = document.getElementById('verdict-conf');
    const verdictBanner = document.getElementById('verdict-banner');

    const barOverripe = document.getElementById('bar-overripe');
    const valOverripe = document.getElementById('val-overripe');
    const itemOverripe = document.getElementById('item-overripe');

    const barRipe = document.getElementById('bar-ripe');
    const valRipe = document.getElementById('val-ripe');
    const itemRipe = document.getElementById('item-ripe');

    const barUnripe = document.getElementById('bar-unripe');
    const valUnripe = document.getElementById('val-unripe');
    const itemUnripe = document.getElementById('item-unripe');

    let isStreaming = true;
    let isPaused = false;
    let fpsCount = 0;
    let lastFpsTime = Date.now();
    let currentFps = "0";

    async function fetchFrame(predict = 0) {
      try {
        const url = "/snapshot?predict=" + predict + "&t=" + Date.now();
        const res = await fetch(url);
        if (!res.ok) throw new Error("Capture failed");

        const predClass = res.headers.get("X-Prediction");
        const conf = parseFloat(res.headers.get("X-Confidence") || "0");
        const latency = parseFloat(res.headers.get("X-Latency") || "0");
        const p95 = parseFloat(res.headers.get("X-P95") || "0");
        const arena = res.headers.get("X-Arena") || "--";
        const scores = (res.headers.get("X-Scores") || "0,0,0").split(",");

        const blob = await res.blob();
        const img = new Image();
        img.onload = () => {
          ctx.drawImage(img, 0, 0, 240, 240);
          URL.revokeObjectURL(img.src);

          fpsCount++;
          const now = Date.now();
          if (now - lastFpsTime >= 1000) {
            currentFps = (fpsCount * 1000 / (now - lastFpsTime)).toFixed(1);
            fpsCount = 0;
            lastFpsTime = now;
          }

          const valLat = document.getElementById('val-latency');
          const valP95 = document.getElementById('val-p95');
          const valArena = document.getElementById('val-arena');
          const valFps = document.getElementById('val-fps');

          if (valLat) valLat.textContent = (predict === 1 || chkAuto.checked) && latency > 0 ? latency.toFixed(1) : "...";
          if (valP95) valP95.textContent = p95 > 0 ? p95.toFixed(1) : "...";
          if (valArena) valArena.textContent = arena !== "--" ? arena : "...";
          if (valFps) valFps.textContent = currentFps;

          if (predict === 1 || chkAuto.checked) {
            updateUI(predClass, conf, scores);
          }

          if (isStreaming && !isPaused) {
            const nextPredict = chkAuto.checked ? 1 : 0;
            setTimeout(() => fetchFrame(nextPredict), nextPredict ? 100 : 50);
          }
        };
        img.src = URL.createObjectURL(blob);
      } catch (err) {
        if (isStreaming && !isPaused) {
          setTimeout(() => fetchFrame(chkAuto.checked ? 1 : 0), 500);
        }
      }
    }

    function updateUI(cls, conf, scores) {
      if (!cls) return;
      verdictClass.textContent = cls.toUpperCase();
      verdictConf.textContent = conf.toFixed(1) + "%";
      verdictBanner.className = "verdict-banner " + cls;

      chipStatus.textContent = "● Live (" + conf.toFixed(0) + "%)";

      if (scores && scores.length === 3) {
        const pOver = Math.max(0, Math.min(100, parseFloat(scores[0]) || 0));
        const pRipe = Math.max(0, Math.min(100, parseFloat(scores[1]) || 0));
        const pUnripe = Math.max(0, Math.min(100, parseFloat(scores[2]) || 0));

        valOverripe.textContent = pOver.toFixed(1) + "%";
        barOverripe.style.width = pOver + "%";

        valRipe.textContent = pRipe.toFixed(1) + "%";
        barRipe.style.width = pRipe + "%";

        valUnripe.textContent = pUnripe.toFixed(1) + "%";
        barUnripe.style.width = pUnripe + "%";

        itemOverripe.classList.toggle('winner', cls === 'overripe');
        itemRipe.classList.toggle('winner', cls === 'ripe');
        itemUnripe.classList.toggle('winner', cls === 'unripe');
      }
    }

    function onActionClick() {
      if (chkAuto.checked) return;

      if (!isPaused) {
        isPaused = true;
        btnAction.textContent = "🔄 Live Preview";
        btnAction.className = "btn-tm btn-secondary";
        reticle.style.display = "none";
        fetchFrame(1);
      } else {
        isPaused = false;
        btnAction.textContent = "📸 Snap & Predict";
        btnAction.className = "btn-tm btn-primary";
        reticle.style.display = "block";
        verdictClass.textContent = "[ Ready ]";
        verdictConf.textContent = "--%";
        verdictBanner.className = "verdict-banner";
        fetchFrame(0);
      }
    }

    function onAutoToggle() {
      if (chkAuto.checked) {
        isPaused = false;
        btnAction.textContent = "⚡ Running Continuous AI...";
        btnAction.disabled = true;
        btnAction.className = "btn-tm btn-secondary";
        reticle.style.display = "none";
      } else {
        btnAction.textContent = "📸 Snap & Predict";
        btnAction.disabled = false;
        btnAction.className = "btn-tm btn-primary";
        reticle.style.display = "block";
      }
    }

    function toggleTuning() {
      const panel = document.getElementById('tuning-panel');
      const arrow = document.getElementById('tuning-arrow');
      panel.classList.toggle('open');
      arrow.textContent = panel.classList.contains('open') ? '▲' : '▼';
    }

    const tunerDebounce = {};
    function onParamChange(param, val) {
      const valLabel = document.getElementById('val-' + param);
      if (valLabel) {
        valLabel.textContent = (parseInt(val) > 0 ? '+' : '') + val;
      }
      clearTimeout(tunerDebounce[param]);
      tunerDebounce[param] = setTimeout(() => {
        fetch('/control?var=' + param + '&val=' + val).catch(() => {});
      }, 80);
    }

    function onCcmChange(val) {
      const valLabel = document.getElementById('val-ccm');
      if (valLabel) {
        valLabel.textContent = val === "1" ? "เปิด (ON)" : "ปิด (OFF)";
        valLabel.style.color = val === "1" ? "#10b981" : "var(--text-muted)";
      }
      fetch('/control?var=ccm&val=' + val).catch(() => {});
    }

    function resetDefaults() {
      const defaults = {
        brightness: 0,
        contrast: 1,
        saturation: 1,
        sharpness: 1,
        ae_level: 0,
        wb_mode: 0
      };
      for (const [key, val] of Object.entries(defaults)) {
        const input = document.getElementById(key === 'wb_mode' ? 'sel-wb' : ('rng-' + key));
        if (input) {
          input.value = val;
          const valLabel = document.getElementById('val-' + key);
          if (valLabel) valLabel.textContent = (val > 0 ? '+' : '') + val;
        }
        fetch('/control?var=' + key + '&val=' + val).catch(() => {});
      }
      const ccmSel = document.getElementById('sel-ccm');
      if (ccmSel) {
        ccmSel.value = "1";
        const valLabel = document.getElementById('val-ccm');
        if (valLabel) {
          valLabel.textContent = "เปิด (ON)";
          valLabel.style.color = "#10b981";
        }
      }
      fetch('/control?var=ccm&val=1').catch(() => {});
    }

    function downloadSnapshot() {
      try {
        const link = document.createElement('a');
        const now = new Date();
        const timeStr = now.getFullYear().toString() +
          String(now.getMonth() + 1).padStart(2, '0') +
          String(now.getDate()).padStart(2, '0') + "_" +
          String(now.getHours()).padStart(2, '0') +
          String(now.getMinutes()).padStart(2, '0') +
          String(now.getSeconds()).padStart(2, '0');

        let clsName = "sample";
        if (verdictClass && verdictClass.textContent && !verdictClass.textContent.includes('[')) {
          clsName = verdictClass.textContent.trim().toLowerCase();
        }
        link.download = "mangosteen_" + clsName + "_" + timeStr + ".jpg";
        link.href = canvas.toDataURL("image/jpeg", 0.95);
        link.click();

        const toast = document.getElementById('toast');
        if (toast) {
          toast.textContent = "💾 บันทึกภาพ (" + link.download + ") แล้ว!";
          toast.classList.add('show');
          setTimeout(() => toast.classList.remove('show'), 2200);
        }
      } catch (err) {
        alert("ไม่สามารถบันทึกภาพได้: " + err.message);
      }
    }

    fetchFrame(0);
  </script>
  <div class="toast" id="toast">💾 บันทึกรูปภาพเรียบร้อยแล้ว!</div>
</body>
</html>
)rawliteral";

// ==============================================================================
// 4. CAMERA INITIALIZATION (240x240 RGB565 DUAL BUFFER)
// ==============================================================================
bool initCamera() {
    pinMode(PWR_ON_PIN, OUTPUT);
    digitalWrite(PWR_ON_PIN, HIGH);
    delay(100);

    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size = FRAMESIZE_240X240;   // 240x240 Sharp Square Frame
    config.jpeg_quality = 12;
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
    config.fb_location = CAMERA_FB_IN_PSRAM;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("[Camera] Init Failed with error 0x%x\r\n", err);
        return false;
    }

    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        s->set_brightness(s, 0);
        s->set_contrast(s, 1);       // Rich contrast
        s->set_saturation(s, 1);     // Vibrant colors
        s->set_sharpness(s, 1);      // Edge sharpness
        s->set_whitebal(s, 1);       // Auto White Balance
        s->set_awb_gain(s, 1);
        s->set_exposure_ctrl(s, 1);  // Auto Exposure
    }

    Serial.print("[Camera] Dual buffer OV2640 initialized at 240x240.\r\n");
    return true;
}

// ==============================================================================
// 5. TENSORFLOW LITE MICRO INITIALIZATION
// ==============================================================================
bool initTFLite() {
    if (g_mangosteen_model_data_len == 0) {
        Serial.print("[TFLite] ERROR: No model loaded!\r\n");
        return false;
    }
    Serial.printf("[TFLite] Loading model (%u bytes)...\r\n", g_mangosteen_model_data_len);
    model = tflite::GetModel(g_mangosteen_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("[TFLite] Schema mismatch! Model: %d, Runtime: %d\r\n",
                      model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    if (psramFound()) {
        tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);
    }
    if (!tensor_arena) {
        tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
    }
    if (!tensor_arena) {
        Serial.print("[TFLite] ERROR: Failed to allocate Tensor Arena!\r\n");
        return false;
    }

    static tflite::MicroErrorReporter micro_error_reporter;
    static tflite::ErrorReporter* error_reporter = &micro_error_reporter;
    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.print("[TFLite] ERROR: AllocateTensors() failed!\r\n");
        return false;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    Serial.printf("[TFLite] Ready! Arena: %u / %u bytes\r\n",
                  interpreter->arena_used_bytes(), (unsigned)kTensorArenaSize);
    return true;
}

// ==============================================================================
// 5.5 TELEMETRY & PROFILING (LATENCY, P95, TENSOR ARENA)
// ==============================================================================
#define LATENCY_HISTORY_MAX 50
static float g_latency_history[LATENCY_HISTORY_MAX];
static int g_latency_count = 0;
static int g_latency_head = 0;

void recordLatency(float lat_ms) {
    if (lat_ms <= 0.0f) return;
    g_latency_history[g_latency_head] = lat_ms;
    g_latency_head = (g_latency_head + 1) % LATENCY_HISTORY_MAX;
    if (g_latency_count < LATENCY_HISTORY_MAX) {
        g_latency_count++;
    }
}

float getP95Latency() {
    if (g_latency_count == 0) return 0.0f;
    float sorted[LATENCY_HISTORY_MAX];
    for (int i = 0; i < g_latency_count; i++) {
        sorted[i] = g_latency_history[i];
    }
    for (int i = 1; i < g_latency_count; i++) {
        float key = sorted[i];
        int j = i - 1;
        while (j >= 0 && sorted[j] > key) {
            sorted[j + 1] = sorted[j];
            j--;
        }
        sorted[j + 1] = key;
    }
    int idx = (int)(0.95f * (g_latency_count - 1) + 0.5f);
    if (idx >= g_latency_count) idx = g_latency_count - 1;
    return sorted[idx];
}

size_t getTensorArenaUsedKB() {
    if (!interpreter) return (kTensorArenaSize / 1024);
    size_t used = interpreter->arena_used_bytes();
    if (used == 0) return (kTensorArenaSize / 1024);
    return (used + 1023) / 1024;
}

// ==============================================================================
// 6. INFERENCE & DOWNSAMPLING HELPER
// ==============================================================================
void runInferenceOnFrame(camera_fb_t *fb, int &best_class, float &max_score, float &latency_ms, float scores[3]) {
    uint16_t *pixels = (uint16_t*)fb->buf;
    int8_t *input_buf = input->data.int8;
    int idx = 0;
    int p_idx = 0;

    // Downsample 240x240 -> 112x112 with byte-swap and INT8 scaling
    for (int y = 0; y < MODEL_INPUT_HEIGHT; y++) {
        int src_y = (y * 240) / MODEL_INPUT_HEIGHT;
        int row_offset = src_y * 240;
        for (int x = 0; x < MODEL_INPUT_WIDTH; x++) {
            int src_x = (x * 240) / MODEL_INPUT_WIDTH;
            uint16_t p = pixels[row_offset + src_x];
            p = (p >> 8) | (p << 8); // Swap endianness for OV2640 DMA

            uint8_t r = ((p >> 11) & 0x1F) << 3;
            uint8_t g = ((p >> 5) & 0x3F) << 2;
            uint8_t b = (p & 0x1F) << 3;

            // Apply 3x3 Color Correction Matrix (Smartphone Calibration)
            uint8_t r_cal = r, g_cal = g, b_cal = b;
            if (g_enable_ccm) {
                applyColorCorrection(r, g, b, r_cal, g_cal, b_cal);
            }

            // Save preview in big-endian format for Python viewer (calibrated if CCM enabled)
            uint16_t p_out = g_enable_ccm ? 
                (((uint16_t)(r_cal & 0xF8) << 8) | ((uint16_t)(g_cal & 0xFC) << 3) | (b_cal >> 3)) : p;
            preview_buf[p_idx++] = (uint8_t)(p_out >> 8);
            preview_buf[p_idx++] = (uint8_t)(p_out & 0xFF);

            input_buf[idx++] = (int8_t)((int16_t)r_cal - 128);
            input_buf[idx++] = (int8_t)((int16_t)g_cal - 128);
            input_buf[idx++] = (int8_t)((int16_t)b_cal - 128);
        }
    }

    int64_t t_start = esp_timer_get_time();
    TfLiteStatus status = interpreter->Invoke();
    int64_t t_end = esp_timer_get_time();

    best_class = 0;
    max_score = -1.0f;
    if (status == kTfLiteOk) {
        latency_ms = (float)(t_end - t_start) / 1000.0f;
        recordLatency(latency_ms);
        for (int i = 0; i < 3; i++) {
            scores[i] = (static_cast<float>(output->data.int8[i]) - output->params.zero_point) * output->params.scale;
            if (scores[i] > max_score) {
                max_score = scores[i];
                best_class = i;
            }
        }
    }
}

// ==============================================================================
// 7. HTTP REQUEST HANDLERS
// ==============================================================================
void handleRoot() {
    server.send_P(200, "text/html", INDEX_HTML);
}

void handleSnapshot() {
    bool do_predict = server.hasArg("predict") && server.arg("predict") == "1";

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        server.send(500, "text/plain", "Camera grab failed");
        return;
    }

    float latency_ms = 0.0f;
    int best_class = 0;
    float max_score = -1.0f;
    float scores[3] = {0};

    if (do_predict && interpreter) {
        runInferenceOnFrame(fb, best_class, max_score, latency_ms, scores);
    }

    if (g_enable_ccm) {
        applyColorCorrectionToFB(fb);
    }

    // Convert 240x240 RGB565 to sharp JPEG
    uint8_t *jpg_buf = NULL;
    size_t jpg_len = 0;
    bool ok = fmt2jpg((uint8_t*)fb->buf, fb->len, fb->width, fb->height, PIXFORMAT_RGB565, 80, &jpg_buf, &jpg_len);
    esp_camera_fb_return(fb);

    if (!ok || !jpg_buf) {
        server.send(500, "text/plain", "JPEG convert failed");
        return;
    }

    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.sendHeader("Access-Control-Expose-Headers", "X-Prediction, X-Confidence, X-Latency, X-P95, X-Arena, X-Scores");
    server.sendHeader("X-Prediction", kClassNames[best_class]);
    server.sendHeader("X-Confidence", String(max_score * 100.0f, 1));
    server.sendHeader("X-Latency", String(latency_ms, 1));
    server.sendHeader("X-P95", String(getP95Latency(), 1));
    server.sendHeader("X-Arena", String((unsigned int)getTensorArenaUsedKB()));
    server.sendHeader("X-Scores", String(scores[0] * 100.0f, 1) + "," + String(scores[1] * 100.0f, 1) + "," + String(scores[2] * 100.0f, 1));

    server.setContentLength(jpg_len);
    server.send(200, "image/jpeg", "");
    WiFiClient client = server.client();
    client.write(jpg_buf, jpg_len);
    client.flush();

    free(jpg_buf);
}

void handleControl() {
    if (!server.hasArg("var") || !server.hasArg("val")) {
        server.send(400, "text/plain", "Missing args");
        return;
    }
    String var = server.arg("var");
    int val = server.arg("val").toInt();

    if (var == "ccm") {
        g_enable_ccm = (val != 0);
        server.sendHeader("Access-Control-Allow-Origin", "*");
        server.send(200, "text/plain", "OK");
        return;
    }

    sensor_t *s = esp_camera_sensor_get();
    if (!s) {
        server.send(500, "text/plain", "Camera sensor not ready");
        return;
    }

    int res = 0;
    if (var == "brightness") {
        res = s->set_brightness(s, val);
    } else if (var == "contrast") {
        res = s->set_contrast(s, val);
    } else if (var == "saturation") {
        res = s->set_saturation(s, val);
    } else if (var == "sharpness") {
        res = s->set_sharpness(s, val);
    } else if (var == "ae_level") {
        res = s->set_ae_level(s, val);
    } else if (var == "wb_mode") {
        res = s->set_wb_mode(s, val);
    } else if (var == "whitebal") {
        res = s->set_whitebal(s, val);
    } else if (var == "lenc") {
        res = s->set_lenc(s, val);
    } else {
        server.send(404, "text/plain", "Unknown parameter");
        return;
    }

    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(res == 0 ? 200 : 500, "text/plain", res == 0 ? "OK" : "ERROR");
}

// ==============================================================================
// 8. SERIAL STREAMING HELPER (FOR PYTHON DESKTOP GUI)
// ==============================================================================
void streamSerialFrame() {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) return;

    int best_class = 0;
    float max_score = -1.0f;
    float latency_ms = 0.0f;
    float scores[3] = {0};

    runInferenceOnFrame(fb, best_class, max_score, latency_ms, scores);
    esp_camera_fb_return(fb);

    Serial.println("---IMG_START---");
    Serial.printf("LEN:%d\n", PREVIEW_BUF_SIZE);
    Serial.printf("CLASS:%s\n", kClassNames[best_class]);
    Serial.printf("CONF:%.1f\n", max_score * 100.0f);
    Serial.printf("LATENCY:%.1f\n", latency_ms);
    Serial.printf("P95:%.1f\n", getP95Latency());
    Serial.printf("ARENA:%u\n", (unsigned int)getTensorArenaUsedKB());
    Serial.printf("SCORES:%.1f,%.1f,%.1f\n", scores[0] * 100.0f, scores[1] * 100.0f, scores[2] * 100.0f);
    Serial.println("---PAYLOAD---");
    Serial.write(preview_buf, PREVIEW_BUF_SIZE);
    Serial.println("\n---IMG_END---");
}

// ==============================================================================
// 9. SETUP & MAIN LOOP
// ==============================================================================
void setup() {
    Serial.begin(115200);
    delay(2000);

    Serial.print("\r\n=======================================================\r\n");
    Serial.print(" LilyGO T-SIMCAM: Mangosteen Edge AI (SoftAP + Serial)\r\n");
    Serial.print("=======================================================\r\n");

    if (!initCamera() || !initTFLite()) {
        Serial.print("[HALT] Init failed.\r\n");
        while (1) delay(1000);
    }

    // Setup Wi-Fi SoftAP
    IPAddress local_ip(192, 168, 4, 1);
    IPAddress gateway(192, 168, 4, 1);
    IPAddress subnet(255, 255, 255, 0);
    WiFi.softAPConfig(local_ip, gateway, subnet);
    WiFi.softAP(ap_ssid, ap_pass);

    Serial.print("\r\n[Wi-Fi] SoftAP Started!\r\n");
    Serial.printf("[Wi-Fi] SSID: %s\r\n", ap_ssid);
    Serial.printf("[Wi-Fi] Password: %s\r\n", ap_pass);
    Serial.printf("[Wi-Fi] Web Dashboard URL: http://%s\r\n\r\n", WiFi.softAPIP().toString().c_str());

    server.on("/", HTTP_GET, handleRoot);
    server.on("/snapshot", HTTP_GET, handleSnapshot);
    server.on("/control", HTTP_GET, handleControl);
    server.begin();
    Serial.print("[Web] HTTP Server listening on port 80.\r\n");
}

void loop() {
    server.handleClient();

    // Serial streaming support for view_camera.py (default false to give Wi-Fi 100% priority)
    static unsigned long last_serial_frame = 0;
    static bool serial_streaming = false;

    // Allow toggling serial stream via serial character commands
    while (Serial.available() > 0) {
        char cmd = (char)Serial.read();
        if (cmd == 's' || cmd == 'S') {
            serial_streaming = !serial_streaming;
        } else if (cmd == ' ' || cmd == 'p' || cmd == 'P') {
            streamSerialFrame();
        }
    }

    if (Serial && serial_streaming && (millis() - last_serial_frame > 150)) {
        last_serial_frame = millis();
        streamSerialFrame();
    }
}
