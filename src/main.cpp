/**
 * Mangosteen Ripeness Classifier with High-Res (240x240) Wi-Fi Dashboard
 * Board: LilyGO T-SIMCAM (ESP32-S3)
 * Access: Connect Wi-Fi "Mangosteen-AI" (pass: 12345678), browse http://192.168.4.1
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

// ==============================================================================
// 3. WEB SERVER & SOFTAP CONFIGURATION
// ==============================================================================
const char *ap_ssid = "Mangosteen-AI";
const char *ap_pass = "12345678";
WebServer server(80);

// Embedded Web Dashboard (HTML5 + CSS3 + Vanilla JS)
static const char PROGMEM INDEX_HTML[] = R"rawliteral(
<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Mangosteen Edge AI (240x240)</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --subtext: #94a3b8;
      --primary: #38bdf8;
      --primary-hover: #0284c7;
      --ripe: #22c55e;
      --unripe: #f59e0b;
      --overripe: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); display: flex; flex-direction: column; align-items: center; min-height: 100vh; padding: 16px; }
    .header { text-align: center; margin-bottom: 16px; }
    .header h1 { font-size: 1.4rem; display: flex; align-items: center; justify-content: center; gap: 8px; }
    .header p { font-size: 0.85rem; color: var(--subtext); margin-top: 4px; }
    .container { width: 100%; max-width: 440px; display: flex; flex-direction: column; align-items: center; gap: 14px; }
    .camera-box { position: relative; width: 320px; height: 320px; background: #000; border-radius: 16px; overflow: hidden; border: 2px solid var(--border); box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5); }
    #cam-canvas { width: 100%; height: 100%; object-fit: cover; }
    .reticle { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 200px; height: 200px; border: 2px dashed rgba(56, 189, 248, 0.85); border-radius: 24px; pointer-events: none; }
    .reticle-label { position: absolute; bottom: 8px; width: 100%; text-align: center; font-size: 0.75rem; color: var(--primary); font-weight: bold; text-shadow: 0 1px 4px rgba(0,0,0,0.9); }
    .btn-row { width: 100%; display: flex; gap: 10px; }
    button { flex: 1; padding: 12px; border-radius: 12px; border: none; font-size: 1rem; font-weight: bold; cursor: pointer; transition: all 0.2s ease; }
    .btn-predict { background: var(--primary); color: #0f172a; }
    .btn-predict:active { background: var(--primary-hover); transform: scale(0.98); }
    .btn-resume { background: #334155; color: var(--text); }
    .toggle-row { width: 100%; display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: var(--card); border-radius: 10px; font-size: 0.9rem; }
    .switch { position: relative; display: inline-block; width: 44px; height: 24px; }
    .switch input { opacity: 0; width: 0; height: 0; }
    .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #475569; transition: .3s; border-radius: 24px; }
    .slider:before { position: absolute; content: ""; height: 18px; width: 18px; left: 3px; bottom: 3px; background-color: white; transition: .3s; border-radius: 50%; }
    input:checked + .slider { background-color: var(--primary); }
    input:checked + .slider:before { transform: translateX(20px); }
    .card { width: 100%; background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 16px; }
    .badge-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .badge { font-size: 1.25rem; font-weight: bold; padding: 4px 12px; border-radius: 8px; background: #334155; }
    .badge.ripe { color: #fff; background: var(--ripe); }
    .badge.unripe { color: #000; background: var(--unripe); }
    .badge.overripe { color: #fff; background: var(--overripe); }
    .conf-text { font-size: 1.25rem; font-weight: bold; color: var(--primary); }
    .bar-bg { width: 100%; height: 10px; background: #334155; border-radius: 6px; overflow: hidden; margin-bottom: 12px; }
    .bar-fill { height: 100%; width: 0%; background: var(--primary); transition: width 0.3s ease; }
    .details { font-size: 0.82rem; color: var(--subtext); display: flex; flex-direction: column; gap: 4px; font-family: monospace; }
    .footer { margin-top: 16px; font-size: 0.75rem; color: #64748b; text-align: center; }
  </style>
</head>
<body>
  <div class="header">
    <h1>🍃 Mangosteen AI Inspector</h1>
    <p>LilyGO T-SIMCAM (ESP32-S3) | MobileNetV2 Edge AI</p>
  </div>

  <div class="container">
    <div class="camera-box">
      <canvas id="cam-canvas" width="240" height="240"></canvas>
      <div class="reticle" id="reticle">
        <div class="reticle-label">จัดตำแหน่งมังคุดให้อยู่ในกรอบ</div>
      </div>
    </div>

    <div class="btn-row">
      <button class="btn-predict" id="btn-action" onclick="onActionClick()">📸 ถ่ายภาพ & Predict</button>
    </div>

    <div class="toggle-row">
      <span>⚡ วิเคราะห์สดต่อเนื่อง (Auto-AI)</span>
      <label class="switch">
        <input type="checkbox" id="chk-auto" onchange="onAutoToggle()">
        <span class="slider"></span>
      </label>
    </div>

    <div class="card">
      <div class="badge-row">
        <span class="badge" id="lbl-class">[ รอถ่ายภาพ ]</span>
        <span class="conf-text" id="lbl-conf">--%</span>
      </div>
      <div class="bar-bg">
        <div class="bar-fill" id="bar-fill"></div>
      </div>
      <div class="details">
        <div id="lbl-latency">⏱️ Inference: -- ms | Stream: -- FPS</div>
        <div id="lbl-dist">📊 overripe: 0% | ripe: 0% | unripe: 0%</div>
      </div>
    </div>
  </div>

  <div class="footer">
    เชื่อมต่อ: Mangosteen-AI (192.168.4.1) | หมุนเกลียวรอบเลนส์เพื่อปรับโฟกัส
  </div>

  <script>
    const canvas = document.getElementById('cam-canvas');
    const ctx = canvas.getContext('2d');
    const reticle = document.getElementById('reticle');
    const btnAction = document.getElementById('btn-action');
    const chkAuto = document.getElementById('chk-auto');
    const lblClass = document.getElementById('lbl-class');
    const lblConf = document.getElementById('lbl-conf');
    const barFill = document.getElementById('bar-fill');
    const lblLatency = document.getElementById('lbl-latency');
    const lblDist = document.getElementById('lbl-dist');

    let isStreaming = true;
    let isPaused = false;
    let fpsCount = 0;
    let lastFpsTime = Date.now();
    let currentFps = "0";

    async function fetchFrame(predict = 0) {
      try {
        const url = `/snapshot?predict=${predict}&t=${Date.now()}`;
        const res = await fetch(url);
        if (!res.ok) throw new Error("Capture failed");

        const predClass = res.headers.get("X-Prediction");
        const conf = parseFloat(res.headers.get("X-Confidence") || "0");
        const latency = parseFloat(res.headers.get("X-Latency") || "0");
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

          if (predict === 1 || chkAuto.checked) {
            updateUI(predClass, conf, latency, scores);
          } else {
            lblLatency.textContent = `⏱️ Stream: ${currentFps} FPS | 240x240 คมชัด`;
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

    function updateUI(cls, conf, latency, scores) {
      if (!cls) return;
      lblClass.textContent = `🎯 ${cls.toUpperCase()}`;
      lblClass.className = `badge ${cls}`;
      lblConf.textContent = `${conf.toFixed(1)}%`;
      barFill.style.width = `${conf}%`;
      lblLatency.textContent = `⏱️ Inference: ${latency.toFixed(1)} ms | Stream: ${currentFps} FPS`;
      if (scores.length === 3) {
        lblDist.textContent = `📊 overripe: ${scores[0]}% | ripe: ${scores[1]}% | unripe: ${scores[2]}%`;
      }
    }

    function onActionClick() {
      if (chkAuto.checked) return;

      if (!isPaused) {
        isPaused = true;
        btnAction.textContent = "🔄 กลับไปดูภาพสด (Live Preview)";
        btnAction.className = "btn-resume";
        reticle.style.display = "none";
        fetchFrame(1);
      } else {
        isPaused = false;
        btnAction.textContent = "📸 ถ่ายภาพ & Predict";
        btnAction.className = "btn-predict";
        reticle.style.display = "block";
        lblClass.textContent = "[ รอถ่ายภาพ ]";
        lblClass.className = "badge";
        lblConf.textContent = "--%";
        barFill.style.width = "0%";
        fetchFrame(0);
      }
    }

    function onAutoToggle() {
      if (chkAuto.checked) {
        isPaused = false;
        btnAction.textContent = "⚡ กำลังวิเคราะห์สดตลอดเวลา...";
        btnAction.disabled = true;
        btnAction.className = "btn-resume";
        reticle.style.display = "none";
      } else {
        btnAction.textContent = "📸 ถ่ายภาพ & Predict";
        btnAction.disabled = false;
        btnAction.className = "btn-predict";
        reticle.style.display = "block";
      }
    }

    fetchFrame(0);
  </script>
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
    config.frame_size = FRAMESIZE_240X240;   // 240x240 Sharp Square Frame (6.25x pixels)
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
        Serial.print("[TFLite] ERROR: No model loaded! Please train your model following TRAINING_GUIDE.md\r\n");
        return false;
    }
    Serial.print("[TFLite] Loading model...\r\n");
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
// 6. HTTP REQUEST HANDLERS
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

    // If predict requested, downsample 240x240 to 96x96 INT8 for MobileNetV2
    if (do_predict && interpreter) {
        uint16_t *pixels = (uint16_t*)fb->buf;
        int8_t *input_buf = input->data.int8;
        int idx = 0;

        // Downsample 240x240 -> 96x96 (2.5x step)
        for (int y = 0; y < 96; y++) {
            int src_y = (y * 240) / 96;
            int row_offset = src_y * 240;
            for (int x = 0; x < 96; x++) {
                int src_x = (x * 240) / 96;
                uint16_t p = pixels[row_offset + src_x];
                p = (p >> 8) | (p << 8);

                uint8_t r = ((p >> 11) & 0x1F) << 3;
                uint8_t g = ((p >> 5) & 0x3F) << 2;
                uint8_t b = (p & 0x1F) << 3;

                input_buf[idx++] = (int8_t)((int16_t)r - 128);
                input_buf[idx++] = (int8_t)((int16_t)g - 128);
                input_buf[idx++] = (int8_t)((int16_t)b - 128);
            }
        }

        int64_t t_start = esp_timer_get_time();
        TfLiteStatus status = interpreter->Invoke();
        int64_t t_end = esp_timer_get_time();

        if (status == kTfLiteOk) {
            latency_ms = (float)(t_end - t_start) / 1000.0f;
            for (int i = 0; i < 3; i++) {
                scores[i] = (static_cast<float>(output->data.int8[i]) - output->params.zero_point) * output->params.scale;
                if (scores[i] > max_score) {
                    max_score = scores[i];
                    best_class = i;
                }
            }
        }
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
    server.sendHeader("Access-Control-Expose-Headers", "X-Prediction, X-Confidence, X-Latency, X-Scores");
    server.sendHeader("X-Prediction", kClassNames[best_class]);
    server.sendHeader("X-Confidence", String(max_score * 100.0f, 1));
    server.sendHeader("X-Latency", String(latency_ms, 1));
    server.sendHeader("X-Scores", String(scores[0] * 100.0f, 1) + "," + String(scores[1] * 100.0f, 1) + "," + String(scores[2] * 100.0f, 1));

    server.setContentLength(jpg_len);
    server.send(200, "image/jpeg", "");
    WiFiClient client = server.client();
    client.write(jpg_buf, jpg_len);

    free(jpg_buf);
}

// ==============================================================================
// 7. SETUP & MAIN LOOP
// ==============================================================================
void setup() {
    Serial.begin(115200);
    delay(2000);

    Serial.print("\r\n=======================================================\r\n");
    Serial.print(" LilyGO T-SIMCAM: 240x240 Wi-Fi SoftAP Inspector\r\n");
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
    server.begin();
    Serial.print("[Web] HTTP Server listening on port 80.\r\n");
}

void loop() {
    server.handleClient();
}
