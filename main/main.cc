/**
 * Mangosteen Ripeness Classifier for Edge AI (ESP32-S3)
 * Target: Basic CNN INT8 (96x96x3)
 * Framework: ESP-IDF + esp-tflite-micro (TensorFlow Lite Micro)
 */

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include "esp_log.h"
#include "esp_system.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// TensorFlow Lite Micro Headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model Data Header
#include "mangosteen_model_data.h"

static const char *TAG = "MANGOSTEEN_EDGE";

// Class Mapping (Alphabetical ordering from tf.keras directory loading)
// DO NOT CHANGE ORDER:
// 0 = overripe
// 1 = ripe
// 2 = unripe
static const char *kClassNames[3] = {
    "overripe",
    "ripe",
    "unripe"
};

// Expected Model Input Specifications
constexpr int kExpectedWidth = 96;
constexpr int kExpectedHeight = 96;
constexpr int kExpectedChannels = 3;
constexpr int kExpectedInputSize = kExpectedWidth * kExpectedHeight * kExpectedChannels; // 27,648 bytes

// Tensor Arena Size (350 KB buffer for model weights metadata and layer activations)
constexpr size_t kTensorArenaSize = 350 * 1024;

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "===================================================");
    ESP_LOGI(TAG, " Mangosteen Ripeness Classifier - Edge AI (ESP32-S3)");
    ESP_LOGI(TAG, " Model: Basic CNN INT8 (96x96 RGB)");
    ESP_LOGI(TAG, " Classes: 0=overripe, 1=ripe, 2=unripe");
    ESP_LOGI(TAG, "===================================================");

    // 1. Check System Memory
    size_t internal_free = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    size_t spiram_free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "Internal RAM free : %u bytes (%.2f KB)", internal_free, internal_free / 1024.0f);
    ESP_LOGI(TAG, "SPIRAM / PSRAM free: %u bytes (%.2f KB)", spiram_free, spiram_free / 1024.0f);

    // 2. Validate Model Data Header & Schema
    ESP_LOGI(TAG, "Model byte array size: %u bytes (%.2f KB)", 
             g_mangosteen_model_data_len, g_mangosteen_model_data_len / 1024.0f);

    const tflite::Model *model = tflite::GetModel(g_mangosteen_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema mismatch! Model schema: %" PRIu32 ", Supported schema: %d",
                 model->version(), TFLITE_SCHEMA_VERSION);
        ESP_LOGE(TAG, "Please re-convert the model using convert_tflite_to_c.py");
        return;
    }
    ESP_LOGI(TAG, "Model schema check PASSED (version %" PRIu32 ")", model->version());

    // 3. Allocate Tensor Arena (Prefer PSRAM if available, fallback to Internal RAM)
    uint8_t *tensor_arena = nullptr;
    if (spiram_free >= kTensorArenaSize) {
        tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (tensor_arena) {
            ESP_LOGI(TAG, "Allocated %u KB Tensor Arena in external PSRAM", (unsigned)(kTensorArenaSize / 1024));
        }
    }

    if (tensor_arena == nullptr) {
        ESP_LOGW(TAG, "PSRAM not available or insufficient; attempting internal RAM allocation...");
        tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (tensor_arena == nullptr) {
            ESP_LOGE(TAG, "FAILED to allocate %u KB Tensor Arena in internal RAM!", (unsigned)(kTensorArenaSize / 1024));
            return;
        }
        ESP_LOGI(TAG, "Allocated %u KB Tensor Arena in internal RAM", (unsigned)(kTensorArenaSize / 1024));
    }

    // 4. Set up Ops Resolver
    // AllOpsResolver includes all built-in ops for initial verification.
    static tflite::AllOpsResolver resolver;

    // 5. Build the MicroInterpreter
    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);

    // 6. Allocate Tensors
    ESP_LOGI(TAG, "Invoking interpreter.AllocateTensors()...");
    TfLiteStatus allocate_status = interpreter.AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() FAILED! (Status: %d)", allocate_status);
        return;
    }

    size_t arena_used = interpreter.arena_used_bytes();
    ESP_LOGI(TAG, "AllocateTensors() SUCCESSFUL!");
    ESP_LOGI(TAG, "Tensor Arena used: %u / %u bytes (%.1f%%)",
             arena_used, kTensorArenaSize, (float)arena_used * 100.0f / (float)kTensorArenaSize);

    // 7. Inspect Input Tensor
    TfLiteTensor *input = interpreter.input(0);
    if (!input) {
        ESP_LOGE(TAG, "Failed to get input tensor!");
        return;
    }

    ESP_LOGI(TAG, "---------------- Input Tensor ----------------");
    ESP_LOGI(TAG, "Shape: [%d, %d, %d, %d]",
             input->dims->data[0], input->dims->data[1],
             input->dims->data[2], input->dims->data[3]);
    ESP_LOGI(TAG, "Type: %d (kTfLiteInt8=%d, kTfLiteFloat32=%d)",
             input->type, kTfLiteInt8, kTfLiteFloat32);
    ESP_LOGI(TAG, "Quantization Scale     : %f", input->params.scale);
    ESP_LOGI(TAG, "Quantization Zero Point: %d", input->params.zero_point);

    // 8. Inspect Output Tensor
    TfLiteTensor *output = interpreter.output(0);
    if (!output) {
        ESP_LOGE(TAG, "Failed to get output tensor!");
        return;
    }

    ESP_LOGI(TAG, "---------------- Output Tensor ---------------");
    ESP_LOGI(TAG, "Shape: [%d, %d]", output->dims->data[0], output->dims->data[1]);
    ESP_LOGI(TAG, "Type: %d (kTfLiteInt8=%d)", output->type, kTfLiteInt8);
    ESP_LOGI(TAG, "Quantization Scale     : %f", output->params.scale);
    ESP_LOGI(TAG, "Quantization Zero Point: %d", output->params.zero_point);
    ESP_LOGI(TAG, "---------------------------------------------");

    // 9. Verification Inference with Synthetic Test Pattern
    ESP_LOGI(TAG, "Running dummy test inference (simulating neutral gray uint8=128 -> int8=0)...");
    int8_t *input_data = input->data.int8;
    for (int i = 0; i < kExpectedInputSize; i++) {
        // Pixel = 128 (middle gray) -> int8 = (128 - 128) = 0
        input_data[i] = 0;
    }

    int64_t start_us = esp_timer_get_time();
    TfLiteStatus invoke_status = interpreter.Invoke();
    int64_t end_us = esp_timer_get_time();

    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() FAILED! (Status: %d)", invoke_status);
        return;
    }

    float latency_ms = (float)(end_us - start_us) / 1000.0f;
    ESP_LOGI(TAG, "Invoke() SUCCESSFUL! Execution time: %.2f ms", latency_ms);

    // 10. Dequantize and Print Output Predictions
    ESP_LOGI(TAG, "Dummy Inference Results:");
    int8_t *output_data = output->data.int8;
    int best_class = -1;
    float best_score = -1.0f;

    for (int i = 0; i < 3; i++) {
        int8_t q_val = output_data[i];
        float score = (static_cast<float>(q_val) - output->params.zero_point) * output->params.scale;
        ESP_LOGI(TAG, "  Class [%d] %-8s: raw=%4d, score=%.4f (%.1f%%)",
                 i, kClassNames[i], q_val, score, score * 100.0f);
        if (score > best_score) {
            best_score = score;
            best_class = i;
        }
    }
    ESP_LOGI(TAG, "Predicted Class: [%d] %s (score: %.4f)",
             best_class, (best_class >= 0 ? kClassNames[best_class] : "UNKNOWN"), best_score);

    ESP_LOGI(TAG, "===================================================");
    ESP_LOGI(TAG, " ESP32-S3 Model Bring-up Stage Complete!");
    ESP_LOGI(TAG, " Next stage: Connect Camera module & Stream frames.");
    ESP_LOGI(TAG, "===================================================");

    // Loop idle
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}
