#ifndef COLOR_MATRIX_H
#define COLOR_MATRIX_H

#include <Arduino.h>
#include "esp_camera.h"

// ==============================================================================
// 3x3 COLOR CORRECTION MATRIX + OFFSET (CALIBRATED FROM SMARTPHONE REFERENCE)
// ==============================================================================
// Equation: [R', G', B']^T = M * [R, G, B]^T + Offset
//
// Matrix derived via Linear Least Squares optimization from paired OV2640 
// vs Smartphone reference photos:
//   R' =  1.304 * R - 0.420 * G + 0.104 * B + 22.6
//   G' =  0.093 * R + 0.665 * G + 0.167 * B + 15.7
//   B' =  0.186 * R - 0.173 * G + 1.199 * B -  7.0
// ==============================================================================

inline void applyColorCorrection(uint8_t r, uint8_t g, uint8_t b, uint8_t &r_out, uint8_t &g_out, uint8_t &b_out) {
    float r_f = (float)r;
    float g_f = (float)g;
    float b_f = (float)b;

    int r_c = (int)(1.304f * r_f - 0.420f * g_f + 0.104f * b_f + 22.6f);
    int g_c = (int)(0.093f * r_f + 0.665f * g_f + 0.167f * b_f + 15.7f);
    int b_c = (int)(0.186f * r_f - 0.173f * g_f + 1.199f * b_f - 7.0f);

    r_out = (uint8_t)(r_c < 0 ? 0 : (r_c > 255 ? 255 : r_c));
    g_out = (uint8_t)(g_c < 0 ? 0 : (g_c > 255 ? 255 : g_c));
    b_out = (uint8_t)(b_c < 0 ? 0 : (b_c > 255 ? 255 : b_c));
}

inline void applyColorCorrectionToFB(camera_fb_t *fb) {
    if (!fb || !fb->buf) return;
    uint16_t *pixels = (uint16_t*)fb->buf;
    int total_pixels = fb->width * fb->height;
    for (int i = 0; i < total_pixels; i++) {
        uint16_t p = pixels[i];
        p = (p >> 8) | (p << 8); // Unswap DMA endianness
        uint8_t r = ((p >> 11) & 0x1F) << 3;
        uint8_t g = ((p >> 5) & 0x3F) << 2;
        uint8_t b = (p & 0x1F) << 3;

        uint8_t r_cal, g_cal, b_cal;
        applyColorCorrection(r, g, b, r_cal, g_cal, b_cal);

        uint16_t p_cal = ((uint16_t)(r_cal & 0xF8) << 8) | ((uint16_t)(g_cal & 0xFC) << 3) | (b_cal >> 3);
        pixels[i] = (p_cal >> 8) | (p_cal << 8); // Restore DMA endianness for fmt2jpg
    }
}

#endif // COLOR_MATRIX_H
