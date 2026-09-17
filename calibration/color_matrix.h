#ifndef COLOR_MATRIX_H
#define COLOR_MATRIX_H

// Color Correction Matrix (Derived from Smartphone Reference Calibration)
inline void applyColorCorrection(uint8_t r, uint8_t g, uint8_t b, uint8_t &r_out, uint8_t &g_out, uint8_t &b_out) {
    float r_f = (float)r;
    float g_f = (float)g;
    float b_f = (float)b;

    int r_c = (int)(1.304f * r_f + -0.420f * g_f + 0.104f * b_f + 22.6f);
    int g_c = (int)(0.093f * r_f + 0.665f * g_f + 0.167f * b_f + 15.7f);
    int b_c = (int)(0.186f * r_f + -0.173f * g_f + 1.199f * b_f + -7.0f);

    r_out = (uint8_t)constrain(r_c, 0, 255);
    g_out = (uint8_t)constrain(g_c, 0, 255);
    b_out = (uint8_t)constrain(b_c, 0, 255);
}

#endif
