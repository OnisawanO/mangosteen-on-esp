from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    esp32_path = r'D:\Y3S1\AIoT\mangosonteen\calibration\esp32_raw.jpg'
    phone_path = r'D:\Y3S1\AIoT\mangosonteen\calibration\phone_ref.jpg'

    if not os.path.exists(esp32_path) or not os.path.exists(phone_path):
        print('[ERROR] Images not found!')
        return

    esp_rgb = np.array(Image.open(esp32_path).convert('RGB'))
    phone_rgb = np.array(Image.open(phone_path).convert('RGB'))

    h_e, w_e = esp_rgb.shape[:2]
    h_p, w_p = phone_rgb.shape[:2]
    print(f'[INFO] ESP32 image: {w_e}x{h_e}')
    print(f'[INFO] Phone image: {w_p}x{h_p}')

    patches_esp = [
        (108, 106, 6, 'White Paper'),
        (56, 120, 8, 'Red Bag'),
        (128, 140, 8, 'Green Blanket'),
        (130, 180, 8, 'Green Dark'),
        (50, 68, 5, 'Blue Fan'),
        (195, 148, 8, 'Black Backpack'),
        (140, 75, 8, 'Floor Tile'),
        (55, 160, 8, 'Gray Sofa')
    ]

    patches_phone = [
        (620, 1030, 15, 'White Paper'),
        (380, 1080, 20, 'Red Bag'),
        (760, 1250, 20, 'Green Blanket'),
        (760, 1480, 20, 'Green Dark'),
        (375, 860, 15, 'Blue Fan'),
        (1120, 1250, 25, 'Black Backpack'),
        (780, 890, 20, 'Floor Tile'),
        (370, 1350, 20, 'Gray Sofa')
    ]

    X_cam = []
    Y_ref = []

    for i, (pe, pp) in enumerate(zip(patches_esp, patches_phone)):
        xe, ye, re, label_e = pe
        xp, yp, rp, label_p = pp

        patch_e = esp_rgb[max(0, ye-re):min(h_e, ye+re), max(0, xe-re):min(w_e, xe+re)]
        patch_p = phone_rgb[max(0, yp-rp):min(h_p, yp+rp), max(0, xp-rp):min(w_p, xp+rp)]

        mean_e = np.mean(patch_e, axis=(0, 1))
        mean_p = np.mean(patch_p, axis=(0, 1))

        X_cam.append(mean_e)
        Y_ref.append(mean_p)
        print(f'[{label_e}] ESP32 RGB={np.round(mean_e, 1)} -> Phone RGB={np.round(mean_p, 1)}')

    X_cam = np.array(X_cam)
    Y_ref = np.array(Y_ref)

    X_aug = np.hstack([X_cam, np.ones((len(X_cam), 1))])
    W, residuals, rank, s = np.linalg.lstsq(X_aug, Y_ref, rcond=None)

    M = W[:3, :].T
    offset = W[3, :]

    print("==============================================")
    print("      3x3 COLOR CORRECTION MATRIX (CCM)")
    print("==============================================")
    print("M =\n", np.round(M, 4))
    print("Offset =", np.round(offset, 2))

    esp_flat = esp_rgb.reshape(-1, 3).astype(np.float32)
    calibrated_flat = np.dot(esp_flat, M.T) + offset
    calibrated_img = np.clip(calibrated_flat, 0, 255).reshape(esp_rgb.shape).astype(np.uint8)

    out_img_path = r'D:\Y3S1\AIoT\mangosonteen\calibration\esp32_calibrated.jpg'
    Image.fromarray(calibrated_img).save(out_img_path)
    print(f'[+] Saved calibrated image to: {out_img_path}')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(esp_rgb)
    axes[0].set_title('1. Original ESP32 OV2640 (Dull / Tinted)', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(calibrated_img)
    axes[1].set_title('2. Color-Calibrated ESP32 (Transformed)', fontsize=12, fontweight='bold', color='#15803d')
    axes[1].axis('off')

    phone_crop = phone_rgb[750:1650, 200:1350]
    axes[2].imshow(phone_crop)
    axes[2].set_title('3. Smartphone Reference (True Colors)', fontsize=12, fontweight='bold', color='#1d4ed8')
    axes[2].axis('off')

    plt.tight_layout()
    comparison_path = r'D:\Y3S1\AIoT\mangosonteen\calibration\calibrated_comparison.png'
    plt.savefig(comparison_path, dpi=200)
    plt.close()
    print(f'[+] Saved comparison plot to: {comparison_path}')

    m00, m01, m02 = M[0,0], M[0,1], M[0,2]
    m10, m11, m12 = M[1,0], M[1,1], M[1,2]
    m20, m21, m22 = M[2,0], M[2,1], M[2,2]
    off0, off1, off2 = offset[0], offset[1], offset[2]

    cpp_code = (
        "// Color Correction Matrix (Derived from Smartphone Reference Calibration)\n"
        "inline void applyColorCorrection(uint8_t r, uint8_t g, uint8_t b, uint8_t &r_out, uint8_t &g_out, uint8_t &b_out) {\n"
        "    float r_f = (float)r;\n"
        "    float g_f = (float)g;\n"
        "    float b_f = (float)b;\n\n"
        f"    int r_c = (int)({m00:.3f}f * r_f + {m01:.3f}f * g_f + {m02:.3f}f * b_f + {off0:.1f}f);\n"
        f"    int g_c = (int)({m10:.3f}f * r_f + {m11:.3f}f * g_f + {m12:.3f}f * b_f + {off1:.1f}f);\n"
        f"    int b_c = (int)({m20:.3f}f * r_f + {m21:.3f}f * g_f + {m22:.3f}f * b_f + {off2:.1f}f);\n\n"
        "    r_out = (uint8_t)constrain(r_c, 0, 255);\n"
        "    g_out = (uint8_t)constrain(g_c, 0, 255);\n"
        "    b_out = (uint8_t)constrain(b_c, 0, 255);\n"
        "}"
    )
    print("==============================================")
    print("        C++ IMPLEMENTATION FOR ESP32")
    print("==============================================")
    print(cpp_code)

    with open(r'D:\Y3S1\AIoT\mangosonteen\calibration\color_matrix.h', 'w') as f:
        f.write('#ifndef COLOR_MATRIX_H\n#define COLOR_MATRIX_H\n\n' + cpp_code + '\n\n#endif\n')

if __name__ == '__main__':
    main()
