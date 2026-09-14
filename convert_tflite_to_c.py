#!/usr/bin/env python3
"""
Convert TFLite model to C/C++ source and header files for ESP32-S3 (TensorFlow Lite Micro).
"""

import sys
import os
import argparse

def convert_tflite_to_c(tflite_path, output_dir, prefix="mangosteen_model"):
    if not os.path.exists(tflite_path):
        print(f"Error: Model file '{tflite_path}' not found.")
        return False

    with open(tflite_path, "rb") as f:
        model_bytes = f.read()

    model_len = len(model_bytes)
    print(f"[+] Loaded '{tflite_path}': {model_len} bytes ({model_len / 1024:.2f} KB)")

    os.makedirs(output_dir, exist_ok=True)
    header_path = os.path.join(output_dir, f"{prefix}_data.h")
    source_path = os.path.join(output_dir, f"{prefix}_data.cc")

    array_name = f"g_{prefix}_data"
    len_name = f"g_{prefix}_data_len"

    # Write Header (.h)
    header_guard = f"{prefix.upper()}_DATA_H_"
    header_content = f"""// Auto-generated from {os.path.basename(tflite_path)}
#ifndef {header_guard}
#define {header_guard}

#ifdef __cplusplus
extern "C" {{
#endif

extern const unsigned char {array_name}[];
extern const unsigned int {len_name};

#ifdef __cplusplus
}}
#endif

#endif  // {header_guard}
"""
    with open(header_path, "w", encoding="utf-8") as f:
        f.write(header_content)
    print(f"[+] Generated: {header_path}")

    # Write Source (.cc) with 16-byte alignment
    with open(source_path, "w", encoding="utf-8") as f:
        f.write(f'// Auto-generated from {os.path.basename(tflite_path)}\n')
        f.write(f'#include "{prefix}_data.h"\n\n')
        f.write(f'// 16-byte alignment is required for TensorFlow Lite Micro tensor arena optimizations\n')
        f.write(f'alignas(16) const unsigned char {array_name}[] = {{\n')
        
        # Write bytes in rows of 12
        bytes_per_line = 12
        for i in range(0, model_len, bytes_per_line):
            chunk = model_bytes[i:i + bytes_per_line]
            hex_vals = [f"0x{b:02x}" for b in chunk]
            line = "  " + ", ".join(hex_vals)
            if i + bytes_per_line < model_len:
                line += ","
            f.write(line + "\n")
            
        f.write("};\n\n")
        f.write(f"const unsigned int {len_name} = {model_len};\n")
    print(f"[+] Generated: {source_path}")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TFLite model to C++ array for ESP-IDF / TFLM")
    parser.add_argument("tflite_path", nargs="?", default="basic_cnn_int8.tflite", help="Path to .tflite model file")
    parser.add_argument("--out-dir", default="src", help="Target output directory (default: src)")
    parser.add_argument("--prefix", default="mangosteen_model", help="C variable prefix")
    args = parser.parse_args()

    success = convert_tflite_to_c(args.tflite_path, args.out_dir, args.prefix)
    if not success:
        sys.exit(1)
