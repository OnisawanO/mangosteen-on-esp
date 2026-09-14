#!/usr/bin/env python3
"""
Automated Terminal Setup Script for Mangosteen Edge AI (ESP32-S3)
Can be run from project root or inside setup/ folder:
    python setup.py
    python setup.py --port COM7
    python setup.py --model models/basic_cnn_int8.tflite
    python setup.py --build-only
"""

import os
import sys
import argparse
import subprocess
import time

# Force UTF-8 encoding on Windows to prevent UnicodeEncodeError with emojis
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Auto-detect Project Root directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(CURRENT_DIR, "platformio.ini")):
    PROJECT_ROOT = CURRENT_DIR
elif os.path.exists(os.path.join(CURRENT_DIR, "..", "platformio.ini")):
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
else:
    PROJECT_ROOT = CURRENT_DIR

def print_banner():
    banner = f"""
===========================================================
   🍃 Mangosteen Edge AI — Automated Terminal Setup 🍃
       Board: LilyGO T-SIMCAM (ESP32-S3) | Web SoftAP
       Project Directory: {PROJECT_ROOT}
===========================================================
"""
    print(banner)

def run_cmd(cmd, step_name, cwd=PROJECT_ROOT):
    print(f"\n[+] {step_name}...")
    print(f"    Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd)
    if res.returncode != 0:
        print(f"\n[-] ERROR: '{step_name}' failed with exit code {res.returncode}.")
        sys.exit(res.returncode)
    print(f"[✓] {step_name} completed successfully.")

def check_dependencies():
    print("\n[Step 1/4] Checking & Installing Dependencies...")
    required_packages = ["pyserial", "numpy", "pillow", "platformio"]
    missing = []
    
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
            
    if missing:
        print(f"[*] Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    else:
        print("[✓] All Python dependencies are installed.")

    # Check PlatformIO CLI
    try:
        subprocess.run(["pio", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("[✓] PlatformIO CLI is ready.")
    except Exception:
        print("[!] PlatformIO not found in PATH. Adding Python Scripts directory...")
        py_scripts = os.path.join(os.path.dirname(sys.executable), "Scripts")
        os.environ["PATH"] = py_scripts + os.pathsep + os.environ["PATH"]

def convert_model(model_override=None):
    print("\n[Step 2/4] Converting TFLite Model to C++ Array...")
    
    if model_override:
        candidates = [
            model_override,
            os.path.join(PROJECT_ROOT, model_override),
            os.path.join(PROJECT_ROOT, "models", model_override)
        ]
    else:
        candidates = [
            os.path.join(PROJECT_ROOT, "models", "mobilenet_v2_alpha35_int8.tflite"),
            os.path.join(PROJECT_ROOT, "models", "mobilenet_v2_alpha025_int8.tflite"),
            os.path.join(PROJECT_ROOT, "models", "basic_cnn_int8.tflite"),
            os.path.join(PROJECT_ROOT, "mobilenet_v2_alpha35_int8.tflite"),
            os.path.join(PROJECT_ROOT, "basic_cnn_int8.tflite"),
        ]

    model_path = None
    for p in candidates:
        if os.path.exists(p):
            model_path = os.path.abspath(p)
            break

    if not model_path:
        # Auto-detect any .tflite in models/
        models_dir = os.path.join(PROJECT_ROOT, "models")
        if os.path.exists(models_dir):
            tflites = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith(".tflite")]
            if tflites:
                model_path = os.path.abspath(tflites[0])

    if not model_path:
        print("\n===========================================================")
        print("[-] NOTICE: ไม่พบไฟล์โมเดล '.tflite' ในโฟลเดอร์ 'models/'")
        print("    กรุณาเทรนโมเดลตามขั้นตอนใน 'TRAINING_GUIDE.md'")
        print("    แล้วนำไฟล์โมเดล '.tflite' (INT8 Quantized) มาวางในโฟลเดอร์ 'models/'")
        print("    ตัวอย่าง: models/my_model_int8.tflite")
        print("    จากนั้นรัน setup.py (หรือดับเบิลคลิก setup.bat) ใหม่อีกครั้ง")
        print("===========================================================\n")
        sys.exit(1)

    print(f"[*] Found model file at: {model_path} ({os.path.getsize(model_path):,} bytes)")
    
    convert_script = os.path.join(PROJECT_ROOT, "convert_tflite_to_c.py")
    if not os.path.exists(convert_script):
        convert_script = os.path.join(CURRENT_DIR, "convert_tflite_to_c.py")
        
    if not os.path.exists(convert_script):
        print("[-] ERROR: 'convert_tflite_to_c.py' not found.")
        sys.exit(1)

    out_src_dir = os.path.join(PROJECT_ROOT, "src")
    subprocess.check_call([sys.executable, convert_script, model_path, "--out-dir", out_src_dir])
    print(f"[✓] Model conversion to '{out_src_dir}/mangosteen_model_data.h' and '.cc' done.")

def detect_port():
    print("\n[Step 3/4] Detecting ESP32-S3 Serial Port...")
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    
    esp_ports = []
    for p in ports:
        desc = p.description.lower()
        vid = p.vid or 0
        pid = p.pid or 0
        if vid == 0x303A or pid == 0x1001 or "usb serial" in desc or "ch340" in desc or "cp210" in desc or "espressif" in desc:
            esp_ports.append(p)

    if len(esp_ports) == 1:
        chosen = esp_ports[0].device
        print(f"[✓] Auto-detected ESP32-S3 on port: {chosen} ({esp_ports[0].description})")
        return chosen
    elif len(esp_ports) > 1:
        print("[?] Multiple potential ESP32 ports found:")
        for idx, p in enumerate(esp_ports):
            print(f"    [{idx + 1}] {p.device} - {p.description}")
        try:
            choice = int(input("Select port number: ")) - 1
            return esp_ports[choice].device
        except Exception:
            return esp_ports[0].device

    print("[!] No USB Serial ports detected automatically.")
    print("    Available ports:")
    for p in ports:
        print(f"    - {p.device}: {p.description}")
    
    user_port = input("Please enter your COM port manually (e.g. COM7), or press Enter to let PlatformIO scan: ").strip()
    return user_port if user_port else None

def build_and_upload(port, build_only=False):
    print(f"\n[Step 4/4] Building & Flashing via PlatformIO in {PROJECT_ROOT}...")
    if build_only:
        run_cmd("pio run", "Building firmware", cwd=PROJECT_ROOT)
        return

    cmd = "pio run -t upload"
    if port:
        cmd += f" --upload-port {port}"
    run_cmd(cmd, "Flashing firmware to ESP32-S3", cwd=PROJECT_ROOT)

def print_finish_instructions():
    msg = """
===========================================================
             🎉 SETUP COMPLETED SUCCESSFULLY! 🎉
===========================================================

ต่อไปคุณสามารถเริ่มใช้งานบอร์ดได้ทันที:

1. เชื่อมต่อ Wi-Fi จากบอร์ด:
   - SSID:     Mangosteen-AI
   - Password: 12345678

2. เปิด Web Browser (มือถือ แท็บเล็ต หรือ คอมพิวเตอร์):
   - ไปที่ URL:  http://192.168.4.1

3. ส่องกล้องที่มังคุด:
   - จัดตำแหน่งให้เข้ากลางกรอบเล็งเป้า
   - กดปุ่ม [📸 ถ่ายภาพ & Predict] เพื่อดูผลทำนายทันที!
===========================================================
"""
    print(msg)

def main():
    parser = argparse.ArgumentParser(description="Automated Terminal Setup for Mangosteen Edge AI")
    parser.add_argument("--port", help="Serial port (e.g. COM7)")
    parser.add_argument("--model", help="Path to custom .tflite model file")
    parser.add_argument("--build-only", action="store_true", help="Compile without uploading")
    args = parser.parse_args()

    print_banner()
    check_dependencies()
    convert_model(model_override=args.model)

    port = args.port
    if not args.build_only and not port:
        port = detect_port()

    build_and_upload(port, args.build_only)
    print_finish_instructions()

if __name__ == "__main__":
    main()
