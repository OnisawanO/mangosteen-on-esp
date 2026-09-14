import sys
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk
import serial
import serial.tools.list_ports
import numpy as np
from PIL import Image, ImageTk

CLASS_COLORS = {
    "ripe": "#2ecc71",      # Green
    "unripe": "#f39c12",    # Amber/Orange
    "overripe": "#e74c3c",  # Red
}

class MangosteenViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mangosteen Edge AI — Live Camera Preview & Predict")
        self.root.geometry("540x740")
        self.root.resizable(False, False)
        self.root.configure(bg="#181825")

        # Determine Serial Port: CLI argument, auto-detection, or default
        cli_port = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
        self.serial_port = cli_port or self.auto_detect_port() or "COM7"
        self.baud_rate = 115200
        self.ser = None
        self.running = False
        
        # Latest frame container (thread-safe)
        self.latest_frame = None
        self.latest_meta = None
        self.lock = threading.Lock()

        # State modes
        self.is_paused = False        # True when a snapshot is frozen for inspection
        self.auto_predict = tk.BooleanVar(value=False)
        self.has_predicted = False
        self.fps = 0.0
        self.fps_counter = 0
        self.fps_timer = time.time()

        self.setup_ui()
        self.start_serial_thread()
        self.root.after(20, self.update_gui)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<space>", lambda event: self.on_predict_toggle())

    def auto_detect_port(self):
        try:
            ports = list(serial.tools.list_ports.comports())
            for p in ports:
                desc = p.description.lower()
                vid = p.vid or 0
                pid = p.pid or 0
                if vid == 0x303A or pid == 0x1001 or "usb serial" in desc or "ch340" in desc or "cp210" in desc or "espressif" in desc:
                    return p.device
            if ports:
                return ports[0].device
        except Exception:
            pass
        return None

    def setup_ui(self):
        # Header
        header = tk.Label(self.root, text="🍃 Mangosteen Ripeness Classifier", 
                          font=("Segoe UI", 16, "bold"), fg="#cdd6f4", bg="#181825")
        header.pack(pady=(12, 2))

        sub = tk.Label(self.root, text=f"LilyGO T-SIMCAM (ESP32-S3) | Port: {self.serial_port}", 
                       font=("Segoe UI", 9), fg="#a6adc8", bg="#181825")
        sub.pack(pady=(0, 8))

        # Camera Canvas (384x384)
        self.canvas_frame = tk.Frame(self.root, bg="#11111b", padx=2, pady=2)
        self.canvas_frame.pack()
        self.canvas = tk.Canvas(self.canvas_frame, width=384, height=384, bg="#11111b", 
                                highlightthickness=0)
        self.canvas.pack()
        self.canvas_text = self.canvas.create_text(192, 192, text=f"Connecting to {self.serial_port}...", 
                                                   fill="#6c7086", font=("Segoe UI", 12))

        # Control Buttons Row
        ctrl_frame = tk.Frame(self.root, bg="#181825")
        ctrl_frame.pack(fill=tk.X, padx=35, pady=(10, 6))

        self.btn_predict = tk.Button(ctrl_frame, text="📸 ถ่ายภาพ & Predict (Spacebar)", 
                                     font=("Segoe UI", 11, "bold"), bg="#89b4fa", fg="#11111b",
                                     activebackground="#b4befe", cursor="hand2", padx=12, pady=6,
                                     command=self.on_predict_toggle)
        self.btn_predict.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 8))

        chk_auto = tk.Checkbutton(ctrl_frame, text="⚡ ทำนายสดตลอดเวลา", variable=self.auto_predict,
                                  font=("Segoe UI", 10), bg="#181825", fg="#cdd6f4",
                                  selectcolor="#313244", activebackground="#181825",
                                  activeforeground="#cdd6f4", command=self.on_auto_toggle)
        chk_auto.pack(side=tk.RIGHT)

        # Prediction Card
        self.card = tk.Frame(self.root, bg="#24273a", padx=16, pady=12, highlightthickness=1, highlightbackground="#363a4f")
        self.card.pack(fill=tk.X, padx=35, pady=4)

        top_row = tk.Frame(self.card, bg="#24273a")
        top_row.pack(fill=tk.X)

        self.lbl_prediction = tk.Label(top_row, text="[ Live Preview ]", 
                                       font=("Segoe UI", 14, "bold"), fg="#89b4fa", bg="#24273a")
        self.lbl_prediction.pack(side=tk.LEFT)

        self.lbl_confidence = tk.Label(top_row, text="จัดภาพให้เข้ากรอบ", 
                                       font=("Segoe UI", 12), fg="#a6adc8", bg="#24273a")
        self.lbl_confidence.pack(side=tk.RIGHT)

        # Confidence Bar
        self.progress = ttk.Progressbar(self.card, orient="horizontal", mode="determinate", length=400)
        self.progress.pack(fill=tk.X, pady=(10, 8))

        # Details Row
        self.lbl_latency = tk.Label(self.card, text="Inference: -- ms | Camera Stream: -- FPS", 
                                    font=("Consolas", 9), fg="#a6adc8", bg="#24273a")
        self.lbl_latency.pack(anchor="w")

        self.lbl_dist = tk.Label(self.card, text="overripe: 0.0%  |  ripe: 0.0%  |  unripe: 0.0%", 
                                 font=("Consolas", 9), fg="#cad3f5", bg="#24273a")
        self.lbl_dist.pack(anchor="w", pady=(4, 0))

        # Footer
        self.lbl_status = tk.Label(self.root, text=f"Port: {self.serial_port} @ 115200 | Press [Spacebar] to Predict", 
                                   font=("Segoe UI", 9), fg="#585b70", bg="#181825")
        self.lbl_status.pack(side=tk.BOTTOM, pady=8)

    def on_auto_toggle(self):
        if self.auto_predict.get():
            self.is_paused = False
            self.btn_predict.config(text="⚡ ทำนายสดตลอดเวลา (เปิดอยู่)", state=tk.DISABLED, bg="#45475a")
        else:
            self.btn_predict.config(text="📸 ถ่ายภาพ & Predict (Spacebar)", state=tk.NORMAL, bg="#89b4fa")

    def on_predict_toggle(self):
        if self.auto_predict.get():
            return
        
        if not self.is_paused:
            # Take snapshot and freeze prediction
            self.is_paused = True
            self.has_predicted = True
            self.btn_predict.config(text="🔄 กลับไปดูภาพสด (Live Preview)", bg="#a6e3a1")
        else:
            # Resume live stream
            self.is_paused = False
            self.btn_predict.config(text="📸 ถ่ายภาพ & Predict (Spacebar)", bg="#89b4fa")
            self.lbl_prediction.config(text="[ Live Preview ]", fg="#89b4fa")
            self.lbl_confidence.config(text="จัดภาพให้เข้ากรอบ", fg="#a6adc8")
            self.progress["value"] = 0

    def start_serial_thread(self):
        self.running = True
        self.thread = threading.Thread(target=self.serial_worker, daemon=True)
        self.thread.start()

    def serial_worker(self):
        while self.running:
            try:
                if not self.ser or not self.ser.is_open:
                    self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1.0)
                    self.ser.dtr = True
                    self.ser.rts = True
                    time.sleep(0.5)

                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if "---IMG_START---" in line:
                    meta = {}
                    while True:
                        header_line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                        if header_line == "---PAYLOAD---":
                            break
                        if ":" in header_line:
                            k, v = header_line.split(":", 1)
                            meta[k] = v

                    img_len = int(meta.get("LEN", 18432))
                    payload = bytearray()
                    while len(payload) < img_len and self.running:
                        chunk = self.ser.read(img_len - len(payload))
                        if not chunk:
                            break
                        payload.extend(chunk)

                    # Consume remaining footer line
                    self.ser.readline()

                    if len(payload) == 18432:
                        # Decode RGB565 (96x96)
                        arr = np.frombuffer(payload, dtype='>u2').reshape((96, 96))
                        r = (((arr >> 11) & 0x1F) * 255 // 31).astype(np.uint8)
                        g = (((arr >> 5) & 0x3F) * 255 // 63).astype(np.uint8)
                        b = ((arr & 0x1F) * 255 // 31).astype(np.uint8)
                        rgb = np.stack([r, g, b], axis=-1)

                        img = Image.fromarray(rgb).resize((384, 384), Image.NEAREST)

                        # Update FPS
                        self.fps_counter += 1
                        now = time.time()
                        if now - self.fps_timer >= 1.0:
                            self.fps = self.fps_counter / (now - self.fps_timer)
                            self.fps_counter = 0
                            self.fps_timer = now

                        meta["FPS"] = f"{self.fps:.1f}"

                        with self.lock:
                            # Only overwrite latest if not paused
                            if not self.is_paused:
                                self.latest_frame = img
                                self.latest_meta = meta
                            elif self.is_paused and not self.has_predicted:
                                # First frame upon pause
                                self.latest_frame = img
                                self.latest_meta = meta

            except Exception as e:
                time.sleep(0.5)

    def update_gui(self):
        frame_to_render = None
        meta_to_render = None

        with self.lock:
            if self.latest_frame is not None:
                frame_to_render = self.latest_frame
                meta_to_render = self.latest_meta

        if frame_to_render is not None and meta_to_render is not None:
            self.photo = ImageTk.PhotoImage(frame_to_render)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

            # Draw targeting reticle guide when in preview
            if not self.is_paused and not self.auto_predict.get():
                # Center guide square (160x160)
                cx, cy = 192, 192
                hw = 80
                self.canvas.create_rectangle(cx - hw, cy - hw, cx + hw, cy + hw,
                                            outline="#89b4fa", width=2, dash=(6, 4))
                self.canvas.create_text(192, cy + hw + 16, text="วางมังคุดตรงกลางกรอบนี้",
                                        fill="#89b4fa", font=("Segoe UI", 9, "bold"))

            # Update Predictions if in Auto mode or Snapshot mode
            if self.is_paused or self.auto_predict.get():
                pred_class = meta_to_render.get("CLASS", "unknown")
                conf = float(meta_to_render.get("CONF", 0.0))
                latency = float(meta_to_render.get("LATENCY", 0.0))
                scores = meta_to_render.get("SCORES", "0,0,0").split(",")

                color = CLASS_COLORS.get(pred_class, "#cdd6f4")
                self.lbl_prediction.config(text=f"🎯 {pred_class.upper()}", fg=color)
                self.lbl_confidence.config(text=f"ความมั่นใจ: {conf:.1f}%", fg=color)
                self.progress["value"] = conf

                fps_str = meta_to_render.get("FPS", "--")
                self.lbl_latency.config(text=f"Inference: {latency:.1f} ms  |  Stream: {fps_str} FPS")
                if len(scores) == 3:
                    self.lbl_dist.config(
                        text=f"overripe: {scores[0]}%  |  ripe: {scores[1]}%  |  unripe: {scores[2]}%"
                    )
            else:
                fps_str = meta_to_render.get("FPS", "--")
                self.lbl_latency.config(text=f"Camera Stream: {fps_str} FPS  |  พร้อมทำนาย")

        if self.running:
            self.root.after(20, self.update_gui)

    def on_close(self):
        self.running = False
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MangosteenViewerApp(root)
    root.mainloop()
