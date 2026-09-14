# 🍃 Mini-Mangosteen-Detect (Edge AI Deployment)

> **Edge AI ระบบจำแนกระดับความสุกของมังคุดแบบเรียลไทม์บนบอร์ด ESP32-S3 (LilyGO T-SIMCAM) ด้วย TensorFlow Lite Micro**

[![PlatformIO](https://img.shields.io/badge/Framework-Arduino-blue.svg)](https://platformio.org/)
[![Hardware](https://img.shields.io/badge/Board-LilyGO%20T--SIMCAM%20(ESP32--S3)-orange.svg)](https://www.lilygo.cc/)
[![AI-Model](https://img.shields.io/badge/Model-MobileNetV2%20INT8%20%7C%20Basic%20CNN-green.svg)](https://www.tensorflow.org/lite/microcontrollers)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 📸 ภาพรวมของระบบ (Overview)

โปรเจกต์นี้เป็นชุดเฟิร์มแวร์สำหรับ **Deploy โมเดล AI จำแนกระดับความสุกของผลมังคุดสด** ลงบนบอร์ดสมองกลฝังตัว **LilyGO T-SIMCAM (ESP32-S3 Dual-Core 240MHz, 8MB PSRAM)** ประมวลผล On-Device 100% ผ่าน **TensorFlow Lite Micro** โดยจำแนกผลมังคุดเป็น 3 ระดับ:
1. **`unripe` (มังคุดดิบ / ด่าง / สายเลือด)**
2. **`ripe` (มังคุดสุกพอดีกิน / เลือดดำ / ม่วงแดง)**
3. **`overripe` (มังคุดสุกเกิน / เนื้อแก้ว / ดำคล้ำ)**

---

## ✨ คุณสมบัติเด่น (Features)

- ⚡ **รันโมเดล AI On-Device:** รองรับโมเดล **MobileNetV2 INT8** ($\alpha=0.35$, ~627 KB) และ **Basic CNN INT8** (~31 KB) จัดสรร Tensor Arena ใน PSRAM (Latency ~45–65 ms)
- 📷 **ภาพสดคมชัด 240×240:** ปรับปรุงความละเอียดกล้องเป็น 240×240 พิกเซล พร้อมระบบ Dual Frame Buffer (`fb_count = 2`, `CAMERA_GRAB_LATEST`) ลื่นไหล ไม่หน่วง
- 🌐 **Wi-Fi SoftAP Web Dashboard:** บอร์ดปล่อย Wi-Fi ชื่อ `Mangosteen-AI` ในตัว สามารถใช้สมาร์ทโฟน แท็บเล็ต หรือโน้ตบุ๊กเปิดเบราว์เซอร์ไปที่ `http://192.168.4.1` เพื่อดูภาพสดและกดทำนายผลได้ทันที
- 🎯 **โหมดถ่ายภาพ & Predict ตามสั่ง (On-Demand):** มีกรอบเล็งเป้า (Target Box) ตรงกลาง ช่วยจัดตำแหน่งผลมังคุดก่อนกด `📸 ถ่ายภาพ & Predict` หรือสลับเป็น `⚡ ทำนายสดตลอดเวลา`
- 🖥️ **Desktop Python GUI:** มีโปรแกรม [`view_camera.py`](view_camera.py) พร้อมระบบตรวจจับพอร์ต Serial อัตโนมัติ สำหรับสตรีมภาพและทดสอบผ่านสาย USB
- 🚀 **1-Click Terminal Setup:** ติดตั้งไลบรารี แปลงโมเดล สแกนพอร์ต และแฟลชลงบอร์ดอัตโนมัติด้วยคำสั่งเดียวผ่าน [`setup.bat`](setup.bat) หรือ `python setup.py`

---

## 📁 โครงสร้างโปรเจกต์ (Repository Structure)

```text
Mini-Mangosteen-Detect/
├── models/                               # โฟลเดอร์รวมโมเดลบีบอัด TFLite INT8
│   ├── mobilenet_v2_alpha35_int8.tflite  # ⭐ โมเดลหลักความแม่นยำสูง (Default, ~627 KB)
│   ├── mobilenet_v2_alpha025_int8.tflite # โมเดลทางเลือกขนาดกะทัดรัด (~622 KB)
│   ├── basic_cnn_int8.tflite             # โมเดลขนาดเล็กพิเศษ (~31 KB)
│   ├── models_tflite_int8.zip            # ไฟล์ Zip รวมโมเดลบีบอัดทั้งหมด
│   └── README.md                         # เอกสารอธิบายโมเดลและ Input/Output Specs
├── src/
│   ├── main.cpp                          # ซอร์สโค้ดเฟิร์มแวร์หลัก (กล้อง, AI, SoftAP, Web Server)
│   ├── mangosteen_model_data.h           # Header ข้อมูลโมเดล
│   └── mangosteen_model_data.cc          # ข้อมูลโมเดล C Byte Array (alignas 16)
├── platformio.ini                        # ไฟล์ตั้งค่า PlatformIO สำหรับ ESP32-S3
├── sdkconfig.defaults                    # ค่าคอนฟิกเริ่มต้น ESP-IDF / FreeRTOS
├── setup.py                              # สคริปต์ติดตั้ง แปลงโมเดล และแฟลชอัตโนมัติ
├── setup.bat                             # ตัวเรียกติดตั้งอัตโนมัติสำหรับ Windows (1-Click)
├── convert_tflite_to_c.py                # เครื่องมือแปลงโมเดล TFLite เป็น C++ array
├── view_camera.py                        # โปรแกรม Python GUI ดูภาพสดผ่านสาย USB (Auto Port)
├── run_viewer.bat                        # รันโปรแกรม Python GUI บน Windows
├── requirements.txt                      # รายการไลบรารี Python ที่ต้องใช้
├── class_names.json                      # ลำดับคลาสผลลัพธ์ (overripe, ripe, unripe)
├── TRAINING_GUIDE.md                     # คู่มือการเทรนโมเดลและ Quantize บน Colab ฉบับสมบูรณ์
├── SETUP_GUIDE.md                        # คู่มือการติดตั้งและแฟลชลงบอร์ดอย่างละเอียด
└── README.md                             # เอกสารแนะนำโปรเจกต์ฉบับนี้
```

---

## ⚡ วิธีเริ่มต้นใช้งาน (Quick Setup)

### ขั้นตอนที่ 0: เทรนโมเดล AI ของคุณ (Train Your Model)
โปรเจกต์นี้ไม่ได้แจกจ่ายน้ำหนักโมเดลสำเร็จรูป เพื่อให้ทุกคนสามารถนำไปเทรนด้วยชุดข้อมูลของตนเองได้อย่างอิสระ:
👉 **ปฏิบัติตามคู่มือ:** [**TRAINING_GUIDE.md**](TRAINING_GUIDE.md) (มีโค้ด Google Colab ครบทุกบรรทัด ตั้งแต่ Data Augmentation จนถึง Full INT8 Quantization)  
เมื่อเทรนเสร็จและดาวน์โหลดไฟล์ `.tflite` มาแล้ว ให้นำมาวางในโฟลเดอร์ `models/` (เช่น `models/my_model_int8.tflite`)

---

### ขั้นตอนที่ 1: เซ็ตอัพและแฟลชลงบอร์ดอัตโนมัติ (แนะนำ)
1. เสียบสาย USB ระหว่างบอร์ด LilyGO T-SIMCAM กับคอมพิวเตอร์
2. **บน Windows:** ดับเบิลคลิกไฟล์ **`setup.bat`**  
   **บน Terminal / Command Line:**
   ```bash
   python setup.py
   ```
   *(สคริปต์จะตรวจเช็คไลบรารี, ค้นหาโมเดลใน `models/`, แปลงเป็น C++ array, ตรวจจับพอร์ต COM และแฟลชเฟิร์มแวร์ให้อัตโนมัติทันที)*

---

### วิธีที่ 2: คอมไพล์และแฟลชด้วยตนเอง (Manual PlatformIO)
หากต้องการคอมไพล์และแฟลชทีละสเต็ปผ่าน PlatformIO:

```bash
# 1. ติดตั้งไลบรารี Python
pip install -r requirements.txt

# 2. แปลงโมเดล TFLite ที่ต้องการเป็น C++ Array
python convert_tflite_to_c.py models/mobilenet_v2_alpha35_int8.tflite --out-dir src

# 3. คอมไพล์และอัปโหลดลงบอร์ด (PlatformIO จะค้นหาพอร์ตให้อัตโนมัติ)
pio run -t upload

# หรือระบุพอร์ตโดยตรง เช่น COM7
pio run -t upload --upload-port COM7
```

---

## 🌐 วิธีเข้าใช้งานหน้า Web Dashboard (ไร้สาย)

เมื่อแฟลชเฟิร์มแวร์เสร็จแล้ว บอร์ดจะเริ่มทำงานและปล่อย Wi-Fi Hotspot อัตโนมัติ:

1. **เชื่อมต่อ Wi-Fi จากตัวบอร์ด:**
   - **ชื่อ Wi-Fi (SSID):** `Mangosteen-AI`
   - **รหัสผ่าน:** `12345678`
2. **เปิดเบราว์เซอร์:**
   - เข้าไปที่ URL: **`http://192.168.4.1`**
3. **การใช้งาน:**
   - เล็งกล้องให้ลูกมังคุดอยู่ตรงกลางกรอบเล็ง
   - กดปุ่ม **`📸 ถ่ายภาพ & Predict`** เพื่ออ่านผลการทำนาย (ระดับความสุก, % ความมั่นใจ, Latency)

---

## 💻 วิธีเปิดดูภาพสดผ่านคอมพิวเตอร์ (Desktop Python GUI)

หากต้องการดูภาพสด 384×384 และควบคุมผ่านสาย USB:
1. เสียบสาย USB ของบอร์ดเข้ากับคอมพิวเตอร์
2. ดับเบิลคลิก **`run_viewer.bat`** (หรือรัน `python view_camera.py`)
3. โปรแกรมจะตรวจจับพอร์ต COM อัตโนมัติและเปิดหน้าต่างภาพสด
4. กดปุ่ม **`Spacebar`** บนคีย์บอร์ดเพื่อสั่งถ่ายภาพและทำนายผล

---

## 🛠️ รายละเอียดพินฮาร์ดแวร์ (Hardware Pinout)

> ⚠️ **หมายเหตุสำคัญ:** บอร์ด **LilyGO T-SIMCAM** ไม่มีชิปควบคุมไฟ AXP2101 และมีตำแหน่งพินต่างจากรุ่น T-Camera S3 ทั่วไป

| ฟังก์ชัน | พิน ESP32-S3 GPIO | คำอธิบาย |
|---|---|---|
| **PWR_ON_PIN** | **GPIO 1** | สั่งจ่ายไฟเลี้ยงบอร์ดและกล้อง (ต้องตั้ง `OUTPUT` และ `HIGH`) |
| **DVP XCLK** | **GPIO 14** | สัญญาณนาฬิกากล้อง External Clock (20 MHz) |
| **DVP PCLK** | **GPIO 13** | สัญญาณ Pixel Clock |
| **DVP VSYNC** | **GPIO 6** | สัญญาณ Vertical Sync |
| **DVP HREF** | **GPIO 7** | สัญญาณ Horizontal Reference |
| **SCCB SIOD (SDA)**| **GPIO 4** | ควบคุมเซนเซอร์กล้องผ่าน I2C Data |
| **SCCB SIOC (SCL)**| **GPIO 5** | ควบคุมเซนเซอร์กล้องผ่าน I2C Clock |
| **DVP D0 - D7** | **11, 9, 8, 10, 12, 17, 16, 15** | พินรับข้อมูลภาพ 8-bit Data Bus |
| **DVP PWDN / RESET**| **-1** | ควบคุมผ่านฮาร์ดแวร์ภายในบอร์ด |

> 💡 **การปรับโฟกัสเลนส์กล้อง (Manual Focus):** เลนส์กล้อง OV2640 ตั้งค่าจากโรงงานมาที่ระยะไกล หากส่องมังคุดในระยะใกล้ (< 20 ซม.) แล้วภาพไม่ชัด **ให้ใช้มือค่อยๆ หมุนเกลียวรอบกระบอกเลนส์** เพื่อปรับโฟกัสให้คมชัด

---

## 🚀 วิธี Push ขึ้น GitHub (เมื่อได้รับลิงก์ Repository)

เมื่อสร้างหรือมีลิงก์ Repository บน GitHub แล้ว ให้เปิด Terminal ที่โฟลเดอร์โปรเจกต์แล้วรันคำสั่งต่อไปนี้:

```bash
# 1. เปลี่ยน Remote URL ไปยัง Repo ใหม่
git remote set-url origin <URL_REPOSITORY_ของคุณ>

# 2. Push ขึ้น branch main
git push -u origin main
```
