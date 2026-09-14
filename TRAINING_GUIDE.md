# 🧠 คู่มือการเทรนโมเดล AI และส่งออกเป็น TFLite INT8 (Training Guide)
**ระบบจำแนกระดับความสุกของมังคุดสำหรับรันบนชิป ESP32-S3 (TensorFlow Lite Micro)**

คู่มือนี้อธิบายขั้นตอนการเตรียมชุดข้อมูล (Dataset), การสร้างและเทรนโมเดล Deep Learning บน **Google Colab (GPU ฟรี)**, การแปลงเป็นโมเดลบีบอัด **Full INT8 Quantization**, จนถึงการนำไฟล์โมเดลมาใส่ในโปรเจกต์เพื่อแฟลชลงบอร์ด ESP32-S3

---

## 📋 สเปกโมเดลสำหรับ ESP32-S3 (Model Specifications)

เพื่อให้โมเดลสามารถทำงานร่วมกับเฟิร์มแวร์ในบอร์ดนี้ได้ 100% โปรดตั้งค่าโมเดลตามนี้:
- **ขนาดภาพขาเข้า (Input Shape):** `96 × 96 × 3` (RGB พิกเซล)
- **การแปลงข้อมูลขาเข้า (Input Normalization):** ค่า `int8` ในช่วง `[-128, 127]`
- **สถาปัตยกรรมที่แนะนำ:** **MobileNetV2** ($\alpha = 0.35$) หรือ **Custom Basic CNN**
- **คลาสผลลัพธ์ (3 คลาส เรียงตามตัวอักษร):**
  - Index `0`: **`overripe`** (มังคุดสุกเกิน / เนื้อแก้ว / ดำคล้ำ)
  - Index `1`: **`ripe`** (มังคุดสุกพอดีกิน / เลือดดำ / ม่วงแดง)
  - Index `2`: **`unripe`** (มังคุดดิบ / ด่าง / สายเลือด)

---

## 📁 ขั้นตอนที่ 1: การจัดเตรียมชุดข้อมูล (Dataset Preparation)

จัดระเบียบโฟลเดอร์ภาพมังคุดของคุณตามโครงสร้างมาตรฐานของ Keras ดังนี้:

```text
dataset/
├── train/
│   ├── overripe/    # ภาพมังคุดสุกงอม / เปลือกดำเข้ม
│   ├── ripe/        # ภาพมังคุดสุกพอดีกิน / เปลือกม่วงแดง
│   └── unripe/      # ภาพมังคุดดิบ / เปลือกเขียวหรือมีสายเลือดแดง
├── val/
│   ├── overripe/
│   ├── ripe/
│   └── unripe/
└── test/
    ├── overripe/
    ├── ripe/
    └── unripe/
```

> 💡 **ข้อแนะนำในการถ่ายภาพ:**
> - ควรถ่ายภาพผลมังคุดในมุมมองหลากหลาย (ด้านบนขั้ว, ด้านข้าง, ด้านก้น)
> - ถ่ายในสภาพแสงธรรมชาติและแสงหลอดไฟ เพื่อให้โมเดลทนทานต่อสภาพแวดล้อม
> - สัดส่วนการแบ่งข้อมูลแนะนำ: Train 70%, Validation 15%, Test 15%

---

## 🚀 ขั้นตอนที่ 2: โค้ดเทรนโมเดลบน Google Colab (ครบทุกขั้นตอน)

เปิด [Google Colab](https://colab.research.google.com/) แล้วเปลี่ยน Runtime เป็น **GPU** (เมนู `Runtime` -> `Change runtime type` -> `T4 GPU`) จากนั้นรันโค้ดตามลำดับ:

### 2.1 ติดตั้งไลบรารีและโหลดชุดข้อมูล
```python
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

IMG_SIZE = (96, 96)
BATCH_SIZE = 16
SEED = 42

# แก้ path ให้ตรงกับโฟลเดอร์ dataset ของคุณ
DATASET_PATH = "/content/dataset"

train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)  # ควรได้ ['overripe', 'ripe', 'unripe']
```

---

### 2.2 การทำ Data Augmentation เพื่อเพิ่มความหลากหลาย
```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
], name="data_augmentation")
```

---

### 2.3 สร้างโมเดล MobileNetV2 ($\alpha=0.35$)
```python
ALPHA = 0.35

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    alpha=ALPHA,
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Freeze น้ำหนักเดิมในรอบแรก

inputs = layers.Input(shape=(96, 96, 3), name="input_layer")
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(3, activation="softmax", name="output_layer")(x)

model = models.Model(inputs, outputs, name="Mangosteen_MobileNetV2")
model.summary()
```

---

### 2.4 ฝึกสอนโมเดล (Phase 1: Transfer Learning + Phase 2: Fine-Tuning)
```python
# --- Phase 1: เทรนเฉพาะ Classification Head ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=25)

# --- Phase 2: ปลดล็อค 20 เลเยอร์สุดท้ายเพื่อ Fine-Tune ---
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=20)
```

---

### 2.5 การแปลงโมเดลเป็น Full INT8 Quantization (ขั้นตอนสำคัญสำหรับ ESP32)
เพื่อให้โมเดลมีขนาดเล็ก (~600 KB) และทำงานได้อย่างรวดเร็วบน ESP32-S3 ต้องบีบอัดเป็น **Full INT8**:

```python
# 1. แยกเฉพาะโครงข่ายสำหรับ Inference (ตัด Data Augmentation ออก)
inference_inputs = layers.Input(shape=(96, 96, 3), name="input_layer")
x = tf.keras.applications.mobilenet_v2.preprocess_input(inference_inputs)
x = model.get_layer(base_model.name)(x, training=False)
x = model.get_layer("global_average_pooling2d")(x)
x = model.get_layer("dropout")(x)
inference_outputs = model.get_layer("output_layer")(x)

export_model = models.Model(inference_inputs, inference_outputs, name="Export_MobileNetV2")

# 2. ฟังก์ชัน Representative Dataset สำหรับคำนวณ Scaling Factors
def representative_data_gen():
    for images, _ in train_ds.take(30):
        for img in images:
            yield [tf.expand_dims(img, 0)]

# 3. ตั้งค่า TFLite Converter เป็น Full Integer Quantization (INT8)
converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# 4. ทำการแปลงและบันทึกไฟล์
tflite_model_int8 = converter.convert()

output_filename = "mobilenet_v2_alpha35_int8.tflite"
with open(output_filename, "wb") as f:
    f.write(tflite_model_int8)

import os
file_size_kb = os.path.getsize(output_filename) / 1024
print(f"🎉 แปลงโมเดลสำเร็จ! ขนาดไฟล์: {file_size_kb:.2f} KB")
```

---

### 2.6 ดาวน์โหลดไฟล์โมเดล `.tflite`
รันคำสั่งดาวน์โหลดไฟล์จาก Colab ลงเครื่องของคุณ:
```python
from google.colab import files
files.download(output_filename)
```

---

## 📥 ขั้นตอนที่ 3: นำโมเดลมาใส่ในโปรเจกต์และแฟลชลงบอร์ด

1. **นำไฟล์โมเดล `.tflite` ที่ดาวน์โหลดมา:**  
   วางไว้ในโฟลเดอร์ `models/` ของโปรเจกต์นี้ เช่น:
   ```text
   Mini-Mangosteen-Detect/
   └── models/
       └── mobilenet_v2_alpha35_int8.tflite   <-- วางไฟล์ที่นี่
   ```
2. **เสียบสายบอร์ด LilyGO T-SIMCAM เข้ากับคอมพิวเตอร์**
3. **รันคำสั่งติดตั้งอัตโนมัติ:**
   - **บน Windows:** ดับเบิลคลิกไฟล์ **`setup.bat`** ได้ทันที
   - **บน Terminal:**
     ```bash
     python setup.py
     ```
   *(สคริปต์จะค้นหาไฟล์โมเดลใน `models/` แปลงเป็น C++ Array และแฟลชลงชิป ESP32-S3 ให้อัตโนมัติทันที)*
4. **เปิดดูผลงาน:**
   - ต่อ Wi-Fi บอร์ด: `Mangosteen-AI` (รหัส `12345678`)
   - เปิด Browser ไปที่ `http://192.168.4.1` เพื่อทดสอบโมเดลของคุณได้ทันที!
