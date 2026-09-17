"""
Custom CNN ("MangosteenNet") Training, Evaluation & INT8 Quantization Pipeline
Target Hardware: LilyGO T-SIMCAM (ESP32-S3) with TensorFlow Lite Micro
Designed for: 3-Class Mangosteen Ripeness (Overripe, Ripe, Unripe)
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

print("==================================================================")
print("🥭 Mangosteen Custom CNN Pipeline (Edge AI on ESP32-S3)")
print(f"TensorFlow Version: {tf.__version__}")
print("==================================================================")

# 1. Dataset Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Mangosteen", "Mangosteen.v1i.folder")
if not os.path.exists(DATASET_DIR):
    DATASET_DIR = os.path.join(BASE_DIR, "Mangosteen.v1i.folder")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR = os.path.join(DATASET_DIR, "test")

for p, name in [(TRAIN_DIR, "Train"), (VAL_DIR, "Valid"), (TEST_DIR, "Test")]:
    if not os.path.exists(p):
        sys.exit(f"Error: {name} directory not found at: {p}")
    print(f"{name} Directory: {p}")

ARTIFACTS_DIR = os.path.join(BASE_DIR, "custom_cnn_artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# 2. Hyperparameters & Configuration
IMG_SIZE = (96, 96)
INPUT_SHAPE = (96, 96, 3)
BATCH_SIZE = 16
EPOCHS = 40
SEED = 42

print(f"\n--- Loading Dataset Splits ({IMG_SIZE[0]}x{IMG_SIZE[1]} RGB) ---")
raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = raw_train_ds.class_names
print(f"Classes: {class_names}")
num_classes = len(class_names)

# Class Weights to handle slight class imbalance
train_labels = []
for _, labels in raw_train_ds:
    train_labels.extend(np.argmax(labels.numpy(), axis=1))

unique_classes = np.unique(train_labels)
class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=train_labels)
class_weight_dict = dict(zip(unique_classes, class_weights))
print("Class weights:", {class_names[k]: round(v, 2) for k, v in class_weight_dict.items()})

# 3. Data Augmentation Pipeline (Domain-Specific for Mangosteens)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom((-0.1, 0.1)),
    layers.RandomBrightness(0.15),
    layers.RandomContrast(0.15),
], name="mangosteen_augmentation")

AUTOTUNE = tf.data.AUTOTUNE
train_ds_proc = raw_train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
).cache().prefetch(buffer_size=AUTOTUNE)

val_ds_proc = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds_proc = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 4. Build Custom CNN ("MangosteenNet")
def build_mangosteen_custom_cnn(input_shape=(96, 96, 3), num_classes=3):
    inputs = layers.Input(shape=input_shape, name="input_rgb")
    
    # Rescale [0, 255] to [0.0, 1.0]
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)
    
    # Stem: Standard Conv (48x48x16)
    x = layers.Conv2D(16, (3, 3), strides=2, padding="same", use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(6.0, name="stem_relu")(x)
    
    def depthwise_block(tensor, filters, strides=1, name=""):
        # Depthwise Conv (Spatial feature extraction)
        d = layers.DepthwiseConv2D((3, 3), strides=strides, padding="same", use_bias=False, name=f"{name}_dw")(tensor)
        d = layers.BatchNormalization(name=f"{name}_dw_bn")(d)
        d = layers.ReLU(6.0, name=f"{name}_dw_relu")(d)
        
        # Pointwise Conv (Channel mixing)
        p = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False, name=f"{name}_pw")(d)
        p = layers.BatchNormalization(name=f"{name}_pw_bn")(p)
        p = layers.ReLU(6.0, name=f"{name}_pw_relu")(p)
        return p

    # Stage 1: 24x24x32
    x = depthwise_block(x, 32, strides=2, name="block_1")
    
    # Stage 2: 12x12x64
    x = depthwise_block(x, 64, strides=2, name="block_2")
    
    # Stage 3: 6x6x96
    x = depthwise_block(x, 96, strides=2, name="block_3")

    # Stage 4: 6x6x128 (Detailed rind texture refinement)
    x = depthwise_block(x, 128, strides=1, name="block_4")
    
    # Global Average Pooling Head (100% compatible with TFLM and accelerated by ESP-NN)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="fc1")(x)
    x = layers.Dropout(0.35, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="mangosteen_custom_cnn")
    return model

print("\n--- Model Architecture Summary ---")
model = build_mangosteen_custom_cnn(INPUT_SHAPE, num_classes)
model.summary(line_length=80)

# 5. Compile & Train with Label Smoothing
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.08)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=loss_fn,
    metrics=["accuracy"]
)

best_model_path = os.path.join(ARTIFACTS_DIR, "best_custom_cnn.keras")
callbacks_list = [
    callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-5,
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
]

print("\n--- Training Custom CNN (MangosteenNet) ---")
history = model.fit(
    train_ds_proc,
    validation_data=val_ds_proc,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks_list,
    verbose=1
)

# 6. Plot Learning Curves
print("\n--- Generating Learning Curves ---")
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(1, len(acc) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(epochs_range, acc, label="Training Accuracy", color="#2563eb", linewidth=2.0)
ax1.plot(epochs_range, val_acc, label="Validation Accuracy", color="#16a34a", linewidth=2.0, linestyle="--")
best_epoch = int(np.argmax(val_acc) + 1)
best_val_acc = float(max(val_acc))
ax1.scatter([best_epoch], [best_val_acc], color="#dc2626", s=80, zorder=5, 
            label=f"Peak Val Acc: {best_val_acc*100:.1f}% (Ep {best_epoch})")
ax1.set_title("Custom CNN Accuracy vs Epochs", fontsize=13, fontweight="bold")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, loss, label="Training Loss", color="#2563eb", linewidth=2.0)
ax2.plot(epochs_range, val_loss, label="Validation Loss", color="#16a34a", linewidth=2.0, linestyle="--")
min_loss_ep = int(np.argmin(val_loss) + 1)
min_val_loss = float(min(val_loss))
ax2.scatter([min_loss_ep], [min_val_loss], color="#dc2626", s=80, zorder=5,
            label=f"Min Val Loss: {min_val_loss:.3f} (Ep {min_loss_ep})")
ax2.set_title("Loss vs Epochs (Label Smoothing 0.08)", fontsize=13, fontweight="bold")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
curve_path = os.path.join(ARTIFACTS_DIR, "custom_cnn_curves.png")
plt.savefig(curve_path, dpi=200)
plt.close()
print(f"[+] Learning curves saved to: {curve_path}")

# 7. Evaluate on Held-Out Test Set
print("\n======================================================")
print("Evaluation on Held-Out Test Set (Float32 Model)")
print("======================================================")
y_true = []
y_pred = []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
cm = confusion_matrix(y_true, y_pred)

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 14, "weight": "bold"})
plt.title("Custom CNN Confusion Matrix (Test Set)", fontsize=12, fontweight="bold")
plt.ylabel("Ground Truth")
plt.xlabel("Predicted")
plt.tight_layout()
cm_path = os.path.join(ARTIFACTS_DIR, "custom_cnn_confusion_matrix.png")
plt.savefig(cm_path, dpi=200)
plt.close()
print(f"[+] Confusion matrix saved to: {cm_path}")

# 8. Full INT8 Quantization for ESP32-S3
print("\n======================================================")
print("Full INT8 Quantization (TFLite Micro)")
print("======================================================")
def representative_data_gen():
    # Representative calibration samples across train and valid
    count = 0
    for images, _ in raw_train_ds.take(20):
        for i in range(images.shape[0]):
            sample = tf.expand_dims(images[i], axis=0)
            yield [sample]
            count += 1
            if count >= 80:
                return

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_int8_model = converter.convert()

TFLITE_PATH = os.path.join(ARTIFACTS_DIR, "mangosteen_custom_cnn_int8.tflite")
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_int8_model)

size_kb = len(tflite_int8_model) / 1024.0
print(f"[+] Quantized INT8 Model Saved: {TFLITE_PATH}")
print(f"[+] Model Size: {size_kb:.2f} KB (Original MobileNetV2 was ~638 KB)")

# Verify INT8 Model Accuracy
print("\n--- Verifying Quantized INT8 Model on Test Set ---")
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
in_scale, in_zero_point = input_details["quantization"]
out_scale, out_zero_point = output_details["quantization"]

int8_preds = []
for images, _ in test_ds:
    for img in images:
        # Quantize input
        img_np = img.numpy()
        img_quant = np.round(img_np / in_scale + in_zero_point).astype(np.int8)
        img_quant = np.expand_dims(img_quant, axis=0)
        
        interpreter.set_tensor(input_details["index"], img_quant)
        interpreter.invoke()
        out_tensor = interpreter.get_tensor(output_details["index"])[0]
        
        # Dequantize or take argmax directly
        int8_preds.append(np.argmax(out_tensor))

int8_preds = np.array(int8_preds)
int8_acc = np.mean(int8_preds == y_true)
print(f"INT8 Quantized Test Accuracy: {int8_acc * 100:.2f}%")

# 9. Export to C Header and Source
print("\n--- Exporting C Array for ESP32-S3 Firmware ---")
c_header = f"""// Auto-generated Mangosteen Custom CNN Model for ESP32-S3
// Resolution: {IMG_SIZE[0]}x{IMG_SIZE[1]} RGB
// Size: {len(tflite_int8_model)} bytes ({size_kb:.2f} KB)
// Classes: {class_names}

#ifndef MANGOSTEEN_MODEL_DATA_H_
#define MANGOSTEEN_MODEL_DATA_H_

#ifdef __cplusplus
extern "C" {{
#endif

extern const unsigned char g_mangosteen_model_data[];
extern const unsigned int g_mangosteen_model_data_len;

#ifdef __cplusplus
}}
#endif

#endif  // MANGOSTEEN_MODEL_DATA_H_
"""

hex_rows = []
for i in range(0, len(tflite_int8_model), 12):
    chunk = tflite_int8_model[i:i+12]
    hex_rows.append("    " + ", ".join([f"0x{b:02x}" for b in chunk]))
c_body = ",\n".join(hex_rows)

c_source = f"""// Auto-generated Mangosteen Custom CNN Model
#include "mangosteen_model_data.h"

alignas(16) const unsigned char g_mangosteen_model_data[] = {{
{c_body}
}};

const unsigned int g_mangosteen_model_data_len = {len(tflite_int8_model)};
"""

header_path = os.path.join(ARTIFACTS_DIR, "mangosteen_model_data.h")
source_path = os.path.join(ARTIFACTS_DIR, "mangosteen_model_data.cc")

with open(header_path, "w", encoding="utf-8") as f:
    f.write(c_header)
with open(source_path, "w", encoding="utf-8") as f:
    f.write(c_source)

print(f"[+] Header generated: {header_path}")
print(f"[+] Source generated: {source_path}")

# Write Markdown Training Summary Report
report_path = os.path.join(ARTIFACTS_DIR, "custom_cnn_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"""# 🥭 Mangosteen Custom CNN Training Report
- **Architecture:** Lightweight Depthwise Separable CNN (MangosteenNet)
- **Input Resolution:** {IMG_SIZE[0]}x{IMG_SIZE[1]}x3 RGB
- **Float32 Peak Val Accuracy:** {best_val_acc*100:.2f}%
- **INT8 Quantized Test Accuracy:** {int8_acc*100:.2f}%
- **INT8 Model Size:** {size_kb:.2f} KB (Reduced from ~638 KB)
- **TFLM / ESP-NN Compatible:** 100% (Conv2D, DepthwiseConv2D, ReLU6, BatchNormalization fused)
""")

print("\n🎉 [SUCCESS] Custom CNN pipeline finished successfully!")
