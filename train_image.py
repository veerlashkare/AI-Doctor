import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

# ==========================================================
# 🧩 CONFIGURATION
# ==========================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
DATA_DIR = "images"
MODEL_PATH = "models/image_model_best.h5"

# ==========================================================
# 🧠 BUILD MODEL FUNCTION
# ==========================================================
def build_image_model(input_shape=(224, 224, 3), num_classes=2):
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False  # Freeze pretrained layers initially
    inp = Input(shape=input_shape)
    x = base(inp, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================================================
# 📦 DATASET PREPARATION
# ==========================================================
train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
).flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='sparse'
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='sparse'
)

# ==========================================================
# ⚙️ BUILD + TRAIN MODEL
# ==========================================================
model = build_image_model(input_shape=(*IMG_SIZE, 3))

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1)

print("🚀 Starting training...\n")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

print("\n✅ Training complete! Best model saved to:", MODEL_PATH)

# ==========================================================
# 🔍 EVALUATE MODEL
# ==========================================================
val_loss, val_acc = model.evaluate(val_gen, verbose=0)
print(f"🎯 Final Validation Accuracy: {val_acc*100:.2f}%")