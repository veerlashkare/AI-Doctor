# train_image_confident.py
"""
Train EfficientNetB0-based cancer image classifier with confidence boosting (fine-tuning)
Saves best model to: models/image_model_best.h5
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from sklearn.utils import class_weight

# ==============================================
# CONFIG
# ==============================================
DATA_DIR = "datasets"           # expects train/, val/, test/ folders inside
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 30
INITIAL_LR = 1e-4
FINE_TUNE_LR = 1e-5
FINE_TUNE_AT = -40              # unfreeze last 40 layers
MODEL_OUT = "models/image_model_best.h5"

os.makedirs("models", exist_ok=True)

# ==============================================
# DATA
# ==============================================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    brightness_range=(0.8, 1.2),
    horizontal_flip=True,
    fill_mode='nearest'
).flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# ==============================================
# MODEL
# ==============================================
def build_model(input_shape=(224,224,3), num_classes=2):
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False
    inp = Input(shape=input_shape)
    x = base(inp, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.15)(x)  # reduced dropout for higher confidence
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    return model, base

model, base = build_model(input_shape=IMG_SIZE+(3,), num_classes=2)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Compute class weights
labels = train_gen.classes
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# Callbacks
cb_checkpoint = ModelCheckpoint(MODEL_OUT, monitor='val_accuracy', save_best_only=True, verbose=1)
cb_earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
cb_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1)

# ==============================================
# STAGE 1: Train top layers
# ==============================================
print("🔹 Stage 1: Training top layers only...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=6,
    class_weight=class_weights,
    callbacks=[cb_checkpoint, cb_earlystop, cb_reduce]
)

# ==============================================
# STAGE 2: Fine-tuning deeper layers
# ==============================================
print("🔹 Stage 2: Fine-tuning deeper layers...")
for layer in base.layers[FINE_TUNE_AT:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[cb_checkpoint, cb_earlystop, cb_reduce]
)

# ==============================================
# EVALUATE
# ==============================================
print("🔹 Evaluating best model...")
model = tf.keras.models.load_model(MODEL_OUT)
loss, acc = model.evaluate(test_gen)
print(f"✅ Test Accuracy: {acc*100:.2f}%")
model.save(MODEL_OUT)
print(f"✅ Saved fine-tuned model at: {MODEL_OUT}")
