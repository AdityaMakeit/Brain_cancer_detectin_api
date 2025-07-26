#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
import os
import math
import shutil
import tensorflow as tf

# ✅ Get project root path
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(BASE_DIR, 'brain_tumor_dataset')

number_of_imgs = {}
for dir in os.listdir(ROOT_DIR):
    number_of_imgs[dir] = len(os.listdir(os.path.join(ROOT_DIR, dir)))
print(number_of_imgs)

# ✅ Function to split data if not already done
def data_Folder(p, split):
    dest_path = os.path.join(BASE_DIR, p)
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
        for dir in os.listdir(ROOT_DIR):
            os.makedirs(os.path.join(dest_path, dir))
            img_list = os.listdir(os.path.join(ROOT_DIR, dir))
            num_imgs = len(img_list)
            num_to_move = min(math.floor(split * num_imgs) - 2, num_imgs)
            images_to_move = np.random.choice(img_list, size=num_to_move, replace=False)
            for img in images_to_move:
                shutil.move(os.path.join(ROOT_DIR, dir, img), os.path.join(dest_path, dir, img))
    else:
        print(f"{p} folder already exists.")

data_Folder("train", 0.7)
data_Folder("val", 0.15)
data_Folder("test", 0.15)

# ✅ CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(36, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ✅ Preprocessing function
def preprocessingImages(path):
    image_data = tf.keras.preprocessing.image.ImageDataGenerator(
        zoom_range=0.2, shear_range=0.2, rescale=1/255, horizontal_flip=True)
    return image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=32, class_mode="binary")

# ✅ Load training/validation/test data (relative paths)
train_data = preprocessingImages(os.path.join(BASE_DIR, "train"))
val_data = preprocessingImages(os.path.join(BASE_DIR, "val"))
test_data = preprocessingImages(os.path.join(BASE_DIR, "test"))

# ✅ Callbacks
es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", min_delta=0.01, patience=6, verbose=1, mode="auto")
model_path = os.path.join(BASE_DIR, "bestmodel.keras")
mc = tf.keras.callbacks.ModelCheckpoint(monitor="val_accuracy", filepath=model_path, verbose=1, save_best_only=True)

# ✅ Train
hs = model.fit(
    x=train_data,
    steps_per_epoch=8,
    epochs=30,
    verbose=1,
    validation_data=val_data,
    validation_steps=16,
    callbacks=[es, mc]
)

# ✅ Save model
model.save(model_path)
print("Model saved successfully!")

# ✅ Load the saved model
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully")

# ✅ Evaluate model
loss, accuracy = model.evaluate(test_data)
print(f"Accuracy of model is {accuracy * 100:.2f}%")

# ✅ Predict function
def predict_image(img_path):
    image = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return "No cancer found." if predictions[0] > 0.5 else "This Brain MRI has Cancer."
