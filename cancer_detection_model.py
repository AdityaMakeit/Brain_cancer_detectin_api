#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
import os
import math
import shutil
import glob
import tensorflow as tf  # Ensure TensorFlow is imported

# ROOT directory where the brain tumor dataset is stored
ROOT_DIR = r"C:\Users\Aditya\Desktop\code\Brain tumor\brain_tumor_dataset"
number_of_imgs = {}

# Count images in each subdirectory
for dir in os.listdir(ROOT_DIR):
    number_of_imgs[dir] = len(os.listdir(os.path.join(ROOT_DIR, dir)))

print(number_of_imgs)

# Function to create folders and move data
def data_Folder(p, split):
    if not os.path.exists("./" + p):
        os.mkdir("./" + p)

        for dir in os.listdir(ROOT_DIR):
            os.makedirs(os.path.join("./" + p, dir))
            img_list = os.listdir(os.path.join(ROOT_DIR, dir))
            num_imgs = len(img_list)

            # Calculate number of images to move
            num_to_move = min(math.floor(split * num_imgs) - 2, num_imgs)
            images_to_move = np.random.choice(a=img_list, size=num_to_move, replace=False)
            for img in images_to_move:
                O = os.path.join(ROOT_DIR, dir, img)
                D = os.path.join("./" + p, dir, img)
                shutil.move(O, D)

    else:
        print(f"{p} folder already exists.")

data_Folder("train", 0.7)
data_Folder("val", 0.15)
data_Folder("test", 0.15)

# CNN model definition
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(filters=36, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(rate=0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dropout(rate=0.25),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Image preprocessing function
def preprocessingImages(path):
    image_data = tf.keras.preprocessing.image.ImageDataGenerator(
        zoom_range=0.2, shear_range=0.2, rescale=1/255, horizontal_flip=True)
    image = image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=32, class_mode="binary")
    return image

# Load training data
train_data = preprocessingImages(r"C:\Users\Aditya\Desktop\code\Brain tumor\train")
val_data = preprocessingImages(r"C:\Users\Aditya\Desktop\code\Brain tumor\val")
test_data = preprocessingImages(r"C:\Users\Aditya\Desktop\code\Brain tumor\test")

# Early stopping and model checkpoint
es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", min_delta=0.01, patience=6, verbose=1, mode="auto")
mc = tf.keras.callbacks.ModelCheckpoint(monitor="val_accuracy", filepath="./bestmodel.keras", verbose=1, save_best_only=True, mode="auto")

# Fit the model
hs = model.fit(
    x=train_data,
    steps_per_epoch=8,
    epochs=30,
    verbose=1,
    validation_data=val_data,
    validation_steps=16,
    callbacks=[es, mc]
)

# Save the trained model
model.save(r"C:\Users\Aditya\Desktop\bestmodel.keras")
print("Model saved successfully!")

# Load the saved model
model = tf.keras.models.load_model(r"C:\Users\Aditya\Desktop\bestmodel.keras")
print("Model loaded successfully")

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_data)
print(f"Accuracy of model is {accuracy * 100:.2f}%")

# Function to predict new images
def predict_image(img_path):
    image = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))  # Resize image
    input_arr = tf.keras.preprocessing.image.img_to_array(image) / 255.0  # Normalize
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    predictions = model.predict(input_arr)

    if predictions[0] > 0.5:
        return "No cancer found."
    else:
        return "This Brain MRI has Cancer."
