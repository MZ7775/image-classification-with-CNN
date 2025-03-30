#pip install tensorflow keras numpy matplotlib opencv-python

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split


# Load images and labels
data = []
labels = []
IMG_SIZE = 100  # Resize images

# Define dataset path
dataset_path = "dataset/"

# Iterate over categories
categories = ["Cat", "Dog"]

for category in categories:
    path = os.path.join(dataset_path, category)
    label = categories.index(category)  # 0 for Cat, 1 for Dog
    
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        try:
            img_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # Resize
            data.append(img_array)
            labels.append(label)
        except Exception as e:
            pass  # Ignore errors

# Convert to numpy arrays
data = np.array(data) / 255.0  # Normalize pixel values
labels = np.array(labels)


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification (Cat/Dog)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)


test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")


def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    prediction = model.predict(img)
    return "Dog" if prediction[0][0] > 0.5 else "Cat"

print(predict_image("test_image.jpg"))  # Replace with an actual image path
