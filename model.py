import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np

# Define the input shape for the images
input_shape = (224, 224, 3)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Load the dataset
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')

# Train the model
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Save the model
model.save('drowsiness_detection_model.h5')