import numpy as np
import cv2
import os

# Define the paths to the drowsy and alert image folders
drowsy_dir = "C:\\Users\\Manushree\\Driver Drowsiness detection prjct2\\Closed_Eyes"
alert_dir = "C:\\Users\\Manushree\\Driver Drowsiness detection prjct2\\Open_Eyes"

# Load the drowsy images and create the drowsy labels
drowsy_images = []
drowsy_labels = []
for filename in os.listdir(drowsy_dir):
    img = cv2.imread(os.path.join(drowsy_dir, filename))
    img = cv2.resize(img, (224, 224))
    drowsy_images.append(img)
    drowsy_labels.append(0)

# Load the alert images and create the alert labels
alert_images = []
alert_labels = []
for filename in os.listdir(alert_dir):
    img = cv2.imread(os.path.join(alert_dir, filename))
    img = cv2.resize(img, (224, 224))
    alert_images.append(img)
    alert_labels.append(1)

# Concatenate the drowsy and alert images and labels
x_train = np.concatenate((drowsy_images, alert_images))
y_train = np.concatenate((drowsy_labels, alert_labels))

# Save the x_train and y_train arrays to .npy files
np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)