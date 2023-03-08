import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained CNN model
model = tf.keras.models.load_model('drowsiness_detection_model.h5')

# Define a function to detect drowsiness in a given image
def detect_drowsiness(img):
    # Preprocess the image
    img = cv2.resize(img, (224, 224))
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)

    # Apply the CNN model to the image
    prediction = model.predict(img)
    score = prediction[0][0]

    # Return the drowsiness score
    return score

# Open the video stream
cap = cv2.VideoCapture(0)

while True:

    # Capture a frame from the video stream
    ret, frame = cap.read()

    # Detect drowsiness in the frame
    score = detect_drowsiness(frame)

    # Display the frame and drowsiness score
    cv2.imshow('Driver Drowsiness Detection', frame)
    cv2.putText(frame, f"Drowsiness score: {score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Wait for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
