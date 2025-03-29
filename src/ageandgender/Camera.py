from tensorflow.keras.models import model_from_json
import cv2
import numpy as np

# Load the trained model
with open("model.json", "r") as json_file:
    loaded_json_model = json_file.read()

model = model_from_json(loaded_json_model)
model.load_weights("Model.h5")

# Load Haar cascade for face detection
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define function to predict age & gender
def predict_age_gender(image):
    # Preprocess image
    image = cv2.resize(image, (128, 128))
    image = image.reshape((1, 128, 128, 3))  # Add batch dimension
    image = image / 255.0  # Normalize

    # Get model predictions
    gender_pred, age_pred = model.predict(image)

    # Gender Prediction (Binary Classification)
    gender = "Male" if gender_pred[0][0] > 0.5 else "Female"

    # Age Prediction (Regression)
    predicted_age = int(age_pred[0][0])  # Convert age to integer

    return gender, predicted_age

# Video Capture Class
class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()  # Capture frame

        if not ret:
            return None

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.3, 5)

        for (x, y, w, h) in faces_detected:
            # Draw rectangle around detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=3)

            # Crop & preprocess face
            roi_gray = gray_img[y:y+h, x:x+w]
            gender, age = predict_age_gender(roi_gray)

            # Display results on frame
            text = f"{gender}, Age: {age}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        resized_img = cv2.resize(frame, (1000, 600))
        _, jpeg = cv2.imencode('.jpg', resized_img)

        return jpeg.tobytes()
