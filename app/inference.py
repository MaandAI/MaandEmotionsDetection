import cv2
import numpy as np
import onnxruntime as ort

EMOTIONS = ["angry", "happy", "fear", "disgust", "surprise", "neutral", "sad"]

session = ort.InferenceSession("model/emotion_model.onnx")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def preprocess(face):

    face = cv2.resize(face, (224, 224))

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    face = face.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    face = (face - mean) / std

    face = np.transpose(face, (2, 0, 1))

    face = np.expand_dims(face, axis=0)

    return face.astype(np.float32)


def predict_emotion(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return "No face detected"

    x, y, w, h = faces[0]

    face = image[y : y + h, x : x + w]

    input_tensor = preprocess(face)

    outputs = session.run(None, {"input": input_tensor})

    pred = np.argmax(outputs[0])

    return EMOTIONS[pred]
