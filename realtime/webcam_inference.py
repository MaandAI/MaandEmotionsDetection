import cv2
import numpy as np
import onnxruntime as ort

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

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


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in faces:
        face = frame[y : y + h, x : x + w]

        input_tensor = preprocess(face)

        outputs = session.run(None, {"input": input_tensor})

        pred = np.argmax(outputs[0])

        emotion = EMOTIONS[pred]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
