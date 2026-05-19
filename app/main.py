from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
from app.inference import predict_emotion

app = FastAPI(title="Emotion Detection API")


@app.get("/")
def read_root():
    return {"message": "Emotion Detection API Running"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()

    print("Client connected")

    import base64

    try:
        while True:

            # receive image as base64 string
            data = await websocket.receive_text()

            try:
                # If it's a data URL, extract the base64 data after the comma
                if "," in data:
                    data = data.split(",", 1)[1]

                image_bytes = base64.b64decode(data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as decode_err:
                print("Decode error:", decode_err)
                await websocket.send_json({"emotion": "Invalid frame encoding"})
                continue

            if frame is None:
                await websocket.send_json({"emotion": "Invalid frame"})
                continue

            # predict
            try:
                emotion = predict_emotion(frame)
                # send result as JSON object
                await websocket.send_json({"emotion": emotion})
            except Exception as pred_err:
                print("Prediction error:", pred_err)
                await websocket.send_json({"emotion": f"Error: {str(pred_err)}"})

    except Exception as e:
        print("Connection closed:", e)