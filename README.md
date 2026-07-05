# 😊 Emotions Detection API

> A real-time facial emotion detection microservice built with **FastAPI** and **ONNX Runtime** — part of the **Maand** mental health support platform.

---

## 📖 Overview

The **Emotions Detection API** is a Python-based backend service that uses a pre-trained deep learning model (exported to ONNX format) to detect human facial emotions in real time. It exposes a **WebSocket endpoint** that accepts live video frames (as Base64-encoded images) and returns the predicted emotion label.

This service is consumed by the **Maand** Next.js frontend, enabling emotion-aware interactions and personalized mental health support.

---

## 🧠 Detected Emotions

The model classifies faces into **7 emotion categories**:

| Label      | Description              |
|------------|--------------------------|
| `angry`    | Anger / Frustration      |
| `disgust`  | Disgust                  |
| `fear`     | Fear / Anxiety           |
| `happy`    | Happiness / Joy          |
| `neutral`  | Neutral / Calm           |
| `sad`      | Sadness / Depression     |
| `surprise` | Surprise / Shock         |

---

## 🗂️ Project Structure

```
emotions_api/
├── app/
│   ├── main.py              # FastAPI app — HTTP & WebSocket endpoints
│   ├── inference.py         # Face detection + ONNX emotion inference logic
│   └── webcam_test.html     # Browser-based webcam test client (served at GET /)
├── model/
│   └── emotion_model.onnx   # Pre-trained ONNX emotion classification model
├── realtime/
│   └── webcam_inference.py  # Standalone local webcam demo (OpenCV window)
├── Dockerfile               # Docker configuration for containerized deployment
├── requirements.txt         # Python dependencies
└── EmotionDetectionModel.py # (Reserved for model training/export scripts)
```

---

## ⚙️ Prerequisites

Make sure you have the following installed:

- **Python 3.10+** (Python 3.11 recommended)
- **pip** (Python package manager)
- A working **webcam** (for the real-time demo)
- *(Optional)* **Docker** for containerized deployment

---

## 🚀 Setup & Running Locally

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd emotions_api
```

### 2. Create & Activate a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The server will be available at **`http://localhost:8000`**

> **Note:** The `--reload` flag enables hot-reloading for development. Remove it in production.

---

## 🌐 API Endpoints

### `GET /`
Returns the built-in browser-based webcam test client (`webcam_test.html`).  
Open **`http://localhost:8000`** in your browser to interactively test emotion detection using your webcam.

---

### `WebSocket /ws`

The core real-time inference endpoint.

**Protocol:** WebSocket  
**URL:** `ws://localhost:8000/ws`

#### How It Works

1. **Client** opens a WebSocket connection to `ws://localhost:8000/ws`.
2. **Client** sends a video frame as a **Base64-encoded string** (data URL format is supported, e.g., `data:image/jpeg;base64,...`).
3. **Server** decodes the image, detects the face using OpenCV Haar Cascades, and runs ONNX inference.
4. **Server** responds with a JSON object:

```json
{ "emotion": "happy" }
```

#### Possible Responses

| Response                            | Meaning                               |
|-------------------------------------|---------------------------------------|
| `{ "emotion": "happy" }`            | Successfully detected emotion         |
| `{ "emotion": "No face detected" }` | Frame received but no face found      |
| `{ "emotion": "Invalid frame" }`    | Could not decode the image            |
| `{ "emotion": "Error: ..." }`       | Internal inference error              |

---

## 🔬 Inference Pipeline

The inference logic in `app/inference.py` follows these steps:

```
Raw Frame (BGR image)
        │
        ▼
┌─────────────────────┐
│  Grayscale Convert  │  (for face detection)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Haar Cascade Face  │  detectMultiScale(scaleFactor=1.1, minNeighbors=5)
│     Detection       │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Crop Face ROI     │  First detected face is used
└─────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Preprocessing                          │
│  • Resize to 224x224                    │
│  • BGR → RGB                            │
│  • Normalize to [0.0, 1.0]             │
│  • ImageNet mean/std normalization      │
│  • Reshape to (1, 3, 224, 224) tensor   │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────┐
│   ONNX Runtime      │  emotion_model.onnx
│   Inference         │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Argmax → Label     │  Returns emotion string
└─────────────────────┘
```

---

## 🖥️ Realtime Standalone Demo (No Server Required)

To run emotion detection locally using your webcam in an **OpenCV window** (no browser needed):

```bash
# From the project root
python realtime/webcam_inference.py
```

- A window titled **"Emotion Recognition"** will open showing your live webcam feed.
- Detected faces are highlighted with a **green bounding box**.
- The predicted emotion label is displayed **above the face**.
- Press **`Q`** to quit.

> **Note:** Run this command from the project root directory (`emotions_api/`) so that the relative path `model/emotion_model.onnx` resolves correctly.

---

## 🐳 Running with Docker

### Build the Image

```bash
docker build -t emotions-api .
```

### Run the Container

```bash
docker run -p 8000:8000 emotions-api
```

The API will be accessible at **`http://localhost:8000`**.

---

## 🧪 Testing the WebSocket via Browser

1. Start the server: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
2. Open **`http://localhost:8000`** in your browser.
3. Allow webcam access when prompted.
4. The built-in test client (`webcam_test.html`) will:
   - Capture frames from your webcam
   - Send them over WebSocket to the server
   - Display the returned emotion label on screen in real time

---

## 📦 Dependencies

| Package                  | Purpose                                   |
|--------------------------|-------------------------------------------|
| `fastapi`                | Web framework for building the API        |
| `uvicorn[standard]`      | ASGI server to run the FastAPI app        |
| `websockets`             | WebSocket support                         |
| `python-multipart`       | Form data / multipart request handling    |
| `numpy`                  | Numerical array operations                |
| `opencv-python-headless` | Image processing & face detection         |
| `onnxruntime`            | Running the ONNX emotion model            |
| `pillow`                 | Image utility support                     |

---

## 🔗 Integration with Maand

This microservice is part of the **Maand** mental health platform. The **Maand Next.js frontend** connects to this API via WebSocket to:

- Detect the user's emotional state during chat sessions
- Adapt AI responses based on detected emotion
- Surface mood insights and tracking over time

> Make sure this service is running **before** launching the Maand frontend so the emotion detection feature works correctly.

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'app'` | Run `uvicorn` from the project root (`emotions_api/`) directory |
| `model/emotion_model.onnx not found` | Ensure the `model/` folder with `emotion_model.onnx` exists at the project root |
| `No face detected` responses | Ensure good lighting, face the camera directly, and check camera permissions |
| Webcam not opening (realtime script) | Verify no other app is using the webcam; try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` |
| Docker build fails | Make sure Docker Desktop is running and you have internet access for pip install |

---

## 📄 License

This project is part of the **Maand FYP** academic project. All rights reserved.
