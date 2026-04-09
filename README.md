# EmotionTrack Engine 🚀

A real-time PyTorch Facial Recognition Agent integrated into a modern React Web Application. 

This project takes a ResNet50 neural network locally trained on facial expressions (RAF-DB/FER datasets) and serves it through a Python Flask backend. The fast Vite+React frontend accesses the local webcam and draws animated bounding-boxes directly over the user's face, or accepts drag-and-drop structural inputs using MTCNN Face Extraction.

## 🛠️ Architecture
- **Backend:** Python, Flask, PyTorch, MTCNN (Facenet)
- **Frontend:** React, Vite, CSS Glassmorphism
- **Model:** Custom-trained ResNet50 (90MB `emotion_model.pth` included)

---

## 🚀 Getting Started (for Team Members)

To run this repository on your own machine, you will need to start both the Python backend and the React frontend locally.

### Prerequisites
1. You must have **Python 3.10+** installed.
2. You must have **Node.js** (npm) installed.

### 1. Setup the Python Backend
Open a terminal in the root folder of this project:
```bash
# Install the required AI & Server packages
pip install torch torchvision facenet-pytorch flask flask-cors pandas pillow

# Start the API server
python app.py
```
*Wait until you see `API Server starting on http://127.0.0.1:5000`.*

### 2. Setup the React Frontend
Open a **second, separate terminal** in the root folder:
```bash
# Move into the React folder
cd frontend

# Install the UI dependencies
npm install

# Start the web server
npm run dev
```

### 3. Open the Dashboard!
Once both servers are running, simply open your internet browser and go to:
**[http://localhost:5173](http://localhost:5173)**

*If attempting to use the Live Camera feature, be sure to click Accept when your browser asks for webcam permission!*
