import base64
import io
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from facenet_pytorch import MTCNN

app = Flask(__name__)
# Enable CORS for the frontend port to access this API
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

print("Loading PyTorch Models into Memory...")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("emotion_model.pth", map_location=device, weights_only=True))
model = model.to(device)
model.eval()

mtcnn = MTCNN(keep_all=False, device=device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route('/api/predict', methods=['POST'])
def predict_emotion():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
            
        img_data = data['image']
        
        # Remove data URI prefix
        if ',' in img_data:
            img_data = img_data.split(',')[1]
            
        img_bytes = base64.b64decode(img_data)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # 1. Detect Face
        boxes, probs = mtcnn.detect(image)
        if boxes is None:
            return jsonify({'faces': []})
            
        # Target the top prediction
        box = [int(b) for b in boxes[0]]
        face_image = image.crop((box[0], box[1], box[2], box[3]))
        
        # 2. Predict Emotion
        face_tensor = transform(face_image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(face_tensor)
            # Fetch softmax confidence
            probs_emotion = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs_emotion, 1)

        emotion = classes[predicted.item()].upper()
        confidence_val = float(confidence.item())

        return jsonify({
            'faces': [{
                # Return standard Canvas plotting coordinates (x, y, width, height)
                'box': [box[0], box[1], box[2]-box[0], box[3]-box[1]],
                'emotion': emotion,
                'confidence': confidence_val
            }]
        })
        
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("API Server starting on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
