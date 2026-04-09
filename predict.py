import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))

model.load_state_dict(torch.load("emotion_model.pth", map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# Initialize MTCNN for face tracking
mtcnn = MTCNN(keep_all=False, device=device)

def predict(image_path):
    image = Image.open(image_path).convert("RGB")

    # --- Face Detection & Cropping ---
    boxes, probs = mtcnn.detect(image)
    if boxes is None:
        print(f"[{image_path}] Error: No face detected in this image!")
        return
        
    # Crop the first (most confident) face detected
    box = [int(b) for b in boxes[0]]
    face_image = image.crop((box[0], box[1], box[2], box[3]))
    # ---------------------------------

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = transform(face_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    print(f"[{image_path}] Predicted Emotion: {classes[predicted.item()].upper()}")

if __name__ == "__main__":
    import sys
    # If the user provides an image argument, test that image
    if len(sys.argv) > 1:
        image_to_test = sys.argv[1]
        predict(image_to_test)
    else:
        # Otherwise run on the default test images
        print("Tip: You can test any image by running: python predict.py your_image.jpg\n")
        predict("test.jpg")
        predict("test2.jpg")
        predict("test3.jpg")
