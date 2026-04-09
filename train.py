import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 1. Transforms (GOOD BALANCE)
# -----------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# 2. Load Dataset
# -----------------------------
train_data = datasets.ImageFolder("data/train", transform=train_transforms)
val_data = datasets.ImageFolder("data/val", transform=val_transforms)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

print("Classes:", train_data.classes)

from collections import Counter

# -----------------------------
# 3. Handle Class Imbalance
# -----------------------------
target_counts = Counter(train_data.targets)
total_samples = len(train_data)
num_classes = len(train_data.classes)

# Compute inverted weights array
class_weights = [total_samples / (num_classes * target_counts[i]) for i in range(num_classes)]
print("Using Class Weights:", class_weights)

# -----------------------------
# 4. Load Pretrained Model
# -----------------------------
from torchvision.models import ResNet50_Weights
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 🔥 BETTER FINE-TUNING
# We unfreeze the whole model but use a very small learning rate for everything 
# except the new fully-connected layer. This avoids batch-norm freezing bugs.
for param in model.parameters():
    param.requires_grad = True

# -----------------------------
# 5. Device (GPU)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Prepare weights tensor
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

print("Using device:", device)

# -----------------------------
# 6. Loss + Optimizer + Scheduler
# -----------------------------
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# 🔥 Differential learning rates
optimizer = optim.Adam([
    {'params': [param for name, param in model.named_parameters() if 'fc' not in name], 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-4}
])

# Scale LR down when validation accuracy stops improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# -----------------------------
# 7. Training Loop (with Best Model tracking)
# -----------------------------
num_epochs = 25
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}")

    epoch_loss = running_loss / total_train
    epoch_acc = 100 * correct_train / total_train
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")

    # -----------------------------
    # 8. Validation Phase
    # -----------------------------
    model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss_val = criterion(outputs, labels)
            
            val_loss += loss_val.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

    val_epoch_loss = val_loss / total_val
    accuracy = 100 * correct_val / total_val
    print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")

    # Step scheduler on validation accuracy
    scheduler.step(accuracy)

    # -----------------------------
    # 9. Save Best Model
    # -----------------------------
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), "emotion_model.pth")
        print(f"🌟 Best model saved with accuracy: {best_acc:.2f}%")

print("Training Complete!")