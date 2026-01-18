"""
⚡ FAST Crack Detection - Optimized for Speed & 92%+ Accuracy
Uses EfficientNet-B0 (proven to work great for this task)
"""

import os
import shutil
import random
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("="*60)
print("⚡ FAST CRACK DETECTION TRAINING")
print("="*60)

# ============== TORCH IMPORTS ==============
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============== CONFIG ==============
DATA_DIR = Path("./dataset")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 25  # EfficientNet converges fast with pretrained weights
IMG_SIZE = 224
LR = 0.001

# ============== DATA PREPARATION ==============
print("\n📁 Preparing data...")

# Strong augmentation for training
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load full dataset
full_dataset = datasets.ImageFolder(str(DATA_DIR))
classes = full_dataset.classes
num_classes = len(classes)
print(f"Classes: {classes}")
print(f"Total images: {len(full_dataset)}")

# Split indices
labels = [s[1] for s in full_dataset.samples]
train_idx, val_idx = train_test_split(
    range(len(full_dataset)), test_size=0.2, stratify=labels, random_state=SEED
)

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

# Custom dataset class to apply different transforms
class SubsetWithTransform(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[self.indices[idx]]
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

train_dataset = SubsetWithTransform(full_dataset, train_idx, train_transforms)
val_dataset = SubsetWithTransform(full_dataset, val_idx, val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ============== MODEL ==============
print("\n🔧 Building EfficientNet-B0 model...")

# EfficientNet-B0 - fast and accurate for binary classification
model = models.efficientnet_b0(weights='IMAGENET1K_V1')

# Freeze early layers (transfer learning)
for param in model.features[:6].parameters():
    param.requires_grad = False

# Replace classifier
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.classifier[1].in_features, num_classes)
)

model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ============== TRAINING ==============
print("\n🚀 Training...")
print("-"*50)

best_acc = 0.0
best_model_state = None

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
    
    train_acc = 100. * train_correct / train_total
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_acc = 100. * val_correct / val_total
    
    scheduler.step()
    
    # Save best
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_state = model.state_dict().copy()
        marker = " ⭐ BEST"
    else:
        marker = ""
    
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%{marker}")
    
    # Early success check
    if val_acc >= 94:
        print(f"\n🎯 Reached {val_acc:.1f}% - stopping early!")
        break

# Load best model
model.load_state_dict(best_model_state)

# ============== FINAL EVALUATION ==============
print("\n" + "="*60)
print("📊 FINAL EVALUATION")
print("="*60)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds) * 100

print(f"\n{'='*40}")
print(f"🎯 FINAL ACCURACY: {accuracy:.2f}%")
print(f"{'='*40}")

if accuracy >= 92:
    print("✅ TARGET ACHIEVED! Accuracy ≥ 92%! 🎉")
else:
    print(f"⚠️ {92-accuracy:.1f}% away from target")

print("\n📋 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

# Save model
torch.save(best_model_state, OUTPUT_DIR / 'best_crack_detector.pt')
print(f"\n✅ Model saved to: {OUTPUT_DIR / 'best_crack_detector.pt'}")

print("\n" + "="*60)
print(f"🏁 DONE! Best Accuracy: {best_acc:.2f}%")
print("="*60)
