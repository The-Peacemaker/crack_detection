# 🔬 Crack Detection using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-95.65%25-brightgreen.svg)](#results)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **High-accuracy crack detection system using EfficientNet-B0 transfer learning for industrial surface inspection.**

![Crack Detection Banner](https://img.shields.io/badge/🔬-Crack%20Detection-orange?style=for-the-badge)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Training](#-training)
- [Inference](#-inference)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements an automated **crack detection system** for industrial surface inspection using deep learning. The model can classify images as either containing **cracks** or being **crack-free** with high accuracy.

### Key Features

- ✅ **95.65% Accuracy** on validation set
- ⚡ **Fast Training** - Converges in ~1 epoch with transfer learning
- 🎯 **High Precision** - 100% precision for crack detection
- 🔧 **Easy to Use** - Simple training and inference scripts
- 📦 **Lightweight** - EfficientNet-B0 backbone (~20MB)

---

## 📊 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | **95.65%** |
| **Precision (crack)** | 100% |
| **Recall (crack)** | 91% |
| **Precision (without_crack)** | 92% |
| **Recall (without_crack)** | 100% |
| **F1-Score (macro)** | 96% |

### Classification Report

```
               precision    recall  f1-score   support

        crack       1.00      0.91      0.95        57
without_crack       0.92      1.00      0.96        58

     accuracy                           0.96       115
    macro avg       0.96      0.96      0.96       115
 weighted avg       0.96      0.96      0.96       115
```

### Confusion Matrix

|  | Predicted: Crack | Predicted: No Crack |
|--|------------------|---------------------|
| **Actual: Crack** | 52 | 5 |
| **Actual: No Crack** | 0 | 58 |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/The-Peacemaker/crack_detection.git
   cd crack_detection
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### Quick Start - Training

```bash
python fast_train.py
```

This will:
- Load the dataset from `./dataset/`
- Train EfficientNet-B0 with transfer learning
- Save the best model to `./output/best_crack_detector.pt`
- Print accuracy and classification report

### Inference on New Images

```python
from inference import CrackDetector

# Load model
detector = CrackDetector('output/best_crack_detector.pt')

# Predict single image
result = detector.predict('path/to/image.jpg')
print(f"Prediction: {result['class']} ({result['confidence']:.1%})")

# Predict folder
results = detector.predict_folder('path/to/folder/')
```

---

## 📁 Dataset

### Structure

```
dataset/
├── crack/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── without_crack/
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```

### Statistics

| Class | Images |
|-------|--------|
| crack | 285 |
| without_crack | 287 |
| **Total** | **572** |

### Data Split

- **Training**: 80% (457 images)
- **Validation**: 20% (115 images)

---

## 🏗️ Model Architecture

### EfficientNet-B0 (Transfer Learning)

```
EfficientNet-B0 (pretrained on ImageNet)
│
├── Features (frozen layers 0-5)
│   └── Convolutional blocks with SE attention
│
├── Features (trainable layers 6+)
│   └── Fine-tuned for crack detection
│
└── Classifier
    ├── Dropout (p=0.3)
    └── Linear(1280 → 2)
```

### Why EfficientNet-B0?

1. **Efficient** - Optimal accuracy/speed trade-off
2. **Pretrained** - Leverages ImageNet features
3. **Compact** - Only ~20MB model size
4. **Fast** - Quick inference time

---

## 🎓 Training

### Configuration

```python
BATCH_SIZE = 32
EPOCHS = 25
IMG_SIZE = 224
LEARNING_RATE = 0.001
OPTIMIZER = AdamW (weight_decay=0.01)
SCHEDULER = CosineAnnealingLR
```

### Data Augmentation

- Random Horizontal Flip (p=0.5)
- Random Vertical Flip (p=0.5)
- Random Rotation (±15°)
- Color Jitter (brightness, contrast, saturation)
- Random Affine (translation, scale)
- Normalization (ImageNet stats)

### Training Command

```bash
python fast_train.py
```

### Expected Output

```
============================================================
⚡ FAST CRACK DETECTION TRAINING
============================================================
Device: cpu

📁 Preparing data...
Classes: ['crack', 'without_crack']
Total images: 572
Train: 457, Val: 115

🔧 Building EfficientNet-B0 model...

🚀 Training...
--------------------------------------------------
Epoch  1/25 | Train Acc: 90.4% | Val Acc: 95.7% ⭐ BEST

🎯 Reached 95.7% - stopping early!

========================================
🎯 FINAL ACCURACY: 95.65%
========================================
✅ TARGET ACHIEVED! Accuracy ≥ 92%! 🎉
```

---

## 🔮 Inference

### Python API

```python
import torch
from torchvision import transforms, models
from PIL import Image

# Load model
model = models.efficientnet_b0()
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.3),
    torch.nn.Linear(1280, 2)
)
model.load_state_dict(torch.load('output/best_crack_detector.pt'))
model.eval()

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Predict
img = Image.open('test_image.jpg').convert('RGB')
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    _, predicted = output.max(1)
    
classes = ['crack', 'without_crack']
print(f"Prediction: {classes[predicted.item()]}")
```

### Using Inference Script

```bash
python inference.py --image path/to/image.jpg
python inference.py --folder path/to/folder/
```

---

## 📂 Project Structure

```
crack_detection/
│
├── dataset/                    # Dataset folder
│   ├── crack/                  # Crack images
│   └── without_crack/          # Non-crack images
│
├── output/                     # Output directory
│   └── best_crack_detector.pt  # Trained model weights
│
├── fast_train.py              # Main training script
├── inference.py               # Inference utilities
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── LICENSE                    # MIT License
```

---

## 🛠️ Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
scikit-learn>=1.0.0
Pillow>=9.0.0
```

---

## 📈 Future Improvements

- [ ] Add YOLOv8 object detection for crack localization
- [ ] Implement Grad-CAM visualization
- [ ] Create web interface with Gradio/Streamlit
- [ ] Add mobile deployment (ONNX/TensorRT)
- [ ] Expand dataset with more crack types
- [ ] Implement ensemble methods

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👏 Acknowledgments

- EfficientNet architecture by Google Research
- PyTorch team for the deep learning framework
- Dataset contributors

---

## 📧 Contact

**The-Peacemaker** - [GitHub](https://github.com/The-Peacemaker)

Project Link: [https://github.com/The-Peacemaker/crack_detection](https://github.com/The-Peacemaker/crack_detection)

---

<p align="center">
  Made with ❤️ for industrial inspection
</p>

<p align="center">
  ⭐ Star this repo if you found it useful!
</p>
