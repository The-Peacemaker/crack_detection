"""
🔮 Crack Detection Inference Module
Easy-to-use inference for crack detection model
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import argparse


class CrackDetector:
    """
    Crack Detection inference class.
    
    Example:
        detector = CrackDetector('output/best_crack_detector.pt')
        result = detector.predict('image.jpg')
        print(result)
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the crack detector.
        
        Args:
            model_path: Path to the trained model weights (.pt file)
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = ['crack', 'without_crack']
        
        # Build model architecture
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.model.classifier[1].in_features, len(self.classes))
        )
        
        # Load weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Model loaded on {self.device}")
    
    def predict(self, image_path: str) -> dict:
        """
        Predict crack/no-crack for a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict with 'class', 'confidence', and 'probabilities'
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            confidence, predicted = probs.max(0)
        
        return {
            'class': self.classes[predicted.item()],
            'confidence': confidence.item(),
            'probabilities': {
                self.classes[i]: probs[i].item() 
                for i in range(len(self.classes))
            },
            'is_crack': predicted.item() == 0
        }
    
    def predict_folder(self, folder_path: str) -> list:
        """
        Predict crack/no-crack for all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            
        Returns:
            List of prediction results
        """
        folder = Path(folder_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        results = []
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() in image_extensions:
                result = self.predict(str(img_path))
                result['image'] = img_path.name
                result['path'] = str(img_path)
                results.append(result)
        
        return results
    
    def predict_batch(self, image_paths: list, batch_size: int = 16) -> list:
        """
        Predict crack/no-crack for multiple images efficiently.
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for inference
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            
            for path in batch_paths:
                img = Image.open(path).convert('RGB')
                tensor = self.transform(img)
                batch_tensors.append(tensor)
            
            batch = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch)
                probs = torch.softmax(outputs, dim=1)
                confidences, predictions = probs.max(1)
            
            for j, path in enumerate(batch_paths):
                results.append({
                    'image': Path(path).name,
                    'path': str(path),
                    'class': self.classes[predictions[j].item()],
                    'confidence': confidences[j].item(),
                    'is_crack': predictions[j].item() == 0
                })
        
        return results


def main():
    """Command-line interface for crack detection."""
    parser = argparse.ArgumentParser(description='Crack Detection Inference')
    parser.add_argument('--model', type=str, default='output/best_crack_detector.pt',
                        help='Path to model weights')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder of images')
    
    args = parser.parse_args()
    
    # Load detector
    detector = CrackDetector(args.model)
    
    if args.image:
        # Single image prediction
        result = detector.predict(args.image)
        print(f"\n🔍 Prediction for: {args.image}")
        print(f"   Class: {result['class']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Is Crack: {'Yes ⚠️' if result['is_crack'] else 'No ✅'}")
        
    elif args.folder:
        # Folder prediction
        results = detector.predict_folder(args.folder)
        
        print(f"\n📁 Results for folder: {args.folder}")
        print("-" * 50)
        
        crack_count = sum(1 for r in results if r['is_crack'])
        
        for r in results:
            status = "🔴 CRACK" if r['is_crack'] else "🟢 OK"
            print(f"  {r['image']}: {status} ({r['confidence']:.1%})")
        
        print("-" * 50)
        print(f"📊 Summary: {crack_count}/{len(results)} images have cracks")
        
    else:
        print("Please provide --image or --folder argument")
        print("Example: python inference.py --image test.jpg")
        print("Example: python inference.py --folder ./test_images/")


if __name__ == '__main__':
    main()
