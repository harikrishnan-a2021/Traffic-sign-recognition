import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Define the same CNN architecture used for training GTSRB dataset
# This is a simplified example for demonstration
class TrafficSignNet(nn.Module):
    def __init__(self, num_classes=43):
        super(TrafficSignNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
    
    
# Load a pre-trained model (you can replace this with your own trained model path)
model_path = Path("gtsrb_cnn.pth")
if not model_path.exists():
    raise FileNotFoundError(
        "Model file not found: gtsrb_cnn.pth. Place the trained weights in the project root."
    )

model = TrafficSignNet()
try:
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
except Exception as exc:
    raise RuntimeError(f"Failed to load model weights from {model_path}: {exc}") from exc
model.eval()    

# List of traffic sign class names (abbreviated for brevity)
classes = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Stop", "Yield", 
    "No entry", "No passing", "Right-of-way", "Roundabout", "Children crossing", "Turn right", 
    "Turn left", "Keep right", "Keep left", "Traffic signals", "Bumpy road", "Slippery road",
    "Road work", "Other warning"
] + ["Class " + str(i) for i in range(19, 43)]  # Generic placeholders for remaining classes


# Load and preprocess the test image
image_path = Path("test_sign.jpg")  # Replace with your test image
if not image_path.exists():
    raise FileNotFoundError(
        "Input image not found: test_sign.jpg. Place a test image in the project root."
    )

try:
    image = Image.open(image_path).convert('RGB')
except Exception as exc:
    raise RuntimeError(f"Failed to open image {image_path}: {exc}") from exc
 
# Transform the image to match model input
transform = transforms.Compose([
    transforms.Resize((32, 32)),           # Resize to model input size
    transforms.ToTensor(),                 # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5),  # Normalize channels
                         (0.5, 0.5, 0.5))
])
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
  
# Predict class using the model
with torch.no_grad():
    outputs = model(input_tensor)
    predicted_class = outputs.argmax(1).item()
 
# Display the result
plt.imshow(image)
plt.title(f"Predicted: {classes[predicted_class]}")
plt.axis('off')
plt.show()
 
# Print predicted label
print("\n🚦 Recognized Traffic Sign:")
print(classes[predicted_class])  