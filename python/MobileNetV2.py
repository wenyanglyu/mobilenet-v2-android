import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json

# Load Pretrained Model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.eval()  # Set to evaluation mode

# Load ImageNet Class Labels
imagenet_labels = []
with open("imagenet_labels.json", "r") as f:
    imagenet_labels = json.load(f)

# Transform Input Image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Load and Preprocess Image
image_path = "D:/game/mobilenet-v2-android/python/Pingzi.jpg"

image = Image.open(image_path)
image = transform(image).unsqueeze(0)  # Add batch dimension

# Make Prediction
with torch.no_grad():
    output = model(image)
    predicted_index = output.argmax(dim=1).item()
    predicted_label = imagenet_labels[str(predicted_index)]  # Convert index to string

print(f"Predicted Class: {predicted_label} (Index: {predicted_index})")
