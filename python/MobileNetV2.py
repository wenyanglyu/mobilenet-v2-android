import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import urllib.request

# Load Pretrained Model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.eval()  # Set to evaluation mode

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load ImageNet Class Labels Dynamically
label_file = "imagenet_classes.txt"
if not os.path.exists(label_file):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", label_file)

# Read ImageNet labels
with open(label_file, "r") as f:
    imagenet_labels = [s.strip() for s in f.readlines()]

# Transform Input Image (Standard Preprocessing for MobileNetV2)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load and Preprocess Image
image_path = "cock.jpeg"
image = Image.open(image_path)
image = transform(image).unsqueeze(0)  # Add batch dimension

# Move input image to the same device as the model
image = image.to(device)

# Make Prediction
with torch.no_grad():
    output = model(image)

# Apply Softmax to Get Probabilities
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Get Top-5 Predictions
top5_prob, top5_catid = torch.topk(probabilities, 5)

# Print Predictions
print("\nTop-5 Predictions:")
for i in range(top5_prob.size(0)):
    print(f"{i+1}. {imagenet_labels[top5_catid[i].item()]} ({top5_prob[i].item()*100:.2f}%)")
