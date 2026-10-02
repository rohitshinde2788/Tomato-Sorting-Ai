import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import sys


# ==========================================
# CONFIG
# ==========================================

MODEL_PATH = "models/best_model.pth"
IMAGE_SIZE = 224


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("====================================")
print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print("====================================")


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

classes = checkpoint["classes"]

print("Classes:", classes)


# ==========================================
# CREATE MODEL
# ==========================================

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


# ==========================================
# IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# GET IMAGE PATH
# ==========================================

if len(sys.argv) < 2:

    print("\nUsage:")
    print("python predict.py image.jpg")

    sys.exit()


image_path = sys.argv[1]


# ==========================================
# LOAD IMAGE
# ==========================================

try:

    image = Image.open(
        image_path
    ).convert("RGB")

except Exception as e:

    print("Error loading image:")
    print(e)

    sys.exit()


# ==========================================
# PREPROCESS IMAGE
# ==========================================

image_tensor = transform(
    image
).unsqueeze(0)

image_tensor = image_tensor.to(device)


# ==========================================
# PREDICTION
# ==========================================

with torch.no_grad():

    output = model(
        image_tensor
    )

    probabilities = torch.softmax(
        output,
        dim=1
    )

    confidence, prediction = torch.max(
        probabilities,
        1
    )

print("\nAll Class Probabilities:")

for i, class_name in enumerate(classes):
    probability = probabilities[0][i].item() * 100

    print(
        f"{class_name}: {probability:.2f}%"
    )

    
# ==========================================
# RESULT
# ==========================================

predicted_class = classes[
    prediction.item()
]

confidence_value = (
    confidence.item() * 100
)


print("\n====================================")
print("       TOMATO PREDICTION")
print("====================================")

print(
    "Predicted Class:",
    predicted_class
)

print(
    "Confidence:",
    f"{confidence_value:.2f}%"
)

print("====================================")