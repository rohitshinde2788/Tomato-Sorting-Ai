import cv2
import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image

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

print("Device:", device)

# ==========================================
# LOAD MODEL
# ==========================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

classes = checkpoint["classes"]

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
# TRANSFORM
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
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Camera could not be opened.")

    exit()

print("\nCamera started.")
print("Press Q to quit.")

# ==========================================
# LIVE LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Could not read frame.")

        break

    # --------------------------------------
    # OpenCV BGR -> RGB
    # --------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------
    # PIL IMAGE
    # --------------------------------------

    image = Image.fromarray(
        rgb_frame
    )

    # --------------------------------------
    # PREPROCESS
    # --------------------------------------

    image_tensor = transform(
        image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        device
    )

    # --------------------------------------
    # PREDICTION
    # --------------------------------------

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

    predicted_class = classes[
        prediction.item()
    ]

    confidence_value = (
        confidence.item() * 100
    )

    # --------------------------------------
    # DISPLAY
    # --------------------------------------

    text = (
        f"{predicted_class} "
        f"{confidence_value:.1f}%"
    )

    cv2.putText(
        frame,
        text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow(
        "Tomato Classification",
        frame
    )

    # --------------------------------------
    # EXIT
    # --------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break

# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()

print("Camera stopped.")