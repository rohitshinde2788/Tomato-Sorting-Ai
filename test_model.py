import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn

# ==========================================
# CONFIG
# ==========================================

TEST_DIR = "dataset/test"
MODEL_PATH = "models/best_model.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 32

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
# TRANSFORM
# ==========================================

test_transform = transforms.Compose([
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
# LOAD TEST DATASET
# ==========================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nClasses:")
print(test_dataset.classes)

print(
    "Test images:",
    len(test_dataset)
)

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
# TEST
# ==========================================

correct = 0
total = 0

class_correct = [0] * len(classes)
class_total = [0] * len(classes)

print("\nTesting started...\n")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

        # Per-class accuracy
        for label, prediction in zip(
            labels,
            predictions
        ):

            label_index = label.item()

            class_total[label_index] += 1

            if label == prediction:
                class_correct[label_index] += 1


# ==========================================
# OVERALL ACCURACY
# ==========================================

accuracy = (
    100 * correct / total
)

print("====================================")
print("TEST RESULTS")
print("====================================")

print(
    f"Test Accuracy: {accuracy:.2f}%"
)

print(
    f"Correct: {correct}/{total}"
)

print("\nClass-wise Accuracy:")

for i, class_name in enumerate(classes):

    if class_total[i] > 0:

        class_accuracy = (
            100 *
            class_correct[i] /
            class_total[i]
        )

        print(
            f"{class_name}: "
            f"{class_accuracy:.2f}% "
            f"({class_correct[i]}/"
            f"{class_total[i]})"
        )

print("====================================")