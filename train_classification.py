import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import Counter


# ==========================================
# CONFIGURATION
# ==========================================

TRAIN_DIR = "dataset/train"
VALID_DIR = "dataset/valid"

MODEL_DIR = "models"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001


# ==========================================
# CREATE MODEL DIRECTORY
# ==========================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# GPU / CPU
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
# TRANSFORMS
# ==========================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


valid_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.225],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# DATASET
# ==========================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

valid_dataset = datasets.ImageFolder(
    VALID_DIR,
    transform=valid_transform
)


print("\nClasses:")
print(train_dataset.classes)

print(
    "\nTraining images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(valid_dataset)
)


# ==========================================
# CHECK CLASS COUNTS
# ==========================================

class_counts = Counter(
    train_dataset.targets
)

print("\nTraining class distribution:")

for class_index, class_name in enumerate(
    train_dataset.classes
):
    print(
        f"{class_name}: "
        f"{class_counts[class_index]}"
    )


# ==========================================
# CLASS WEIGHTS
# ==========================================

total_samples = len(train_dataset)
num_classes = len(train_dataset.classes)

class_weights = []

for class_index in range(num_classes):

    count = class_counts[class_index]

    weight = total_samples / (
        num_classes * count
    )

    class_weights.append(weight)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)


print("\nClass weights:")

for class_name, weight in zip(
    train_dataset.classes,
    class_weights
):
    print(
        f"{class_name}: "
        f"{weight.item():.4f}"
    )


# ==========================================
# DATALOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ==========================================
# MODEL
# ==========================================

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


# ==========================================
# FINAL CLASSIFICATION LAYER
# ==========================================

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)


model = model.to(device)


# ==========================================
# LOSS FUNCTION
# ==========================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ==========================================
# OPTIMIZER
# ==========================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# TRAINING
# ==========================================

best_accuracy = 0.0


print("\n====================================")
print("5-Class Training Started")
print("====================================\n")


for epoch in range(EPOCHS):

    # ======================================
    # TRAIN
    # ======================================

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0


    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        running_loss += loss.item()


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_accuracy = (
        100 * correct / total
    )


    # ======================================
    # VALIDATION
    # ======================================

    model.eval()

    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for images, labels in valid_loader:

            images = images.to(device)
            labels = labels.to(device)


            outputs = model(images)


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_accuracy = (
        100 * val_correct / val_total
    )


    # ======================================
    # PRINT RESULTS
    # ======================================

    epoch_loss = (
        running_loss /
        len(train_loader)
    )


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Acc: {val_accuracy:.2f}%"
    )


    # ======================================
    # SAVE BEST MODEL
    # ======================================

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "classes":
                    train_dataset.classes,

                "image_size":
                    IMAGE_SIZE
            },

            os.path.join(
                MODEL_DIR,
                "best_model.pth"
            )
        )


        print(
            "✓ Best model saved!"
        )


# ==========================================
# TRAINING COMPLETE
# ==========================================

print("\n====================================")
print("Training completed!")

print(
    f"Best Validation Accuracy: "
    f"{best_accuracy:.2f}%"
)

print("====================================")

print(
    "\nModel saved at:"
    "\nmodels/best_model.pth"
)