import os

DATASET_DIR = "dataset"

classes = [
    "Damaged",
    "Old",
    "Ripe",
    "Unripe",
    "No_Tomato"
]

splits = ["train", "valid", "test"]

print("\n===== DATASET CHECK =====\n")

total_all = 0

for split in splits:
    print(f"--- {split.upper()} ---")

    split_total = 0

    for class_name in classes:
        folder = os.path.join(DATASET_DIR, split, class_name)

        if not os.path.exists(folder):
            count = 0
            print(f"{class_name:12} : FOLDER NOT FOUND")
        else:
            files = [
                f for f in os.listdir(folder)
                if f.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                )
            ]

            count = len(files)
            print(f"{class_name:12} : {count}")

        split_total += count

    print(f"TOTAL {split}: {split_total}\n")
    total_all += split_total

print("==========================")
print(f"TOTAL DATASET: {total_all}")
print("==========================")