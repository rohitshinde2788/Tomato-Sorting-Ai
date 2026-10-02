from pathlib import Path
from PIL import Image, ImageDraw
import random
import math

DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("dataset_samples")

CLASSES = [
    "Damaged",
    "Old",
    "Ripe",
    "Unripe"
]

SAMPLES_PER_CLASS = 12
THUMB_SIZE = 180
COLUMNS = 4


def get_images(folder):
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    return [
        p for p in folder.iterdir()
        if p.suffix.lower() in extensions
    ]


OUTPUT_DIR.mkdir(exist_ok=True)


for class_name in CLASSES:

    folder = DATASET_DIR / "train" / class_name

    images = get_images(folder)

    samples = random.sample(
        images,
        min(SAMPLES_PER_CLASS, len(images))
    )

    rows = math.ceil(
        len(samples) / COLUMNS
    )

    canvas = Image.new(
        "RGB",
        (
            COLUMNS * THUMB_SIZE,
            rows * (THUMB_SIZE + 30)
        ),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    for index, image_path in enumerate(samples):

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            image.thumbnail(
                (THUMB_SIZE - 10,
                 THUMB_SIZE - 10)
            )

            x = (
                (index % COLUMNS)
                * THUMB_SIZE
            )

            y = (
                (index // COLUMNS)
                * (THUMB_SIZE + 30)
            )

            image_x = (
                x +
                (THUMB_SIZE - image.width) // 2
            )

            image_y = (
                y +
                (THUMB_SIZE - image.height) // 2
            )

            canvas.paste(
                image,
                (image_x, image_y)
            )

            draw.text(
                (x + 5, y + THUMB_SIZE),
                image_path.name[:22],
                fill="black"
            )

        except Exception as e:

            print(
                "Could not read:",
                image_path,
                e
            )

    output_file = (
        OUTPUT_DIR /
        f"{class_name}.jpg"
    )

    canvas.save(
        output_file,
        quality=95
    )

    print(
        f"Created: {output_file}"
    )


print("\nSample generation completed.")