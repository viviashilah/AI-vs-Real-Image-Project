import os
import shutil
import random
from sklearn.model_selection import train_test_split

# ==================================================
# PATH DATASET
# ==================================================
AI_DIR = r"D:\Porto-project\AI vs Real-Image\dataset\Ai_generated_dataset"
REAL_DIR = r"D:\Porto-project\AI vs Real-Image\dataset\real_dataset"

# ==================================================
# OUTPUT FOLDER
# ==================================================
OUTPUT_DIR = "final_dataset"

# ==================================================
# SPLIT RATIO
# ==================================================
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ==================================================
# CREATE OUTPUT FOLDER
# ==================================================
classes = ['ai', 'real']

for split in ['train', 'val', 'test']:
    for cls in classes:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, cls),
            exist_ok=True
        )

# ==================================================
# FUNCTION TO COLLECT IMAGES
# ==================================================
def collect_images(base_dir):

    image_paths = []

    categories = os.listdir(base_dir)

    for category in categories:

        category_path = os.path.join(base_dir, category)

        if os.path.isdir(category_path):

            for img in os.listdir(category_path):

                img_path = os.path.join(category_path, img)

                image_paths.append(img_path)

    return image_paths

# ==================================================
# LOAD ALL IMAGES
# ==================================================
ai_images = collect_images(AI_DIR)
real_images = collect_images(REAL_DIR)

print(f"Total AI Images   : {len(ai_images)}")
print(f"Total Real Images : {len(real_images)}")

# ==================================================
# SPLIT FUNCTION
# ==================================================
def split_and_copy(images, label):

    random.shuffle(images)

    # TRAIN
    train_imgs, temp_imgs = train_test_split(
        images,
        test_size=(1 - TRAIN_SPLIT),
        random_state=42
    )

    # VALIDATION + TEST
    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=0.5,
        random_state=42
    )

    # ---------------------------------------------
    # COPY TRAIN
    # ---------------------------------------------
    for img_path in train_imgs:

        filename = os.path.basename(img_path)

        destination = os.path.join(
            OUTPUT_DIR,
            'train',
            label,
            filename
        )

        shutil.copy(img_path, destination)

    # ---------------------------------------------
    # COPY VALIDATION
    # ---------------------------------------------
    for img_path in val_imgs:

        filename = os.path.basename(img_path)

        destination = os.path.join(
            OUTPUT_DIR,
            'val',
            label,
            filename
        )

        shutil.copy(img_path, destination)

    # ---------------------------------------------
    # COPY TEST
    # ---------------------------------------------
    for img_path in test_imgs:

        filename = os.path.basename(img_path)

        destination = os.path.join(
            OUTPUT_DIR,
            'test',
            label,
            filename
        )

        shutil.copy(img_path, destination)

    print(f"\nLabel : {label}")
    print(f"Train : {len(train_imgs)}")
    print(f"Val   : {len(val_imgs)}")
    print(f"Test  : {len(test_imgs)}")

# ==================================================
# RUN SPLIT
# ==================================================
split_and_copy(ai_images, 'ai')
split_and_copy(real_images, 'real')

print("\nDataset splitting completed!")