import os
import shutil

VAL_DIR = "./data/tiny-imagenet-200/val"
IMG_DIR = os.path.join(VAL_DIR, "images")
ANN_FILE = os.path.join(VAL_DIR, "val_annotations.txt")

with open(ANN_FILE) as f:
    lines = f.readlines()

for line in lines:
    img, cls = line.split()[:2]
    cls_dir = os.path.join(VAL_DIR, cls)
    os.makedirs(cls_dir, exist_ok=True)
    shutil.move(
        os.path.join(IMG_DIR, img),
        os.path.join(cls_dir, img)
    )

os.rmdir(IMG_DIR)
print("Tiny-ImageNet validation set fixed!")
