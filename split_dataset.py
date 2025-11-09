import os, shutil, random

BASE_DIR = "datasets_full"
DEST_DIR = "datasets"

# Create dirs if not exist
for split in ["train", "val", "test"]:
    for cls in ["Benign", "Malignant"]:
        os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)

# % split
train_split, val_split, test_split = 0.7, 0.15, 0.15

for cls in ["Benign", "Malignant"]:
    src_dir = os.path.join(BASE_DIR, cls)
    files = os.listdir(src_dir)
    random.shuffle(files)
    n_total = len(files)
    n_train = int(train_split * n_total)
    n_val = int(val_split * n_total)
    n_test = n_total - n_train - n_val

    print(f"{cls}: {n_total} images → train:{n_train}, val:{n_val}, test:{n_test}")

    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train+n_val],
        "test": files[n_train+n_val:]
    }

    for split, items in splits.items():
        for fname in items:
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(DEST_DIR, split, cls, fname)
            shutil.copy(src_path, dst_path)
