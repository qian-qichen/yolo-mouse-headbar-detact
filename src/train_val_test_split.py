import os
import argparse
import random
import shutil
from pathlib import Path

def create_symlink(src, dest):
    """Create a symbolic link."""
    src = os.path.abspath(src)
    try:
        os.symlink(src, dest)
    except FileExistsError:
        print(f"Symbolic link {dest} already exists. Skipping creation.")

def copy_file(src, dest):
    """Copy file from src to dest."""
    if not dest.exists():
        shutil.copy2(src, dest)
    else:
        print(f"File {dest} already exists. Skipping copy.")

def split_dataset(source_dir, target_dir, train_ratio, val_ratio, test_ratio, link_method="symlink"):
    """Split dataset into train, val, and test sets."""
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    # Find all .txt files and corresponding images
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    annotation_files = list(source_dir.glob("*.txt"))
    paired_files = []

    for annotation_file in annotation_files:
        image_file = source_dir / (annotation_file.stem + ".png")
        # print(image_file)
        if image_file.exists():
            paired_files.append((annotation_file, image_file))
    # print(paired_files)
    # Shuffle files
    random.shuffle(paired_files)

    # Split files
    total_files = len(paired_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)

    train_files = paired_files[:train_end]
    val_files = paired_files[train_end:val_end]
    test_files = paired_files[val_end:]

    # Create target directories
    for split in ["train", "val", "test"]:
        (target_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (target_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Create symlinks or copy files
    for split, files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
        for annotation_file, image_file in files:
            # print(annotation_file,image_file)
            img_dest = target_dir / "images" / split / image_file.name
            ann_dest = target_dir / "labels" / split / annotation_file.name
            if link_method == "symlink":
                create_symlink(image_file, img_dest)
                create_symlink(annotation_file, ann_dest)
            elif link_method == "copy":
                copy_file(image_file, img_dest)
                copy_file(annotation_file, ann_dest)
            else:
                raise ValueError("link_method must be 'symlink' or 'copy'")

def main():
    parser = argparse.ArgumentParser(description="Split YOLO dataset into train, val, and test sets.")
    parser.add_argument("source_dir", type=str, help="Source directory containing YOLO annotations and images.")
    parser.add_argument("target_dir", type=str, help="Target directory to store train, val, and test sets.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of training set (default: 0.7).")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Ratio of validation set (default: 0.2).")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of test set (default: 0.1).")
    parser.add_argument("-m", "--link_method", type=str, choices=["symlink", "copy"], default="symlink",
                        help="Link method: 'symlink' for symbolic link, 'copy' for file copy (default: symlink).")

    args = parser.parse_args()

    split_dataset(args.source_dir, args.target_dir, args.train_ratio, args.val_ratio, args.test_ratio, args.link_method)

if __name__ == "__main__":
    main()