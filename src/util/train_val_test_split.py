import os
import argparse
import random
import shutil
from pathlib import Path
from tqdm import tqdm
from itertools import chain
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
IMG_EXTENTIONS = ['.png', '.jpg','.jpeg']
def split_dataset(source_dirs, target_dir, train_ratio, val_ratio, test_ratio, link_method="symlink", img_extention=IMG_EXTENTIONS):
    """
    Split dataset into train, val, and test sets for YOLO-style datasets.

    Args:
        source_dirs (str, Path, or list): One or more directories containing YOLO annotation (.txt) files and corresponding image files.
        target_dir (str or Path): Directory where the split datasets (train/val/test) will be stored.
        train_ratio (float): Proportion of the dataset to use for training. Should be between 0 and 1.
        val_ratio (float): Proportion of the dataset to use for validation. Should be between 0 and 1.
        test_ratio (float): Proportion of the dataset to use for testing. Should be between 0 and 1.
        link_method (str): 'symlink' to create symbolic links, 'copy' to copy files. Default is 'symlink'.
        img_extention (str or list): Image file extension(s) to look for (e.g., '.png', '.jpg'). Default supports common image formats.

    Raises:
        ValueError: If the sum of train_ratio, val_ratio, and test_ratio does not equal 1.0.
        ValueError: If link_method is not 'symlink' or 'copy'.

    This function will:
        1. Find all annotation files and their corresponding image files in the source directories.
        2. Shuffle and split the dataset according to the provided ratios.
        3. Create the necessary directory structure in the target directory.
        4. Copy or symlink the files into train, val, and test folders for both images and labels.
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    # Find all .txt files and corresponding images
    if isinstance(source_dirs, (str, Path)):
        source_dirs = [Path(source_dirs)]
    source_dirs = [Path(s) for s in source_dirs]
    target_dir = Path(target_dir)
    annotation_files_ = [list(s.glob("*.txt")) for s in source_dirs]
    annotation_files = list(chain.from_iterable(annotation_files_))
    paired_files = []

    if isinstance(img_extention,str):
        for annotation_file in tqdm(annotation_files):
            image_file = annotation_file.with_suffix(img_extention)
            # print(image_file)
            if image_file.exists():
                paired_files.append((annotation_file, image_file))
    else:
        for annotation_file in tqdm(annotation_files):
            for ext in img_extention:
                image_file = annotation_file.with_suffix(ext)
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
        for annotation_file, image_file in tqdm(files):
            # print(annotation_file,image_file)
            img_dest = target_dir / "images" / split / f"{image_file.parent.name}_{image_file.name}"
            ann_dest = target_dir / "labels" / split /  f"{annotation_file.parent.name}_{annotation_file.name}"
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
    parser.add_argument('-s',"--source_dirs",nargs="+", type=str, help="Source directory containing YOLO annotations and images.")
    parser.add_argument('-t',"--target_dir", type=str, help="Target directory to store train, val, and test sets.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training set (default: 0.7).")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of validation set (default: 0.2).")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of test set (defa lt: 0.1).")
    parser.add_argument("-m", "--link_method", type=str, choices=["symlink", "copy"], default="symlink",
                        help="Link method: 'symlink' for symbolic link, 'copy' for file copy (default: symlink).")

    args = parser.parse_args()

    split_dataset(args.source_dirs, args.target_dir, args.train_ratio, args.val_ratio, args.test_ratio, args.link_method)

if __name__ == "__main__":
    main()