import os
import shutil
from pathlib import Path


def create_directory_structure():
    """Create the required directory structure if it doesn't exist."""
    directories = [
        "dataset-v2/train/images",
        "dataset-v2/train/labels",
        "dataset-v2/test/images",
        "dataset-v2/test/labels",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def copy_files(src_dir, train_dir, test_dir):
    """Copy files from source directory to train and test directories."""
    test_folder = "apriltags_p3_mp4"

    # Walk through all subdirectories in the labels directory
    for root, dirs, files in os.walk(src_dir):
        # Get the subfolder name
        subfolder = os.path.basename(root)

        if not files:  # Skip if no files in directory
            continue

        # Determine if this is a test or train subfolder
        is_test = subfolder == test_folder
        target_dir = test_dir if is_test else train_dir

        print(f"Processing {subfolder} -> {'test' if is_test else 'train'}")

        # Process each file
        for file in files:
            # Skip non-image and non-label files
            if not (file.endswith((".txt", ".jpg", ".png", ".jpeg"))):
                continue

            src_path = os.path.join(root, file)

            # Determine if it's an image or label file
            is_label = file.endswith(".txt")
            dst_dir = os.path.join(target_dir, "labels" if is_label else "images")

            # Create unique filename to avoid conflicts
            base_name = f"{subfolder}_{file}"
            dst_path = os.path.join(dst_dir, base_name)

            # Copy the file
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"Error copying {src_path}: {e}")


def main():
    os.chdir("..")
    # Create directory structure
    create_directory_structure()

    # Set up paths
    labels_dir = "labels"
    train_dir = "dataset-v2/train"
    test_dir = "dataset-v2/test"

    # Copy files
    copy_files(labels_dir, train_dir, test_dir)

    # Print summary
    train_images = len(os.listdir("dataset-v2/train/images"))
    train_labels = len(os.listdir("dataset-v2/train/labels"))
    test_images = len(os.listdir("dataset-v2/test/images"))
    test_labels = len(os.listdir("dataset-v2/test/labels"))

    print("\nDataset creation complete!")
    print(f"Train set: {train_images} images, {train_labels} labels")
    print(f"Test set: {test_images} images, {test_labels} labels")


if __name__ == "__main__":
    main()
