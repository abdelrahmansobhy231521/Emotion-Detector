import cv2
import os
from albumentations import Compose, ShiftScaleRotate, HorizontalFlip

# Main directory containing all subfolders with images
main_dataset_dir = r'D:\CodAlpha\Task2\Training and testing\Training'  # Change this if needed

# Define the augmentation pipeline
augmentation_pipeline = Compose([
    ShiftScaleRotate(
        shift_limit=0.2,
        scale_limit=0.2,
        rotate_limit=0,  # No rotation
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        p=1.0
    ),
    HorizontalFlip(p=0.5)
])

# Iterate over each folder in the main directory
for subfolder in os.listdir(main_dataset_dir):
    subfolder_path = os.path.join(main_dataset_dir, subfolder)

    # Skip if it's not a folder
    if not os.path.isdir(subfolder_path):
        continue

    print(f"📁 Processing folder: {subfolder}")

    # Process each image in the subfolder
    for filename in os.listdir(subfolder_path):
        if not filename.lower().endswith(('jpg', 'jpeg', 'png')):
            continue

        image_path = os.path.join(subfolder_path, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Skipping unreadable image: {image_path}")
            continue

        # Convert image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        base_name, ext = os.path.splitext(filename)

        # Generate 20 augmented versions
        for i in range(1, 21):
            augmented = augmentation_pipeline(image=image)
            aug_img = augmented['image']
            aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)

            # Create save path in the same folder
            aug_filename = f"{base_name}_aug_{i}{ext}"
            aug_path = os.path.join(subfolder_path, aug_filename)

            cv2.imwrite(aug_path, aug_img_bgr)

    print(f"✅ Done with folder: {subfolder}\n")
