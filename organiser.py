import os
import shutil

# Root source directory with subfolders containing image files
source_root = r'D:\CodAlpha\Task2\MelSpectrograms'
destination_root = r'D:\CodAlpha\Task2\Training and testing\Training'

# Mapping from emotion codes (from the 3rd field in the filename) to emotion labels
emotion_code_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Ensure destination root exists
os.makedirs(destination_root, exist_ok=True)

# Traverse all subdirectories
for root, dirs, files in os.walk(source_root):
    for filename in files:
        if filename.lower().endswith('.png'):
            try:
                parts = filename.split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    emotion_name = emotion_code_map.get(emotion_code, 'unknown')

                    # Create target folder based on emotion name
                    target_folder = os.path.join(destination_root, emotion_name)
                    os.makedirs(target_folder, exist_ok=True)

                    # Move file to the new location
                    src_path = os.path.join(root, filename)
                    dst_path = os.path.join(target_folder, filename)
                    shutil.move(src_path, dst_path)

                    print(f"Moved {filename} → {emotion_name}")
                else:
                    print(f"Skipping {filename}: unexpected format")
            except Exception as e:
                print(f"Error with {filename}: {e}")
