"""Save important artifacts to Google Drive for persistence."""
import os
import shutil
import sys

# Mount Google Drive
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as e:
    print(f"Could not mount Google Drive: {e}")
    print("Run this script in Google Colab.")
    sys.exit(1)

# Create Zensei directory on Drive
drive_dir = "/content/drive/MyDrive/zensei_artifacts"
os.makedirs(drive_dir, exist_ok=True)

# Copy tokenizer files
tokenizer_dir = os.path.join(drive_dir, "tokenizer")
os.makedirs(tokenizer_dir, exist_ok=True)

files_to_copy = [
    ("data/tokenizer/zensei_ja_sp.model", tokenizer_dir),
    ("data/tokenizer/zensei_ja_sp.vocab", tokenizer_dir),
]

for src, dst_dir in files_to_copy:
    if os.path.exists(src):
        shutil.copy2(src, dst_dir)
        print(f"Saved: {src} -> {dst_dir}/")
    else:
        print(f"Not found: {src}")

print(f"\nAll artifacts saved to Google Drive: {drive_dir}")
