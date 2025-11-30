import os
import shutil
from pathlib import Path

# Define source and destination directories
source_dir = Path("input_original")
dest_dir = Path("input_sorted")

# Create destination directory if it doesn't exist
dest_dir.mkdir(exist_ok=True)

# Get all files from source directory and sort alphabetically
files = sorted([f for f in source_dir.iterdir() if f.is_file()])

# Copy files with new names
for index, file_path in enumerate(files, start=1):
    # Generate new filename with leading zeros (page001.jpg, page002.jpg, etc.)
    new_filename = f"page{index:03d}.jpg"
    dest_path = dest_dir / new_filename
    
    # Copy the file
    shutil.copy2(file_path, dest_path)
    print(f"Copied {file_path.name} -> {new_filename}")

print(f"\nTotal files copied: {len(files)}")

