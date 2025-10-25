import cv2
import os
import glob

# Input folder with JPG images
input_folder = "images/checkerboard_test"
# Output folder for PGM images
output_folder = "images/checkerboard_test_pgm"
os.makedirs(output_folder, exist_ok=True)

# Find all JPG/JPEG files
jpg_files = glob.glob(os.path.join(input_folder, "*.jpg")) + \
            glob.glob(os.path.join(input_folder, "*.jpeg"))

for jpg_file in jpg_files:
    # Read in grayscale
    img = cv2.imread(jpg_file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to read {jpg_file}")
        continue

    # Construct output path with .pgm extension
    base_name = os.path.basename(jpg_file)
    pgm_file = os.path.join(output_folder, os.path.splitext(base_name)[0] + ".pgm")

    # Save as PGM
    cv2.imwrite(pgm_file, img)
    print(f"Converted {jpg_file} → {pgm_file}")

print("All JPG images converted to PGM!")
