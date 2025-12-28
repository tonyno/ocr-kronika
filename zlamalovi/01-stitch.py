import cv2
import os
import glob
import numpy as np

def enhance_features(img):
    """Enhance image features to help with feature detection"""
    # Convert to grayscale for processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Convert back to BGR
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced_bgr

def stitch_document_images(image_folder, output_name="stitched_document.jpg", max_images=None, enhance=True):
    print("Loading images...")
    
    # 1. Get all image paths and sort them
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    # Reverse if needed (depends on capture order)
    # image_paths.reverse()
    print(f"Processing {len(image_paths)} images:")
    print(image_paths)
    
    if not image_paths:
        print("No images found! Check the folder path and file extension.")
        return

    images = []
    # 2. Load images and optionally resize to save memory
    scale_factor = 0.5  # 1.0 = Original size, 0.5 = Half size
    
    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not load {path}")
            continue
            
        if scale_factor != 1.0:
            width = int(img.shape[1] * scale_factor)
            height = int(img.shape[0] * scale_factor)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        
        # Enhance features if requested
        if enhance:
            img = enhance_features(img)
            
        images.append(img)
        print(f"Loaded image {i+1}/{len(image_paths)}: {os.path.basename(path)} ({img.shape[1]}x{img.shape[0]})")

    if len(images) < 2:
        print("Need at least 2 images to stitch!")
        return

    print(f"\nFound {len(images)} images. Starting stitching process...")
    print("This may take a while depending on resolution...")

    # Try different stitcher modes
    modes_to_try = [
        (cv2.Stitcher_SCANS, "SCANS (document mode)"),
        (cv2.Stitcher_PANORAMA, "PANORAMA (default mode)"),
    ]
    
    for mode, mode_name in modes_to_try:
        print(f"\nTrying {mode_name}...")
        stitcher = cv2.Stitcher_create(mode=mode)
        
        # Set confidence threshold (lower = more lenient, default is 1.0)
        # This helps when images have less overlap
        try:
            stitcher.setPanoConfidenceThresh(0.3)  # Lower threshold for better matching
        except:
            pass  # Some OpenCV versions don't support this
        
        # 4. Stitch
        status, result = stitcher.stitch(images)

        if status == cv2.Stitcher_OK:
            print(f"Stitching successful with {mode_name}!")
            cv2.imwrite(output_name, result)
            print(f"Saved as {output_name}")
            print(f"Result size: {result.shape[1]}x{result.shape[0]}")
            return True
        else:
            # Error handling
            error_messages = {
                1: "ERR_NEED_MORE_IMGS - Not enough matching features found",
                2: "ERR_HOMOGRAPHY_EST_FAIL - Could not estimate transformation",
                3: "ERR_CAMERA_PARAMS_ADJUST_FAIL - Camera parameter adjustment failed"
            }
            print(f"Stitching failed with {mode_name}. Error code: {status}")
            print(f"  {error_messages.get(status, 'Unknown error')}")
    
    # If all modes failed, try pairwise stitching
    print("\nAll modes failed. Attempting pairwise stitching as fallback...")
    if len(images) >= 2:
        print("Trying to stitch first two images...")
        stitcher = cv2.Stitcher_create(mode=cv2.Stitcher_SCANS)
        status, result = stitcher.stitch(images[:2])
        
        if status == cv2.Stitcher_OK:
            print("Pairwise stitching successful! You may need to stitch in smaller batches.")
            cv2.imwrite(output_name.replace('.jpg', '_pairwise.jpg'), result)
            print(f"Saved as {output_name.replace('.jpg', '_pairwise.jpg')}")
            return True
        else:
            print(f"Pairwise stitching also failed. Error code: {status}")
    
    print("\nStitching failed with all methods.")
    print("Tips:")
    print("  - Ensure images have at least 30% overlap")
    print("  - Check that images are in correct order")
    print("  - Try reducing the number of images processed at once")
    print("  - Ensure images have sufficient detail/texture")
    return False

# --- USAGE ---
# Replace with the path to your folder containing the images
folder_path = "input/" 
# Process all images (or set max_images=3 to test with just 3)
stitch_document_images(folder_path, max_images=3, enhance=True)