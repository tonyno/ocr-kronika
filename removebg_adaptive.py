import cv2
import numpy as np
import sys
import os

def remove_paper_background_adaptive(input_path, output_path, block_size=11, c_value=2, fill_strokes=True):
    """
    Remove paper background from scanned handwritten document using adaptive thresholding.
    Ensures white background with black text.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output image
        block_size: Size of neighborhood for adaptive threshold (must be odd, default 11)
        c_value: Constant subtracted from mean (sensitivity). Lower = more sensitive/fills more (default 2)
        fill_strokes: Apply morphological closing to fill in stroke gaps (default True)
    """
    # Read the image
    img = cv2.imread(input_path)
    
    if img is None:
        raise ValueError(f"Could not read image from {input_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1
    
    # Adaptive thresholding
    # THRESH_BINARY_INV: pixels above threshold become 0 (black), below become 255 (white)
    # This gives us white text on black background, which we'll invert
    binary = cv2.adaptiveThreshold(
        gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        block_size,
        c_value
    )
    
    # Fill in the strokes using morphological closing
    if fill_strokes:
        # First pass: fill small gaps
        kernel1 = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel1, iterations=1)
        
        # Second pass: fill slightly larger gaps
        kernel2 = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel2, iterations=1)
    
    # Remove small noise (opening)
    kernel_noise = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise)
    
    # Check if background is black (most pixels are black) and invert if needed
    # Sample corners and edges to determine background color
    h, w = cleaned.shape
    corners = np.concatenate([
        cleaned[0:min(50, h//10), 0:min(50, w//10)].flatten(),  # top-left
        cleaned[0:min(50, h//10), max(0, w-50):w].flatten(),    # top-right
        cleaned[max(0, h-50):h, 0:min(50, w//10)].flatten(),    # bottom-left
        cleaned[max(0, h-50):h, max(0, w-50):w].flatten()       # bottom-right
    ])
    # If more than 50% of corner pixels are black (0), background is black, so invert
    if np.mean(corners) < 127:
        cleaned = cv2.bitwise_not(cleaned)
    
    # Convert back to RGB for saving
    result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
    
    # Save the result
    cv2.imwrite(output_path, result)
    print(f"Background removed! Output saved to: {output_path}")
    print(f"  Parameters: block_size={block_size}, c_value={c_value}, fill_strokes={fill_strokes}")


if __name__ == "__main__":
    # Default values
    input_path = './input'
    output_path = './output'
    block_size = 11
    c_value = 2
    fill_strokes = True
    
    try:
        for file in os.listdir(input_path):
            if file.endswith('.jpg') or file.endswith('.png'):
           
                file_with_path = os.path.join(input_path, file)
                output_file_with_path = os.path.join(output_path, file)
                print(f"Processing {file} from {file_with_path} to {output_file_with_path}")
                remove_paper_background_adaptive(file_with_path, output_file_with_path, block_size, c_value, fill_strokes)
        # remove_paper_background_adaptive(input_path, output_path, block_size, c_value, fill_strokes)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

