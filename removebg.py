import cv2
import numpy as np
from PIL import Image
import sys

def remove_paper_background(input_path, output_path, method='adaptive', block_size=11, c_value=2, fill_strokes=True):
    """
    Remove paper background from scanned handwritten document.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output image
        method: 'adaptive' (default) or 'otsu' - thresholding method
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
    
    if method == 'adaptive':
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
        
        # Adaptive thresholding - works well for varying lighting
        # Lower C value = more sensitive = fills strokes better
        # Using THRESH_BINARY - we'll invert at the end to ensure correct colors
        binary = cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            block_size,
            c_value
        )
    else:
        # Otsu's thresholding - automatically determines threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Fill in the strokes using morphological closing
    if fill_strokes:
        # Use a small kernel to fill gaps inside strokes without merging separate letters
        # First pass: fill small gaps
        kernel1 = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel1, iterations=1)
        
        # Second pass: fill slightly larger gaps (adjust size if needed)
        kernel2 = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel2, iterations=1)
    
    # Remove small noise (opening) - use very small kernel to avoid affecting text
    kernel_noise = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise)
    
    # Invert to ensure white background and black text
    # THRESH_BINARY_INV produces white text on black, so we invert it
    cleaned = cv2.bitwise_not(cleaned)
    
    # Convert back to RGB for saving
    result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
    
    # Save the result
    cv2.imwrite(output_path, result)
    print(f"Background removed! Output saved to: {output_path}")
    print(f"  Parameters: block_size={block_size}, c_value={c_value}, fill_strokes={fill_strokes}")


def remove_paper_background_advanced(input_path, output_path):
    """
    Advanced method: Remove paper background while preserving text quality.
    Uses bilateral filter and adaptive thresholding for better results.
    """
    # Read the image
    img = cv2.imread(input_path)
    
    if img is None:
        raise ValueError(f"Could not read image from {input_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,  # block size
        10   # C constant
    )
    
    # Invert to ensure white background and black text
    binary = cv2.bitwise_not(binary)
    
    # Convert to RGB
    result = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    # Save
    cv2.imwrite(output_path, result)
    print(f"Background removed (advanced method)! Output saved to: {output_path}")


if __name__ == "__main__":
    # Default values
    input_path = '20251128_200048.jpg'
    output_path = 'output.png'
    method = 'adaptive'
    block_size = 11
    c_value = 2
    fill_strokes = True
    
    # Allow command line arguments
    # Usage: python removebg.py [input] [output] [method] [block_size] [c_value] [fill_strokes]
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    if len(sys.argv) > 3:
        method = sys.argv[3]  # 'adaptive', 'otsu', or 'advanced'
    if len(sys.argv) > 4:
        block_size = int(sys.argv[4])
    if len(sys.argv) > 5:
        c_value = int(sys.argv[5])
    if len(sys.argv) > 6:
        fill_strokes = sys.argv[6].lower() in ('true', '1', 'yes', 'y')
    
    try:
        if method == 'advanced':
            remove_paper_background_advanced(input_path, output_path)
        else:
            remove_paper_background(input_path, output_path, method, block_size, c_value, fill_strokes)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
