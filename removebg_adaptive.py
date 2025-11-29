import cv2
import numpy as np
import sys

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


def remove_borders(input_path, output_path, min_line_length_ratio=0.3, line_thickness=3, use_hough=False):
    """
    Remove border lines from a scanned document image.
    Detects and removes long horizontal and vertical lines (borders) around the text.
    
    Args:
        input_path: Path to input image (should have white background, black text)
        output_path: Path to save output image
        min_line_length_ratio: Minimum line length as ratio of image dimension (default 0.3 = 30%)
                              Lines shorter than this won't be removed
        line_thickness: Thickness of lines to detect/remove (default 3 pixels)
        use_hough: If True, use Hough Line Transform for more precise detection (slower but more accurate)
    """
    # Read the image
    img = cv2.imread(input_path)
    
    if img is None:
        raise ValueError(f"Could not read image from {input_path}")
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    h, w = gray.shape
    min_horizontal_length = int(w * min_line_length_ratio)
    min_vertical_length = int(h * min_line_length_ratio)
    
    # Create mask for lines to remove (start with all zeros/black)
    line_mask = np.zeros_like(gray)
    
    if use_hough:
        # Method 1: Hough Line Transform (more precise but slower)
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_horizontal_length, line_thickness))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_thickness, min_vertical_length))
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine horizontal and vertical lines
        line_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
        
        # Use HoughLinesP for more precise line detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                minLineLength=min(min_horizontal_length, min_vertical_length),
                                maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is mostly horizontal or vertical
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dx > dy and dx >= min_horizontal_length:  # Horizontal line
                    cv2.line(line_mask, (x1, y1), (x2, y2), 255, line_thickness)
                elif dy > dx and dy >= min_vertical_length:  # Vertical line
                    cv2.line(line_mask, (x1, y1), (x2, y2), 255, line_thickness)
    else:
        # Method 2: Morphological operations (faster)
        # Detect horizontal lines using morphological opening with horizontal kernel
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_horizontal_length, line_thickness))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines using morphological opening with vertical kernel
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_thickness, min_vertical_length))
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine horizontal and vertical lines
        line_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
        
        # Dilate the mask slightly to ensure we remove the entire line
        dilate_kernel = np.ones((line_thickness + 2, line_thickness + 2), np.uint8)
        line_mask = cv2.dilate(line_mask, dilate_kernel, iterations=1)
    
    # Invert the mask: we want to keep everything except the lines
    # Then use inpainting to fill in the removed lines with background color
    # For binary images, we can simply set detected line pixels to white (255)
    result = gray.copy()
    result[line_mask > 0] = 255  # Replace detected lines with white background
    
    # Optional: Use inpainting for smoother results (works better for grayscale images)
    # Uncomment if you want smoother line removal:
    # result = cv2.inpaint(gray, line_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # Convert back to RGB if original was RGB
    if len(img.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    
    # Save the result
    cv2.imwrite(output_path, result)
    print(f"Borders removed! Output saved to: {output_path}")
    print(f"  Parameters: min_line_length_ratio={min_line_length_ratio}, line_thickness={line_thickness}, use_hough={use_hough}")


if __name__ == "__main__":
    # Default values
    input_path = '20251128_200048.jpg'
    output_path = 'output.png'
    block_size = 11
    c_value = 2
    fill_strokes = True
    
    # Allow command line arguments
    # Usage: python removebg_adaptive.py [input] [output] [block_size] [c_value] [fill_strokes]
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    if len(sys.argv) > 3:
        block_size = int(sys.argv[3])
    if len(sys.argv) > 4:
        c_value = int(sys.argv[4])
    if len(sys.argv) > 5:
        fill_strokes = sys.argv[5].lower() in ('true', '1', 'yes', 'y')
    
    try:
        # Step 1: Remove paper background
        remove_paper_background_adaptive(input_path, output_path, block_size, c_value, fill_strokes)
        
        # Step 2: Remove borders from the output
        import os
        import shutil
        base_name = os.path.splitext(output_path)[0]
        ext = os.path.splitext(output_path)[1]
        temp_output = f"{base_name}_temp{ext}"
        
        remove_borders(output_path, temp_output, min_line_length_ratio=0.3, line_thickness=3, use_hough=False)
        
        # Replace original output with border-removed version
        shutil.move(temp_output, output_path)
        print(f"\nFinal output (with borders removed) saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

