import cv2
import numpy as np
import sys
import os

def detect_and_remove_borders(gray, min_line_length_ratio=0.25):
    """
    Detect and remove border lines (vertical and horizontal) from the image.
    Finds the main content area and removes lines/rectangles around it.
    
    Args:
        gray: Grayscale image
        min_line_length_ratio: Minimum line length as ratio of image dimension (default 0.25)
    
    Returns:
        Mask with borders removed (255 where content is, 0 where borders are)
    """
    h, w = gray.shape
    mask = np.ones((h, w), dtype=np.uint8) * 255
    
    # Create a binary version for line detection
    # Use adaptive threshold to get text and lines
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Detect long vertical lines (potential left/right borders)
    # Use a tall, thin kernel to detect vertical lines
    vertical_kernel_length = min(int(h * 0.4), 200)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, vertical_kernel_length))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    
    # Detect long horizontal lines (potential top/bottom borders)
    # Use a wide, short kernel to detect horizontal lines
    horizontal_kernel_length = min(int(w * 0.4), 200)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_length, 3))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    # Remove vertical lines near left/right edges (likely borders)
    edge_margin = int(w * 0.08)  # Check within 8% of edges
    vertical_mask = vertical_lines.copy()
    # Only consider lines in the edge regions
    vertical_mask[:, edge_margin:w-edge_margin] = 0
    # Remove these lines from the mask
    mask[vertical_mask > 0] = 0
    
    # Remove horizontal lines near top/bottom edges (likely borders)
    edge_margin_v = int(h * 0.08)  # Check within 8% of edges
    horizontal_mask = horizontal_lines.copy()
    # Only consider lines in the edge regions
    horizontal_mask[edge_margin_v:h-edge_margin_v, :] = 0
    # Remove these lines from the mask
    mask[horizontal_mask > 0] = 0
    
    # Also use HoughLinesP for more precise detection of straight border lines
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    min_line_length = int(min(h, w) * min_line_length_ratio)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=int(min(h, w) * 0.2),
                            minLineLength=min_line_length,
                            maxLineGap=30)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Check if it's a vertical line (mostly vertical)
            if abs(x2 - x1) < 15 and abs(y2 - y1) > h * min_line_length_ratio:
                x_pos = (x1 + x2) // 2
                # If near left or right edge, it's likely a border
                if x_pos < w * 0.12 or x_pos > w * 0.88:
                    # Remove this line from mask with wider stroke
                    cv2.line(mask, (x1, y1), (x2, y2), 0, 20)
            
            # Check if it's a horizontal line (mostly horizontal)
            elif abs(y2 - y1) < 15 and abs(x2 - x1) > w * min_line_length_ratio:
                y_pos = (y1 + y2) // 2
                # If near top or bottom edge, it's likely a border
                if y_pos < h * 0.12 or y_pos > h * 0.88:
                    # Remove this line from mask with wider stroke
                    cv2.line(mask, (x1, y1), (x2, y2), 0, 20)
    
    # Dilate the removed areas slightly to ensure we remove the full border width
    kernel = np.ones((5, 5), np.uint8)
    mask_inv = 255 - mask
    mask_inv = cv2.dilate(mask_inv, kernel, iterations=2)
    mask = 255 - mask_inv
    
    return mask


def remove_paper_background_adaptive(input_path, output_path, block_size=11, c_value=2, fill_strokes=True):
    """
    Remove paper background from scanned handwritten document using adaptive thresholding.
    Ensures white background with black text, preserves grayscale for antialiasing,
    and removes border lines/rectangles.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output image (will be saved as JPEG)
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
    
    # Detect and remove borders
    border_mask = detect_and_remove_borders(gray)
    
    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1
    
    # Create a mask for text detection using adaptive thresholding
    # This will help us identify text vs background
    binary_mask = cv2.adaptiveThreshold(
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
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel1, iterations=1)
        
        # Second pass: fill slightly larger gaps
        kernel2 = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel2, iterations=1)
    
    # Remove small noise (opening)
    kernel_noise = np.ones((1, 1), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_noise)
    
    # Check if background is black (most pixels are black) and invert if needed
    # Sample corners and edges to determine background color
    h, w = gray.shape
    corners = np.concatenate([
        binary_mask[0:min(50, h//10), 0:min(50, w//10)].flatten(),  # top-left
        binary_mask[0:min(50, h//10), max(0, w-50):w].flatten(),    # top-right
        binary_mask[max(0, h-50):h, 0:min(50, w//10)].flatten(),    # bottom-left
        binary_mask[max(0, h-50):h, max(0, w-50):w].flatten()       # bottom-right
    ])
    # If more than 50% of corner pixels are black (0), background is black, so invert
    if np.mean(corners) < 127:
        binary_mask = cv2.bitwise_not(binary_mask)
    
    # Apply border mask - remove borders from the binary mask
    binary_mask = cv2.bitwise_and(binary_mask, border_mask)
    
    # Create grayscale output with antialiasing
    # We want to preserve the original grayscale values for smooth text edges
    # Strategy: use original grayscale where text is detected, pure white elsewhere
    
    # Normalize the original grayscale to enhance contrast while preserving smoothness
    # Apply slight contrast enhancement to make text darker
    gray_enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=-20)
    
    # Create a soft mask from binary_mask for smoother blending
    # Dilate the binary mask slightly to include edge pixels
    kernel_soft = np.ones((2, 2), np.uint8)
    text_mask_soft = cv2.dilate(binary_mask, kernel_soft, iterations=1)
    text_mask_soft = cv2.GaussianBlur(text_mask_soft.astype(np.float32), (3, 3), 0.5)
    text_mask_soft = text_mask_soft / 255.0
    
    # Blend: where there's text, use enhanced grayscale (darker), elsewhere use white
    # Invert enhanced grayscale so darker pixels (text) become darker
    gray_inverted = 255 - gray_enhanced
    
    # Create result: blend between inverted grayscale (text) and white (background)
    result_gray = (text_mask_soft * gray_inverted + (1 - text_mask_soft) * 255).astype(np.uint8)
    
    # Invert back to get black text on white background
    result_gray = 255 - result_gray
    
    # Ensure 100% white background: set all non-text pixels to pure white
    # Use the original binary_mask (not the soft one) to ensure clean white background
    result_gray[binary_mask == 0] = 255
    
    # Convert to RGB (3 channels) for JPEG saving
    result = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2RGB)
    
    # Ensure output path ends with .jpg
    if not output_path.lower().endswith('.jpg') and not output_path.lower().endswith('.jpeg'):
        output_path = os.path.splitext(output_path)[0] + '.jpg'
    
    # Save as JPEG with high quality to preserve grayscale antialiasing
    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Background removed! Output saved to: {output_path}")
    print(f"  Parameters: block_size={block_size}, c_value={c_value}, fill_strokes={fill_strokes}")


if __name__ == "__main__":
    # Default values
    input_path = './input'
    output_path = './output'
    block_size = 80
    c_value = 12
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

