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
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Detect long vertical lines (potential left/right borders)
    vertical_kernel_length = min(int(h * 0.4), 200)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, vertical_kernel_length))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    
    # Detect long horizontal lines (potential top/bottom borders)
    horizontal_kernel_length = min(int(w * 0.4), 200)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_length, 3))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    # Remove vertical lines near left/right edges (likely borders)
    edge_margin = int(w * 0.08)
    vertical_mask = vertical_lines.copy()
    vertical_mask[:, edge_margin:w-edge_margin] = 0
    mask[vertical_mask > 0] = 0
    
    # Remove horizontal lines near top/bottom edges (likely borders)
    edge_margin_v = int(h * 0.08)
    horizontal_mask = horizontal_lines.copy()
    horizontal_mask[edge_margin_v:h-edge_margin_v, :] = 0
    mask[horizontal_mask > 0] = 0
    
    # Use HoughLinesP for more precise detection of straight border lines
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    min_line_length = int(min(h, w) * min_line_length_ratio)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=int(min(h, w) * 0.2),
                            minLineLength=min_line_length, maxLineGap=30)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Check if it's a vertical line (mostly vertical)
            if abs(x2 - x1) < 15 and abs(y2 - y1) > h * min_line_length_ratio:
                x_pos = (x1 + x2) // 2
                if x_pos < w * 0.12 or x_pos > w * 0.88:
                    cv2.line(mask, (x1, y1), (x2, y2), 0, 20)
            
            # Check if it's a horizontal line (mostly horizontal)
            elif abs(y2 - y1) < 15 and abs(x2 - x1) > w * min_line_length_ratio:
                y_pos = (y1 + y2) // 2
                if y_pos < h * 0.12 or y_pos > h * 0.88:
                    cv2.line(mask, (x1, y1), (x2, y2), 0, 20)
    
    # Dilate the removed areas slightly to ensure we remove the full border width
    kernel = np.ones((5, 5), np.uint8)
    mask_inv = 255 - mask
    mask_inv = cv2.dilate(mask_inv, kernel, iterations=2)
    mask = 255 - mask_inv
    
    return mask


def remove_paper_background(input_path, output_path, block_size=80, c_value=12, fill_strokes=True):
    """
    Remove paper background from scanned handwritten document.
    Ensures pure white background (255) with black text (with grayscale antialiasing).
    Removes border lines/rectangles around the main content.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output image (will be saved as JPEG)
        block_size: Size of neighborhood for adaptive threshold (must be odd, default 80)
        c_value: Constant subtracted from mean (sensitivity, default 12)
        fill_strokes: Apply morphological closing to fill in stroke gaps (default True)
    """
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image from {input_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    
    # Detect and remove borders
    border_mask = detect_and_remove_borders(gray.astype(np.uint8))
    
    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1
    
    # Create binary mask to identify text vs background
    # THRESH_BINARY_INV: pixels above threshold -> 0, below threshold -> 255
    binary_mask = cv2.adaptiveThreshold(
        gray.astype(np.uint8),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_value
    )
    
    # Fill in the strokes using morphological closing
    if fill_strokes:
        kernel1 = np.ones((2, 2), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel1, iterations=1)
        kernel2 = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel2, iterations=1)
    
    # Remove small noise
    kernel_noise = np.ones((1, 1), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_noise)
    
    # Determine if binary_mask needs inversion
    # Sample corners to check if background is black (0) or white (255)
    corners = np.concatenate([
        binary_mask[0:min(50, h//10), 0:min(50, w//10)].flatten(),
        binary_mask[0:min(50, h//10), max(0, w-50):w].flatten(),
        binary_mask[max(0, h-50):h, 0:min(50, w//10)].flatten(),
        binary_mask[max(0, h-50):h, max(0, w-50):w].flatten()
    ])
    
    # If corners are mostly black (0), background is black, so invert
    # We want: binary_mask 255 = text, 0 = background
    if np.mean(corners) < 127:
        binary_mask = cv2.bitwise_not(binary_mask)
    
    # Apply border mask to remove borders
    binary_mask = cv2.bitwise_and(binary_mask, border_mask)
    
    # STEP 1: Determine the correct orientation
    # Check original grayscale to see if we have dark text on light bg, or light text on dark bg
    # Sample corners (usually background) and center (usually has text)
    corner_samples = np.concatenate([
        gray[0:min(50, h//10), 0:min(50, w//10)].flatten(),
        gray[0:min(50, h//10), max(0, w-50):w].flatten(),
        gray[max(0, h-50):h, 0:min(50, w//10)].flatten(),
        gray[max(0, h-50):h, max(0, w-50):w].flatten()
    ])
    center_samples = gray[h//4:3*h//4, w//4:3*w//4].flatten()
    
    avg_corner = np.mean(corner_samples)
    avg_center = np.mean(center_samples)
    
    # If corners are darker than center, we likely have light text on dark background
    # We want dark text on light background, so we'll need to invert
    original_has_light_text = avg_corner < avg_center
    
    # STEP 2: Create properly oriented grayscale
    # We want: dark values where text is, light values where background is
    if original_has_light_text:
        gray = 255.0 - gray
    
    # STEP 3: Verify and fix binary_mask orientation
    # binary_mask should have 255 where text is (dark pixels), 0 where background is (light pixels)
    text_pixels_in_gray = gray[binary_mask == 255]
    bg_pixels_in_gray = gray[binary_mask == 0]
    
    if len(text_pixels_in_gray) > 100 and len(bg_pixels_in_gray) > 100:
        avg_text_in_gray = np.mean(text_pixels_in_gray)
        avg_bg_in_gray = np.mean(bg_pixels_in_gray)
        
        # If text pixels are lighter than background pixels, binary_mask is inverted
        if avg_text_in_gray > avg_bg_in_gray:
            binary_mask = cv2.bitwise_not(binary_mask)
            # Re-apply border mask
            binary_mask = cv2.bitwise_and(binary_mask, border_mask)
    
    # STEP 4: Create result with pure white background and dark text
    # Start with pure white (255) everywhere
    result = np.ones((h, w), dtype=np.float32) * 255.0
    
    # In text regions: use grayscale values (which are now dark)
    # Preserve antialiasing by using original grayscale values
    text_pixels = binary_mask == 255
    result[text_pixels] = np.clip(gray[text_pixels] * 0.8, 0, 255)
    
    # Convert to uint8
    result = result.astype(np.uint8)
    
    # STEP 5: Force ALL background to pure white (255)
    bg_pixels = binary_mask == 0
    result[bg_pixels] = 255
    
    # STEP 6: Final verification and correction
    final_text = result[binary_mask == 255]
    final_bg = result[binary_mask == 0]
    
    if len(final_text) > 100 and len(final_bg) > 100:
        avg_final_text = np.mean(final_text)
        avg_final_bg = np.mean(final_bg)
        
        # If text is lighter than background, invert everything
        if avg_final_text > avg_final_bg:
            result = 255 - result
            # Force background to white again
            result[bg_pixels] = 255
    
    # STEP 7: Final aggressive cleanup - ensure background is 100% white
    result[bg_pixels] = 255
    # Also force any light pixels in background areas
    light_in_bg = (result > 200) & bg_pixels
    result[light_in_bg] = 255
    
    # STEP 8: One more verification - if result is inverted, fix it
    final_check_text = result[binary_mask == 255]
    final_check_bg = result[binary_mask == 0]
    if len(final_check_text) > 100 and len(final_check_bg) > 100:
        if np.mean(final_check_text) > np.mean(final_check_bg):
            # Result is inverted - fix it
            result = 255 - result
            result[binary_mask == 0] = 255
    
    # STEP 9: Absolute final pass - force background to white
    result[binary_mask == 0] = 255
    
    # STEP 10: Force corners to white (they should always be background)
    corner_size = min(100, h//10, w//10)
    result[0:corner_size, 0:corner_size] = 255  # top-left
    result[0:corner_size, w-corner_size:w] = 255  # top-right
    result[h-corner_size:h, 0:corner_size] = 255  # bottom-left
    result[h-corner_size:h, w-corner_size:w] = 255  # bottom-right
    
    # STEP 11: FINAL CHECK - Look at actual output
    # Check corners (should be white/light) and overall brightness
    corner_pixels = np.concatenate([
        result[0:corner_size, 0:corner_size].flatten(),
        result[0:corner_size, w-corner_size:w].flatten(),
        result[h-corner_size:h, 0:corner_size].flatten(),
        result[h-corner_size:h, w-corner_size:w].flatten()
    ])
    avg_corner_brightness = np.mean(corner_pixels)
    overall_brightness = np.mean(result)
    
    # If corners are dark OR overall image is too dark, the image is likely inverted
    # Also check: if most pixels are dark, it's inverted
    dark_pixel_ratio = np.sum(result < 128) / (h * w)
    
    # If corners are dark (< 200) OR most of image is dark (> 60%), invert it
    if avg_corner_brightness < 200 or dark_pixel_ratio > 0.6:
        result = 255 - result
        # Force background to white
        result[binary_mask == 0] = 255
        # Force corners to white
        result[0:corner_size, 0:corner_size] = 255
        result[0:corner_size, w-corner_size:w] = 255
        result[h-corner_size:h, 0:corner_size] = 255
        result[h-corner_size:h, w-corner_size:w] = 255
        # Force any light pixels in background to white
        light_bg = (result > 200) & (binary_mask == 0)
        result[light_bg] = 255
    
    # STEP 12: One more absolute check - ensure corners are white
    result[0:corner_size, 0:corner_size] = 255
    result[0:corner_size, w-corner_size:w] = 255
    result[h-corner_size:h, 0:corner_size] = 255
    result[h-corner_size:h, w-corner_size:w] = 255
    
    # Convert to RGB (3 channels) for JPEG saving
    result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    
    # FINAL FINAL CHECK: Ensure background in RGB is pure white (255, 255, 255)
    # Check if any background pixel is not white
    bg_mask_3d = np.stack([binary_mask == 0] * 3, axis=2)
    # Force all background pixels to pure white in RGB
    result_rgb[bg_mask_3d] = 255
    
    # Also check corners in RGB and force to white
    result_rgb[0:corner_size, 0:corner_size, :] = 255
    result_rgb[0:corner_size, w-corner_size:w, :] = 255
    result_rgb[h-corner_size:h, 0:corner_size, :] = 255
    result_rgb[h-corner_size:h, w-corner_size:w, :] = 255
    
    # Ensure output path ends with .jpg
    if not output_path.lower().endswith('.jpg') and not output_path.lower().endswith('.jpeg'):
        output_path = os.path.splitext(output_path)[0] + '.jpg'
    
    # Save as JPEG with high quality to preserve grayscale antialiasing
    cv2.imwrite(output_path, result_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
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
        # Process all image files in the input directory
        for file in os.listdir(input_path):
            if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg') or file.lower().endswith('.png'):
                file_with_path = os.path.join(input_path, file)
                output_file_with_path = os.path.join(output_path, file)
                print(f"Processing {file}...")
                remove_paper_background(file_with_path, output_file_with_path, block_size, c_value, fill_strokes)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

