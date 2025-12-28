#!/usr/bin/env python3
"""
Create a multi-page PDF from all images in a folder.
Each image -> its own A4 page (portrait).
Open the resulting PDF in Affinity Publisher/Designer.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont

A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297
DPI = 300  # print quality
IMAGE_SCALE = 1.02  # increase image size by 5%


def mm_to_pixels(mm, dpi=DPI):
    return int(mm / 25.4 * dpi)


PAGE_WIDTH_PX = mm_to_pixels(A4_WIDTH_MM)
PAGE_HEIGHT_PX = mm_to_pixels(A4_HEIGHT_MM)


def find_content_center(img: Image.Image, debug_name: str = None):
    """
    Find the center of the content in the image by detecting non-background areas.
    Uses blur and thresholding to create a bounding box around all content.
    Returns (center_x, center_y) relative to the image.
    
    Args:
        img: Input image
        debug_name: Optional filename base for saving debug images
    """
    # Convert to grayscale for analysis
    gray = img.convert("L")
    
    # Apply blur to smooth out noise and help detect content regions
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=10))
    
    # Save blurred image for debugging
    if debug_name:
        debug_dir = Path("debug_content_detection")
        debug_dir.mkdir(exist_ok=True)
        blurred_path = debug_dir / f"{debug_name}_blurred.png"
        blurred.save(blurred_path)
        print(f"  Saved blurred image: {blurred_path}")
    
    # Convert to numpy array
    img_array = np.array(blurred)
    
    # Threshold: consider pixels darker than this as content
    # For white background images, content is typically darker
    # Using a threshold that distinguishes content from white background
    threshold = 240  # Pixels darker than this are considered content
    
    # Create binary mask: 1 for content, 0 for background
    mask = img_array < threshold
    
    # Save thresholded mask for debugging
    if debug_name:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_path = debug_dir / f"{debug_name}_mask.png"
        mask_img.save(mask_path)
        print(f"  Saved thresholded mask: {mask_path}")
    
    # Find bounding box of content
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        # No content detected, return image center
        if debug_name:
            print(f"  No content detected in {debug_name}, using image center")
        return img.width // 2, img.height // 2
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Calculate center of content bounding box
    content_center_x = (x_min + x_max) // 2
    content_center_y = (y_min + y_max) // 2
    
    # Save visualization with bounding box and center point
    if debug_name:
        # Create a copy of the original image for visualization
        vis_img = img.copy()
        from PIL import ImageDraw
        draw = ImageDraw.Draw(vis_img)
        
        # Draw bounding box
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        
        # Draw center point
        center_size = 10
        draw.ellipse(
            [
                content_center_x - center_size,
                content_center_y - center_size,
                content_center_x + center_size,
                content_center_y + center_size,
            ],
            fill="blue",
            outline="blue",
        )
        
        vis_path = debug_dir / f"{debug_name}_bbox_center.png"
        vis_img.save(vis_path)
        print(f"  Saved bounding box visualization: {vis_path}")
        print(f"  Content center: ({content_center_x}, {content_center_y})")
    
    return content_center_x, content_center_y


def load_and_fit_image(path: Path, page_number: int):
    img = Image.open(path).convert("RGB")

    # Find content center in original image (before resizing)
    # Use filename without extension as debug name
    debug_name = path.stem
    content_center_x_orig, content_center_y_orig = find_content_center(img, debug_name=debug_name)

    # Calculate target size: 105% of A4 page size
    # This allows the image to be 5% larger and potentially overflow the page
    target_width = int(PAGE_WIDTH_PX * IMAGE_SCALE)
    target_height = int(PAGE_HEIGHT_PX * IMAGE_SCALE)

    # Resize image to fit target size while preserving aspect ratio
    # Use LANCZOS resampling for high quality (no thumbnail to preserve quality)
    img_ratio = img.width / img.height
    target_ratio = target_width / target_height

    if img_ratio > target_ratio:
        # Image is wider - fit to width
        new_width = target_width
        new_height = int(img.height * (target_width / img.width))
        scale_factor = target_width / img.width
    else:
        # Image is taller - fit to height
        new_height = target_height
        new_width = int(img.width * (target_height / img.height))
        scale_factor = target_height / img.height

    # Resize with high-quality resampling (not thumbnail)
    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Calculate content center position after resizing
    content_center_x_resized = content_center_x_orig * scale_factor
    content_center_y_resized = content_center_y_orig * scale_factor

    # Create white A4 background
    page = Image.new("RGB", (PAGE_WIDTH_PX, PAGE_HEIGHT_PX), "white")

    # Calculate horizontal offset: odd pages move 1cm right, even pages move 1cm left
    # 1cm = 10mm
    offset_mm = 7 if page_number % 2 == 1 else -7  # Odd: +10mm (right), Even: -10mm (left)
    offset_px = mm_to_pixels(offset_mm)

    # Position image so content center aligns with page center (with offset)
    # Page center coordinates
    page_center_x = PAGE_WIDTH_PX // 2
    page_center_y = PAGE_HEIGHT_PX // 2

    # Calculate position: content center should be at page center + offset
    x = int(page_center_x - content_center_x_resized + offset_px)
    y = int(page_center_y - content_center_y_resized)

    page.paste(img, (x, y))

    # Add page number at the bottom (4mm from bottom)
    draw = ImageDraw.Draw(page)
    
    # Try to use a nice font, fallback to default if not available
    try:
        # Try to use a system font (adjust path for your system if needed)
        font_size = mm_to_pixels(4)  # 4mm font size
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # Page number text
    if page_number >= 5:
        page_text = str(page_number-4)
        
        # Get text bounding box to center it
        bbox = draw.textbbox((0, 0), page_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position: 4mm from bottom, centered horizontally with same offset
        bottom_margin_px = mm_to_pixels(8)
        text_x = int(page_center_x - text_width // 2 + offset_px)
        text_y = PAGE_HEIGHT_PX - bottom_margin_px - text_height
        
        # Draw page number (70% black = 30% brightness)
        draw.text((text_x, text_y), page_text, fill=(77, 77, 77), font=font)
    
    print("Processed image: ", path.name, f"(page {page_number})")

    return page


def main():
    input_dir = Path("output_cleaned_fixed")
    output_pdf = Path("output.pdf")

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in exts]
    )

    #image_paths = image_paths[:12]

    if not image_paths:
        raise SystemExit(f"No images found in: {input_dir}")

    pages = [load_and_fit_image(p, page_number=i+1) for i, p in enumerate(image_paths)]

    # Save as multi-page PDF with maximum quality
    first, rest = pages[0], pages[1:]
    first.save(
        output_pdf,
        "PDF",
        save_all=True,
        append_images=rest,
        resolution=DPI,  # 300 DPI for print quality
        quality=100,  # Maximum quality (100%)
    )

    print(f"Created PDF with {len(pages)} pages: {output_pdf}")


if __name__ == "__main__":
    main()
