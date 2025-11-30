#!/usr/bin/env python3
"""
Create a multi-page PDF from all images in a folder.
Each image -> its own A4 page (portrait).
Open the resulting PDF in Affinity Publisher/Designer.
"""

import argparse
from pathlib import Path

from PIL import Image

A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297
DPI = 300  # print quality
IMAGE_SCALE = 1.05  # increase image size by 5%


def mm_to_pixels(mm, dpi=DPI):
    return int(mm / 25.4 * dpi)


PAGE_WIDTH_PX = mm_to_pixels(A4_WIDTH_MM)
PAGE_HEIGHT_PX = mm_to_pixels(A4_HEIGHT_MM)


def load_and_fit_image(path: Path):
    img = Image.open(path).convert("RGB")

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
    else:
        # Image is taller - fit to height
        new_height = target_height
        new_width = int(img.width * (target_height / img.height))

    # Resize with high-quality resampling (not thumbnail)
    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Create white A4 background
    page = Image.new("RGB", (PAGE_WIDTH_PX, PAGE_HEIGHT_PX), "white")

    # Center the image on the page (may overflow as requested)
    x = (PAGE_WIDTH_PX - img.width) // 2
    y = (PAGE_HEIGHT_PX - img.height) // 2
    page.paste(img, (x, y))
    print("Processed image: ", path.name)

    return page


def main():
    input_dir = Path("output_cleaned_fixed")
    output_pdf = Path("output.pdf")

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in exts]
    )

    if not image_paths:
        raise SystemExit(f"No images found in: {input_dir}")

    pages = [load_and_fit_image(p) for p in image_paths]

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
