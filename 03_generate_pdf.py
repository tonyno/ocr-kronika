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


def mm_to_pixels(mm, dpi=DPI):
    return int(mm / 25.4 * dpi)


PAGE_WIDTH_PX = mm_to_pixels(A4_WIDTH_MM)
PAGE_HEIGHT_PX = mm_to_pixels(A4_HEIGHT_MM)


def load_and_fit_image(path: Path):
    img = Image.open(path).convert("RGB")

    # Fit into A4 page while keeping aspect ratio
    img.thumbnail((PAGE_WIDTH_PX, PAGE_HEIGHT_PX), Image.LANCZOS)

    # Create white A4 background
    page = Image.new("RGB", (PAGE_WIDTH_PX, PAGE_HEIGHT_PX), "white")

    # Center the image on the page
    x = (PAGE_WIDTH_PX - img.width) // 2
    y = (PAGE_HEIGHT_PX - img.height) // 2
    page.paste(img, (x, y))

    return page


def main():
    input_dir = Path("output_cleaned")
    output_pdf = Path("output.pdf")

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in exts]
    )

    if not image_paths:
        raise SystemExit(f"No images found in: {input_dir}")

    pages = [load_and_fit_image(p) for p in image_paths]

    # Save as multi-page PDF
    first, rest = pages[0], pages[1:]
    first.save(
        output_pdf,
        "PDF",
        save_all=True,
        append_images=rest,
        resolution=DPI,
    )

    print(f"Created PDF with {len(pages)} pages: {output_pdf}")


if __name__ == "__main__":
    main()
