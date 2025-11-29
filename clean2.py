import cv2
import numpy as np
from pathlib import Path
import argparse


def auto_crop(img, margin=40):
    """
    Automatically crop the page area by detecting the main bright region.
    margin: number of pixels to keep around detected content.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Slight blur to smooth noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold just for finding the page boundary
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find external contours
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback – return original if nothing is detected
        return img

    # Largest contour is assumed to be the page
    page_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(page_contour)

    # Add margin but keep inside image bounds
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img.shape[1] - x, w + 2 * margin)
    h = min(img.shape[0] - y, h + 2 * margin)

    cropped = img[y:y + h, x:x + w]
    return cropped


def clean_page(input_path: Path, output_path: Path):
    # 1) Load image
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"Could not read {input_path}")
        return

    # 1b) Auto-crop to page area
    # img = auto_crop(img, margin=40)

    # 2) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Optional: normalize contrast a bit (helps if some pages are very dark/light)
    gray = cv2.normalize(gray, None, alpha=0, beta=255,
                         norm_type=cv2.NORM_MINMAX)

    # 3) Adaptive threshold – this removes background & makes text black/white
    # blockSize must be odd; C is a small constant subtracted from the mean.
    bw = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=51,   # size of neighbourhood (try 21, 31, 41)
        C=5             # local threshold adjustment (try 5–20)
    )

    # Now bw is white background (255) and dark text (0).

    # 4) Thicken handwriting a little so it prints better
    kernel = np.ones((2, 2), np.uint8)
    bw = cv2.dilate(bw, kernel, iterations=1)

    # 5) (Optional) Remove very small isolated dots (noise)
    # This will keep strokes but kill tiny specks.
    # Comment out if it removes details you care about.
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(
        255 - bw, connectivity=8
    )
    sizes = stats[1:, -1]  # skip background
    nb_components -= 1

    min_size = 90  # pixels; tune this if you see too much noise
    cleaned = np.zeros(bw.shape, dtype=np.uint8) + 255  # start as all white

    for i in range(nb_components):
        if sizes[i] >= min_size:
            cleaned[output == i + 1] = 0  # keep this component as black

    # 6) Save as PNG (lossless, good for printing)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cleaned)
    print(f"Saved cleaned page to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean scanned/photographed handwritten pages."
    )
    parser.add_argument(
        "input",
        help="Input file or folder with images (jpg/jpeg/png)."
    )
    parser.add_argument(
        "output",
        help="Output file or folder."
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if in_path.is_file():
        # Single file
        if out_path.is_dir():
            out_file = out_path / (in_path.stem + "_clean.png")
        else:
            out_file = out_path
        clean_page(in_path, out_file)

    elif in_path.is_dir():
        # Folder → process all images
        out_path.mkdir(parents=True, exist_ok=True)
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        for img_file in sorted(in_path.iterdir()):
            if img_file.suffix.lower() in exts:
                out_file = out_path / (img_file.stem + "_clean.png")
                clean_page(img_file, out_file)
    else:
        print("Input path does not exist.")


if __name__ == "__main__":
    main()
