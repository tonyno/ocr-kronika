import cv2
import numpy as np
from pathlib import Path





def clean_page(input_path: Path, output_path: Path):
    # 1) Load image
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"Could not read {input_path}")
        return

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
        blockSize=83,   # size of neighbourhood (try 21, 31, 41)
        C=9             # local threshold adjustment (try 5–20)
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

    # 6) Anti-aliasing: slight blur to create gray edge pixels and smoother text
    # (3,3) kernel and sigma 0 = very mild; try (5,5) if you want even smoother edges
    antialiased = cv2.GaussianBlur(cleaned, (3, 3), 0)

    # Save as PNG (lossless, good for printing)
    # Compression level 3: balances file size (~5 MB) with encoding speed
    # PNG compression is lossless, so quality remains perfect at any level
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), antialiased, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    print(f"Saved cleaned page to {output_path}")


def main():
    in_path = Path("input_sorted")
    out_path = Path("output_cleaned")

    if in_path.is_dir():
        # Folder → process all images
        out_path.mkdir(parents=True, exist_ok=True)
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        for img_file in sorted(in_path.iterdir()):
            if img_file.suffix.lower() in exts:
                if not any(page_num in img_file.stem for page_num in ['125', '126', '127', '140', '143', '295', '322', '323', '324']):
                    continue
                out_file = out_path / (img_file.stem + "_clean.png")
                clean_page(img_file, out_file)
    else:
        print("Input path does not exist.")


if __name__ == "__main__":
    main()
