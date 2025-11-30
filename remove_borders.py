import cv2
import numpy as np
from pathlib import Path


def remove_page_border(in_path: str, out_path: str) -> None:
    """
    Load a scanned page image, detect long straight lines near the border
    (page frame / book edge) and paint them over in white.
    """

    img = cv2.imread(in_path)
    if img is None:
        raise ValueError(f"Cannot read image: {in_path}")

    # Work in grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    h, w = gray.shape

    # Detect (probabilistic) Hough lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min(w, h) // 2,   # only long lines
        maxLineGap=20
    )

    result = img.copy()

    # How close a line has to be to image edge to be considered a border
    edge_margin = int(0.08 * min(w, h))   # ~8 % of size
    # How thick we overpaint the border line (tune if needed)
    erase_thickness = 25

    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx, dy = x2 - x1, y2 - y1

            # classify line orientation
            if abs(dx) > 5 * abs(dy):  # horizontal-ish
                near_edge = (min(y1, y2) < edge_margin) or (max(y1, y2) > h - edge_margin)
            elif abs(dy) > 5 * abs(dx):  # vertical-ish
                near_edge = (min(x1, x2) < edge_margin) or (max(x1, x2) > w - edge_margin)
            else:
                continue  # diagonal line, probably handwriting -> ignore

            # If it is a long line close to image border, erase it
            if near_edge:
                cv2.line(result, (x1, y1), (x2, y2), (255, 255, 255), erase_thickness)

    cv2.imwrite(out_path, result)


if __name__ == "__main__":
    # Single image
    remove_page_border("page281a_clean.png", "page281a_noborder.png")

    # OR: batch process a folder
    # from glob import glob
    # for fname in glob("input_folder/*.png"):
    #     out = Path("output_folder") / Path(fname).name
    #     remove_page_border(fname, str(out))
