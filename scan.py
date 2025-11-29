import base64
import sys
import time
from pathlib import Path

from openai import OpenAI

# Configure your model here.
# "gpt-4.1-mini" is a good balance of price/quality and supports vision.
MODEL_NAME = "gpt-4.1-mini"

client = OpenAI()  # Uses OPENAI_API_KEY from environment


def image_file_to_data_url(path: Path) -> str:
    """
    Read an image from disk and convert it to a base64 data URL
    that the OpenAI vision models accept as `image_url`.
    """
    suffix = path.suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suffix in [".png"]:
        mime = "image/png"
    elif suffix in [".webp"]:
        mime = "image/webp"
    else:
        # Fallback – most images will still work as jpeg
        mime = "image/jpeg"

    with path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{b64}"


def ocr_one_image(image_path: Path) -> str:
    """
    Perform OCR of a single handwritten page using the OpenAI vision model.
    Returns plain text.
    """
    data_url = image_file_to_data_url(image_path)

    prompt = (
        "You are an OCR engine for handwritten family history documents.\n"
        "Read ALL legible handwritten text from this image and output it as clean plain text.\n\n"
        "- Preserve paragraphs and obvious line breaks.\n"
        "- Do NOT add commentary, explanations, or translation.\n"
        "- If a word is unclear, make your best guess and append a question mark (?) to it.\n"
        "- Keep the original language as written."
    )

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )

    # Collect text from the response
    text_chunks = []
    for output_item in response.output:
        for content_item in output_item.content:
            # For output_text content, we get a `.text` field
            if hasattr(content_item, "text") and content_item.text:
                text_chunks.append(content_item.text)

    return "\n".join(text_chunks).strip()


def find_images_in_dir(input_dir: Path):
    """
    Find all jpeg/png images in a directory, sorted by filename.
    """
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    files = []
    for pattern in patterns:
        files.extend(input_dir.glob(pattern))
    return sorted(files)


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python batch_ocr_family_history.py /path/to/images output.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    images = find_images_in_dir(input_dir)
    if not images:
        print(f"No JPG/PNG images found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(images)} images. Starting OCR...")

    # Track processing times for each image
    processing_times = []

    # Open output file once and append all pages
    with output_file.open("w", encoding="utf-8") as out_f:
        for i, img_path in enumerate(images, start=1):
            print(f"[{i}/{len(images)}] OCR: {img_path.name} ...", end="", flush=True)
            
            # Start timing
            start_time = time.time()
            
            try:
                text = ocr_one_image(img_path)
            except Exception as e:
                # Record time even for errors
                elapsed_time = time.time() - start_time
                processing_times.append(elapsed_time)
                
                print(f"\n  ERROR on {img_path.name}: {e}", file=sys.stderr)
                # Write a marker into the output so you know one page failed
                out_f.write(
                    f"\n\n----- PAGE {i}: {img_path.name} (ERROR: {e}) -----\n\n"
                )
                # Small delay in case of transient errors / rate limits
                time.sleep(2)
                continue

            # Record processing time
            elapsed_time = time.time() - start_time
            processing_times.append(elapsed_time)

            # Write a clear separator between pages
            out_f.write(f"\n\n----- PAGE {i}: {img_path.name} -----\n\n")
            out_f.write(text)
            out_f.write("\n")

            print(f" done. ({elapsed_time:.2f}s)")

            # Optional: gentle pause to avoid rate-limits if you hit them
            # time.sleep(0.5)

    # Calculate and display statistics
    if processing_times:
        total_time = sum(processing_times)
        average_time = total_time / len(processing_times)
        print(f"\nAll done. Combined OCR saved to: {output_file}")
        print(f"\nProcessing statistics:")
        print(f"  Total pages processed: {len(processing_times)}")
        print(f"  Total processing time: {total_time:.2f}s")
        print(f"  Average time per page: {average_time:.2f}s")
    else:
        print(f"\nAll done. Combined OCR saved to: {output_file}")


if __name__ == "__main__":
    main()
