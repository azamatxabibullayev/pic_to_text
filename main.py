import cv2
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def image_to_text(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    gray = cv2.bitwise_not(gray)

    gray = cv2.medianBlur(gray, 1)

    text = pytesseract.image_to_string(
        gray,
        lang="uzb+uzb_cyrl+rus+eng",
        config="--oem 3 --psm 6"
    )

    return text.strip()


def main():
    folder = "images"
    output_folder = "results"

    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_files:
        print(f"No images found in '{folder}' folder.")
        return

    for filename in image_files:
        image_path = os.path.join(folder, filename)
        text = image_to_text(image_path)

        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Processed: {filename}")

    print("\nAll images converted successfully! Results saved in 'results/' folder.")


if __name__ == "__main__":
    main()
