import easyocr
import os


def image_to_text(image_path, reader):
    results = reader.readtext(image_path, detail=1)
    text = "\n".join([res[1] for res in results])
    return text


def main():
    reader = easyocr.Reader(['ru', 'en'], gpu=False)

    folder = "images"
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Folder '{folder}' created. Put your images inside and rerun.")
        return

    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("Images not found in the folder.")
        return

    for filename in image_files:
        path = os.path.join(folder, filename)
        print(f"\nReading: {filename}")
        text = image_to_text(path, reader)
        print(f"Extracted Text:\n{text}\n{'=' * 60}")
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(folder, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved text to {txt_filename}")


if __name__ == "__main__":
    main()
