# Pic to Text OCR Project

This project is a **Python-based OCR tool** that converts images (PNG, JPG, TIFF, BMP) to text files. It supports multiple languages (Uzbek, Russian, English) and preserves layout, paragraphs, spacing, punctuation, quotes, and emojis.

It uses **Tesseract OCR** as the primary engine and **EasyOCR** as a fallback for hard-to-read images.

---

## Features

* Supports **Uzbek (Cyrillic), Russian, English**.
* Preserves **paragraphs** and **indentation**.
* Maintains proper **punctuation spacing**.
* Handles **smart quotes** and **dashes**.
* Detects **emojis** and preserves them in output.
* Fallback using **EasyOCR** if Tesseract fails.

---

## Installation

1. Clone the repository or download the code:

```bash
git clone https://github.com/yourusername/pic-to-text.git
cd pic-to-text
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract):

   * Windows: Download the installer and set `TESSERACT_CMD` in `main.py`.
   * Linux/macOS: Install via package manager (`sudo apt install tesseract-ocr`).
   * For Uzbek, Russian, English, etc., download the corresponding .traineddata files and place them in the tessdata folder of your Tesseract installation.

---

## Usage

1. Place your images in the `images` folder.
2. Run the main script:

```bash
python main.py
```

3. Results are saved as `.txt` files in the `results` folder.

---



## Notes

* Make sure Tesseract is installed and the path is correct.
* Emojis are extracted using `regex` and added to the text output.
* For large images, the script automatically enhances contrast and reduces noise.

