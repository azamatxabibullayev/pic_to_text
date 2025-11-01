import os
import math
import re
from statistics import median
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
import pytesseract

try:
    import easyocr

    EASY_AVAILABLE = True
except Exception:
    EASY_AVAILABLE = False

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
LANGS = "uzb+uzb_cyrl+rus+eng"
SCALE = 1.6
LINE_Y_TOL = 18
USE_TESSERACT_FALLBACK = True
PRESERVE_LEADING_INDENT = True
TAB_WIDTH = 4

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

_ALPHA_RE = re.compile(r"[A-Za-zА-Яа-яЁёЎўҚқҒғҲҳЇїІіҢң]", re.UNICODE)
_CYR_RE = re.compile(r"[А-Яа-яЁёЎўҚқҒғҲҳЇїІіҢң]", re.UNICODE)
_LAT_RE = re.compile(r"[A-Za-z]", re.UNICODE)


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    if SCALE != 1.0:
        img = cv2.resize(img, (int(img.shape[1] * SCALE), int(img.shape[0] * SCALE)), interpolation=cv2.INTER_LINEAR)
    return img


def detect_boxes_easy(image: np.ndarray) -> List[Dict[str, Any]]:
    h, w = image.shape[:2]

    if EASY_AVAILABLE:
        try:
            reader = easyocr.Reader(["ru", "en"], gpu=False)
            raw = reader.readtext(image, detail=1)
        except Exception as e:
            print(f"EasyOCR failed ({e}), using Tesseract fallback detection.")
            raw = []

        boxes = []
        for item in raw:
            box_pts, text, conf = item[0], item[1], float(item[2] or 0.0)
            xs = [int(round(p[0])) for p in box_pts]
            ys = [int(round(p[1])) for p in box_pts]
            boxes.append({
                "left": max(0, min(xs)),
                "top": max(0, min(ys)),
                "right": min(w, max(xs)),
                "bottom": min(h, max(ys)),
                "text": text.strip(),
                "conf": conf
            })
        if boxes:
            return boxes

    data = pytesseract.image_to_data(image, lang=LANGS, config="--oem 3 --psm 3", output_type=pytesseract.Output.DICT)
    boxes = []
    for i in range(len(data["text"])):
        txt = str(data["text"][i]).strip()
        if not txt:
            continue
        left, top = int(data["left"][i]), int(data["top"][i])
        width, height = int(data["width"][i]), int(data["height"][i])
        if width > 0 and height > 0:
            boxes.append({
                "left": left, "top": top,
                "right": left + width, "bottom": top + height,
                "text": txt, "conf": float(data["conf"][i]) if data["conf"][i].isdigit() else 0.0
            })
    return boxes


def ocr_box_tesseract(img_gray: np.ndarray, box: Dict[str, Any], psm: int = 7) -> Tuple[str, float]:
    x1, y1, x2, y2 = box["left"], box["top"], box["right"], box["bottom"]
    h, w = img_gray.shape[:2]
    pad = max(2, int((y2 - y1) * 0.08))
    x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
    crop = img_gray[y1:y2, x1:x2]

    guess_text = box.get("text", "")
    if _CYR_RE.search(guess_text):
        lang = "uzb_cyrl+rus+eng"
    elif _LAT_RE.search(guess_text):
        lang = "uzb+eng"
    else:
        lang = LANGS

    config = f"--oem 3 --psm {psm}"
    try:
        text = pytesseract.image_to_string(crop, lang=lang, config=config).strip()
    except Exception:
        text = ""

    conf = 0.0
    try:
        d = pytesseract.image_to_data(crop, lang=lang, config=config, output_type=pytesseract.Output.DICT)
        valid = [float(c) for c in d["conf"] if c.replace(".", "", 1).isdigit() and float(c) >= 0]
        if valid:
            conf = sum(valid) / len(valid)
    except Exception:
        pass

    return text, conf


def group_into_lines(boxes: List[Dict[str, Any]], y_tol: int = LINE_Y_TOL) -> List[List[Dict[str, Any]]]:
    if not boxes:
        return []
    boxes_sorted = sorted(boxes, key=lambda b: (b["top"] + b["bottom"]) // 2)
    lines, current = [], [boxes_sorted[0]]
    for b in boxes_sorted[1:]:
        cy = (b["top"] + b["bottom"]) // 2
        mean_cy = sum((r["top"] + r["bottom"]) // 2 for r in current) / len(current)
        if abs(cy - mean_cy) <= y_tol:
            current.append(b)
        else:
            lines.append(sorted(current, key=lambda x: x["left"]))
            current = [b]
    if current:
        lines.append(sorted(current, key=lambda x: x["left"]))
    return lines


def build_line_grid(line_boxes: List[Dict[str, Any]], texts: List[str]) -> str:
    char_widths = []
    for b, t in zip(line_boxes, texts):
        w = b["right"] - b["left"]
        if len(t.strip()) > 0:
            char_widths.append(w / max(1, len(t.strip())))
    avg_char_w = median(char_widths) if char_widths else 8.0

    max_x = max((b["right"] for b in line_boxes), default=0)
    cols = int(math.ceil(max_x / avg_char_w)) + 4
    grid = [" "] * cols

    for b, t in zip(line_boxes, texts):
        col = int(round(b["left"] / avg_char_w))
        for i, ch in enumerate(t):
            pos = col + i
            if pos >= len(grid):
                grid.extend([" "] * (pos - len(grid) + 1))
            grid[pos] = ch
    return "".join(grid).rstrip()


def postprocess_join_letters(line: str) -> str:
    tokens = line.split(" ")
    i, out = 0, []
    while i < len(tokens):
        if len(tokens[i]) == 1 and _ALPHA_RE.search(tokens[i]):
            run = []
            while i < len(tokens) and len(tokens[i]) == 1 and _ALPHA_RE.search(tokens[i]):
                run.append(tokens[i])
                i += 1
            if len(run) >= 3:
                out.append("".join(run))
            else:
                out.extend(run)
        else:
            out.append(tokens[i])
            i += 1
    return " ".join([t for t in out if t is not None])


def process_image_file(path: str, out_dir: str):
    print(f"Processing: {os.path.basename(path)}")
    img = read_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    boxes = detect_boxes_easy(img)
    if not boxes:
        text = pytesseract.image_to_string(gray, lang=LANGS, config="--oem 3 --psm 3")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + ".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text.strip())
        print("Fallback (full Tesseract):", out_path)
        return

    if USE_TESSERACT_FALLBACK:
        for b in boxes:
            t_text, t_conf = ocr_box_tesseract(gray, b)
            if t_text:
                if t_conf >= 30 or len(t_text) > len(b.get("text", "")):
                    b["text"] = t_text

    lines = group_into_lines(boxes, y_tol=LINE_Y_TOL)

    out_lines = []
    prev_center = None
    median_height = np.median([b["bottom"] - b["top"] for b in boxes]) if boxes else 15

    for line_boxes in lines:
        texts = [b["text"] for b in line_boxes]
        line_str = build_line_grid(line_boxes, texts)
        line_str = postprocess_join_letters(line_str)
        center = sum((b["top"] + b["bottom"]) // 2 for b in line_boxes) / len(line_boxes)
        if prev_center and center - prev_center > median_height * 1.8:
            out_lines.append("")
        out_lines.append(line_str)
        prev_center = center

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print("Wrote:", out_path)


def process_folder(input_folder: str = "images", output_folder: str = "results"):
    files = [f for f in os.listdir(input_folder) if
             f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))]
    if not files:
        print("No images found in", input_folder)
        return

    for fn in sorted(files):
        path = os.path.join(input_folder, fn)
        try:
            process_image_file(path, output_folder)
        except Exception as e:
            print(f"Failed: {fn} ({e})")


if __name__ == "__main__":
    process_folder("images", "results")
