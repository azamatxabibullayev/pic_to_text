import os
import re
import cv2
import numpy as np
import pytesseract
import regex
from PIL import Image

try:
    import easyocr

    EASY_AVAILABLE = True
except Exception:
    EASY_AVAILABLE = False

try:
    import ftfy

    FTFY_AVAILABLE = True
except Exception:
    FTFY_AVAILABLE = False

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

LANGS = "uzb+uzb_cyrl+rus+eng"
SCALE = 1.75
PARA_GAP_MULT = 1.8
PSM_BLOCK = 6
PSM_LINE = 7
USE_EASYOCR_DETECT_FALLBACK = True
USE_TESSERACT_FOR_FINAL_TEXT = True

_CYR_RE = re.compile(r"[А-Яа-яЁёЎўҚқҒғҲҳЇїІіҢң]", re.UNICODE)
_LAT_RE = re.compile(r"[A-Za-z]", re.UNICODE)
_EMOJI_RE = regex.compile(r"\p{Emoji}", flags=regex.UNICODE)

_DASH_PATTERN = re.compile(r"\s*(?:--|—|–|-\s*-|_{2,})\s*")
_ELLIPSIS = re.compile(r"(?<!\.)\.{3}(?!\.)")
_GARBAGE_SINGLETONS = set(list("®©™•·¤§^`~|\\/<>$@#&*_="))


def read_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if SCALE != 1.0:
        img = cv2.resize(img, (int(img.shape[1] * SCALE), int(img.shape[0] * SCALE)), interpolation=cv2.INTER_LINEAR)
    if img.ndim == 3 and img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        a = a.astype(np.float32) / 255.0
        bg = np.full_like(a, 255.0)
        b = (b * a + bg * (1 - a)).astype(np.uint8)
        g = (g * a + bg * (1 - a)).astype(np.uint8)
        r = (r * a + bg * (1 - a)).astype(np.uint8)
        img = cv2.merge([b, g, r])
    return img


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def auto_invert_for_dark_theme(gray):
    return 255 - gray if gray.mean() < 120 else gray


def preprocess_for_ocr(img):
    gray = to_gray(img)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=40)
    gray = auto_invert_for_dark_theme(gray)
    return gray


def detect_words_and_lines_tesseract(gray):
    d = pytesseract.image_to_data(gray, lang=LANGS, config=f"--oem 3 --psm {PSM_BLOCK} -c preserve_interword_spaces=1",
                                  output_type=pytesseract.Output.DICT)
    n = len(d["text"])
    if n == 0: return []

    lines_map = {}
    for i in range(n):
        txt = (d["text"][i] or "").strip()
        level = int(d.get("level", [0])[i])
        if level != 5: continue
        block, par, line_no, word_no = int(d["block_num"][i]), int(d["par_num"][i]), int(d["line_num"][i]), int(
            d["word_num"][i])
        left, top, w, h = int(d["left"][i]), int(d["top"][i]), int(d["width"][i]), int(d["height"][i])
        conf_raw = str(d["conf"][i])
        conf = float(conf_raw) if conf_raw.replace(".", "", 1).lstrip("-").isdigit() else -1.0
        key = (block, par, line_no)
        word = {"left": left, "top": top, "right": left + w, "bottom": top + h, "text": txt, "conf": conf,
                "word_num": word_no}
        if key not in lines_map:
            lines_map[key] = {"left": left, "top": top, "right": left + w, "bottom": top + h, "words": [word],
                              "conf_vals": [conf if conf >= 0 else 0.0]}
        else:
            L = lines_map[key]
            L["left"] = min(L["left"], left)
            L["top"] = min(L["top"], top)
            L["right"] = max(L["right"], left + w)
            L["bottom"] = max(L["bottom"], top + h)
            L["words"].append(word)
            if conf >= 0:
                L["conf_vals"].append(conf)

    lines = []
    for key, item in sorted(lines_map.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        words_sorted = sorted(item["words"], key=lambda w: (w["left"], w["word_num"]))
        avg_conf = sum(item["conf_vals"]) / max(1, len(item["conf_vals"]))
        guess = " ".join([w["text"].strip() for w in words_sorted if w["text"].strip()])
        lines.append({"left": item["left"], "top": item["top"], "right": item["right"], "bottom": item["bottom"],
                      "words": words_sorted, "conf": avg_conf, "text": guess})
    return lines


def smart_quotes_and_dashes(s: str) -> str:
    out = []
    open_dbl = True
    open_sgl = True
    for i, ch in enumerate(s):
        if ch == '"':
            out.append("“" if open_dbl else "”");
            open_dbl = not open_dbl
        elif ch == "'":
            prev = s[i - 1] if i > 0 else " "
            nxt = s[i + 1] if i + 1 < len(s) else " "
            if prev.isalnum() and nxt.isalnum():
                out.append("’")
            else:
                out.append("‘" if open_sgl else "’");
                open_sgl = not open_sgl
        else:
            out.append(ch)
    s = "".join(out)
    s = _DASH_PATTERN.sub(" — ", s)
    s = re.sub(r"\s*—\s*", " — ", s)
    s = _ELLIPSIS.sub("…", s)
    return s


def ensure_spaces_around_emojis(s: str) -> str:
    s = regex.sub(r"(?<!\s)(\p{Emoji})", r" \1", s)
    s = regex.sub(r"(\p{Emoji})(?![\s,.;:?!\)\]\}])", r"\1 ", s)
    s = re.sub(r"[ \t]{3,}", "  ", s)
    return s


def tidy_punctuation_spacing(s: str) -> str:
    s = re.sub(r"\s+([,.;:?!%])", r"\1", s)
    s = re.sub(r"\s+([”’»\)\]\}])", r"\1", s)
    s = re.sub(r"([\(«„“\[\{])\s+", r"\1", s)
    s = re.sub(r"([,.;:?!”’\)\]\}—])([^\s.,;:?!])", r"\1 \2", s)
    s = re.sub(r"[ \t]{3,}", "  ", s)
    return s


def postprocess_line(s: str) -> str:
    if FTFY_AVAILABLE: s = ftfy.fix_text(s)
    s = smart_quotes_and_dashes(s)
    s = tidy_punctuation_spacing(s)
    s = ensure_spaces_around_emojis(s)
    return s.rstrip()


def estimate_space_width(lines):
    widths = []
    for b in lines:
        txt = (b.get("final_text") or "").strip("\n")
        if not txt: continue
        line_width_px = max(1, b["right"] - b["left"])
        avg_char = line_width_px / max(1, len(txt))
        widths.append(avg_char)
    return np.median(widths) * 0.55 if widths else 8.0


def prepend_indentation(lines):
    if not lines: return []
    min_left = min(b["left"] for b in lines)
    space_px = estimate_space_width(lines)
    out = []
    for b in lines:
        txt = b.get("final_text", "")
        if not txt: out.append(""); continue
        delta = max(0, b["left"] - min_left)
        n_spaces = int(round(delta / max(1.0, space_px)))
        out.append((" " * n_spaces) + txt)
    return out


def process_image_file(path: str, out_dir: str):
    print(f"Processing: {os.path.basename(path)}")
    img = read_image(path)
    gray = preprocess_for_ocr(img)

    lines = detect_words_and_lines_tesseract(gray)
    if not lines and USE_EASYOCR_DETECT_FALLBACK:
        try:
            reader = easyocr.Reader(["ru", "en"], gpu=False)
            raw = reader.readtext(img, detail=1)
            lines = []
            for item in raw:
                pts, text, conf = item[0], item[1], float(item[2] or 0.0)
                xs = [int(round(p[0])) for p in pts];
                ys = [int(round(p[1])) for p in pts]
                lines.append(
                    {"left": min(xs), "top": min(ys), "right": max(xs), "bottom": max(ys), "text": text.strip(),
                     "conf": conf, "words": []})
        except Exception:
            pass

    if not lines:
        text = pytesseract.image_to_string(gray, lang=LANGS, config=f"--oem 3 --psm {PSM_BLOCK}")
        text_lines = [postprocess_line(t) for t in text.splitlines()]
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + ".txt")
        with open(out_path, "w", encoding="utf-8") as f: f.write("\n".join(text_lines).strip())
        print("Fallback:", out_path)
        return

    lines = sorted(lines, key=lambda b: ((b["top"] + b["bottom"]) // 2, b["left"]))
    median_height = np.median([b["bottom"] - b["top"] for b in lines]) if lines else 16
    paragraph_markers = []

    for idx, b in enumerate(lines):
        base_text = b.get("text", "")
        final_text = postprocess_line(base_text)
        b["final_text"] = final_text
        center = (b["top"] + b["bottom"]) / 2.0
        if idx > 0 and center - prev_center > median_height * PARA_GAP_MULT:
            paragraph_markers.append(idx)
        prev_center = center if 'prev_center' in locals() else center

    indented_lines = prepend_indentation(lines)
    out_lines = []
    for i, line_text in enumerate(indented_lines):
        if i in paragraph_markers: out_lines.append("")
        out_lines.append(line_text)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print("Wrote:", out_path)


def process_folder(input_folder="images", output_folder="results"):
    os.makedirs(output_folder, exist_ok=True)
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
