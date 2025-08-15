#!/usr/bin/env python3
"""
Spec Packet Builder + OCR (Python 3.8 Compatible)
-------------------------------------------------
- Converts a photo/scan of your handwritten list into a CSV (via OCR)
- Downloads spec PDFs (or uses URLs from the CSV)
- Stamps the Section (WC1, L1, etc.) on the top-right of every page
- Merges PDFs in the exact CSV order

USAGE EXAMPLES:
    # OCR only -> CSV
    python spec_packet_builder_py38.py --image order.jpg --csv items.csv --skip-build

    # OCR -> Packet in one go
    python spec_packet_builder_py38.py --image order.jpg --out packet.pdf --workdir ./work

    # CSV -> Packet (no OCR)
    python spec_packet_builder_py38.py --csv items.csv --out packet.pdf --workdir ./work

DEPENDENCIES:
    pip install requests beautifulsoup4 PyMuPDF pypdf opencv-python pytesseract numpy

PLUS install the Tesseract binary:
- macOS: brew install tesseract
- Windows: UB-Mannheim installer from the Tesseract wiki
- Linux: sudo apt-get install tesseract-ocr
"""
import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

import numpy as np
import cv2
import pytesseract
from pytesseract import Output

try:
    import fitz  # PyMuPDF
except Exception:
    print("PyMuPDF (fitz) is required. Install with: pip install PyMuPDF")
    raise

try:
    from pypdf import PdfReader, PdfWriter
except Exception:
    print("pypdf is required. Install with: pip install pypdf")
    raise


UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

BRAND_ALIASES = {
    "am. std": "American Standard",
    "am std": "American Standard",
    "amstd": "American Standard",
    "american standard": "American Standard",
    "delta": "Delta",
    "zurn": "Zurn",
    "centoco": "Centoco",
    "plumberex": "Plumberex",
    "aquatic": "Aquatic",
    "sloan": "Sloan",
    "kohler": "Kohler",
    "steris": "Steris",
    "steris/williams": "Steris",
    "sterling": "Sterling",
    "elkay": "Elkay",
    "american std": "American Standard",
}

KNOWN_BRANDS = set(BRAND_ALIASES.values())

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def sanitize_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "file"

def looks_like_pdf_url(u: str) -> bool:
    if not u:
        return False
    parsed = urlparse(u)
    if parsed.scheme not in ("file", "http", "https"):
        return False
    if parsed.path.lower().endswith(".pdf"):
        return True
    return False

def ddg_search_pdf(query: str, max_results: int = 5) -> List[str]:
    q = f"{query} filetype:pdf"
    url = "https://duckduckgo.com/html/"
    try:
        resp = requests.post(url, data={"q": q}, headers={"User-Agent": UA}, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"[search] DuckDuckGo failed for '{query}': {e}")
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    out: List[str] = []
    for a in soup.select("a.result__a"):
        href = a.get("href")
        if not href:
            continue
        out.append(href)
        if len(out) >= max_results:
            break
    return out

def head_is_pdf(url: str) -> bool:
    try:
        r = requests.head(url, headers={"User-Agent": UA}, allow_redirects=True, timeout=20)
        ctype = r.headers.get("Content-Type","").lower()
        if "pdf" in ctype:
            return True
        return url.lower().endswith(".pdf")
    except Exception:
        return url.lower().endswith(".pdf")

def download_pdf(url: str, dest: Path) -> Path:
    if url.startswith("file://"):
        local = Path(url.replace("file://", ""))
        if not local.exists():
            raise FileNotFoundError(f"Local file not found: {local}")
        return local

    filename = sanitize_filename(Path(urlparse(url).path).name or "download.pdf")
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    out = dest / filename

    with requests.get(url, headers={"User-Agent": UA}, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return out

def overlay_section_on_pdf(pdf_in: Path, section: str, pdf_out: Path) -> None:
    doc = fitz.open(pdf_in)
    fontname = "helv"
    fontsize = 12
    margin = 18
    for page in doc:
        rect = page.rect
        text = f"{section}"
        text_width = page.get_text_length(text, fontname=fontname, fontsize=fontsize)
        x = rect.x1 - margin - text_width
        y = rect.y0 + margin
        pad = 3
        box = fitz.Rect(x - pad, y - pad, x + text_width + pad, y + fontsize + pad)
        page.draw_rect(box, color=(1,1,1), fill=(1,1,1), fill_opacity=0.75, width=0)
        page.insert_text((x, y + fontsize*0.8), text, fontname=fontname, fontsize=fontsize,
                         color=(0,0,0), fill_opacity=1.0)
    doc.save(pdf_out)
    doc.close()

def merge_pdfs(pdf_paths: List[Path], out_path: Path) -> None:
    writer = PdfWriter()
    for p in pdf_paths:
        try:
            reader = PdfReader(str(p))
            for page in reader.pages:
                writer.add_page(page)
        except Exception as e:
            print(f"[merge] Skipping '{p}' due to error: {e}")
    with open(out_path, "wb") as f:
        writer.write(f)

def guess_brand_query(brand: str) -> str:
    if not brand:
        return ""
    b = brand.strip().lower()
    return BRAND_ALIASES.get(b, brand.title())

def search_and_download(item: str, brand: str, downloads_dir: Path) -> Optional[Path]:
    brand_q = guess_brand_query(brand)
    q = f'{item} "{brand_q}" specification pdf'
    print(f"[search] {q}")
    urls = ddg_search_pdf(q, max_results=6)
    candidates = [u for u in urls if looks_like_pdf_url(u) or head_is_pdf(u)]
    if not candidates:
        candidates = urls
    for u in candidates:
        try:
            p = download_pdf(u, downloads_dir)
            print(f"[downloaded] {u} -> {p}")
            return p
        except Exception as e:
            print(f"[warn] Could not download '{u}': {e}")
            continue
    return None

# -----------------------------
# OCR
# -----------------------------

SECTION_RE = re.compile(r"^[A-Z]{1,3}\d+[A-Z]?$")

def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=15)
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 41, 11)
    return th

def normalize_brand(token: str) -> Optional[str]:
    t = token.strip().lower().rstrip(",.:;")
    if not t:
        return None
    if t in BRAND_ALIASES:
        return BRAND_ALIASES[t]
    for kb in KNOWN_BRANDS:
        if t == kb.lower():
            return kb
        if t.replace(" ", "") == kb.replace(" ", "").lower():
            return kb
    return None

def ocr_to_rows(image_path: Path) -> List[Dict[str,str]]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    proc = preprocess_for_ocr(img)

    custom_oem_psm = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(proc, output_type=Output.DICT, config=custom_oem_psm)

    rows: List[Dict[str,str]] = []
    current_section: Optional[str] = None
    last_line_key: Optional[Tuple[int,int,int]] = None
    line_words: List[Dict[str,int]] = []

    def flush_line():
        nonlocal line_words, current_section, rows
        if not line_words:
            return
        texts = [w['text'] for w in line_words if str(w['text']).strip()]
        if not texts:
            line_words = []
            return

        line_words_sorted = sorted(line_words, key=lambda w: w['left'])
        section_token = None
        for w in line_words_sorted[:3]:
            t = str(w['text']).strip().upper()
            if SECTION_RE.match(t):
                section_token = t
                break
        if section_token:
            current_section = section_token

        brand = None
        texts_clean = [tw.strip() for tw in texts if tw.strip()]

        for n in (2,1):
            if len(texts_clean) >= n:
                cand = " ".join(texts_clean[-n:])
                br = normalize_brand(cand)
                if br:
                    brand = br
                    texts_clean = texts_clean[:-n]
                    break
        if brand is None and texts_clean:
            br = normalize_brand(texts_clean[-1])
            if br:
                brand = br
                texts_clean = texts_clean[:-1]

        if section_token and texts_clean and texts_clean[0].upper() == section_token:
            texts_clean = texts_clean[1:]

        item_desc = " ".join(texts_clean).strip()

        if current_section and item_desc:
            rows.append({"Section": current_section, "Item": item_desc, "Brand": brand or ""})

        line_words = []

    for i in range(len(data['text'])):
        try:
            conf = int(float(data['conf'][i]))
        except Exception:
            conf = -1
        if conf < 40:
            continue
        text = data['text'][i]
        if not text or not str(text).strip():
            continue
        block = data['block_num'][i]
        par = data['par_num'][i]
        line = data['line_num'][i]
        left = data['left'][i]

        key = (block, par, line)
        word_info = {"text": text, "left": int(left)}

        if last_line_key is None:
            last_line_key = key
        if key != last_line_key:
            flush_line()
            last_line_key = key

        line_words.append(word_info)

    flush_line()

    last_brand = ""
    for r in rows:
        if r["Brand"]:
            last_brand = r["Brand"]
        else:
            r["Brand"] = last_brand
    return rows

def write_rows_to_csv(rows: List[Dict[str,str]], csv_path: Path) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Section","Item","Brand","URL"])
        w.writeheader()
        for r in rows:
            r2 = {**r, "URL": ""}
            w.writerow(r2)

# -----------------------------
# Build packet
# -----------------------------

def build_packet(csv_path: Path, out_pdf: Path, workdir: Path) -> None:
    ensure_dir(workdir)
    downloads = ensure_dir(workdir / "downloads")
    stamped   = ensure_dir(workdir / "stamped")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        has_header = False
        if header:
            cols = [c.strip().lower() for c in header]
            has_header = ("section" in cols and "item" in cols and "brand" in cols)
        if has_header:
            f.seek(0)
            dreader = csv.DictReader(f)
            rows = list(dreader)
        else:
            f.seek(0)
            rows = []
            for row in csv.reader(f):
                if not row or all((str(c).strip()=="" for c in row)):
                    continue
                row = list(row) + [""]*(4-len(row))
                rows.append({"Section":row[0], "Item":row[1], "Brand":row[2], "URL":row[3]})

    ordered_paths: List[Path] = []
    for i, r in enumerate(rows, start=1):
        section = (r.get("Section") or r.get("section") or "").strip()
        item    = (r.get("Item") or r.get("item") or "").strip()
        brand   = (r.get("Brand") or r.get("brand") or "").strip()
        url     = (r.get("URL") or r.get("url") or "").strip()

        if not (section and item):
            print(f"[skip #{i}] Missing section or item -> {r}")
            continue

        src_pdf: Optional[Path] = None
        if url:
            try:
                src_pdf = download_pdf(url, downloads)
            except Exception as e:
                print(f"[warn] URL failed for row #{i}: {e}")
        if src_pdf is None:
            src_pdf = search_and_download(item, brand, downloads)

        if src_pdf is None:
            print(f"[ERROR] Could not find a PDF for row #{i}: Section={section} Item={item} Brand={brand}")
            continue

        stamped_path = stamped / f"{sanitize_filename(section)}__{sanitize_filename(src_pdf.name)}"
        try:
            overlay_section_on_pdf(src_pdf, section, stamped_path)
            ordered_paths.append(stamped_path)
        except Exception as e:
            print(f"[warn] stamping failed for {src_pdf}: {e}")
            ordered_paths.append(src_pdf)

    if not ordered_paths:
        raise SystemExit("No PDFs were processed. Aborting merge.")

    merge_pdfs(ordered_paths, out_pdf)
    print(f"[DONE] Packet created: {out_pdf}")

def main() -> None:
    ap = argparse.ArgumentParser(description="Build a merged spec packet from an OCR'd image or CSV list.")
    ap.add_argument("--image", help="Path to a photo/scan of the handwritten list to OCR into a CSV.")
    ap.add_argument("--csv", help="CSV path for input or OCR output (default: items_from_ocr.csv).")
    ap.add_argument("--out", default="SpecPacket.pdf", help="Output merged PDF filename (default: SpecPacket.pdf)")
    ap.add_argument("--workdir", default="./work", help="Working directory for downloads/stamped PDFs")
    ap.add_argument("--skip-build", action="store_true", help="If set with --image, perform OCR only and skip building the packet.")
    args = ap.parse_args()

    workdir  = Path(args.workdir).expanduser().resolve()
    out_pdf  = Path(args.out).expanduser().resolve()

    csv_path: Optional[Path] = None
    if args.image:
        image_path = Path(args.image).expanduser().resolve()
        rows = ocr_to_rows(image_path)
        if args.csv:
            csv_path = Path(args.csv).expanduser().resolve()
        else:
            csv_path = Path(str(image_path.with_suffix('')) + "_items.csv")
        write_rows_to_csv(rows, csv_path)
        print(f"[OCR] Wrote CSV: {csv_path}")
        if args.skip_build:
            return

    if not csv_path:
        if not args.csv:
            print("Error: provide --csv, or use --image to OCR.")
            sys.exit(2)
        csv_path = Path(args.csv).expanduser().resolve()

    build_packet(csv_path, out_pdf, workdir)

if __name__ == "__main__":
    main()
