# streamlit_datasheet_ec_extractor.py
"""
Datasheet → Electrical Characteristics
Streamlit app (single-file) that:
 - accepts PDF or image uploads
 - extracts candidate electrical-characteristics from tables (pdfplumber)
 - falls back to OCR (pytesseract + pdf2image) for scanned pages
 - normalizes numbers/units simply
 - shows editable table and export as JSON/CSV

Notes:
 - For best results install tesseract-ocr on your system if using scanned PDFs.
 - Camelot/tabula are not used here to keep this simple and easier to run.
"""

import io
import re
import json
import tempfile
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd
from PIL import Image
import pdfplumber

# Optional OCR imports (used if text extraction fails)
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    import numpy as np
    import cv2
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# --- Helpers ---------------------------------------------------------------

UNIT_TOKENS = [
    "v", "mv", "a", "ma", "ua", "µa", "ohm", "Ω", "kohm", "kΩ",
    "hz", "khz", "mhz", "pf", "nf", "uf", "w", "db"
]

NUMERIC_RE = re.compile(r"[-+]?(\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")

def normalize_unit(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.strip().lower()
    # simplify micro symbol
    s = s.replace("µ", "u")
    s = s.replace("ohm", "Ω")
    s = s.replace("kohm", "kΩ")
    # small mapping / cleaning
    if s in ["v","mv","kv","uv","uv"]:
        return s.upper()
    # return verbatim if looks like unit
    if any(tok in s for tok in UNIT_TOKENS):
        return s
    return s

def parse_numeric_from_string(s: str) -> Optional[float]:
    if not s: return None
    s = s.replace("−", "-").replace("–", "-").replace(",", "")
    m = NUMERIC_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except:
        return None

def simple_row_confidence(row: Dict[str, Any]) -> float:
    # basic heuristic: presence of parameter and at least one numeric value increases confidence
    score = 0.1
    if row.get("parameter_name"): score += 0.4
    if row.get("typ") is not None: score += 0.2
    if row.get("min") is not None: score += 0.15
    if row.get("max") is not None: score += 0.15
    return min(1.0, score)

# --- Extraction strategies -------------------------------------------------

def extract_using_pdfplumber_bytes(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Use pdfplumber to extract textual tables and heuristically find electrical-characteristics.
    Returns list of extracted parameter dicts.
    """
    out = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            # 1) try structured table extraction
            tables = page.extract_tables()
            if tables:
                for t in tables:
                    if len(t) < 2:  # need header + rows
                        continue
                    header = [ (c or "").strip().lower() for c in t[0] ]
                    # check if table looks like electrical-characteristics header
                    if any("parameter" in h or "symbol" in h for h in header) and any(h in ("min","typ","max","unit","units") or "typ" in h for h in header):
                        # map header indices
                        def find(hints):
                            for idx, h in enumerate(header):
                                for opt in hints:
                                    if opt in h:
                                        return idx
                            return None
                        param_idx = find(["parameter","symbol","description"]) or 0
                        min_idx = find(["min"])
                        typ_idx = find(["typ","typical"])
                        max_idx = find(["max"])
                        unit_idx = find(["unit","units"])
                        tc_idx = find(["condition","test","notes","test conditions"])
                        for row in t[1:]:
                            if not any(row): continue
                            param = (row[param_idx] or "").strip() if param_idx is not None else ""
                            min_v = parse_numeric_from_string(row[min_idx]) if min_idx is not None and row[min_idx] else None
                            typ_v = parse_numeric_from_string(row[typ_idx]) if typ_idx is not None and row[typ_idx] else None
                            max_v = parse_numeric_from_string(row[max_idx]) if max_idx is not None and row[max_idx] else None
                            unit_s = (row[unit_idx] or "").strip() if unit_idx is not None and row[unit_idx] else None
                            tc = (row[tc_idx] or "").strip() if tc_idx is not None and row[tc_idx] else ""
                            rec = {
                                "parameter_name": param,
                                "min": min_v,
                                "typ": typ_v,
                                "max": max_v,
                                "unit": normalize_unit(unit_s),
                                "test_conditions": tc,
                                "source_page": pno
                            }
                            rec["confidence"] = simple_row_confidence(rec)
                            out.append(rec)
            # 2) fallback: scan lines of text for candidate lines with units & numeric tokens
            text = page.extract_text() or ""
            if text:
                for line in text.splitlines():
                    low = line.lower()
                    if any(u in low for u in UNIT_TOKENS) and any(k in low for k in ["typ","typical","min","max","test",":"]):
                        # try to split param : values
                        if ":" in line:
                            left, right = line.split(":", 1)
                            param = left.strip()
                            nums = NUMERIC_RE.findall(right)
                            first_num = parse_numeric_from_string(right)
                            unit = None
                            for u in UNIT_TOKENS:
                                if u in right.lower():
                                    unit = u
                                    break
                            rec = {
                                "parameter_name": param,
                                "min": None,
                                "typ": first_num,
                                "max": None,
                                "unit": normalize_unit(unit),
                                "test_conditions": None,
                                "source_page": pno,
                                "confidence": 0.45
                            }
                            out.append(rec)
    return out

def ocr_extract_from_pdf_bytes(pdf_bytes: bytes, max_pages: int = 10) -> List[Dict[str, Any]]:
    """
    Convert PDF to images and run OCR. Attempt a very simple heuristic:
    - Use pytesseract to get line-level text and search for lines containing unit tokens & numeric values.
    - Use naive splitting heuristics to find parameter name and numeric fields.
    """
    if not OCR_AVAILABLE:
        return []
    out = []
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=200)
    except Exception:
        return out
    for pno, pil_img in enumerate(pages[:max_pages], start=1):
        img = np.array(pil_img.convert("RGB"))
        # optional preprocessing: grayscale + threshold
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # slight blur + adaptive threshold to improve OCR on some scans
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
        ocr_result = pytesseract.image_to_string(th)
        for line in ocr_result.splitlines():
            low = line.lower()
            if any(u in low for u in UNIT_TOKENS) and re.search(r"\d", line):
                # parse colon separated or dash separated
                param = None
                typ_v = None
                min_v = None
                max_v = None
                unit = None
                if ":" in line:
                    left, right = line.split(":",1)
                    param = left.strip()
                    typ_v = parse_numeric_from_string(right)
                    for u in UNIT_TOKENS:
                        if u in right.lower():
                            unit = u
                            break
                else:
                    # try "Parameter  typ  value unit" style by splitting
                    parts = re.split(r"\s{2,}|\t", line)
                    if len(parts) >= 2:
                        param = parts[0].strip()
                        typ_v = parse_numeric_from_string(parts[1])
                        for u in UNIT_TOKENS:
                            if u in parts[1].lower():
                                unit = u
                                break
                rec = {
                    "parameter_name": param or line.strip(),
                    "min": min_v,
                    "typ": typ_v,
                    "max": max_v,
                    "unit": normalize_unit(unit),
                    "test_conditions": None,
                    "source_page": pno,
                    "confidence": 0.35
                }
                out.append(rec)
    return out

# --- UI & glue ------------------------------------------------------------

st.set_page_config(page_title="Datasheet → Electrical Characteristics", layout="wide")
st.title("Datasheet → Electrical Characteristics")
st.markdown("Upload a datasheet (PDF or image). The app will try to extract **electrical characteristics** (Parameter, Min/Typ/Max, Unit, Test conditions).")

uploaded_file = st.file_uploader("Upload datasheet (PDF, PNG, JPG)", type=["pdf","png","jpg","jpeg"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    st.sidebar.markdown(f"**File:** {uploaded_file.name}  •  **Size:** {len(file_bytes)/1024:.1f} KB")

    with st.spinner("Extracting with pdfplumber..."):
        try:
            parsed = extract_using_pdfplumber_bytes(file_bytes)
        except Exception as e:
            st.error(f"pdfplumber extraction failed: {e}")
            parsed = []

    # If nothing found, try OCR fallback
    if (not parsed) and OCR_AVAILABLE:
        with st.spinner("No table-like structures found — trying OCR fallback (pytesseract)..."):
            try:
                parsed = ocr_extract_from_pdf_bytes(file_bytes, max_pages=8)
            except Exception as e:
                st.error(f"OCR extraction failed: {e}")
                parsed = []

    if not parsed:
        st.warning("No electrical-characteristics were detected automatically. You can still enter manually below.")
        parsed = []

    # Normalize into dataframe
    df = pd.DataFrame(parsed)
    # ensure expected columns
    for c in ["parameter_name","min","typ","max","unit","test_conditions","source_page","confidence"]:
        if c not in df.columns:
            df[c] = None

    # Show preview and allow editing
    st.markdown("### Extracted candidates (editable)")
    if df.shape[0] == 0:
        st.info("No rows detected — add a row manually")
        df = pd.DataFrame(columns=["parameter_name","min","typ","max","unit","test_conditions","source_page","confidence"])
    edited = st.experimental_data_editor(df, num_rows="dynamic")  # editable table

    # Clean up edited rows and compute normalized values
    def cleaned_rows(df_in: pd.DataFrame) -> List[Dict[str,Any]]:
        outlist = []
        for _, row in df_in.iterrows():
            param = str(row.get("parameter_name") or "").strip()
            if param == "":  # skip empty
                continue
            min_v = parse_numeric_from_string(str(row.get("min") or "")) if row.get("min") is not None else None
            typ_v = parse_numeric_from_string(str(row.get("typ") or "")) if row.get("typ") is not None else None
            max_v = parse_numeric_from_string(str(row.get("max") or "")) if row.get("max") is not None else None
            unit = normalize_unit(str(row.get("unit") or "")) if row.get("unit") else None
            tc = str(row.get("test_conditions") or "")
            source = int(row.get("source_page")) if row.get("source_page") not in (None,"") else None
            rec = {
                "parameter_name": param,
                "min": min_v,
                "typ": typ_v,
                "max": max_v,
                "unit": unit,
                "test_conditions": tc if tc else None,
                "source_page": source,
            }
            rec["confidence"] = simple_row_confidence(rec)
            outlist.append(rec)
        return outlist

    final_list = cleaned_rows(edited)
    st.markdown("### Final structured output preview")
    if final_list:
        st.json(final_list)
        df_final = pd.DataFrame(final_list)
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            st.download_button("Download JSON", data=json.dumps(final_list, indent=2), file_name="electrical_characteristics.json", mime="application/json")
        with col2:
            csv_bytes = df_final.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv_bytes, file_name="electrical_characteristics.csv", mime="text/csv")
        with col3:
            st.dataframe(df_final)
    else:
        st.info("No final rows (empty). Use the editor above to add electrical characteristics manually.")

    st.markdown("---")
    st.markdown("**Notes & tips**")
    st.markdown("""
    - If your datasheet is scanned, install Tesseract OCR on your system and the `pytesseract` and `pdf2image` python packages.  
    - This simple prototype relies on heuristics. For better accuracy, consider integrating `camelot` for well-formed PDF tables or a document layout model (LayoutLMv3) for complex layouts.  
    - You can paste/edit parameter names and values in the table above — the app will export exactly what you confirm.
    """)
else:
    st.info("Upload a datasheet to get started. (PDFs with selectable text work best.)")
