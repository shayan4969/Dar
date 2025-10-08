# App.py
# Streamlit DAR generator — robust PDF reading + optional OCR + improved extraction
#
# Recommended install:
# pip install streamlit pandas PyPDF2 pdfplumber reportlab openpyxl pytesseract pdf2image pillow
# Also install Tesseract binary (OS package) if you want OCR:
# - Ubuntu: sudo apt install tesseract-ocr
# - macOS (brew): brew install tesseract
#
# Run:
# streamlit run App.py

import streamlit as st
import pandas as pd
import re
from io import BytesIO
from datetime import datetime

# Optional libraries — attempt imports, set flags
PYPDF2_AVAILABLE = False
PDFPLUMBER_AVAILABLE = False
REPORTLAB_AVAILABLE = False
PYTESSERACT_AVAILABLE = False
PDF2IMAGE_AVAILABLE = False
PIL_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except Exception:
    PYPDF2_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except Exception:
    PDFPLUMBER_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except Exception:
    PDF2IMAGE_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

st.set_page_config(page_title="DAR from Datasheet — PDF Reader", layout="wide")
st.title("DAR Generator — PDF reading (text, tables, optional OCR)")

st.markdown(
    """
Upload a datasheet (PDF recommended) or a spreadsheet (CSV/XLSX).  
This app will attempt to read text and tables from the PDF using multiple strategies:
- `pdfplumber` (best for table extraction in text PDFs),
- `PyPDF2` (fallback for text extraction),
- optional OCR using `pytesseract` + `pdf2image` if the PDF is scanned.
"""
)

# ----------------------------
# Regex / helpers / heuristics
# ----------------------------
SECTION_KEYWORDS = [
    r"Electrical Characteristics",
    r"Static Electrical Characteristics",
    r"DC Characteristics",
    r"AC Electrical Characteristics",
    r"Electrical specifications",
    r"Electrical Characteristics \(continued\)",
]

NUM_UNIT_PATTERN = r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)(?:\s?([mMkKuUnpµμ]?)(V|A|Hz|kHz|MHz|GHz|Ω|ohm|°C|C|mW|W)?)"

def extract_text_with_pdfplumber(pdf_bytes):
    """Extract text from PDF using pdfplumber; also gather table objects."""
    if not PDFPLUMBER_AVAILABLE:
        return "", []
    text_all = []
    tables = []
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as doc:
            for page in doc.pages:
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
                text_all.append(txt)
                try:
                    page_tables = page.extract_tables() or []
                except Exception:
                    page_tables = []
                for t in page_tables:
                    # convert to DataFrame if not empty
                    df = pd.DataFrame(t).dropna(how="all")
                    if not df.empty:
                        tables.append(df)
    except Exception:
        return "", []
    return "\n".join(filter(None, text_all)), tables

def extract_text_with_pypdf2(pdf_bytes):
    """Extract concatenated page text using PyPDF2."""
    if not PYPDF2_AVAILABLE:
        return ""
    out_lines = []
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        for p in reader.pages:
            try:
                txt = p.extract_text() or ""
            except Exception:
                txt = ""
            out_lines.append(txt)
    except Exception:
        return ""
    return "\n".join(filter(None, out_lines))

def ocr_pdf_to_text(pdf_bytes, max_pages=10):
    """Run OCR (pytesseract) on PDF pages converted to images using pdf2image.
    Limits to max_pages for performance."""
    if not (PYTESSERACT_AVAILABLE and PDF2IMAGE_AVAILABLE and PIL_AVAILABLE):
        return ""
    try:
        images = convert_from_bytes(pdf_bytes, first_page=1, last_page=max_pages)
    except Exception:
        return ""
    texts = []
    for img in images:
        try:
            txt = pytesseract.image_to_string(img)
        except Exception:
            txt = ""
        texts.append(txt)
    return "\n".join(filter(None, texts))

def find_pages_with_heading(text_pages, section_patterns=SECTION_KEYWORDS):
    """Given per-page text (list), return page indices where heading occurs."""
    pages = []
    for i, t in enumerate(text_pages):
        if not t:
            continue
        for pat in section_patterns:
            if re.search(pat, t, re.IGNORECASE):
                pages.append(i)
                break
    return pages

def heuristic_table_parse_block(text_block):
    """Parse a block of text into a table-like DataFrame using whitespace heuristics."""
    lines = [l for l in text_block.splitlines() if l.strip()]
    if not lines:
        return pd.DataFrame()
    # find header line
    header_idx = 0
    for i, l in enumerate(lines[:10]):
        if re.search(r"(Parameter|Symbol|Min|Typ|Max|Unit|Condition)", l, re.IGNORECASE):
            header_idx = i
            break
    marker = " ::: "
    normalized = [re.sub(r"[ \t]{2,}", marker, l.strip()) for l in lines[header_idx:]]
    header_cols = [c.strip() for c in normalized[0].split(marker) if c.strip()]
    if not header_cols:
        header_cols = [f"Col{i}" for i in range(6)]
    rows = []
    for l in normalized[1:]:
        parts = [p.strip() for p in l.split(marker)]
        if len(parts) < len(header_cols):
            parts += [""] * (len(header_cols) - len(parts))
        rows.append(parts[:len(header_cols)])
    df = pd.DataFrame(rows, columns=header_cols)
    df = df.replace("", pd.NA).dropna(how="all").reset_index(drop=True)
    return df

def map_table_to_standard(df):
    """Map table headers into standard columns: Parameter, Symbol, Condition, Min, Typ, Max, Unit, Notes"""
    mapping = {}
    for c in df.columns:
        lc = str(c).lower()
        if "param" in lc or "test" in lc or "item" in lc or "characteristic" in lc:
            mapping[c] = "Parameter"
        elif "symbol" in lc:
            mapping[c] = "Symbol"
        elif "cond" in lc:
            mapping[c] = "Condition"
        elif "min" in lc and "max" not in lc:
            mapping[c] = "Min"
        elif "typ" in lc:
            mapping[c] = "Typ"
        elif "max" in lc:
            mapping[c] = "Max"
        elif "unit" in lc:
            mapping[c] = "Unit"
        elif "note" in lc or "remark" in lc or "comment" in lc:
            mapping[c] = "Notes"
    df2 = df.rename(columns=mapping)
    for col in ["Parameter", "Symbol", "Condition", "Min", "Typ", "Max", "Unit", "Notes"]:
        if col not in df2.columns:
            df2[col] = ""
    # keep columns in that order
    return df2[["Parameter", "Symbol", "Condition", "Min", "Typ", "Max", "Unit", "Notes"]]

def parse_first_number(s):
    if s is None:
        return None
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    m = re.search(NUM_UNIT_PATTERN, s)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    m2 = re.search(r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)", s)
    if m2:
        try:
            return float(m2.group(1))
        except:
            return None
    return None

# PDF reading orchestration: try pdfplumber -> PyPDF2 -> OCR (only if needed)
def extract_text_and_tables_from_pdf(pdf_bytes, try_ocr_if_no_text=True):
    """Return tuple (per_page_texts: list[str], tables: list[DataFrame])"""
    per_page_texts = []
    tables = []
    # 1) try pdfplumber for both text and tables (preferred)
    if PDFPLUMBER_AVAILABLE:
        try:
            text_all, tbls = extract_text_with_pdfplumber(pdf_bytes)
            # pdfplumber returned a single large text string; split into pages if possible by page breaks
            # note: pdfplumber extract_text returned per page strings earlier; here we used join. We'll instead re-open to get per-page strings:
            try:
                with pdfplumber.open(BytesIO(pdf_bytes)) as doc:
                    per_page_texts = [(page.extract_text() or "") for page in doc.pages]
            except Exception:
                # fallback: use the combined text split crudely
                per_page_texts = text_all.split("\f") if text_all else []
            tables = tbls or []
            if any(per_page_texts):
                return per_page_texts, tables
        except Exception:
            # continue to PyPDF2
            pass

    # 2) Try PyPDF2 text extraction
    if PYPDF2_AVAILABLE:
        try:
            text_all = extract_text_with_pypdf2(pdf_bytes)
            # split into pages by form feed or attempt to extract pages separately
            try:
                reader = PdfReader(BytesIO(pdf_bytes))
                per_page_texts = []
                for p in reader.pages:
                    per_page_texts.append(p.extract_text() or "")
            except Exception:
                per_page_texts = text_all.split("\f") if text_all else []
            if any(per_page_texts):
                return per_page_texts, tables
        except Exception:
            pass

    # 3) If no text and OCR is available and requested, run OCR (may be slow)
    if try_ocr_if_no_text and PYTESSERACT_AVAILABLE and PDF2IMAGE_AVAILABLE and PIL_AVAILABLE:
        try:
            txt = ocr_pdf_to_text(pdf_bytes, max_pages=10)
            if txt:
                # split crude pages by blank lines or leave as single block
                per_page_texts = txt.split("\f") if "\f" in txt else [txt]
                return per_page_texts, tables
        except Exception:
            pass

    return per_page_texts, tables

# ----------------------------
# UI: file upload
# ----------------------------
uploaded = st.file_uploader("Upload datasheet (PDF preferred) or CSV/XLSX", type=["pdf", "csv", "xlsx", "xls"])
if not uploaded:
    st.info("Upload a datasheet to begin. PDF parsing uses pdfplumber/PyPDF2. Install optional OCR libs for scanned PDFs.")
    st.stop()

file_bytes = uploaded.read()
filename = uploaded.name
st.write("Uploaded:", filename)

# If PDF: attempt to extract
extracted_df = pd.DataFrame()
if filename.lower().endswith(".pdf"):
    st.info("Reading PDF — trying multiple extraction strategies...")
    per_page_texts, tables = extract_text_and_tables_from_pdf(file_bytes, try_ocr_if_no_text=True)

    # show what methods are available
    st.write("Available PDF libs:", {
        "pdfplumber": PDFPLUMBER_AVAILABLE,
        "PyPDF2": PYPDF2_AVAILABLE,
        "pytesseract+pdf2image": (PYTESSERACT_AVAILABLE and PDF2IMAGE_AVAILABLE and PIL_AVAILABLE)
    })

    # quick debug: show which pages had text
    pages_with_text = [i for i,t in enumerate(per_page_texts) if (t and t.strip())] if per_page_texts else []
    st.write("Pages with text extracted:", pages_with_text if pages_with_text else "none")

    # If pdfplumber returned structured table(s), prefer them
    df_candidate = pd.DataFrame()
    if tables:
        st.success(f"Found {len(tables)} table(s) via pdfplumber — trying the largest one.")
        # pick largest table by rows
        df_candidate = max(tables, key=lambda t: t.shape[0])
        df_candidate = df_candidate.fillna("").astype(str)
        # if first row looks like header, apply it
        first_row = df_candidate.iloc[0].tolist()
        if any(re.search(r"[A-Za-z]", str(x)) for x in first_row):
            df_candidate.columns = first_row
            df_candidate = df_candidate.drop(df_candidate.index[0]).reset_index(drop=True)

    # If no structured tables, try to locate "Electrical Characteristics" heading in page texts and parse that block
    if (df_candidate is None or df_candidate.empty) and per_page_texts:
        st.info("Searching for 'Electrical Characteristics' heading in extracted page texts...")
        heading_pages = find_pages_with_heading(per_page_texts, SECTION_KEYWORDS)
        st.write("Heading found on pages:", heading_pages if heading_pages else "none")
        # build text block from heading pages and nearby pages
        if heading_pages:
            # prefer the first occurrence
            p = heading_pages[0]
            start = max(0, p - 1)
            end = min(len(per_page_texts) - 1, p + 1)
            block = "\n\n".join(per_page_texts[start:end+1])
            df_candidate = heuristic_table_parse_block(block)
        else:
            # try entire document text heuristics (less likely to be neat)
            block = "\n\n".join(per_page_texts)
            df_candidate = heuristic_table_parse_block(block)

    # Map to standard columns if we have candidate table
    if df_candidate is not None and not df_candidate.empty:
        try:
            extracted_df = map_table_to_standard(df_candidate)
            st.success("Converted candidate table into standard DAR columns.")
        except Exception as e:
            st.warning(f"Failed to map table to standard columns: {e}")
            extracted_df = pd.DataFrame()
    else:
        st.warning("No usable table extracted from PDF automatically. You can add rows manually.")

else:
    # CSV/XLSX path
    try:
        if filename.lower().endswith(".csv"):
            df_in = pd.read_csv(BytesIO(file_bytes))
        else:
            df_in = pd.read_excel(BytesIO(file_bytes))
        st.subheader("Preview of uploaded spreadsheet")
        st.dataframe(df_in.head(30))
        extracted_df = map_table_to_standard(df_in)
    except Exception as e:
        st.error(f"Failed to parse uploaded spreadsheet: {e}")
        extracted_df = pd.DataFrame()

# ----------------------------
# Editable DAR table UI
# ----------------------------
st.subheader("Extracted / Suggested Parameters")
if extracted_df.empty:
    st.info("No automatic extraction — start with a sample row or paste CSV lines.")
    extracted_df = pd.DataFrame([{
        "Parameter": "VCC",
        "Symbol": "",
        "Condition": "",
        "Min": "3.0",
        "Typ": "3.3",
        "Max": "3.6",
        "Unit": "V",
        "Notes": ""
    }])

# ensure columns exist
for col in ["Parameter", "Symbol", "Condition", "Min", "Typ", "Max", "Unit", "Notes"]:
    if col not in extracted_df.columns:
        extracted_df[col] = ""

extracted_df["Measured"] = ""
extracted_df["Pass/Fail"] = ""
extracted_df["_AutoSuggestion"] = ""

# Allow editing
try:
    edited = st.experimental_data_editor(extracted_df, num_rows="dynamic")
except Exception:
    st.info("experimental_data_editor not available — use CSV editor fallback")
    csv_text = st.text_area("Edit parameter CSV (Parameter,Symbol,Condition,Min,Typ,Max,Unit,Notes,Measured,Pass/Fail)", extracted_df.to_csv(index=False))
    try:
        edited = pd.read_csv(BytesIO(csv_text.encode()))
    except Exception:
        st.error("Edited CSV parse failed. Reverting to suggestion.")
        edited = extracted_df.copy()

# ----------------------------
# Auto Pass/Fail suggestions
# ----------------------------
st.subheader("Auto Pass/Fail suggestions")
auto_list = []
for _, row in edited.iterrows():
    min_v = parse_first_number(row.get("Min", ""))
    max_v = parse_first_number(row.get("Max", ""))
    measured_v = parse_first_number(row.get("Measured", ""))
    if measured_v is None:
        suggestion = "Unknown"
    else:
        if (min_v is not None) and (max_v is not None):
            suggestion = "Pass" if (min_v <= measured_v <= max_v) else "Fail"
        elif max_v is not None:
            suggestion = "Pass" if measured_v <= max_v else "Fail"
        elif min_v is not None:
            suggestion = "Pass" if measured_v >= min_v else "Fail"
        else:
            suggestion = "Unknown"
    auto_list.append(suggestion)
edited["_AutoSuggestion"] = auto_list

st.dataframe(edited)

# ----------------------------
# DAR metadata inputs
# ----------------------------
st.subheader("DAR Metadata")
c1, c2, c3 = st.columns(3)
with c1:
    project = st.text_input("Project", value="Project-X")
    component = st.text_input("Component / Part", value="")
    part_number = st.text_input("Part Number", value="")
with c2:
    reviewer = st.text_input("Reviewer", value="")
    date = st.date_input("Date", value=datetime.today())
    priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
with c3:
    status = st.selectbox("Status", ["Open", "In Progress", "Resolved", "Closed"])
    overall_result = st.selectbox("Overall Result", ["Conditional", "Pass", "Fail"])

meta = {
    "project": project,
    "component": component,
    "part_number": part_number,
    "reviewer": reviewer,
    "date": date.isoformat(),
    "priority": priority,
    "status": status,
    "overall_result": overall_result,
}

# ----------------------------
# Export (safe buttons)
# ----------------------------
def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

def generate_pdf_bytes(meta, df):
    if not REPORTLAB_AVAILABLE:
        return None
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    margin = 15 * mm
    y = h - margin
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, f"DAR — {meta.get('project','')} / {meta.get('component','')}")
    y -= 8 * mm
    c.setFont("Helvetica", 9)
    c.drawString(margin, y, f"Part: {meta.get('part_number','')}  Reviewer: {meta.get('reviewer','')}  Date: {meta.get('date','')}")
    y -= 8 * mm
    cols_x = [margin, margin + 70*mm, margin + 110*mm, margin + 140*mm]
    headers = ["Parameter", "Spec (Min/Typ/Max/Unit)", "Measured", "Pass/Fail"]
    c.setFont("Helvetica-Bold", 10)
    for i, hh in enumerate(headers):
        c.drawString(cols_x[i], y, hh)
    y -= 6 * mm
    c.setFont("Helvetica", 9)
    for _, r in df.iterrows():
        if y < 30*mm:
            c.showPage()
            y = h - margin
        spec = f"Min:{r.get('Min','')} Typ:{r.get('Typ','')} Max:{r.get('Max','')} {r.get('Unit','') or ''}"
        c.drawString(cols_x[0], y, str(r.get("Parameter",""))[:40])
        c.drawString(cols_x[1], y, spec[:50])
        c.drawString(cols_x[2], y, str(r.get("Measured",""))[:15])
        pf = r.get("Pass/Fail") or r.get("_AutoSuggestion","")
        c.drawString(cols_x[3], y, str(pf)[:12])
        y -= 6 * mm
    c.save()
    buf.seek(0)
    return buf.read()

colA, colB, colC = st.columns([1,1,1])
with colA:
    csv_bytes = df_to_csv_bytes(edited.assign(**meta))
    st.download_button(
        label="Download DAR (CSV)",
        data=csv_bytes,
        file_name=f"dar_{project or 'project'}_{part_number or 'part'}.csv",
        mime="text/csv",
    )

with colB:
    if REPORTLAB_AVAILABLE:
        pdf_bytes = generate_pdf_bytes(meta, edited)
        if pdf_bytes:
            st.download_button(
                label="Download DAR (PDF)",
                data=pdf_bytes,
                file_name=f"dar_{project or 'project'}_{part_number or 'part'}.pdf",
                mime="application/pdf",
            )
        else:
            st.info("PDF generation returned empty.")
    else:
        st.info("PDF export disabled. Install reportlab (`pip install reportlab`) to enable.")

with colC:
    try:
        # prefer Excel if openpyxl is present
        import openpyxl  # type: ignore
        excel_buf = BytesIO()
        edited.assign(**meta).to_excel(excel_buf, index=False, engine="openpyxl")
        excel_buf.seek(0)
        st.download_button(
            label="Download DAR (XLSX)",
            data=excel_buf,
            file_name=f"dar_{project or 'project'}_{part_number or 'part'}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception:
        # fallback to CSV
        st.download_button(
            label="Download DAR (CSV fallback)",
            data=csv_bytes,
            file_name=f"dar_{project or 'project'}_{part_number or 'part'}.csv",
            mime="text/csv",
        )

st.success("DAR ready — edit above and export.")

# final tips about installed libs
st.markdown("**Environment info (available libs):**")
st.write({
    "pdfplumber": PDFPLUMBER_AVAILABLE,
    "PyPDF2": PYPDF2_AVAILABLE,
    "pytesseract+pdf2image": (PYTESSERACT_AVAILABLE and PDF2IMAGE_AVAILABLE and PIL_AVAILABLE),
    "reportlab": REPORTLAB_AVAILABLE
})

if not (PDFPLUMBER_AVAILABLE or PYPDF2_AVAILABLE):
    st.info("Install PyPDF2/pdfplumber to enable better PDF parsing: pip install PyPDF2 pdfplumber")
if (PYTESSERACT_AVAILABLE and (not PDF2IMAGE_AVAILABLE or not PIL_AVAILABLE)):
    st.info("pdf2image or Pillow not found; OCR may be unavailable.")
if not PYTESSERACT_AVAILABLE:
    st.info("Install pytesseract and the tesseract binary for OCR: pip install pytesseract; then install tesseract via OS package manager.")
