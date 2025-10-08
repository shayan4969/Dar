# App.py
# Streamlit app: Upload a datasheet (PDF/CSV/XLSX), auto-extract Electrical Characteristics table, and generate DAR
#
# Requires:
# pip install streamlit pandas PyPDF2 pdfplumber reportlab openpyxl
#
# Run:
# streamlit run App.py

import streamlit as st
import pandas as pd
import re
from io import BytesIO
from datetime import datetime
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
import base64

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except Exception:
    PDFPLUMBER_AVAILABLE = False

# ----------------------------------------------------
# Streamlit Config
# ----------------------------------------------------
st.set_page_config(page_title="DAR from Datasheet", layout="wide")
st.title("DAR Generator — Improved Electrical Characteristics Extraction")
st.markdown("Upload a **datasheet (PDF, CSV, or XLSX)** and get a DAR table automatically extracted from its *Electrical Characteristics* section.")

# ----------------------------------------------------
# Helper patterns and functions
# ----------------------------------------------------
SECTION_KEYWORDS = [
    r"Electrical Characteristics",
    r"Static Electrical Characteristics",
    r"DC Characteristics",
    r"AC Electrical Characteristics",
    r"Electrical specifications",
]

NUM_UNIT_PATTERN = r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)(?:\s?([mMkKuUnpµμ]?)(V|A|Hz|kHz|MHz|GHz|Ω|ohm|°C|C|mW|W)?)"


def extract_text_with_pypdf2(pdf_bytes):
    text = ""
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        for p in reader.pages:
            txt = p.extract_text() or ""
            text += txt + "\n"
    except Exception as e:
        st.warning(f"PDF text extraction failed: {e}")
    return text


def find_section_pages(pdf_bytes):
    pages = []
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        for i, p in enumerate(reader.pages):
            txt = p.extract_text() or ""
            for kw in SECTION_KEYWORDS:
                if re.search(kw, txt, re.IGNORECASE):
                    pages.append(i)
                    break
    except Exception:
        pass
    return sorted(set(pages))


def extract_tables_with_pdfplumber(pdf_bytes, pages):
    tables = []
    if not PDFPLUMBER_AVAILABLE:
        return tables
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for p in pages:
                if p < 0 or p >= len(pdf.pages):
                    continue
                page = pdf.pages[p]
                tlist = page.extract_tables()
                for t in tlist:
                    df = pd.DataFrame(t).dropna(how="all")
                    if not df.empty:
                        tables.append(df)
    except Exception as e:
        st.warning(f"pdfplumber warning: {e}")
    return tables


def heuristic_table_parse(text):
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return pd.DataFrame()

    # find header
    header_idx = None
    for i, l in enumerate(lines[:10]):
        if re.search(r"(Parameter|Symbol|Min|Typ|Max|Unit)", l, re.IGNORECASE):
            header_idx = i
            break
    if header_idx is None:
        header_idx = 0

    split_marker = " :: "
    norm = []
    for l in lines[header_idx:]:
        l = re.sub(r"[ \t]{2,}", split_marker, l.strip())
        norm.append(l)

    header_cols = [c.strip() for c in norm[0].split(split_marker) if c.strip()]
    if not header_cols:
        header_cols = [f"Col{i}" for i in range(6)]

    rows = []
    for l in norm[1:]:
        parts = [p.strip() for p in l.split(split_marker)]
        if len(parts) < len(header_cols):
            parts += [""] * (len(header_cols) - len(parts))
        rows.append(parts[:len(header_cols)])

    df = pd.DataFrame(rows, columns=header_cols)
    df = df.replace("", pd.NA).dropna(how="all").reset_index(drop=True)
    return df


def map_columns(df):
    mapping = {}
    for c in df.columns:
        lc = c.lower()
        if "param" in lc:
            mapping[c] = "Parameter"
        elif "symbol" in lc:
            mapping[c] = "Symbol"
        elif "cond" in lc:
            mapping[c] = "Condition"
        elif "min" in lc:
            mapping[c] = "Min"
        elif "typ" in lc:
            mapping[c] = "Typ"
        elif "max" in lc:
            mapping[c] = "Max"
        elif "unit" in lc:
            mapping[c] = "Unit"
        elif "note" in lc or "remark" in lc:
            mapping[c] = "Notes"
    df = df.rename(columns=mapping)
    for c in ["Parameter", "Min", "Typ", "Max", "Unit"]:
        if c not in df.columns:
            df[c] = ""
    return df[["Parameter", "Min", "Typ", "Max", "Unit"]]


def parse_numeric_unit(s):
    if not isinstance(s, str):
        return None
    s = s.strip()
    m = re.search(NUM_UNIT_PATTERN, s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except:
        return None


def auto_pass_fail(min_v, max_v, measured):
    if measured is None:
        return "Unknown"
    try:
        if min_v is not None and max_v is not None:
            return "Pass" if min_v <= measured <= max_v else "Fail"
        elif max_v is not None:
            return "Pass" if measured <= max_v else "Fail"
        elif min_v is not None:
            return "Pass" if measured >= min_v else "Fail"
        else:
            return "Unknown"
    except:
        return "Unknown"


# ----------------------------------------------------
# File Upload
# ----------------------------------------------------
uploaded = st.file_uploader("Upload Datasheet (PDF preferred)", type=["pdf", "csv", "xlsx"])
if not uploaded:
    st.info("Upload a datasheet to start.")
    st.stop()

file_bytes = uploaded.read()
filename = uploaded.name
st.write(f"**Uploaded:** {filename}")

rows = []

# ----------------------------------------------------
# If PDF, extract Electrical Characteristics table
# ----------------------------------------------------
if filename.lower().endswith(".pdf"):
    st.info("Analyzing PDF...")
    section_pages = find_section_pages(file_bytes)
    st.write("Detected section pages:", section_pages or "none")

    # Try pdfplumber
    tables = []
    if PDFPLUMBER_AVAILABLE:
        with st.spinner("Extracting tables with pdfplumber..."):
            tables = extract_tables_with_pdfplumber(file_bytes, section_pages)
    if tables:
        df = max(tables, key=lambda t: t.shape[0])  # pick largest table
        df = df.fillna("").astype(str)
        # header normalization
        first_row = df.iloc[0].tolist()
        if any(re.search("[A-Za-z]", x) for x in first_row):
            df.columns = first_row
            df = df.drop(0).reset_index(drop=True)
        df = map_columns(df)
    else:
        # fallback: heuristic
        text = extract_text_with_pypdf2(file_bytes)
        df = heuristic_table_parse(text)
        df = map_columns(df)
else:
    # Spreadsheet input
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(file_bytes))
        else:
            df = pd.read_excel(BytesIO(file_bytes))
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()
    df = map_columns(df)

# ----------------------------------------------------
# Convert extracted data into editable DAR table
# ----------------------------------------------------
st.subheader("Extracted Parameters (editable)")
if df.empty:
    st.warning("No table extracted. Add rows manually.")
    df = pd.DataFrame([{"Parameter": "VCC", "Min": "3.0", "Typ": "3.3", "Max": "3.6", "Unit": "V"}])

df["Measured"] = ""
df["Pass/Fail"] = ""
df["Comments"] = ""

edited = st.experimental_data_editor(df, num_rows="dynamic")
st.markdown("---")

# ----------------------------------------------------
# Auto pass/fail suggestions
# ----------------------------------------------------
st.subheader("Auto Pass/Fail (optional)")
auto_results = []
for _, r in edited.iterrows():
    min_v = parse_numeric_unit(str(r.get("Min", "")))
    max_v = parse_numeric_unit(str(r.get("Max", "")))
    measured = parse_numeric_unit(str(r.get("Measured", "")))
    result = auto_pass_fail(min_v, max_v, measured)
    auto_results.append(result)
edited["_Auto"] = auto_results
st.dataframe(edited)

# ----------------------------------------------------
# DAR Metadata
# ----------------------------------------------------
st.subheader("DAR Metadata")
col1, col2, col3 = st.columns(3)
with col1:
    project = st.text_input("Project", "Project-X")
    component = st.text_input("Component", "")
    part_number = st.text_input("Part Number", "")
with col2:
    reviewer = st.text_input("Reviewer", "")
    date = st.date_input("Date", datetime.today())
    priority = st.selectbox("Priority", ["Low", "Medium", "High"])
with col3:
    status = st.selectbox("Status", ["Open", "Closed"])
    overall_result = st.selectbox("Overall Result", ["Pass", "Fail", "Conditional"])

# ----------------------------------------------------
# Export Functions
# ----------------------------------------------------
def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode()

def make_pdf(meta, df):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    margin = 15 * mm
    y = h - margin

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, f"DAR — {meta['project']} / {meta['component']}")
    y -= 8 * mm
    c.setFont("Helvetica", 9)
    c.drawString(margin, y, f"Part: {meta['part_number']} | Reviewer: {meta['reviewer']} | Date: {meta['date']}")
    y -= 8 * mm

    headers = ["Parameter", "Spec", "Measured", "Pass/Fail"]
    cols_x = [margin, margin + 60*mm, margin + 105*mm, margin + 135*mm]
    c.setFont("Helvetica-Bold", 10)
    for i, htxt in enumerate(headers):
        c.drawString(cols_x[i], y, htxt)
    y -= 6 * mm
    c.setFont("Helvetica", 9)

    for _, r in df.iterrows():
        if y < 30 * mm:
            c.showPage()
            y = h - margin
        spec = f"Min:{r.get('Min','')} Typ:{r.get('Typ','')} Max:{r.get('Max','')} {r.get('Unit','')}"
        c.drawString(cols_x[0], y, str(r.get("Parameter",""))[:40])
        c.drawString(cols_x[1], y, spec[:40])
        c.drawString(cols_x[2], y, str(r.get("Measured",""))[:15])
        c.drawString(cols_x[3], y, str(r.get("Pass/Fail",""))[:10])
        y -= 6 * mm

    c.save()
    buffer.seek(0)
    return buffer.read()

# ----------------------------------------------------
# Download Buttons
# ----------------------------------------------------
meta = {
    "project": project,
    "component": component,
    "part_number": part_number,
    "reviewer": reviewer,
    "date": date.isoformat(),
}

colA, colB = st.columns(2)
with colA:
    if st.button("Download DAR CSV"):
        csv_bytes = df_to_csv_bytes(edited)
        b64 = base64.b64encode(csv_bytes).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="dar_{project or "project"}_{part_number or "part"}.csv">📥 Download DAR CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

with colB:
    if st.button("Download DAR PDF"):
        pdf_bytes = make_pdf(meta, edited)
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="dar_{project or "project"}_{part_number or "part"}.pdf">📄 Download DAR PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

st.success("✅ Ready! You can now edit, auto-check, and export DARs from datasheets.")
