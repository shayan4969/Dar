# App.py
# Streamlit DAR generator — improved extraction + robust downloads
#
# Install recommended deps:
# pip install streamlit pandas PyPDF2 pdfplumber reportlab openpyxl
#
# Run:
# streamlit run App.py

import streamlit as st
import pandas as pd
import re
from io import BytesIO
from datetime import datetime
import base64

# Optional libraries (may not be present in all environments)
PYPDF2_AVAILABLE = False
PDFPLUMBER_AVAILABLE = False
REPORTLAB_AVAILABLE = False

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

st.set_page_config(page_title="DAR from Datasheet", layout="wide")
st.title("DAR Generator — Improved Electrical Characteristics Extraction")

st.markdown(
    """
Upload a datasheet (PDF recommended) or a spreadsheet (CSV/XLSX).  
The app will try to extract the *Electrical Characteristics* table and convert it to editable DAR rows.
"""
)

# ----------------------
# Patterns & helpers
# ----------------------
SECTION_KEYWORDS = [
    r"Electrical Characteristics",
    r"Static Electrical Characteristics",
    r"DC Characteristics",
    r"AC Electrical Characteristics",
    r"Electrical specifications",
]

NUM_UNIT_PATTERN = r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)(?:\s?([mMkKuUnpµμ]?)(V|A|Hz|kHz|MHz|GHz|Ω|ohm|°C|C|mW|W)?)"


def notify_missing_pdf_libs():
    st.warning(
        "PyPDF2/pdfplumber are not installed — PDF parsing will be limited.\n\n"
        "Install them with:\n\n`pip install PyPDF2 pdfplumber`\n\n"
        "CSV/XLSX uploads will still work."
    )


def extract_text_pypdf2(pdf_bytes):
    if not PYPDF2_AVAILABLE:
        return ""
    out = []
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        for p in reader.pages:
            try:
                txt = p.extract_text() or ""
            except Exception:
                txt = ""
            out.append(txt)
    except Exception:
        return ""
    return "\n".join(out)


def find_section_pages(pdf_bytes):
    pages = []
    if not PYPDF2_AVAILABLE:
        return pages
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


def extract_tables_pdfplumber(pdf_bytes, pages):
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
    except Exception:
        pass
    return tables


def heuristic_table_from_text(text_block):
    lines = [l for l in text_block.splitlines() if l.strip()]
    if not lines:
        return pd.DataFrame()
    # pick header line heuristically
    header_idx = 0
    for i, l in enumerate(lines[:10]):
        if re.search(r"(Parameter|Symbol|Min|Typ|Max|Unit|Condition)", l, re.IGNORECASE):
            header_idx = i
            break
    # normalize multiple spaces -> marker
    marker = " ::: "
    normalized = []
    for l in lines[header_idx:]:
        normalized.append(re.sub(r"[ \t]{2,}", marker, l.strip()))
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
    # map headers to standard names where possible
    mapping = {}
    for c in df.columns:
        lc = str(c).lower()
        if "param" in lc or "test" in lc or "item" in lc:
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
    # keep the core columns
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


def generate_pdf_bytes(meta, df):
    if not REPORTLAB_AVAILABLE:
        return None
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    margin = 15 * mm
    y = h - margin
    c.setFont("Helvetica-Bold", 14)
    title = f"DAR — {meta.get('project','')} / {meta.get('component','')}"
    c.drawString(margin, y, title)
    y -= 8 * mm
    c.setFont("Helvetica", 9)
    hdr = f"Part: {meta.get('part_number','')}  Reviewer: {meta.get('reviewer','')}  Date: {meta.get('date','')}"
    c.drawString(margin, y, hdr)
    y -= 8 * mm
    # table header
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


# ----------------------
# Upload UI
# ----------------------
uploaded = st.file_uploader("Upload Datasheet (PDF preferred) or CSV/XLSX", type=["pdf", "csv", "xlsx", "xls"])
if not uploaded:
    if not (PYPDF2_AVAILABLE or PDFPLUMBER_AVAILABLE):
        notify_missing_pdf_libs()
    st.stop()

file_bytes = uploaded.read()
filename = uploaded.name
st.write("Uploaded:", filename)

# If PDF and libs missing, notify user
if filename.lower().endswith(".pdf") and not (PYPDF2_AVAILABLE or PDFPLUMBER_AVAILABLE):
    st.warning("PDF uploaded but PyPDF2/pdfplumber are not available. Automatic PDF parsing will be skipped.")
    proceed = st.button("Proceed and enter parameters manually")
    if not proceed:
        st.stop()

# ----------------------
# Extraction flow
# ----------------------
extracted_df = pd.DataFrame()
if filename.lower().endswith(".pdf") and (PYPDF2_AVAILABLE or PDFPLUMBER_AVAILABLE):
    st.info("Attempting to extract Electrical Characteristics from PDF...")
    pages = find_section_pages(file_bytes) if PYPDF2_AVAILABLE else []
    st.write("Candidate pages found:", pages if pages else "none")
    df_candidate = pd.DataFrame()
    # try pdfplumber first if available and pages discovered
    if PDFPLUMBER_AVAILABLE and pages:
        with st.spinner("Trying pdfplumber table extraction..."):
            tables = extract_tables_pdfplumber(file_bytes, pages)
        if tables:
            # pick biggest table
            df_candidate = max(tables, key=lambda t: t.shape[0])
            # header normalization
            df_candidate = df_candidate.fillna("").astype(str)
            first = df_candidate.iloc[0].tolist()
            if any(re.search(r"[A-Za-z]", str(x)) for x in first):
                df_candidate.columns = first
                df_candidate = df_candidate.drop(df_candidate.index[0]).reset_index(drop=True)
    # fallback to text heuristics using PyPDF2
    if (df_candidate is None or df_candidate.empty) and PYPDF2_AVAILABLE:
        with st.spinner("Falling back to text-based extraction (PyPDF2)..."):
            txt = extract_text_pypdf2(file_bytes)
            if txt:
                # try to extract chunk near the heading first
                chunk = txt
                # try to find heading
                for kw in SECTION_KEYWORDS:
                    m = re.search(kw, txt, re.IGNORECASE)
                    if m:
                        start = max(0, m.start() - 200)
                        chunk = txt[start:start + 8000]  # window after heading
                        break
                df_candidate = heuristic_table_from_text(chunk)
    if not df_candidate.empty:
        try:
            extracted_df = map_table_to_standard(df_candidate)
        except Exception:
            extracted_df = pd.DataFrame()
else:
    # CSV/XLSX path
    try:
        if filename.lower().endswith(".csv"):
            df_in = pd.read_csv(BytesIO(file_bytes))
        else:
            df_in = pd.read_excel(BytesIO(file_bytes))
        # attempt mapping
        extracted_df = map_table_to_standard(df_in)
    except Exception as e:
        st.warning(f"Failed to parse spreadsheet automatically: {e}")
        extracted_df = pd.DataFrame()

# ----------------------
# Prepare editable DAR table
# ----------------------
st.subheader("Extracted / Suggested Parameters")
if extracted_df.empty:
    st.info("No automatic table extracted — add rows manually or paste CSV-lines.")
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

# Ensure required columns
for c in ["Parameter", "Symbol", "Condition", "Min", "Typ", "Max", "Unit", "Notes"]:
    if c not in extracted_df.columns:
        extracted_df[c] = ""

# Add editable columns
extracted_df["Measured"] = ""
extracted_df["Pass/Fail"] = ""
extracted_df["_AutoSuggestion"] = ""

# Use experimental_data_editor if present
try:
    edited = st.experimental_data_editor(extracted_df, num_rows="dynamic")
except Exception:
    st.info("Your Streamlit version may not support experimental_data_editor. Use CSV edit fallback.")
    csv_text = st.text_area("Edit parameter CSV (Parameter,Symbol,Condition,Min,Typ,Max,Unit,Notes,Measured,Pass/Fail)", extracted_df.to_csv(index=False))
    try:
        edited = pd.read_csv(BytesIO(csv_text.encode()))
    except Exception:
        st.error("Failed to parse edited CSV. Reverting to suggested table.")
        edited = extracted_df.copy()

# ----------------------
# Auto pass/fail suggestions
# ----------------------
st.subheader("Auto Pass/Fail Suggestions")
auto_list = []
for _, row in edited.iterrows():
    min_v = parse_first_number(row.get("Min", ""))
    max_v = parse_first_number(row.get("Max", ""))
    measured_v = parse_first_number(row.get("Measured", ""))
    suggestion = "Unknown"
    try:
        if measured_v is not None:
            if (min_v is not None) and (max_v is not None):
                suggestion = "Pass" if (min_v <= measured_v <= max_v) else "Fail"
            elif max_v is not None:
                suggestion = "Pass" if measured_v <= max_v else "Fail"
            elif min_v is not None:
                suggestion = "Pass" if measured_v >= min_v else "Fail"
            else:
                suggestion = "Unknown"
    except Exception:
        suggestion = "Unknown"
    auto_list.append(suggestion)
edited["_AutoSuggestion"] = auto_list

st.dataframe(edited)

# ----------------------
# DAR metadata
# ----------------------
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

# ----------------------
# Export: CSV & PDF (use st.download_button to avoid inline HTML)
# ----------------------
def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def make_pdf_for_download(meta, df):
    # returns bytes or None if reportlab missing
    return generate_pdf_bytes(meta, df) if REPORTLAB_AVAILABLE else None


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
        pdf_bytes = make_pdf_for_download(meta, edited)
        if pdf_bytes:
            st.download_button(
                label="Download DAR (PDF)",
                data=pdf_bytes,
                file_name=f"dar_{project or 'project'}_{part_number or 'part'}.pdf",
                mime="application/pdf",
            )
        else:
            st.info("PDF generation returned empty. Check reportlab installation.")
    else:
        st.info("PDF export disabled — install reportlab (`pip install reportlab`) to enable.")

with colC:
    st.download_button(
        label="Download DAR (Excel)",
        data=BytesIO(edited.assign(**meta).to_excel(index=False, engine="openpyxl") if 'openpyxl' in globals() else df_to_csv_bytes(edited.assign(**meta))),
        file_name=f"dar_{project or 'project'}_{part_number or 'part'}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.success("DAR prepared — edit values above and export using the buttons.")
if not (PYPDF2_AVAILABLE or PDFPLUMBER_AVAILABLE):
    st.info("Tip: install PyPDF2 and pdfplumber for better PDF extraction: `pip install PyPDF2 pdfplumber`")
if not REPORTLAB_AVAILABLE:
    st.info("Tip: install reportlab to enable PDF export: `pip install reportlab`")
