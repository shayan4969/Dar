# dar_from_datasheet_improved.py
# Streamlit app: upload a datasheet (PDF/CSV/XLSX), auto-extract Electrical Characteristics table
#
# Requires:
# pip install streamlit pandas PyPDF2 openpyxl reportlab pdfplumber
#
# Run:
# streamlit run dar_from_datasheet_improved.py

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

# optional import, used for table extraction
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except Exception:
    PDFPLUMBER_AVAILABLE = False

st.set_page_config(page_title="DAR from Datasheet (improved extraction)", layout="wide")
st.title("DAR Generator — improved Electrical Characteristics extraction")
st.markdown("Upload a PDF datasheet. The app will try to find an *Electrical Characteristics* table and convert it into DAR parameter rows.")

# ---------- helpers ----------
SECTION_KEYWORDS = [
    r"Electrical Characteristics",
    r"Electrical Characteristics \(continued\)",
    r"Electrical Characteristics Table",
    r"Static Electrical Characteristics",
    r"AC Electrical Characteristics",
    r"DC Characteristics",
    r"Electrical specifications",
    r"Electrical characteristics",
]

NUM_UNIT_PATTERN = r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)(?:\s?([mMkKuUnpµμ]?)(V|A|Hz|kHz|MHz|GHz|Ω|ohm|°C|C|mW|W)?)"
# fairly permissive numeric+unit regex

def extract_text_with_pypdf2(pdf_bytes):
    text = ""
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        for p in reader.pages:
            try:
                txt = p.extract_text()
                if txt:
                    text += txt + "\n"
            except Exception:
                continue
    except Exception as e:
        st.warning(f"PyPDF2 reading error: {e}")
    return text

def find_section_pages_by_text(pdf_bytes, section_patterns=SECTION_KEYWORDS):
    """
    Return a set of page indices that contain the section heading.
    """
    pages = set()
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        for i, p in enumerate(reader.pages):
            try:
                txt = p.extract_text() or ""
            except Exception:
                txt = ""
            for pat in section_patterns:
                if re.search(pat, txt, re.IGNORECASE):
                    pages.add(i)
    except Exception:
        pass
    return sorted(list(pages))

def extract_tables_with_pdfplumber(pdf_bytes, pages_to_try):
    """
    Use pdfplumber to extract tables from the given page numbers (0-based).
    Return list of DataFrames.
    """
    tables = []
    if not PDFPLUMBER_AVAILABLE:
        return tables
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as doc:
            for p in pages_to_try:
                if p < 0 or p >= len(doc.pages):
                    continue
                page = doc.pages[p]
                # first attempt: page.extract_tables()
                tlist = page.extract_tables()
                if tlist:
                    for t in tlist:
                        # convert table (list of lists) into df (clean)
                        df = pd.DataFrame(t)
                        # drop rows that are all None
                        df = df.dropna(how="all")
                        if not df.empty:
                            df = df.reset_index(drop=True)
                            tables.append(df)
                # also try searching area below the heading by bounding boxes if needed (omitted complexity)
    except Exception as e:
        st.warning(f"pdfplumber extraction warning: {e}")
    return tables

def heuristic_line_table_parse(section_text):
    """
    Given a block of text (section), attempt to parse it into rows/columns using whitespace splitting heuristics.
    Returns a pandas DataFrame.
    """
    lines = [l for l in (section_text.splitlines()) if l.strip()]
    if not lines:
        return pd.DataFrame()

    # skip lines that look like titles or footers commonly
    # find candidate header line: look for line with words like 'Min', 'Typ', 'Max', 'Unit', 'Condition', 'Symbol'
    header_idx = None
    for i,ln in enumerate(lines[:8]):  # usually header appears early
        if re.search(r"\b(Min|Typ|Max|Unit|Condition|Symbol|Parameter|Test|Limits)\b", ln, re.IGNORECASE):
            header_idx = i
            break

    # fallback: pick first 0..2 lines as header if they contain multiple words
    if header_idx is None:
        header_idx = 0

    header_line = lines[header_idx]
    # Normalize whitespace: replace sequences of 2+ spaces/tabs by a marker
    # Many datasheets use columnar alignment with multiple spaces; use that to split
    split_marker = " ::COL:: "
    normalized = []
    for ln in lines[header_idx:]:
        # replace 2 or more spaces/tabs with marker
        nl = re.sub(r"[ \t]{2,}", split_marker, ln.strip())
        normalized.append(nl)

    # split header into columns
    header_cols = normalized[0].split(split_marker)
    header_cols = [c.strip() for c in header_cols if c.strip()]
    # ensure reasonable column names
    if not header_cols:
        header_cols = ["Column" + str(i) for i in range(6)]

    # now parse subsequent lines into columns using same marker
    rows = []
    for ln in normalized[1:]:
        parts = [p.strip() for p in ln.split(split_marker)]
        # If parts fewer than header cols, try splitting by two or more spaces fallback
        if len(parts) < len(header_cols):
            parts = [p.strip() for p in re.split(r"[ \t]{2,}", ln.strip())]
        # If still fewer, pad with empty
        if len(parts) < len(header_cols):
            parts += [""]*(len(header_cols)-len(parts))
        rows.append(parts[:len(header_cols)])

    # Build DataFrame
    df = pd.DataFrame(rows, columns=header_cols)
    # simple cleaning: drop rows that are all empty
    df = df.replace("", pd.NA).dropna(how="all").reset_index(drop=True)
    return df

def extract_section_text_near_pages(pdf_bytes, pages, context_pages=1):
    """
    Collect text from pages around pages list (within context_pages)
    """
    text_blocks = []
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        n = len(reader.pages)
        for p in pages:
            for q in range(max(0, p-context_pages), min(n, p+context_pages+1)):
                try:
                    txt = reader.pages[q].extract_text() or ""
                except Exception:
                    txt = ""
                if txt:
                    text_blocks.append(txt)
    except Exception:
        pass
    combined = "\n\n".join(text_blocks)
    return combined

def map_extracted_df_to_standard(df):
    """
    Try to rename columns into: Parameter, Symbol, Condition, Min, Typ, Max, Unit, Notes
    Returns a cleaned DataFrame with those columns where available.
    """
    col_map = {}
    lower_cols = [c.lower() for c in df.columns.astype(str)]
    for i, c in enumerate(df.columns):
        lc = str(c).lower()
        if any(k in lc for k in ["parameter", "param", "item", "test"]):
            col_map[c] = "Parameter"
        elif any(k in lc for k in ["symbol", "sig", "abbrev"]):
            col_map[c] = "Symbol"
        elif any(k in lc for k in ["condition", "conditions", "test condition"]):
            col_map[c] = "Condition"
        elif re.search(r"\bmin\b", lc):
            col_map[c] = "Min"
        elif re.search(r"\btyp\b", lc):
            col_map[c] = "Typ"
        elif re.search(r"\bmax\b", lc):
            col_map[c] = "Max"
        elif any(k in lc for k in ["unit","units"]):
            col_map[c] = "Unit"
        elif any(k in lc for k in ["note","notes","remark","comment"]):
            col_map[c] = "Notes"
        else:
            # if header looks like numeric column (contains digits or %), consider Min/Typ/Max heuristics
            if re.search(r"\b(min|typ|max)\b", lc):
                # map generically
                pass
    # Apply mapping where possible
    df2 = df.copy()
    # rename mapped columns
    if col_map:
        df2 = df2.rename(columns=col_map)
    # Ensure presence of core columns
    for c in ["Parameter","Symbol","Condition","Min","Typ","Max","Unit","Notes"]:
        if c not in df2.columns:
            df2[c] = ""
    # Reorder
    df2 = df2[["Parameter","Symbol","Condition","Min","Typ","Max","Unit","Notes"]]
    return df2

# numeric parsing helpers
def parse_numeric_with_unit(s):
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    s = s.strip()
    if not s:
        return (None, None)
    # common patterns: "-40", "-40 to 85", "3.0", "1.2 V", "0.8/1.2"
    # take first numeric occurrence and unit if present
    m = re.search(NUM_UNIT_PATTERN, s)
    if m:
        val = m.group(1)
        unit_prefix = m.group(2) or ""
        unit_type = m.group(3) or ""
        # normalize micro symbol
        unit_prefix = unit_prefix.replace("µ","u").replace("μ","u")
        unit = (unit_prefix + unit_type).strip()
        try:
            f = float(val)
            # apply prefix multipliers
            if unit.startswith("m") and unit.endswith("V"):  # mV
                f = f * 1e-3
            # this is simplistic — for display we keep unit separately
        except:
            f = None
        return f, unit
    # try to parse ranges like "-40 to 85"
    m2 = re.search(r"(-?\d+\.?\d*)\s*(to|-)\s*(-?\d+\.?\d*)", s)
    if m2:
        try:
            f1 = float(m2.group(1))
            f2 = float(m2.group(3))
            return (f1 + f2)/2.0, None
        except:
            pass
    return None, None

def auto_pass_fail(spec_min, spec_max, measured_value, tol_pct=2.0):
    """
    Basic pass/fail:
    - If spec_min and spec_max both present: pass if spec_min <= measured <= spec_max
    - If only spec_max present: pass if measured <= spec_max + tol
    - If only spec_min present: pass if measured >= spec_min - tol
    tol_pct: if no explicit range, allow +/- tol_pct %
    Returns "Pass"/"Fail"/"Unknown"
    """
    try:
        if measured_value is None:
            return "Unknown"
        if spec_min is not None and spec_max is not None:
            return "Pass" if (spec_min - 1e-12) <= measured_value <= (spec_max + 1e-12) else "Fail"
        if spec_max is not None:
            # small tolerance as percentage of spec_max
            tol = abs(spec_max) * tol_pct/100.0
            return "Pass" if measured_value <= spec_max + tol else "Fail"
        if spec_min is not None:
            tol = abs(spec_min) * tol_pct/100.0
            return "Pass" if measured_value >= spec_min - tol else "Fail"
        # no numerical spec, unknown
        return "Unknown"
    except Exception:
        return "Unknown"

# ---------- UI: upload ----------
uploaded = st.file_uploader("Upload datasheet (PDF recommended) or spreadsheet (CSV/XLSX)", type=["pdf","csv","xls","xlsx"])
if not uploaded:
    st.info("Upload a datasheet to start. I'll try to find the Electrical Characteristics table.")
    st.stop()

file_bytes = uploaded.read()
filename = uploaded.name
st.write("Uploaded:", filename, f"— {len(file_bytes):,} bytes")

# ---------- Main extraction flow ----------
extracted_param_rows = []

# If PDF: try to find section pages and extract tables
if filename.lower().endswith(".pdf"):
    st.info("Processing PDF...")
    # 1) find pages that mention Electrical Characteristics
    pages_with_section = find_section_pages_by_text(file_bytes)
    st.write("Detected section on pages:", pages_with_section or "none detected by simple text search")
    # Also include next page (tables often spill)
    pages_to_try = pages_with_section.copy()
    for p in pages_with_section:
        pages_to_try += [p+1, p-1]
    pages_to_try = sorted(set([p for p in pages_to_try if p >= 0]))
    # 2) Try pdfplumber table extraction
    pdf_tables = []
    if PDFPLUMBER_AVAILABLE and pages_to_try:
        with st.spinner("Running pdfplumber table extraction on candidate pages..."):
            pdf_tables = extract_tables_with_pdfplumber(file_bytes, pages_to_try)
        st.write(f"pdfplumber found {len(pdf_tables)} candidate table(s).")
    else:
        if not PDFPLUMBER_AVAILABLE:
            st.info("pdfplumber not installed — falling back to heuristic extraction. Install pdfplumber for better results.")

    # 3) If pdfplumber returned tables, attempt to map them
    used_table = None
    if pdf_tables:
        # pick the table with most columns / reasonable shape
        best = None
        best_score = -1
        for t in pdf_tables:
            r,c = t.shape
            score = c * r
            if score > best_score:
                best_score = score
                best = t
        used_table = best
        st.success("Using table extracted by pdfplumber.")
        # cleanup: convert None to empty strings, fill first row if headerless
        used_table = used_table.fillna("").astype(str)
        # If the first row looks like headers (contains words), use them
        firstrow = used_table.iloc[0].tolist()
        if any(re.search(r"[A-Za-z]", str(x)) for x in firstrow) and not all(re.match(r"^-?\d", str(x).strip()) for x in firstrow):
            # treat as header
            df_table = used_table.copy()
            df_table.columns = df_table.iloc[0].tolist()
            df_table = df_table.drop(df_table.index[0]).reset_index(drop=True)
        else:
            # no header; create generic ones
            df_table = used_table.copy()
            df_table.columns = [f"Col{i}" for i in range(1, df_table.shape[1]+1)]
    else:
        # 4) Fallback heuristic: extract text around pages and parse lines
        st.info("Falling back to text-based heuristic parsing of the section.")
        surrounding_text = extract_section_text_near_pages(file_bytes, pages_with_section or [0], context_pages=1)
        # Try to find the block starting at the heading
        found_block = ""
        for pat in SECTION_KEYWORDS:
            m = re.search(pat + r".{0,200}", surrounding_text, re.IGNORECASE | re.DOTALL)
            if m:
                # start index
                idx = m.start()
                found_block = surrounding_text[idx:]
                break
        if not found_block:
            # fallback: use whole surrounding_text
            found_block = surrounding_text or extract_text_with_pypdf2(file_bytes)
        # cut block until next heading (heuristic: blank line followed by capitalized heading)
        # find next occurrence of two newlines + capitalized word
        next_sec = re.search(r"\n\s*\n[A-Z][A-Za-z0-9 \-]{3,40}\n", found_block)
        if next_sec:
            found_block = found_block[:next_sec.start()]
        # Now parse table-like lines
        df_table = heuristic_line_table_parse(found_block)
        if df_table.empty:
            st.warning("Heuristic parsing didn't find a clean table. You can still manually add parameters or paste CSV-lines.")
        else:
            st.success("Heuristic table parse produced a candidate table.")

    # If df_table exists, normalize
    if 'df_table' in locals() and not df_table.empty:
        # Map columns to standard names
        df_mapped = map_extracted_df_to_standard(df_table)
        # Trim whitespace
        df_mapped = df_mapped.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        # Convert rows into parameter suggestions
        for _, r in df_mapped.iterrows():
            param = r.get("Parameter") or r.get("Symbol") or ""
            # combine min/typ/max into Spec text
            spec_parts = []
            if str(r.get("Min")).strip():
                spec_parts.append("Min: " + str(r.get("Min")))
            if str(r.get("Typ")).strip():
                spec_parts.append("Typ: " + str(r.get("Typ")))
            if str(r.get("Max")).strip():
                spec_parts.append("Max: " + str(r.get("Max")))
            spec_text = "; ".join(spec_parts).strip()
            unit = r.get("Unit","")
            comments = str(r.get("Notes","")) + (" | " + str(r.get("Condition","")) if r.get("Condition") else "")
            extracted_param_rows.append({
                "Parameter": param,
                "Spec": spec_text,
                "Measured": "",
                "Unit": unit,
                "Pass/Fail": "",
                "Comments": comments
            })
    else:
        st.info("No table automatically extracted. You can still edit / add parameters below.")

# If spreadsheet format: attempt to map columns (reuse previous simpler heuristics)
else:
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(file_bytes))
        else:
            df = pd.read_excel(BytesIO(file_bytes))
        st.subheader("Spreadsheet preview")
        st.dataframe(df.head(50))
        # try map
        lower_cols = [c.lower() for c in df.columns.astype(str)]
        def find_col(keywords):
            for k in keywords:
                for c in df.columns:
                    if k in str(c).lower():
                        return c
            return None
        col_param = find_col(["param","parameter","item","test"])
        col_min = find_col(["min"])
        col_typ = find_col(["typ"])
        col_max = find_col(["max"])
        col_unit = find_col(["unit"])
        col_cond = find_col(["condition","cond"])
        col_note = find_col(["note","notes","comment"])
        # build rows
        if col_param:
            for _, rr in df.iterrows():
                spec_parts = []
                if col_min and pd.notna(rr.get(col_min,'')): spec_parts.append("Min: "+str(rr.get(col_min)))
                if col_typ and pd.notna(rr.get(col_typ,'')): spec_parts.append("Typ: "+str(rr.get(col_typ)))
                if col_max and pd.notna(rr.get(col_max,'')): spec_parts.append("Max: "+str(rr.get(col_max)))
                spec_text = "; ".join(spec_parts)
                extracted_param_rows.append({
                    "Parameter": rr.get(col_param,""),
                    "Spec": spec_text,
                    "Measured": "",
                    "Unit": rr.get(col_unit,"") if col_unit else "",
                    "Pass/Fail": "",
                    "Comments": str(rr.get(col_note,"")) + (" | " + str(rr.get(col_cond,"")) if col_cond else "")
                })
    except Exception as e:
        st.warning(f"Spreadsheet parse warning: {e}")

# ---------- Present extracted parameter table for editing ----------
st.subheader("Extracted / Suggested Parameters (editable)")
if extracted_param_rows:
    df_suggested = pd.DataFrame(extracted_param_rows)
else:
    df_suggested = pd.DataFrame([
        {"Parameter":"VCC","Spec":"Typ: 3.3 V; Max: 3.6 V","Measured":"","Unit":"V","Pass/Fail":"","Comments":"Example fallback"},
        {"Parameter":"IDD","Spec":"Typ: 120 mA","Measured":"","Unit":"mA","Pass/Fail":"","Comments":""}
    ])

# Let user edit via experimental_data_editor (if available)
try:
    df_edited = st.experimental_data_editor(df_suggested, num_rows="dynamic")
except Exception:
    st.info("interactive table not supported — using CSV text edit fallback")
    csv_text = st.text_area("Edit parameter rows as CSV (Parameter,Spec,Measured,Unit,Pass/Fail,Comments)", df_suggested.to_csv(index=False))
    try:
        df_edited = pd.read_csv(BytesIO(csv_text.encode()))
    except Exception:
        st.error("Failed to parse CSV; using original suggestions.")
        df_edited = df_suggested.copy()

# Attempt auto pass/fail suggestions for numeric specs when user provides Measured
st.subheader("Auto pass/fail (optional)")
st.write("If you enter numeric values in the 'Measured' column, the app will attempt to determine Pass/Fail using Min/Max from Spec.")

# Parse spec min/typ/max into numeric for each row
parsed_specs = []
for idx, row in df_edited.iterrows():
    spec_text = str(row.get("Spec",""))
    # try to find Min, Typ, Max numbers in spec_text
    min_val, typ_val, max_val = (None, None, None)
    # search for Min: ... , Typ: ... , Max: ...
    m_min = re.search(r"[Mm]in[:\s]*([^\;,\n]+)", spec_text)
    m_typ = re.search(r"[Tt]yp[:\s]*([^\;,\n]+)", spec_text)
    m_max = re.search(r"[Mm]ax[:\s]*([^\;,\n]+)", spec_text)
    if m_min:
        v,u = parse_numeric_with_unit(m_min.group(1))
        min_val = v
    if m_typ:
        v,u = parse_numeric_with_unit(m_typ.group(1))
        typ_val = v
    if m_max:
        v,u = parse_numeric_with_unit(m_max.group(1))
        max_val = v
    # fallback: if spec contains a single number with unit
    if min_val is None and max_val is None:
        v,u = parse_numeric_with_unit(spec_text)
        if v is not None:
            # treat as typ
            typ_val = v

    # parse measured column
    measured_raw = row.get("Measured","")
    measured_num, _ = parse_numeric_with_unit(str(measured_raw))
    suggestion = auto_pass_fail(min_val, max_val, measured_num, tol_pct=2.0)
    parsed_specs.append({
        "min": min_val,
        "typ": typ_val,
        "max": max_val,
        "measured": measured_num,
        "suggestion": suggestion
    })

# Add suggestions to displayed df (non-destructive: show in a separate column)
df_display = df_edited.copy()
df_display["_AutoSuggestion"] = [p["suggestion"] for p in parsed_specs]

st.dataframe(df_display)

# ---------- DAR metadata ----------
st.subheader("DAR Metadata")
col1, col2, col3 = st.columns(3)
with col1:
    project = st.text_input("Project Name", "Project-X")
    component = st.text_input("Component / Part", "")
    part_number = st.text_input("Part Number", "")
    datasheet_ref = st.text_input("Datasheet Reference (filename or URL)", filename)
with col2:
    reviewer = st.text_input("Reviewer", "")
    date = st.date_input("Date", datetime.today())
    priority = st.selectbox("Priority", ["Low","Medium","High","Critical"])
with col3:
    status = st.selectbox("Status", ["Open","In Progress","Resolved","Closed"])
    overall_result = st.selectbox("Overall Result", ["Conditional","Pass","Fail"])

# ---------- Export helpers ----------
def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode()

def generate_simple_pdf(entry_meta, param_df):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    margin = 15 * mm
    y = h - margin

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, f"DAR — {entry_meta.get('project','')} / {entry_meta.get('component','')}")
    y -= 8 * mm
    c.setFont("Helvetica", 9)
    c.drawString(margin, y, f"Part: {entry_meta.get('part_number','')}    Reviewer: {entry_meta.get('reviewer','')}    Date: {entry_meta.get('date','')}")
    y -= 8 * mm

    # table header
    c.setFont("Helvetica-Bold", 10)
    cols_x = [margin, margin + 60*mm, margin + 105*mm, margin + 135*mm]
    headers = ["Parameter","Spec","Measured","Pass/Fail"]
    for i, htext in enumerate(headers):
        c.drawString(cols_x[i], y, htext)
    y -= 6 * mm
    c.setFont("Helvetica", 9)

    for idx, r in param_df.iterrows():
        if y < 30 * mm:
            c.showPage()
            y = h - margin
        c.drawString(cols_x[0], y, str(r.get("Parameter",""))[:40])
        c.drawString(cols_x[1], y, str(r.get("Spec",""))[:40])
        c.drawString(cols_x[2], y, str(r.get("Measured",""))[:18])
        c.drawString(cols_x[3], y, str(r.get("Pass/Fail",""))[:10])
        y -= 6 * mm

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()

# ---------- action buttons ----------
colA, colB, colC = st.columns([1,1,1])
with colA:
    if st.button("Download DAR as CSV"):
        meta = {
            "project": project,
            "component": component,
            "part_number": part_number,
            "datasheet": datasheet_ref,
            "reviewer": reviewer,
            "date": date.isoformat(),
            "priority": priority,
            "status": status,
            "overall_result": overall_result
        }
        df_export = df_edited.copy()
        # add auto suggestion and metadata
        df_export["_AutoSuggestion"] = df_display["_AutoSuggestion"]
        for k,v in meta.items():
            df_export[k] = v
        csv_bytes = df_to_csv_bytes(df_export)
        b64 = base64.b64encode(csv_bytes).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="dar_{project or \"project\"}_{part_number or \"part\"}.csv">Download DAR CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

with colB:
    if st.button("Generate & Download PDF"):
        meta = {
            "project": project,
            "component": component,
            "part_number": part_number,
            "datasheet": datasheet_ref,
            "reviewer": reviewer,
            "date": date.isoformat(),
            "priority": priority,
            "status": status,
            "overall_result": overall_result
        }
        # include auto suggestion column into PDF table
        pdf_bytes = generate_simple_pdf(meta, df_edited.assign(**{"Pass/Fail": df_display["_AutoSuggestion"]}))
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="dar_{project or \"project\"}_{part_number or \"part\"}.pdf">Download DAR PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

with colC:
    if st.button("Copy DAR CSV to clipboard (for quick paste)"):
        df_export = df_edited.copy()
        df_export["_AutoSuggestion"] = df_display["_AutoSuggestion"]
        st.text_area("DAR CSV (copy this)", df_export.to_csv(index=False), height=250)

st.markdown("---")
st.success("Preview below. Make edits above and re-download when ready.")
st.dataframe(df_display)

st.markdown("## Notes & next improvements")
st.write("""
- OCR support with Tesseract would allow scanned datasheets to be parsed.
- pdfplumber yields best table extraction for text-based PDFs — install it for best results.
- Further improvement: use ML/NLP (table structure models) to map multi-line cells and merged cells reliably.
- We can add 'smart' tolerances per-parameter (user-provided) rather than global %.
- Want me to save DARs to SQLite and keep history? I can add that next.
""")
