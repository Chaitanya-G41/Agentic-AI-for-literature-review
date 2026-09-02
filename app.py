"""
NLP-05: Agentic AI for Automated Research Paper Analysis and Literature Review
Interactive Streamlit Web Dashboard (Step 1 & Step 2 Prototype)
"""

import os
import json
import streamlit as st

# Configure Streamlit Page
st.set_page_config(
    page_title="NLP-05 Agentic AI Literature Review",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title { font-size: 26px; font-weight: bold; color: #1A365D; }
    .sub-title { font-size: 16px; color: #0D9488; font-weight: bold; margin-bottom: 20px; }
    .card-box { background-color: #F8FAFC; border-left: 4px solid #0D9488; padding: 15px; border-radius: 6px; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

# Title & Metadata Header
st.markdown('<div class="main-title">📚 NLP-05: Agentic AI for Automated Research Paper Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Department of CSE, R.V. College of Engineering | Mentors: Dr. Sagar B M, Prof. Kavi Narayana Murthy, Dr. Chethana R Murthy</div>', unsafe_allow_html=True)

# Sidebar Options
st.sidebar.title("🛠️ Project Controls")
st.sidebar.info("**Current Phase:** Step 1 (Paper Collection) & Step 2 (2-Stage PDF Ingestion Pipeline) Active")

api_key = st.sidebar.text_input("Google Gemini API Key (Optional)", type="password")
if api_key:
    os.environ["GEMINI_API_KEY"] = api_key
    st.sidebar.success("Gemini API Key Set!")

# Tabs Breakdown
tab1, tab2, tab3, tab4 = st.tabs([
    "📥 Step 1: Paper Collection", 
    "⚙️ Step 2: PDF Parsing & Schema Extraction", 
    "📊 Extracted Paper Schemas", 
    "🗺️ Ingestion Pipeline Architecture"
])

# Tab 1: Step 1 Paper Collection
with tab1:
    st.header("Step 1: Automated Paper Collection & External API Ingestion")
    st.write("Fetch benchmark research papers via **arXiv API**, **Semantic Scholar API**, or upload raw paper PDFs directly.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("1. Download Benchmark Corpus (12 Papers)")
        if st.button("🚀 Download 12 Agentic AI Papers via arXiv API"):
            with st.spinner("Fetching papers from arXiv API..."):
                from src.ingestion.downloader import download_benchmark_corpus
                download_benchmark_corpus()
                st.success("12 Benchmark Papers Downloaded Successfully into `data/pdf_corpus/`!")

    with col2:
        st.subheader("2. Direct PDF File Upload")
        uploaded_files = st.file_uploader("Upload Paper PDFs", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            os.makedirs("data/pdf_corpus", exist_ok=True)
            for file in uploaded_files:
                filepath = os.path.join("data/pdf_corpus", file.name)
                with open(filepath, "wb") as f:
                    f.write(file.getbuffer())
            st.success(f"Saved {len(uploaded_files)} custom PDF(s) into `data/pdf_corpus/`!")

    st.subheader("📁 Corpus Files Currently in Local Storage")
    corpus_dir = "data/pdf_corpus"
    if os.path.exists(corpus_dir):
        files = [f for f in os.listdir(corpus_dir) if f.endswith(".pdf")]
        if files:
            st.write(f"Total Papers Available: **{len(files)}**")
            st.dataframe({"Filename": files}, use_container_width=True)
        else:
            st.info("No PDF files found yet. Click Download or Upload PDFs above.")

# Tab 2: Step 2 PDF Parser & Schema Extractor
with tab2:
    st.header("Step 2: 2-Stage PDF Ingestion Pipeline (GROBID + Gemini Schema Mapper)")
    st.write("Converts 2-column PDFs into clean structural TEI XML/JSON chunks, pre-filters methodology/limitation candidates, and maps them into validated Pydantic `PaperSchema` JSON files.")

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("⚙️ Run Stage 1 PDF Layout Parser"):
            with st.spinner("Parsing PDF layouts into structural JSON..."):
                from src.ingestion.parser import process_pdf_directory
                process_pdf_directory()
                st.success("Stage 1 Parsing Complete! Parsed JSONs written to `data/extracted_schemas/`")

    with c2:
        if st.button("🧠 Run Stage 2 Semantic Extraction Agent"):
            with st.spinner("Extracting PaperSchemas via Gemini / Heuristics..."):
                from src.agents.extraction import process_all_parsed_json
                process_all_parsed_json()
                st.success("Stage 2 Extraction Complete! PaperSchema JSONs updated!")

# Tab 3: Extracted Schemas Preview
with tab3:
    st.header("Extracted Pydantic Paper Schemas")
    schema_dir = "data/extracted_schemas"
    if os.path.exists(schema_dir):
        schema_files = [f for f in os.listdir(schema_dir) if f.endswith("_schema.json")]
        if schema_files:
            selected_file = st.selectbox("Select Paper Schema to View:", schema_files)
            filepath = os.path.join(schema_dir, selected_file)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            st.markdown(f'<div class="card-box"><b>Paper Title:</b> {data.get("paper_title")}<br><b>Venue:</b> {data.get("venue")} | <b>Authors:</b> {", ".join(data.get("authors", []))}</div>', unsafe_allow_html=True)
            
            sc1, sc2 = st.columns(2)
            with sc1:
                st.subheader("Core Problem & Methodology")
                st.write(f"**Problem:** {data.get('core_problem')}")
                st.write(f"**Methodology:** {data.get('methodology_summary')}")
                st.write(f"**Datasets:** {', '.join(data.get('datasets_eval', []))}")

            with sc2:
                st.subheader("Results & Explicit Limitations")
                st.write(f"**Results:** {data.get('empirical_results')}")
                st.warning(f"**Limitations:** {data.get('explicit_limitations')}")

            with st.expander("View Full Raw JSON Schema"):
                st.json(data)
        else:
            st.info("No schema JSON files found. Run Step 2 Stage 1 & Stage 2 above.")
    else:
        st.info("No extracted schemas directory found.")

# Tab 4: Architecture Flowchart
with tab4:
    st.header("2-Stage Ingestion Pipeline Architecture Flowchart")
    st.markdown("""
    ```mermaid
    flowchart TD
        A[User PDF Upload / arXiv API Fetcher] --> B[Stage 1: GROBID TEI XML / PyMuPDF Layout Parser]
        B --> C[Section Boundary & Paragraph Chunk Indexer]
        C --> D[Candidate Block Keyword Pre-Filter]
        D --> E[Stage 2: Gemini 2.0 Flash / Groq LLM Schema Extractor]
        E --> F[Validated Pydantic PaperSchema JSON Output]
    ```
    """)
