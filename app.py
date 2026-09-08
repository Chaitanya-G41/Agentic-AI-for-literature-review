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
st.sidebar.info("**Current Phase:** Phase 2 Step 3 (Grounded Summarizer Agent & Global Memory Store) Active")

api_key = st.sidebar.text_input("Google Gemini API Key(s) (Optional, comma-separated)", type="password", key="gemini_api_key_input")
cleaned_api_key = api_key.strip().strip('"').strip("'") if api_key else ""
if cleaned_api_key:
    st.session_state["gemini_api_keys"] = cleaned_api_key
    os.environ["GEMINI_API_KEYS"] = cleaned_api_key
    os.environ["GEMINI_API_KEY"] = cleaned_api_key.split(",")[0].strip()
    keys_cnt = len([k for k in cleaned_api_key.split(",") if k.strip()])
    st.sidebar.success(f"Gemini API Key Pool Configured! ({keys_cnt} key(s))")
else:
    try:
        from dotenv import load_dotenv, find_dotenv
        env_file = find_dotenv(usecwd=True)
        if env_file:
            load_dotenv(env_file, override=True)
    except Exception:
        pass

    env_keys = os.environ.get("GEMINI_API_KEYS", "") or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    if env_keys.strip():
        st.session_state["gemini_api_keys"] = env_keys.strip()
        st.sidebar.success(f"Gemini API Key loaded from .env file! (...{env_keys.strip()[-4:]})")
    else:
        st.sidebar.warning("⚠️ No API Key detected in `.env` or sidebar input.\n\nAdd `GEMINI_API_KEY=your_key` in `.env` or paste it above to run LLM summarization!")

# Tabs Breakdown
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📥 Step 1: Paper Collection", 
    "⚙️ Step 2: PDF Parsing & Schema Extraction", 
    "📊 Extracted Paper Schemas", 
    "📝 Step 3: Summarizer Agent & Global Memory",
    "🗺️ Pipeline Architecture Flowchart"
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
                ui_api_keys = st.session_state.get("gemini_api_keys", None)
                from src.agents.extraction import process_all_parsed_json
                process_all_parsed_json(api_keys=ui_api_keys)
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

# Tab 4: Step 3 Grounded Summarizer Agent & Global Memory Store
with tab4:
    st.header("Step 3: Grounded Summarizer Agent & Global Memory Store")
    st.write("Generates multi-pillar grounded summaries (~600–1000 words) anchored in source paper text, registered into Global Memory Store for downstream agents.")

    schema_dir = "data/extracted_schemas"
    available_schemas = [f for f in os.listdir(schema_dir) if f.endswith("_schema.json")] if os.path.exists(schema_dir) else []
    paper_id_map = {f.replace("_schema.json", ""): f for f in available_schemas}

    st.subheader("🎯 Selective Summarization Controls (Save API Quota)")
    selected_pids = st.multiselect(
        "Select Specific Paper(s) to Summarize:",
        options=list(paper_id_map.keys()),
        default=[list(paper_id_map.keys())[0]] if paper_id_map else [],
        help="Select 1 or 2 papers to test without exhausting API keys!"
    )

    sm_col1, sm_col2, sm_col3 = st.columns([1.2, 1.2, 1])
    with sm_col1:
        if st.button("🎯 Summarize Selected Paper(s) Only"):
            if not selected_pids:
                st.warning("Please select at least one paper above.")
            else:
                ui_api_keys = st.session_state.get("gemini_api_keys", None)
                if not ui_api_keys:
                    st.warning("⚠️ No API key entered in sidebar! Summaries will use Heuristic Engine only.")
                with st.spinner(f"Generating grounded summary for {len(selected_pids)} selected paper(s)..."):
                    from src.agents.summarizer import process_all_schemas
                    process_all_schemas(target_paper_ids=selected_pids, api_keys=ui_api_keys)
                    st.success(f"Summarizer Agent Complete for {len(selected_pids)} paper(s)!")

    with sm_col2:
        if st.button("🚀 Summarize ALL Papers in Corpus"):
            ui_api_keys = st.session_state.get("gemini_api_keys", None)
            if not ui_api_keys:
                st.warning("⚠️ No API key entered in sidebar! Summaries will use Heuristic Engine only.")
            with st.spinner("Generating grounded summaries for all papers..."):
                from src.agents.summarizer import process_all_schemas
                process_all_schemas(api_keys=ui_api_keys)
                st.success("Summarizer Agent Finished for all papers!")

    with sm_col3:
        from src.agents.memory_store import get_global_memory_store
        mem_store = get_global_memory_store()
        all_ids = mem_store.get_all_paper_ids()
        st.metric("Papers in Memory Store", len(all_ids))

    summary_dir = "data/summaries"
    if os.path.exists(summary_dir):
        sum_files = [f for f in os.listdir(summary_dir) if f.endswith("_summary.json")]
        if sum_files:
            st.subheader("📄 Grounded Multi-Pillar Summary Inspector")
            selected_sum_file = st.selectbox("Select Paper Summary:", sum_files)
            sum_filepath = os.path.join(summary_dir, selected_sum_file)
            with open(sum_filepath, "r", encoding="utf-8") as f:
                sum_data = json.load(f)

            src_tag = sum_data.get("summary_source", "Unknown Engine")
            if "LLM" in src_tag or "Gemini" in src_tag:
                st.success(f"🤖 **Generated by LLM Engine**: {src_tag}")
            else:
                st.warning(f"⚡ **Generated by Heuristic Engine Fallback**: {src_tag}")

            st.markdown(f'<div class="card-box"><b>Paper Title:</b> {sum_data.get("paper_title")}<br><b>Citation Anchor:</b> <code>{sum_data.get("grounded_citation")}</code> | <b>Venue:</b> {sum_data.get("venue")}</div>', unsafe_allow_html=True)

            m_tab1, m_tab2 = st.tabs(["📖 Rendered Markdown Summary", "🧩 Reusable Structured Components (For Downstream Agents)"])
            with m_tab1:
                st.markdown(sum_data.get("markdown_full_summary", "No summary available."))
            with m_tab2:
                st.json(sum_data)
        else:
            st.info("No summary JSON files found. Click 'Run Summarizer Agent' above.")
    else:
        st.info("No summaries directory found.")

# Tab 5: Architecture Flowchart
with tab5:
    st.header("Phase 2 Multi-Agent Architecture Flowchart")
    st.markdown("""
    ```mermaid
    flowchart TD
        A[Paper PDFs] --> B[Stage 1: Layout & Chunk Parser]
        B --> C[Stage 2: Semantic Extraction Agent]
        C --> D[PaperSchema JSON]
        D --> E[Step 3: Grounded Summarizer Agent]
        E --> F[Global LangGraph Memory Store]
        F --> G[Comparative Analysis Agent]
        F --> H[Research Gap Agent]
        F --> I[Report Generation Agent]
    ```
    """)

