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
st.sidebar.info("**Current Phase:** Phase 2 Step 4 (Comparative Analysis Agent & Multi-Paper Synthesis) Active")

# Load environment variables early
try:
    from dotenv import load_dotenv, find_dotenv
    env_file = find_dotenv(usecwd=True)
    if env_file:
        load_dotenv(env_file, override=True)
except Exception:
    pass

# 1. NVIDIA API Key Input (Primary)
nv_key_input = st.sidebar.text_input("NVIDIA API Key (Primary - build.nvidia.com)", type="password", key="nvidia_api_key_input")
cleaned_nv_key = nv_key_input.strip().strip('"').strip("'") if nv_key_input else ""
if cleaned_nv_key:
    os.environ["NVIDIA_API_KEY"] = cleaned_nv_key
    st.sidebar.success(f"NVIDIA NIM Active! (...{cleaned_nv_key[-4:]})")
else:
    env_nv = os.environ.get("NVIDIA_API_KEY", "") or os.environ.get("NVAPI_KEY", "")
    if env_nv.strip():
        st.sidebar.success(f"NVIDIA API Key loaded from .env! (...{env_nv.strip()[-4:]})")
    else:
        st.sidebar.warning("⚠️ No NVIDIA API Key detected in `.env` or sidebar.")

# 2. Gemini API Key Input (Secondary Fallback)
with st.sidebar.expander("Google Gemini Fallback Key (Optional)"):
    api_key = st.text_input("Gemini API Key", type="password", key="gemini_api_key_input")
    cleaned_api_key = api_key.strip().strip('"').strip("'") if api_key else ""
    if cleaned_api_key:
        st.session_state["gemini_api_keys"] = cleaned_api_key
        os.environ["GEMINI_API_KEY"] = cleaned_api_key
        st.success(f"Gemini Fallback Configured! (...{cleaned_api_key[-4:]})")
    else:
        env_gemini = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        if env_gemini.strip():
            st.info(f"Gemini Key loaded from .env (...{env_gemini.strip()[-4:]})")

# Tabs Breakdown
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📥 Step 1: Paper Collection", 
    "⚙️ Step 2: PDF Parsing & Schema Extraction", 
    "📊 Extracted Paper Schemas", 
    "📝 Step 3: Summarizer Agent & Global Memory",
    "🔬 Step 4: Comparative Analysis Agent",
    "🗺️ Pipeline Architecture Flowchart"
])

# Tab 1: Step 1 Paper Collection & Relevance Screening
with tab1:
    st.header("Step 1: Research Topic Ingestion & Relevance Screening Agent")
    st.write(
        "Define your research project title/topic to autonomously discover, rank, and download "
        "the most relevant and recent research papers from arXiv via **Relevance Screening Agent**."
    )

    # 1. Topic Definition & Screening Controls
    st.subheader("🎯 Research Project Title / Topic Query")
    t_col1, t_col2 = st.columns([3, 1])
    with t_col1:
        project_title = st.text_input(
            "Enter your Research Topic / Project Title:",
            value=st.session_state.get("project_title", "Agentic AI with Multi-Agent Systems and Tool Use"),
            placeholder="e.g. Multi-Modal Vision-Language Agents for Autonomous Web Navigation",
            help="The Relevance Screening Agent calculates semantic similarity between this title and candidate papers."
        )
    with t_col2:
        top_k = st.number_input("Target Papers (Top X):", min_value=1, max_value=20, value=6, step=1)

    s_col1, s_col2 = st.columns([1, 1])
    with s_col1:
        recency_pref = st.slider(
            "Recency Preference Bonus:",
            min_value=0.0,
            max_value=0.35,
            value=0.20,
            step=0.05,
            help="Higher value gives newer (2024-2026) papers a boost in ranking alongside semantic similarity."
        )
    with s_col2:
        max_search = st.slider(
            "Candidates to Screen from arXiv:",
            min_value=10,
            max_value=40,
            value=25,
            step=5,
            help="Number of initial candidates fetched from arXiv before vector screening."
        )

    # Screening Action Button
    if st.button("🔍 Run Relevance Screening Agent", type="primary"):
        if not project_title.strip():
            st.warning("Please enter a research topic or project title.")
        else:
            with st.spinner(f"Querying arXiv and screening candidates against '{project_title}'..."):
                from src.ingestion.screening_agent import RelevanceScreeningAgent
                screener = RelevanceScreeningAgent(recency_weight=recency_pref)
                candidates = screener.search_arxiv(project_title, max_candidates=max_search)
                if not candidates:
                    st.error("No papers found on arXiv matching your query. Try broadening your keywords.")
                else:
                    ranked = screener.score_and_rank(project_title, candidates)
                    st.session_state["screened_candidates"] = ranked
                    st.session_state["project_title"] = project_title
                    st.success(f"Screened and ranked {len(ranked)} candidate papers! Review the top matches below.")

    # 2. Display Candidate Screening Results Table
    if "screened_candidates" in st.session_state and st.session_state["screened_candidates"]:
        ranked_papers = st.session_state["screened_candidates"]
        st.markdown("---")
        st.subheader("📋 Screened Papers Ranked by Relevance & Recency")
        st.info("The top papers are pre-selected based on your Target Count. Adjust selections as needed and click Download.")

        # Selection checkboxes using a multi-select or data_editor
        default_selected_ids = [p["arxiv_id"] for p in ranked_papers[:top_k]]
        
        selected_paper_ids = st.multiselect(
            "Select Papers to Download into Corpus:",
            options=[p["arxiv_id"] for p in ranked_papers],
            default=default_selected_ids,
            format_func=lambda x: next((f"[{p['relevance_score']}% match, {p['published_year']}] {p['title'][:80]}" for p in ranked_papers if p["arxiv_id"] == x), x)
        )

        # Show detailed table
        table_rows = []
        for p in ranked_papers:
            table_rows.append({
                "Selected": "✅ Yes" if p["arxiv_id"] in selected_paper_ids else "No",
                "Overall Score": f"{p['relevance_score']}%",
                "Semantic Match": f"{p['semantic_score']}%",
                "Year": p["published_year"],
                "Paper Title": p["title"],
                "Authors": ", ".join(p["authors"][:3]) + ("..." if len(p["authors"]) > 3 else ""),
                "arXiv ID": p["arxiv_id"],
                "Published": p["published_date"]
            })
        st.dataframe(table_rows, width="stretch")

        with st.expander("📖 Inspect Abstracts of Screened Papers"):
            for p in ranked_papers:
                st.markdown(f"**[{p['relevance_score']}% Match | {p['published_year']}] {p['title']}** (`{p['arxiv_id']}`)")
                st.write(p["abstract"])
                st.markdown("---")

        d_col1, d_col2 = st.columns([1.5, 1])
        with d_col1:
            if st.button(f"📥 Download & Ingest Selected ({len(selected_paper_ids)} Papers)", type="primary"):
                if not selected_paper_ids:
                    st.warning("Please select at least one paper to download.")
                else:
                    papers_to_download = [p for p in ranked_papers if p["arxiv_id"] in selected_paper_ids]
                    with st.spinner(f"Downloading {len(papers_to_download)} selected PDF(s) into data/pdf_corpus/..."):
                        from src.ingestion.screening_agent import RelevanceScreeningAgent
                        screener = RelevanceScreeningAgent()
                        saved = screener.download_papers(papers_to_download)
                        st.success(f"Successfully downloaded {len(saved)} paper(s) into `data/pdf_corpus/`!")

        with d_col2:
            if st.button("🧹 Clear Corpus Directory"):
                corpus_dir = "data/pdf_corpus"
                if os.path.exists(corpus_dir):
                    for f in os.listdir(corpus_dir):
                        if f.endswith(".pdf"):
                            try:
                                os.remove(os.path.join(corpus_dir, f))
                            except Exception:
                                pass
                    st.info("Cleared `data/pdf_corpus/`. Ready for fresh topic ingestion!")

    # 3. Existing Local Corpus Preview & Benchmark Ingestion
    st.markdown("---")
    c_col1, c_col2 = st.columns([1, 1])
    with c_col1:
        st.subheader("📚 Benchmark Corpus Fallback")
        if st.button("🚀 Download 12 Default Agentic AI Benchmark Papers"):
            with st.spinner("Fetching benchmark papers..."):
                from src.ingestion.downloader import download_benchmark_corpus
                download_benchmark_corpus()
                st.success("12 Benchmark Papers Downloaded Successfully into `data/pdf_corpus/`!")

    with c_col2:
        st.subheader("📤 Manual PDF Upload")
        uploaded_files = st.file_uploader("Upload Custom PDFs", type=["pdf"], accept_multiple_files=True)
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
            st.write(f"Total Papers Available in `data/pdf_corpus/`: **{len(files)}**")
            st.dataframe({"Filename": files}, width="stretch")
        else:
            st.info("No PDF files found yet in `data/pdf_corpus/`.")

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

# Tab 5: Step 4 Comparative Analysis Agent
with tab5:
    st.header("Step 4: Comparative Analysis Agent & Cross-Paper Synthesis")
    st.write("Generates structured cross-paper comparison matrices, benchmark dataset overlap analysis, paradigm classification, and limitation cluster synthesis from grounded summaries in Global Memory.")

    from src.agents.memory_store import get_global_memory_store
    mem_store = get_global_memory_store()
    all_summaries = mem_store.get_all_summaries()

    num_summaries = len(all_summaries)

    if num_summaries == 0:
        st.warning("⚠️ No paper summaries found in Global Memory. Please run **Step 3: Grounded Summarizer Agent** in the previous tab first.")
    elif num_summaries == 1:
        st.info("ℹ️ Exactly 1 paper summary currently available in Global Memory. The Comparative Analysis Agent will generate a single-paper baseline matrix. To see multi-paper cross-comparison and dataset overlap, summarize additional papers in Step 3.")
    else:
        st.success(f"✅ **{num_summaries} Paper Summaries Available** in Global Memory for cross-paper comparative analysis.")

    if num_summaries > 0:
        st.subheader("🎯 Comparative Analysis Selection")
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            selected_comp_ids = st.multiselect(
                "Select Summarized Paper(s) to Compare:",
                options=list(all_summaries.keys()),
                default=list(all_summaries.keys()),
                help="Select 2 or more papers to generate cross-paper matrix, or select 1 for single-paper baseline."
            )
        with col_c2:
            st.write("")
            st.write("")
            run_comp_btn = st.button("🔬 Run Comparative Analysis Agent", type="primary", width="stretch")

        if run_comp_btn:
            if not selected_comp_ids:
                st.warning("Please select at least one paper to analyze.")
            else:
                ui_api_keys = st.session_state.get("gemini_api_keys", None)
                with st.spinner(f"Synthesizing comparative analysis across {len(selected_comp_ids)} paper(s)..."):
                    from src.agents.comparative import run_comparative_analysis
                    comp_result = run_comparative_analysis(target_paper_ids=selected_comp_ids, api_keys=ui_api_keys)
                    st.success(f"Comparative Analysis Complete for {len(selected_comp_ids)} paper(s)!")

    # Check for existing analysis data in memory or disk
    comp_data = mem_store.get_comparative_analysis()
    if not comp_data and os.path.exists("data/comparative_analysis.json"):
        try:
            with open("data/comparative_analysis.json", "r", encoding="utf-8") as cf:
                comp_data = json.load(cf)
        except Exception:
            pass

    if comp_data:
        st.subheader("📑 Cross-Paper Comparative Analysis Dashboard")
        src_engine = comp_data.get("analysis_source", "Unknown Engine")
        if "LLM" in src_engine or "Gemini" in src_engine:
            st.success(f"🤖 **Generated by LLM Engine**: {src_engine}")
        else:
            st.info(f"⚡ **Generated by Grounded Heuristic Engine**: {src_engine}")

        c_tab1, c_tab2, c_tab3, c_tab4, c_tab5 = st.tabs([
            "📊 5-Column Comparison Table",
            "📖 Narrative Synthesis Report",
            "🧩 Dataset Overlap Matrix",
            "⚠️ Limitation Clusters",
            "🔍 Raw JSON Output"
        ])

        with c_tab1:
            st.subheader("Automated 5-Column Literature Comparison Table")
            meth_rows = comp_data.get("methodology_comparison", [])
            met_rows = comp_data.get("metrics_comparison", [])
            
            records = []
            for m_item, met_item in zip(meth_rows, met_rows):
                records.append({
                    "Paper & Citation": f"{m_item.get('paper_title')} ({m_item.get('grounded_citation')})",
                    "Architecture Paradigm": f"{m_item.get('paradigm')} — {m_item.get('key_innovation')[:80]}",
                    "Benchmark Datasets": ", ".join(met_item.get("evaluated_datasets", [])) if met_item.get("evaluated_datasets") else "N/A",
                    "Quantitative Results & Baselines": f"{met_item.get('quantitative_results')[:140]}... | Baseline: {met_item.get('baseline_comparison')[:80]}...",
                    "Method Name / ID": m_item.get("method_name")
                })
            if records:
                st.dataframe(records, width="stretch")
            
            st.markdown("#### Rendered Academic Markdown Table")
            rep = comp_data.get("markdown_comparison_report", "")
            table_lines = [line for line in rep.splitlines() if line.strip().startswith("|")]
            if table_lines:
                table_md = "\n".join(table_lines)
                st.markdown(table_md, unsafe_allow_html=True)
            elif rep:
                st.markdown(rep, unsafe_allow_html=True)

        with c_tab2:
            st.subheader("Full Cross-Paper Synthesis & Methodological Narrative")
            st.markdown(comp_data.get("markdown_comparison_report", "No report available."), unsafe_allow_html=True)

        with c_tab3:
            st.subheader("Benchmark Dataset Overlap & Cross-Evaluation Matrix")
            ds_matrix = comp_data.get("dataset_overlap_matrix", {})
            if ds_matrix:
                shared = {ds: pids for ds, pids in ds_matrix.items() if len(pids) > 1}

                if shared:
                    st.markdown("##### 🤝 High-Consensus Shared Datasets (Evaluated across multiple papers)")
                    for ds, pids in shared.items():
                        st.markdown(f"- **{ds}**: Evaluated in `{len(pids)}` papers (`{', '.join(pids)}`)")
                
                st.markdown("##### 📌 All Evaluated Benchmark Datasets")
                ds_records = [{"Dataset": ds, "Evaluated in Papers": ", ".join(pids), "Paper Count": len(pids)} for ds, pids in ds_matrix.items()]
                st.dataframe(ds_records, width="stretch")
            else:
                st.info("No evaluated datasets registered in comparative analysis.")

        with c_tab4:
            st.subheader("Grouped Limitation Clusters Across Papers")
            lim_themes = comp_data.get("limitation_themes", [])
            if lim_themes:
                for theme in lim_themes:
                    with st.expander(f"⚠️ {theme.get('theme')} (Frequency: {theme.get('frequency')} paper(s))", expanded=True):
                        st.write(f"**Impacted Papers:** {', '.join(theme.get('citations', []))} (`{', '.join(theme.get('paper_ids', []))}`)")
                        st.write(f"**Grounded Pattern:** {theme.get('description')}")
            else:
                st.info("No limitation themes identified.")

        with c_tab5:
            st.subheader("Raw Comparative Analysis JSON Object")
            st.json(comp_data)

# Tab 6: Architecture Flowchart
with tab6:
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

