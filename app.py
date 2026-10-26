"""

NLP-05: Agentic AI for Automated Research Paper Analysis and Literature Review

Interactive Streamlit Web Dashboard

"""

import os

import re

import json

import html

from datetime import datetime

import streamlit as st

import streamlit.components.v1 as components

# Configure Streamlit Page

st.set_page_config(

    page_title="Agentic AI - Automated Literature Review",

    page_icon="📚",

    layout="wide",

    initial_sidebar_state="expanded"

)

# Custom Styling for Academic Presentation

st.markdown("""

<style>

    .main-title { font-size: 30px; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }

    .sub-title { font-size: 15px; color: #4B5563; margin-bottom: 20px; }

    .agent-card { background-color: #F8FAFC; border-left: 4px solid #2563EB; padding: 15px; border-radius: 4px; margin-bottom: 15px; }

    .metric-box { background-color: #EFF6FF; border: 1px solid #BFDBFE; padding: 10px; border-radius: 6px; text-align: center; }

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

def render_mermaid_component(mermaid_code: str, uid: str, timeout_ms: int = 12000, height: int = 520):
    """
    Renders an interactive Mermaid.js diagram in a Streamlit HTML component.
    Implements:
    - Independent initialization and instance caching for fast multi-tab rendering.
    - Visibility-aware deferred execution: never times out while waiting in a background tab.
    - Dual visibility detection: ResizeObserver + 250ms polling interval for instant tab-switch rendering.
    - Specific, actionable error taxonomy: CDN network error, initialization error, grammar syntax error, layout error, timeout.
    - CSS typography enforcement: clean dark background #0E1117, 12px body labels, balanced font sizes.
    - Full raw code fallback expander.
    """
    json_code = json.dumps(mermaid_code)
    escaped_code = html.escape(mermaid_code)

    html_content = f"""
    <div id="mermaid-root-{uid}" style="width: 100%; margin: 4px 0; background-color: #0E1117; font-family: Inter, system-ui, -apple-system, sans-serif;">
      <style>
        html, body {{
          margin: 0 !important;
          padding: 0 !important;
          background-color: #0E1117 !important;
          color: #E2E8F0 !important;
          font-family: Inter, system-ui, -apple-system, sans-serif !important;
          overflow-x: auto !important;
          overflow-y: hidden !important;
        }}
        #mermaid-render-{uid} {{
          display: flex;
          justify-content: center;
          align-items: center;
          width: 100%;
          min-height: 120px;
          background-color: #0E1117;
        }}
        #mermaid-render-{uid} svg {{
          display: block !important;
          margin: 0 auto !important;
          height: auto !important;
        }}
        #mermaid-render-{uid} .cluster-label text,
        #mermaid-render-{uid} .cluster text {{
          font-family: Inter, system-ui, -apple-system, sans-serif !important;
          font-weight: 700 !important;
          fill: #1E293B !important;
        }}
        #mermaid-render-{uid} .edgeLabel {{
          background-color: #1E293B !important;
          padding: 2px 6px !important;
          border-radius: 4px !important;
          border: 1px solid #475569 !important;
          font-size: 10px !important;
          font-weight: 500 !important;
          color: #CBD5E1 !important;
          box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
        }}
        #mermaid-render-{uid} .edgeLabel rect {{
          fill: #1E293B !important;
          stroke: #475569 !important;
          rx: 4px !important;
        }}
      </style>
      <div id="mermaid-render-{uid}">
        <div id="loading-{uid}" style="color: #64748B; font-size: 13px; padding: 20px; text-align: center;">
          <div style="margin-bottom: 6px;">⏳ Rendering diagram...</div>
          <div style="font-size: 11px; color: #94A3B8;">(Activates automatically when tab is selected)</div>
        </div>
      </div>
      <div id="mermaid-fallback-{uid}" style="display: none; background: #0E1117; border: 1px solid #334155; border-radius: 6px; padding: 14px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
          <span style="font-size: 13px; font-weight: bold; color: #DC2626;" id="fallback-status-{uid}">
            ⚠️ Rendering Issue Detected
          </span>
          <span style="font-size: 11px; color: #94A3B8; max-width: 60%; text-align: right;" id="fallback-reason-{uid}"></span>
        </div>
        <div style="font-size: 12px; color: #CBD5E1; margin-bottom: 8px;">
          The visual layout could not be completed. Raw Mermaid diagram definition:
        </div>
        <pre style="margin: 0; padding: 12px; background: #0F172A; color: #F8FAFC; border-radius: 4px; overflow-x: auto; font-size: 12px; line-height: 1.45;"><code>{escaped_code}</code></pre>
      </div>
    </div>
    <script type="module">
      (async function() {{
        const uid = "{uid}";
        const code = {json_code};
        let isTerminated = false;
        let isRendered = false;
        let observer = null;
        let pollTimer = null;
        let activeRenderTimeout = null;
        const renderEl = document.getElementById(`mermaid-render-${{uid}}`);
        const fallbackEl = document.getElementById(`mermaid-fallback-${{uid}}`);
        const statusEl = document.getElementById(`fallback-status-${{uid}}`);
        const reasonEl = document.getElementById(`fallback-reason-${{uid}}`);
        const loadingEl = document.getElementById(`loading-${{uid}}`);

        function activateFallback(status, reason) {{
          if (isTerminated || isRendered) return;
          isTerminated = true;
          if (observer) observer.disconnect();
          if (pollTimer) clearInterval(pollTimer);
          if (activeRenderTimeout) clearTimeout(activeRenderTimeout);
          if (loadingEl) loadingEl.remove();
          if (renderEl) renderEl.style.display = 'none';
          if (fallbackEl) {{
            fallbackEl.style.display = 'block';
            if (statusEl) statusEl.textContent = status;
            if (reasonEl) reasonEl.textContent = reason ? reason : '';
          }}
        }}

        // 1. Dynamic CDN import with window-level caching
        let mermaid;
        try {{
          if (window.mermaid) {{
            mermaid = window.mermaid;
          }} else {{
            const mermaidModule = await import('https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs');
            mermaid = mermaidModule.default || mermaidModule;
            window.mermaid = mermaid;
          }}
        }} catch (cdnErr) {{
          activateFallback("🌐 CDN Network Error", "Failed to load Mermaid from CDN: " + (cdnErr.message || String(cdnErr)));
          return;
        }}
        if (isTerminated) return;

        // 2. Safe initialization with strict security and clean typography
        try {{
          mermaid.initialize({{
            startOnLoad: false,
            theme: 'base',
            securityLevel: 'strict',
            suppressErrorRendering: true,
            fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
            flowchart: {{ htmlLabels: true, curve: 'basis', padding: 14 }},
            themeVariables: {{
              fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
              fontSize: '12px',
              background: '#0E1117',
              darkBkgColor: '#0E1117',
              primaryColor: '#EFF6FF',
              primaryTextColor: '#1E3A8A',
              primaryBorderColor: '#2563EB',
              lineColor: '#64748B',
              secondaryColor: '#CCFBF1',
              tertiaryColor: '#FEF3C7',
              mindmapBranchColor0: '#1E3A8A',
              mindmapBranchColor1: '#0F766E',
              mindmapBranchColor2: '#D97706',
              mindmapBranchColor3: '#7C3AED',
              mindmapBranchColor4: '#DC2626'
            }}
          }});
        }} catch (initErr) {{
          activateFallback("⚙️ Initialization Error", initErr.message || String(initErr));
          return;
        }}
        if (isTerminated) return;

        // 3. Fast syntax parse check with descriptive line/column errors
        try {{
          await mermaid.parse(code);
        }} catch (parseErr) {{
          let errStr = parseErr.str || parseErr.message || "Grammar syntax error";
          if (parseErr.hash && parseErr.hash.loc) {{
            errStr += ` (Line ${{parseErr.hash.loc.first_line}}, Col ${{parseErr.hash.loc.first_column}})`;
          }}
          activateFallback("⚠️ Syntax Parse Error", errStr);
          return;
        }}
        if (isTerminated) return;

        // 4. Visibility-aware layout execution
        async function tryRenderDiagram() {{
          if (isRendered || isTerminated) return;
          const containerWidth = document.documentElement.clientWidth || document.body.clientWidth;
          if (containerWidth <= 50) {{
            return;
          }}
          if (pollTimer) {{
            clearInterval(pollTimer);
            pollTimer = null;
          }}
          if (!activeRenderTimeout) {{
            activeRenderTimeout = setTimeout(() => {{
              if (!isRendered && !isTerminated) {{
                activateFallback("⏱️ Diagram Rendering Timeout", "Render exceeded 12 seconds after tab became visible.");
              }}
            }}, 12000);
          }}

          try {{
            const svgId = `svg_${{uid}}_${{Date.now()}}_${{Math.random().toString(36).substring(2, 7)}}`;
            const {{ svg }} = await mermaid.render(svgId, code);
            if (isTerminated) return;
            isRendered = true;
            if (activeRenderTimeout) clearTimeout(activeRenderTimeout);
            if (observer) observer.disconnect();
            if (loadingEl) loadingEl.remove();

            renderEl.innerHTML = svg;
            const renderedSvg = renderEl.querySelector('svg');
            if (renderedSvg) {{
              const vb = renderedSvg.viewBox ? renderedSvg.viewBox.baseVal : null;
              const natW = (vb && vb.width > 0) ? vb.width : (parseFloat(renderedSvg.getAttribute('width')) || 850);
              const maxW = Math.min(Math.max(natW, 600), 1080);
              renderedSvg.style.maxWidth = `${{maxW}}px`;
              renderedSvg.style.width = '100%';
              renderedSvg.style.height = 'auto';
              renderedSvg.style.display = 'block';
              renderedSvg.style.margin = '0 auto';
            }}
            window.scrollTo(0, 0);
          }} catch (renderErr) {{
            if (activeRenderTimeout) clearTimeout(activeRenderTimeout);
            activateFallback("📐 Layout / Rendering Error", renderErr.message || String(renderErr));
          }}
        }}

        // Immediate attempt
        await tryRenderDiagram();

        // If inactive, monitor tab selection via ResizeObserver, resize events, and polling
        if (!isRendered) {{
          if (typeof ResizeObserver !== 'undefined') {{
            observer = new ResizeObserver(() => tryRenderDiagram());
            observer.observe(document.documentElement);
            observer.observe(document.body);
          }}
          window.addEventListener('resize', () => tryRenderDiagram());
          window.addEventListener('focus', () => tryRenderDiagram());
          document.addEventListener('visibilitychange', () => tryRenderDiagram());
          pollTimer = setInterval(() => {{
            if (!isRendered && !isTerminated) {{
              tryRenderDiagram();
            }} else if (pollTimer) {{
              clearInterval(pollTimer);
            }}
          }}, 250);
        }}
      }})();
    </script>
    """
    components.html(html_content, height=height, scrolling=True)

def render_report_markdown_with_diagrams(markdown_text: str, base_uid: str = "report_doc"):

    """

    Renders report markdown with interactive, perfectly aligned Mermaid diagrams inline.

    Splits text by ```mermaid blocks: renders prose via st.markdown and diagrams via render_mermaid_component.

    """

    pattern = r"```mermaid\s*\n(.*?)\n```"

    parts = re.split(pattern, markdown_text, flags=re.DOTALL)



    diag_idx = 0

    for i, part in enumerate(parts):

        if i % 2 == 0:

            if part.strip():

                st.markdown(part, unsafe_allow_html=True)

        else:

            diag_idx += 1

            code = part.strip()

            m_id = re.search(r"%% diagram_id:\s*([a-zA-Z0-9_-]+)", code)

            diag_uid = f"{base_uid}_{m_id.group(1)}" if m_id else f"{base_uid}_{diag_idx}"



            num_lines = len(code.splitlines())

            is_flowchart = bool(m_id and m_id.group(1) == "methodology_flowchart")

            is_gaps = bool(m_id and m_id.group(1) == "research_gaps_flowchart")

            h = 520 if is_flowchart else (460 if is_gaps else min(max(num_lines * 22, 360), 500))

            render_mermaid_component(code, uid=diag_uid, height=h)

            if not is_flowchart and not is_gaps:

                with st.expander(f"🔍 View / Copy Raw Mermaid Code ({m_id.group(1) if m_id else f'Diagram {diag_idx}'})"):

                    st.code(code, language="mermaid")

def extract_mermaid_diagrams_by_id(markdown_text: str) -> dict:

    """

    Safely extracts Mermaid blocks keyed by their unique '%% diagram_id: <name>' marker,

    with robust semantic signature fallback for order-independent retrieval.

    """

    diagrams = {}

    pattern = r'```mermaid\s*\n(.*?)\n```'

    blocks = re.findall(pattern, markdown_text, re.DOTALL)

    for block in blocks:

        id_match = re.search(r'%%\s*diagram_id:\s*([a-zA-Z0-9_\-]+)', block)

        if id_match:

            diagrams[id_match.group(1)] = block.strip()

        else:

            if 'mindmap' in block or 'paradigmMain' in block or 'benchmarkMain' in block:

                diagrams.setdefault('themes_mindmap', block.strip())

            elif 'Layer1' in block or 'Layer2' in block or 'limitationStyle' in block or 'research_gaps' in block:

                diagrams.setdefault('research_gaps_flowchart', block.strip())

            elif 'rootStyle' in block or 'paperCard' in block or 'paradigmStyle' in block:

                diagrams.setdefault('methodology_flowchart', block.strip())

    return diagrams

# Workflow Tabs Configuration (Expanded to 7 Steps)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([

    "📂 Corpus Selection",

    "🔍 Document Grounding Engine",

    "🧠 Structured Information Extraction",

    "📊 Global Memory & Verification",

    "📑 Step 3: Single-Paper Summaries",

    "📈 Step 4: Comparative Matrix",

    "📑 Step 5 & 6: Literature Review & Diagrams"

])

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

    st.header("Step 1: Document Grounding & Verification Engine")

    st.markdown("Verifies extracted text coverage, layout boundaries, and visual page rendering.")

    if "current_pdf_path" in st.session_state and os.path.exists(st.session_state["current_pdf_path"]):

        pdf_path = st.session_state["current_pdf_path"]

        pdf_filename = os.path.basename(pdf_path)

        from src.agents.parser import PDFParserAgent

        parser = PDFParserAgent()

        col_run, col_status = st.columns([1, 2])

        with col_run:

            run_parse = st.button("🚀 Run Layout & Text Extraction", key="btn_run_parser")

        parsed_data = parser.load_parsed_data(pdf_filename)

        if run_parse:

            with st.spinner(f"Extracting layout and text from `{pdf_filename}`..."):

                parsed_data = parser.parse(pdf_path)

                st.success(f"Extraction Complete! Extracted {parsed_data.total_pages} pages using `{parsed_data.parser_used}`.")

        if parsed_data:

            st.subheader(f"Extracted Grounding Data: `{pdf_filename}`")

            m1, m2, m3, m4 = st.columns(4)

            m1.metric("Total Pages", parsed_data.total_pages)

            m2.metric("Parser Engine", parsed_data.parser_used)

            m3.metric("Extracted Sections", len(parsed_data.sections))

            m4.metric("Total Text Length", f"{len(parsed_data.full_text):,} chars")

            st.divider()

            col_pages, col_raw = st.columns([1, 1])

            with col_pages:

                st.markdown("### 📄 Page-by-Page Grounded Content")

                if parsed_data.total_pages > 1:

                    selected_page_idx = st.slider("Select Page to Inspect:", 1, parsed_data.total_pages, 1)

                else:

                    selected_page_idx = 1

                    st.info("Document has 1 page.")

                page_text = parsed_data.pages.get(selected_page_idx, "No text extracted for this page.")

                st.text_area(f"Raw Text Content - Page {selected_page_idx}:", page_text, height=350)

            with col_raw:

                st.markdown("### 📑 Section Boundaries")

                if parsed_data.sections:

                    for sec in parsed_data.sections[:8]:

                        with st.expander(f"📍 {sec.title} (Page {sec.page_number})"):

                            st.write(sec.content[:400] + ("..." if len(sec.content) > 400 else ""))

                else:

                    st.info("No section boundaries detected. Full-text mode active.")

    else:

        st.warning("Please select or upload a PDF in the '📂 Corpus Selection' tab to begin.")

with tab3:

    st.header("Step 2: Structured Information Extraction")

    st.markdown("Pydantic Schema Validation for verifiable academic extraction.")

    if "current_pdf_path" in st.session_state and os.path.exists(st.session_state["current_pdf_path"]):

        pdf_path = st.session_state["current_pdf_path"]

        pdf_filename = os.path.basename(pdf_path)

        paper_id = pdf_filename.split("_")[0] if "_" in pdf_filename else os.path.splitext(pdf_filename)[0]

        from src.agents.parser import PDFParserAgent

        from src.agents.extraction import SchemaExtractionAgent

        from src.agents.llm import LLMManager, GeminiLLMManager

        llm = LLMManager()

        extractor = SchemaExtractionAgent(llm_manager=llm)

        col_ext_btn, col_ext_stat = st.columns([1, 2])

        with col_ext_btn:

            run_ext = st.button("⚡ Extract Pydantic Schema", key="btn_run_extraction")

        existing_schema = extractor.load_schema(paper_id)

        if run_ext:

            with st.spinner(f"Extracting structured schema for `{paper_id}`..."):

                parser = PDFParserAgent()

                parsed_data = parser.load_parsed_data(pdf_filename)

                if not parsed_data:

                    parsed_data = parser.parse(pdf_path)

                existing_schema = extractor.extract(parsed_data)

                st.success(f"Structured schema generated successfully for `{existing_schema.paper_title}`!")

        if existing_schema:

            st.subheader(f"📌 {existing_schema.paper_title}")

            st.markdown(f"**Authors:** {', '.join(existing_schema.authors)} | **Year:** {existing_schema.year} | **Venue:** {existing_schema.venue}")

            tab_prob, tab_meth, tab_eval, tab_gaps = st.tabs([

                "🎯 Problem Statement",

                "🔬 Proposed Method",

                "📊 Evaluation & Benchmarks",

                "⚠️ Limitations & Stated Gaps"

            ])

            with tab_prob:

                st.markdown(f"**Primary Problem:** {existing_schema.problem_statement}")

            with tab_meth:

                st.markdown(f"**Architecture / Framework:** `{existing_schema.proposed_method.architecture}`")

                st.markdown(f"**Description:** {existing_schema.proposed_method.description}")

                if existing_schema.proposed_method.key_innovations:

                    st.markdown("**Key Innovations:**")

                    for inn in existing_schema.proposed_method.key_innovations:

                        st.markdown(f"- {inn}")

            with tab_eval:

                st.markdown(f"**Datasets Used:** {', '.join(existing_schema.evaluation.datasets)}")

                st.markdown(f"**Baselines Compared:** {', '.join(existing_schema.evaluation.baselines)}")

                if existing_schema.evaluation.quantitative_results:

                    st.markdown("**Quantitative Results:**")

                    for k, v in existing_schema.evaluation.quantitative_results.items():

                        st.markdown(f"- **{k}:** {v}")

            with tab_gaps:

                st.markdown("**Limitations & Constraints:**")

                for lim in existing_schema.limitations:

                    st.markdown(f"- {lim}")

                st.markdown("**Stated Future Work & Gaps:**")

                for gap in existing_schema.future_work:

                    st.markdown(f"- {gap}")

            with st.expander("🔍 View Raw JSON Schema"):

                st.json(existing_schema.model_dump())

    else:

        st.warning("Please select or upload a PDF in the '📂 Corpus Selection' tab to begin.")

with tab4:

    st.header("Step 2 Verification: Global Memory & Benchmark Corpus")

    st.markdown("Verifies the global state of extracted schemas across the entire literature review corpus.")

    from src.agents.memory_store import get_global_memory_store

    mem_store = get_global_memory_store()

    all_schemas = mem_store.get_all_schemas()

    col_sync, col_status = st.columns([1, 2])

    with col_sync:

        if st.button("🔄 Sync with Disk Schemas", key="btn_sync_memory"):

            mem_store.reload_from_disk()

            all_schemas = mem_store.get_all_schemas()

            st.success(f"Synced! Memory Store now holds {len(all_schemas)} papers.")

    st.subheader(f"📚 Registered Papers in Memory: {len(all_schemas)}")

    if all_schemas:

        import pandas as pd

        rows = []

        for pid, s in all_schemas.items():

            rows.append({

                "Paper ID": pid,

                "Title": s.get("paper_title", "N/A"),

                "Year": s.get("year", "N/A"),

                "Architecture": s.get("proposed_method", {}).get("architecture", "N/A") if isinstance(s.get("proposed_method"), dict) else getattr(s.get("proposed_method"), "architecture", "N/A"),

                "Limitations Tracked": len(s.get("limitations", [])),

                "Future Gaps Tracked": len(s.get("future_work", []))

            })

        df = pd.DataFrame(rows)

        st.dataframe(df, use_container_width=True)

        selected_pid = st.selectbox("Inspect Full Grounded Extraction from Store:", list(all_schemas.keys()))

        if selected_pid:

            st.json(all_schemas[selected_pid])

    else:

        st.info("No schemas currently registered in Global Memory. Run extraction in Step 2 or click 'Sync with Disk Schemas'.")

with tab5:

    st.header("Step 3: Grounded Single-Paper Summaries")

    st.markdown("Single-paper multi-section structured summaries with grounded citations and failure mode analysis.")

    from src.agents.memory_store import get_global_memory_store

    from src.agents.summarizer import SinglePaperSummarizerAgent, run_summarizer_agent

    from src.agents.llm import LLMManager

    mem_store = get_global_memory_store()

    summaries = mem_store.get_all_summaries()

    col_sum_actions, _ = st.columns([1, 1])

    with col_sum_actions:

        if st.button("🚀 Summarize Current Selected Paper", key="btn_summarize_current"):

            if "current_pdf_path" in st.session_state and os.path.exists(st.session_state["current_pdf_path"]):

                p_path = st.session_state["current_pdf_path"]

                p_filename = os.path.basename(p_path)

                p_id = p_filename.split("_")[0] if "_" in p_filename else os.path.splitext(p_filename)[0]

                schema_dir = "data/extracted_schemas"

                schema_path = os.path.join(schema_dir, f"{p_id}_schema.json")

                if not os.path.exists(schema_path) and os.path.exists(schema_dir):

                    candidates = [os.path.join(schema_dir, f) for f in os.listdir(schema_dir) if f.startswith(p_id) and f.endswith("_schema.json")]

                    if candidates:

                        schema_path = candidates[0]

                if not os.path.exists(schema_path):

                    st.error(f"No extracted schema found for paper '{p_id}' in '{schema_dir}'. Please run Step 2 (Structured Information Extraction) first.")

                else:

                    llm = LLMManager()

                    with st.spinner(f"Generating Grounded Summary for {p_id}..."):

                        s_agent = SinglePaperSummarizerAgent(llm_manager=llm)

                        sum_obj = s_agent.summarize_paper(p_id)

                        st.success(f"Summary Generated and Saved for {p_id}!")

                        st.rerun()

            else:

                st.warning("Please select a paper first.")

    st.subheader(f"📑 Stored Single-Paper Summaries: {len(summaries)}")

    if summaries:

        selected_sum_id = st.selectbox("Select Paper Summary to View:", list(summaries.keys()), key="select_sum_id")

        if selected_sum_id:

            s_data = summaries[selected_sum_id]

            st.markdown(f"### 📖 {s_data.get('paper_title', selected_sum_id)}")

            st.caption(f"Citation: `{s_data.get('grounded_citation', 'N/A')}` | Paper ID: `{selected_sum_id}`")

            t_tldr, t_meth, t_results, t_fail = st.tabs([

                "📌 Executive TL;DR",

                "🔬 Detailed Methodology",

                "📊 Empirical Results",

                "⚠️ Failure Modes & Limitations"

            ])

            with t_tldr:

                st.write(s_data.get("executive_summary", "No summary available."))

            with t_meth:

                st.write(s_data.get("methodology_deep_dive", "No methodology available."))

            with t_results:

                st.write(s_data.get("empirical_results", "No results available."))

            with t_fail:

                st.write(s_data.get("limitations_and_failures", "No limitations available."))

            with st.expander("🔍 View Raw Summary JSON"):

                st.json(s_data)

    else:

        st.info("No summaries found. Run Step 3 from the demo CLI or generate one above.")

with tab6:

    st.header("Step 4: Cross-Paper Comparative Analysis Matrix")

    st.markdown("Pydantic-grounded comparative matrix cross-referencing benchmarks, methods, and limitations.")

    from src.agents.memory_store import get_global_memory_store

    from src.agents.comparative import run_comparative_analysis

    mem_store = get_global_memory_store()

    comp_analysis = mem_store.get_comparative_analysis()

    if st.button("🚀 Re-Run Comparative Analysis Agent", key="btn_run_comp"):

        with st.spinner("Executing Cross-Paper Comparative Agent across all summaries..."):

            run_comparative_analysis()

            st.success("Comparative Analysis Matrix updated successfully!")

            st.rerun()

    if comp_analysis:

        m_rows = comp_analysis.get("methodology_comparison", [])

        m1, m2, m3 = st.columns(3)

        m1.metric("Total Papers Analyzed", len(m_rows))

        m2.metric("Cross-Cutting Limitation Themes", len(comp_analysis.get("limitation_themes", [])))

        m3.metric("Shared Benchmark Datasets", len(comp_analysis.get("dataset_overlap_matrix", {})))

        st.divider()

        tab_matrix, tab_themes, tab_datasets, tab_md = st.tabs([

            "📋 5-Column Comparative Matrix",

            "🧩 Limitation Themes & Trade-offs",

            "📊 Benchmark Dataset Consensus",

            "📑 Full Markdown Report"

        ])

        with tab_matrix:

            if m_rows:

                import pandas as pd

                matrix_data = []

                for r in m_rows:

                    matrix_data.append({

                        "Paper ID": r.get("paper_id"),

                        "Method": r.get("method_name"),

                        "Paradigm": r.get("paradigm"),

                        "Citation": r.get("grounded_citation"),

                        "Strengths / Gains": r.get("primary_strengths")[:150] + "..." if len(r.get("primary_strengths", "")) > 150 else r.get("primary_strengths"),

                        "Failure Modes": r.get("failure_modes")[:150] + "..." if len(r.get("failure_modes", "")) > 150 else r.get("failure_modes")

                    })

                st.dataframe(pd.DataFrame(matrix_data), use_container_width=True)

        with tab_themes:

            for theme in comp_analysis.get("limitation_themes", []):

                with st.expander(f"⚠️ {theme.get('theme')} ({theme.get('frequency')} papers affected)"):

                    st.markdown(f"**Impacted Papers:** {', '.join(theme.get('paper_ids', []))}")

                    st.markdown(f"**Synthesized Architectural Bottleneck:**\n{theme.get('synthesized_observation')}")

        with tab_datasets:

            ds_matrix = comp_analysis.get("dataset_overlap_matrix", {})

            if ds_matrix:

                import pandas as pd

                ds_rows = [{"Dataset": k, "Evaluating Papers Count": len(v), "Paper IDs": ", ".join(v)} for k, v in ds_matrix.items()]

                st.dataframe(pd.DataFrame(ds_rows).sort_values("Evaluating Papers Count", ascending=False), use_container_width=True)

        with tab_md:

            st.markdown(comp_analysis.get("markdown_comparison_report", "No markdown report generated."))

    else:

        st.info("No comparative analysis matrix on disk. Click the button above to execute Agent 4.")

# -----------------------------------------------------------------------------

# Tab 7: Literature Review Synthesis & Structural Diagrams

# -----------------------------------------------------------------------------

with tab7:

    st.header("Step 5 & 6: Literature Review Synthesis & Structural Diagrams")

    st.markdown("Publication-grade Literature Review Report enriched with verified, interactive Mermaid.js diagrams.")

    from src.agents.memory_store import get_global_memory_store

    from src.agents.research_gap import run_research_gap_agent

    from src.agents.report_generator import generate_literature_review

    from src.agents.comparative import run_comparative_analysis

    from src.agents.llm_manager import GeminiLLMManager, LLMManager

    from src.agents.schemas import PaperSchema, ComprehensivePaperSummary, ComparativeAnalysis

    schema_dir = "data/extracted_schemas"

    comp_path = "data/comparative_analysis.json"

    # Identify current selected paper and all available schemas

    current_pdf = st.session_state.get("current_pdf_path", "")

    current_pid = ""

    if current_pdf:

        base = os.path.basename(current_pdf)

        current_pid = base.split("_")[0] if "_" in base else os.path.splitext(base)[0]

    available_schema_files = [f for f in os.listdir(schema_dir) if f.endswith("_schema.json")] if os.path.exists(schema_dir) else []

    all_available_pids = sorted(list(set([sf.replace("_schema.json", "").split("_")[0] for sf in available_schema_files])))

    # 1. Literature Review Scope Selection

    scope_options = []

    if current_pid:

        scope_options.append(f"Current Selected Paper ({current_pid})")

    scope_options.append("Full Benchmark Corpus (All Papers)")

    scope_options.append("Custom Multi-Paper Selection")

    col_scope, _ = st.columns([2, 1])

    with col_scope:

        selected_scope = st.radio(

            "Literature Review Synthesis Scope:",

            scope_options,

            index=0,

            horizontal=True,

            key="review_scope_choice",

            help="Choose whether to generate a deep-dive review for the selected paper or an aggregated review across multiple papers."

        )

    if selected_scope.startswith("Current Selected Paper"):

        target_pids = [current_pid]

        scope_key = current_pid

        report_path = f"data/reports/literature_review_{current_pid}.md"

    elif selected_scope == "Full Benchmark Corpus (All Papers)":

        target_pids = all_available_pids

        scope_key = "corpus"

        report_path = "data/reports/literature_review_report.md"

    else:

        mult_pids = st.multiselect("Select Papers to Include in Synthesis:", all_available_pids, default=[current_pid] if current_pid in all_available_pids else all_available_pids[:3])

        target_pids = mult_pids or ([current_pid] if current_pid else all_available_pids)

        scope_key = "_".join(sorted(target_pids)[:3])

        report_path = "data/reports/literature_review_custom.md"

    # 2. Staleness Detection & Validation Check

    report_is_stale = False

    staleness_reasons = []

    if os.path.exists(report_path):

        report_mtime = os.path.getmtime(report_path)

        with open(report_path, "r", encoding="utf-8") as rf:

            first_lines = "".join([rf.readline() for _ in range(6)])

            match = re.search(r"Corpus Coverage:\s*(\d+)", first_lines)

            if match:

                report_paper_count = int(match.group(1))

                if report_paper_count != len(target_pids):

                    report_is_stale = True

                    staleness_reasons.append(f"Report covers {report_paper_count} papers, but {len(target_pids)} papers are selected.")

    else:

        # Check if legacy report exists and matches

        legacy_path = "data/reports/literature_review_report.md"

        if len(target_pids) == len(all_available_pids) and os.path.exists(legacy_path):

            report_path = legacy_path

    if report_is_stale:

        st.warning("⚠️ **Report Staleness Warning:**\n" + "\n".join([f"- {r}" for r in staleness_reasons]) + "\n\n*Click 'Synthesize Fresh Literature Review' below to update the report and diagrams.*")

    # 3. Synthesis Action Controls

    col_synth_btn, col_synth_status = st.columns([1, 2])

    with col_synth_btn:

        trigger_synth = st.button("🚀 Synthesize Fresh Literature Review", key="btn_trigger_synthesis", help="Executes Agent 5 (Research Gaps) and Agent 6 (Report Generator) atomically for the selected scope.")

    if trigger_synth:

        llm = LLMManager()

        with st.status(f"Executing Report & Diagram Synthesis for {len(target_pids)} Paper(s)...", expanded=True) as status_box:

            st.write(f"Target Papers: {', '.join(target_pids)}")

            mem_store = get_global_memory_store()

            # Retrieve schemas for target_pids

            raw_schemas = mem_store.get_all_schemas()

            if not raw_schemas and os.path.exists(schema_dir):

                for sf in available_schema_files:

                    with open(os.path.join(schema_dir, sf), "r", encoding="utf-8") as f_s:

                        s_data = json.load(f_s)

                        raw_schemas[s_data.get("paper_id", sf)] = s_data

            typed_schemas = []

            for s_dict in raw_schemas.values():

                s_obj = PaperSchema(**s_dict) if isinstance(s_dict, dict) else s_dict

                if any(p in s_obj.paper_id for p in target_pids):

                    typed_schemas.append(s_obj)

            # Retrieve summaries for target_pids

            raw_summaries = mem_store.get_all_summaries()

            sum_dir = "data/summaries"

            if not raw_summaries and os.path.exists(sum_dir):

                for sum_f in os.listdir(sum_dir):

                    if sum_f.endswith("_summary.json"):

                        with open(os.path.join(sum_dir, sum_f), "r", encoding="utf-8") as f_sum:

                            sm_data = json.load(f_sum)

                            raw_summaries[sm_data.get("paper_id", sum_f)] = sm_data

            typed_summaries = []

            for sm_dict in raw_summaries.values():

                sm_pid = getattr(sm_dict, "paper_id", "") or (sm_dict.get("paper_id", "") if isinstance(sm_dict, dict) else "")

                if any(p in sm_pid for p in target_pids):

                    try:

                        typed_summaries.append(ComprehensivePaperSummary(**sm_dict) if isinstance(sm_dict, dict) else sm_dict)

                    except Exception:

                        typed_summaries.append(sm_dict)

            # Execute comparative analysis for target papers

            typed_comp = run_comparative_analysis(target_paper_ids=target_pids)

            # Synthesize matching research gaps

            typed_gaps = run_research_gap_agent(typed_schemas, llm_manager=llm)

            # Synthesize final literature review

            final_report = generate_literature_review(typed_schemas, typed_summaries, typed_comp, typed_gaps, llm_manager=llm, output_filepath=report_path)

            st.success("✅ Literature Review Report & Diagrams Synthesized Successfully!")

            st.rerun()

    # 4. Report & Diagrams Display

    if os.path.exists(report_path):

        with open(report_path, "r", encoding="utf-8") as rf:

            report_content = rf.read()

        d_tab1, d_tab2, d_tab3, d_tab4 = st.tabs([

            "📊 Interactive Diagrams (Mermaid.js)",

            "📈 Empirical Charts & Graphs",

            "🗺️ Geographical & Spatial Maps",

            "📖 Full Markdown Literature Review"

        ])

        with d_tab1:

            st.subheader("Structural Knowledge & Methodology Diagrams")

            extracted_diags = extract_mermaid_diagrams_by_id(report_content)

            from src.agents.report_generator import (
                generate_methodology_flowchart,
                generate_themes_mindmap,
                generate_research_gaps_flowchart
            )

            # Guarantee all three diagrams are available even if the report markdown was partial
            if "methodology_flowchart" not in extracted_diags or "themes_mindmap" not in extracted_diags or "research_gaps_flowchart" not in extracted_diags:
                mem_store = get_global_memory_store()
                comp_data = mem_store.get_comparative_analysis()
                if not comp_data:
                    comp_data = run_comparative_analysis(target_paper_ids=target_pids)
                cur_schemas = [mem_store.get_schema(p) for p in target_pids if mem_store.get_schema(p)]

                if "methodology_flowchart" not in extracted_diags:
                    d1_code = generate_methodology_flowchart(comp_data)
                    if d1_code:
                        extracted_diags["methodology_flowchart"] = d1_code
                if "themes_mindmap" not in extracted_diags:
                    d2_code = generate_themes_mindmap(comp_data)
                    if d2_code:
                        extracted_diags["themes_mindmap"] = d2_code
                if "research_gaps_flowchart" not in extracted_diags:
                    d3_code = generate_research_gaps_flowchart(comp_data, None, schemas=cur_schemas)
                    if d3_code:
                        extracted_diags["research_gaps_flowchart"] = d3_code

            titles = {
                "methodology_flowchart": "Diagram 1: Architectural Paradigms & Methodology Flowchart",
                "themes_mindmap": "Diagram 2: Research Themes Mind Map",
                "research_gaps_flowchart": "Diagram 3: Empirical Constraints & Research Gaps Lineage"
            }
            main_diagram_order = ["methodology_flowchart", "themes_mindmap", "research_gaps_flowchart"]

            # Guarantee all three tabs are always rendered and selectable
            sub_tabs = st.tabs([titles[k] for k in main_diagram_order])
            for idx, k in enumerate(main_diagram_order):
                with sub_tabs[idx]:
                    st.markdown(f"**{titles[k]}**")
                    code = extracted_diags.get(k, "")
                    if k == "methodology_flowchart":
                        render_mermaid_component(code, uid=f"diag_{k}", timeout_ms=10000, height=520)
                    elif k == "themes_mindmap":
                        render_mermaid_component(code, uid=f"diag_{k}", timeout_ms=10000, height=480)
                        with st.expander("🔍 View / Copy Raw Mermaid Code"):
                            st.code(code, language="mermaid")
                    elif k == "research_gaps_flowchart":
                        render_mermaid_component(code, uid=f"diag_{k}", timeout_ms=10000, height=460)
                    else:
                        render_mermaid_component(code, uid=f"diag_{k}", timeout_ms=10000, height=580)
                        with st.expander("🔍 View / Copy Raw Mermaid Code"):
                            st.code(code, language="mermaid")

        with d_tab2:

            st.subheader("Empirical Quantitative Benchmarks & Baseline Comparisons")

            figures_dir = "data/figures"

            chart_files = []

            if os.path.exists(figures_dir):

                for root, _, files in os.walk(figures_dir):

                    for f in files:

                        if f.startswith("chart_") and f.endswith(".png"):

                            chart_files.append(os.path.join(root, f).replace("\\", "/"))

            # Filter strictly to target paper(s)

            active_charts = [c for c in chart_files if any(p in c for p in target_pids)]

            if active_charts:

                for c_path in sorted(active_charts):

                    p_name = os.path.basename(os.path.dirname(c_path)) or "Paper"

                    st.markdown(f"#### Quantitative Results: {p_name}")

                    st.image(c_path, use_container_width=True)

                    st.caption(f"Figure: Quantitative Benchmark Comparison for {p_name}")

                    st.markdown("---")

            else:

                st.info(f"ℹ️ No quantitative benchmark chart found for the active target ({', '.join(target_pids)}). The paper did not report structured numerical evaluation tables.")

        with d_tab3:

            st.subheader("Geographical & Regional Spatial Projections")

            figures_dir = "data/figures"

            map_files = []

            if os.path.exists(figures_dir):

                for root, _, files in os.walk(figures_dir):

                    for f in files:

                        if f.startswith("spatial_map_") and f.endswith(".png"):

                            map_files.append(os.path.join(root, f).replace("\\", "/"))

            # Filter strictly to target paper(s)

            active_maps = [m for m in map_files if any(p in m for p in target_pids)]

            if active_maps:

                for m_path in sorted(active_maps):

                    p_name = os.path.basename(os.path.dirname(m_path)) or "Paper"

                    st.markdown(f"#### Spatial Climate & Regional Analysis: {p_name}")

                    st.image(m_path, use_container_width=True)

                    st.caption(f"Figure: Regional Spatial Distribution & Projections ({p_name})")

                    st.markdown("---")

            else:

                st.info("ℹ️ **No Geographical / Spatial Distribution Data for Selected Paper(s)**\n\n"

                        "Spatial maps are dynamically generated exclusively for research papers with genuine "

                        "geographic, meteorological, or regional dimensions (e.g., regional climate projections, "

                        "epidemiological surveys across geographic zones).\n\n"

                        "For purely computational, algorithmic, or mathematical papers, geographical maps are "

                        "deliberately omitted to preserve academic integrity and prevent hallucinated spatial associations.")

        with d_tab4:

            st.subheader("Publication-Ready Literature Review Draft")

            render_report_markdown_with_diagrams(report_content, base_uid=f"report_{scope_key}")

            st.download_button(

                label="📥 Download Literature Review Report (Markdown)",

                data=report_content,

                file_name=os.path.basename(report_path),

                mime="text/markdown",

                use_container_width=True

            )

    else:

        st.info(f"ℹ️ No literature review report found on disk for {target_pids} yet. Click 'Synthesize Fresh Literature Review' above to run Agent 5 and Agent 6.")

