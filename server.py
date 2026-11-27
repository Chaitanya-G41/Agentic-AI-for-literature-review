"""
FastAPI Backend Server for NLP-05 Scholar's Digital Binder
Serves API endpoints for corpus, schemas, comparative analysis, reports, figures, PDF streaming,
and interactive agent execution.
"""

import os
import glob
import json
import re
import time
import datetime
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Body, BackgroundTasks, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

app = FastAPI(title="Scholar's Digital Binder API", version="1.0.0")

# Enable CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_DIR = os.path.join(DATA_DIR, "pdf_corpus")
SCHEMAS_DIR = os.path.join(DATA_DIR, "extracted_schemas")
SUMMARIES_DIR = os.path.join(DATA_DIR, "summaries")
FIGURES_DIR = os.path.join(DATA_DIR, "figures")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend_v2", "dist")
if not os.path.exists(FRONTEND_DIST):
    FRONTEND_DIST = os.path.join(BASE_DIR, "frontend", "dist")

# Mount figures static folder
if os.path.exists(FIGURES_DIR):
    app.mount("/static/figures", StaticFiles(directory=FIGURES_DIR), name="figures")


class ScreeningRequest(BaseModel):
    query: str
    max_candidates: int = 25
    recency_weight: float = 0.20
    inclusion_criteria: Optional[str] = ""
    exclusion_criteria: Optional[str] = ""
    max_results: Optional[int] = None


class SaveScreeningRequest(BaseModel):
    query: str
    inclusion_criteria: Optional[str] = ""
    exclusion_criteria: Optional[str] = ""
    candidates: List[Dict[str, Any]]


class DownloadRequest(BaseModel):
    papers: List[Dict[str, Any]]
    query: Optional[str] = ""


@app.get("/api/status")
def get_status():
    """Returns server and pipeline status."""
    nv_key = os.environ.get("NVIDIA_API_KEY", "") or os.environ.get("NVAPI_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    
    pdf_count = len(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    schema_count = len(glob.glob(os.path.join(SCHEMAS_DIR, "*_schema.json")))
    summary_count = len(glob.glob(os.path.join(SUMMARIES_DIR, "*_summary.json")))
    report_exists = os.path.exists(os.path.join(REPORTS_DIR, "literature_review_report.md"))

    return {
        "status": "ready",
        "nvidia_configured": bool(nv_key.strip()),
        "gemini_configured": bool(gemini_key.strip()),
        "counts": {
            "pdfs": pdf_count,
            "schemas": schema_count,
            "summaries": summary_count,
            "has_report": report_exists
        }
    }


def classify_paradigm(title: str, abstract: str = "") -> Dict[str, str]:
    """Classifies paper into aesthetic pastel badge paradigm."""
    text = f"{title} {abstract}".lower()
    if any(k in text for k in ["react", "reflexion", "reasoning", "reflection", "tree of thought", "deliberate"]):
        return {
            "name": "Reasoning & Reflection",
            "tag": "reasoning",
            "color": "lavender",
            "bg": "#EEF2FF",
            "border": "#818CF8",
            "text": "#4338CA"
        }
    elif any(k in text for k in ["tool", "toolformer", "gorilla", "api", "grounding", "retrieval", "self-rag"]):
        return {
            "name": "Tool & API Grounding",
            "tag": "tools",
            "color": "sage",
            "bg": "#ECFDF5",
            "border": "#34D399",
            "text": "#065F46"
        }
    elif any(k in text for k in ["chatdev", "metagpt", "autogen", "multi-agent", "coordination", "collaborative", "communicative"]):
        return {
            "name": "Multi-Agent Coordination",
            "tag": "multi-agent",
            "color": "amber",
            "bg": "#FEF3C7",
            "border": "#F59E0B",
            "text": "#92400E"
        }
    elif any(k in text for k in ["bench", "agentbench", "swe-bench", "evaluation", "interactive evaluation"]):
        return {
            "name": "Interactive Benchmarking",
            "tag": "benchmarks",
            "color": "rose",
            "bg": "#FFE4E6",
            "border": "#FB7185",
            "text": "#9F1239"
        }
    else:
        return {
            "name": "Embodied & Autonomous",
            "tag": "embodied",
            "color": "periwinkle",
            "bg": "#F0FDF4",
            "border": "#4ADE80",
            "text": "#166534"
        }


@app.get("/api/corpus")
def get_corpus():
    """Returns list of papers with rich schema metadata, figures, and extraction status."""
    if not os.path.exists(PDF_DIR):
        return {"papers": []}

    pdf_files = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    papers = []

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        paper_id = filename.split("_")[0]
        
        parsed_file = os.path.join(SCHEMAS_DIR, f"{paper_id}_parsed.json")
        schema_file = os.path.join(SCHEMAS_DIR, f"{paper_id}_schema.json")
        summary_file = os.path.join(SUMMARIES_DIR, f"{paper_id}_summary.json")

        meta = {
            "paper_id": paper_id,
            "filename": filename,
            "size_kb": round(os.path.getsize(pdf_path) / 1024, 1),
            "is_parsed": os.path.exists(parsed_file),
            "is_extracted": os.path.exists(schema_file),
            "is_summarized": os.path.exists(summary_file),
            "generation_mode": "UNKNOWN",
            "title": filename.replace(".pdf", "").replace("_", " "),
            "authors": [],
            "year": 2023,
            "core_problem": "",
            "methodology": "",
            "limitations": "",
            "paradigm": classify_paradigm(filename),
            "figures": []
        }

        if os.path.exists(schema_file):
            try:
                with open(schema_file, "r", encoding="utf-8") as f:
                    s_data = json.load(f)
                    meta["title"] = s_data.get("paper_title") or meta["title"]
                    meta["authors"] = s_data.get("authors") or []
                    meta["core_problem"] = s_data.get("core_problem") or ""
                    meta["methodology"] = s_data.get("methodology_summary") or ""
                    meta["limitations"] = s_data.get("explicit_limitations") or ""
                    meta["generation_mode"] = s_data.get("generation_mode") or "UNKNOWN"
                    meta["model_name"] = s_data.get("model_name")
                    meta["paradigm"] = classify_paradigm(meta["title"], meta["core_problem"])
            except Exception:
                pass

        paper_fig_dir = os.path.join(FIGURES_DIR, paper_id)
        if os.path.exists(paper_fig_dir):
            figs = [
                f"/static/figures/{paper_id}/{f}"
                for f in os.listdir(paper_fig_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            meta["figures"] = figs[:4]

        papers.append(meta)

    return {"papers": papers}


@app.get("/api/paper/{paper_id}")
def get_paper_detail(paper_id: str):
    """Returns granular sections, schema, and summary for deep inspection."""
    schema_file = os.path.join(SCHEMAS_DIR, f"{paper_id}_schema.json")
    parsed_file = os.path.join(SCHEMAS_DIR, f"{paper_id}_parsed.json")
    summary_file = os.path.join(SUMMARIES_DIR, f"{paper_id}_summary.json")

    result: Dict[str, Any] = {"paper_id": paper_id}

    if os.path.exists(schema_file):
        try:
            with open(schema_file, "r", encoding="utf-8") as f:
                result["schema"] = json.load(f)
        except Exception as e:
            result["schema_error"] = str(e)

    if os.path.exists(parsed_file):
        try:
            with open(parsed_file, "r", encoding="utf-8") as f:
                result["parsed"] = json.load(f)
        except Exception as e:
            result["parsed_error"] = str(e)

    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                result["summary"] = json.load(f)
        except Exception as e:
            result["summary_error"] = str(e)

    matches = glob.glob(os.path.join(PDF_DIR, f"{paper_id}*.pdf"))
    if matches:
        result["pdf_filename"] = os.path.basename(matches[0])

    return result


@app.get("/api/pdf/{filename}")
def stream_pdf(filename: str):
    """Streams PDF directly for in-browser rendering with highlighter provenance."""
    pdf_path = os.path.join(PDF_DIR, filename)
    if not os.path.exists(pdf_path):
        matches = glob.glob(os.path.join(PDF_DIR, f"{filename}*.pdf"))
        if matches:
            pdf_path = matches[0]
        else:
            raise HTTPException(status_code=404, detail="PDF not found")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=os.path.basename(pdf_path),
        headers={"Content-Disposition": "inline"}
    )


@app.get("/api/report")
def get_report():
    """Returns final synthesized Literature Review Report, Mermaid diagrams, and structured diagram_data."""
    report_path = os.path.join(REPORTS_DIR, "literature_review_report.md")
    if not os.path.exists(report_path):
        return {"content": "", "diagrams": {}, "diagram_data": {}}

    with open(report_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    diagrams = {}
    pattern = r"```mermaid\s*\n(.*?)\n```"
    blocks = re.findall(pattern, markdown_text, re.DOTALL)
    for block in blocks:
        id_match = re.search(r"%%\s*diagram_id:\s*([a-zA-Z0-9_\-]+)", block)
        if id_match:
            diagrams[id_match.group(1)] = block.strip()
        else:
            if "mindmap" in block or "paradigmMain" in block or "benchmarkMain" in block:
                diagrams.setdefault("themes_mindmap", block.strip())
            elif "Layer1" in block or "Layer2" in block or "research_gaps" in block:
                diagrams.setdefault("research_gaps_flowchart", block.strip())
            elif "rootStyle" in block or "paperCard" in block:
                diagrams.setdefault("methodology_flowchart", block.strip())

    # Load persisted structured diagram_data or generate dynamically from active review
    diag_data = {}
    diag_data_file = os.path.join(REPORTS_DIR, "diagram_data.json")
    if os.path.exists(diag_data_file):
        try:
            with open(diag_data_file, "r", encoding="utf-8") as f:
                diag_data = json.load(f)
        except Exception:
            pass

    if not diag_data:
        try:
            from src.agents.report_generator import build_dynamic_diagram_data
            from src.agents.extraction import PaperSchema
            active_ids = _get_active_corpus_ids()
            if active_ids:
                schema_files = [os.path.join(SCHEMAS_DIR, f"{pid}_schema.json") for pid in active_ids if os.path.exists(os.path.join(SCHEMAS_DIR, f"{pid}_schema.json"))]
            else:
                schema_files = sorted(glob.glob(os.path.join(SCHEMAS_DIR, "*_schema.json")))
            loaded_schemas = []
            for sf in schema_files:
                try:
                    with open(sf, "r", encoding="utf-8") as s_file:
                        loaded_schemas.append(PaperSchema(**json.load(s_file)))
                except Exception:
                    pass
            diag_data = build_dynamic_diagram_data(schemas=loaded_schemas)
        except Exception:
            diag_data = {}

    return {
        "content": markdown_text,
        "report": markdown_text,
        "diagrams": diagrams,
        "diagram_data": diag_data
    }


@app.get("/api/comparative")
def get_comparative_matrix():
    """Returns structured comparative analysis matrix."""
    comp_file = os.path.join(DATA_DIR, "comparative_analysis.json")
    if not os.path.exists(comp_file):
        return {}
    with open(comp_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/memory")
def get_global_memory():
    """Returns global cross-paper memory store."""
    mem_file = os.path.join(DATA_DIR, "global_memory_store.json")
    if not os.path.exists(mem_file):
        return {}
    with open(mem_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/screen")
def screen_topic(req: ScreeningRequest):
    """Runs Relevance Screening Agent on topic with recency bias and inclusion/exclusion criteria."""
    try:
        from src.ingestion.screening_agent import RelevanceScreeningAgent
        agent = RelevanceScreeningAgent(recency_weight=req.recency_weight)
        limit = req.max_results or req.max_candidates
        candidates = agent.search_arxiv(req.query, max_candidates=limit)
        if not candidates:
            return {
                "candidates": [],
                "papers": [],
                "results": [],
                "query": req.query,
                "inclusion_criteria": req.inclusion_criteria or "",
                "exclusion_criteria": req.exclusion_criteria or "",
                "prisma": {"total_identified": 0, "screened": 0, "included": 0, "maybe": 0, "excluded": 0}
            }
        ranked = agent.score_and_rank(req.query, candidates, recency_weight=req.recency_weight)
        evaluated = agent.evaluate_screening(
            ranked,
            user_query=req.query,
            inclusion_criteria=req.inclusion_criteria or "",
            exclusion_criteria=req.exclusion_criteria or ""
        )

        inc_count = sum(1 for c in evaluated if c.get("decision") == "Include")
        may_count = sum(1 for c in evaluated if c.get("decision") == "Maybe")
        exc_count = sum(1 for c in evaluated if c.get("decision") == "Exclude")

        prisma = {
            "total_identified": len(evaluated),
            "screened": len(evaluated),
            "included": inc_count,
            "maybe": may_count,
            "excluded": exc_count
        }

        # Cache screening state to disk
        screening_file = os.path.join(DATA_DIR, "screening_results.json")
        try:
            with open(screening_file, "w", encoding="utf-8") as f:
                json.dump({
                    "query": req.query,
                    "inclusion_criteria": req.inclusion_criteria or "",
                    "exclusion_criteria": req.exclusion_criteria or "",
                    "prisma": prisma,
                    "candidates": evaluated,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }, f, indent=2)
        except Exception:
            pass

        return {
            "candidates": evaluated,
            "papers": evaluated,
            "results": evaluated,
            "query": req.query,
            "inclusion_criteria": req.inclusion_criteria or "",
            "exclusion_criteria": req.exclusion_criteria or "",
            "prisma": prisma
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/save-screening")
def save_screening_api(req: SaveScreeningRequest):
    """Persists user-curated screening decisions, criteria, and reasons."""
    try:
        inc_count = sum(1 for c in req.candidates if c.get("decision") == "Include")
        may_count = sum(1 for c in req.candidates if c.get("decision") == "Maybe")
        exc_count = sum(1 for c in req.candidates if c.get("decision") == "Exclude")
        prisma = {
            "total_identified": len(req.candidates),
            "screened": len(req.candidates),
            "included": inc_count,
            "maybe": may_count,
            "excluded": exc_count
        }
        screening_file = os.path.join(DATA_DIR, "screening_results.json")
        data = {
            "query": req.query,
            "inclusion_criteria": req.inclusion_criteria or "",
            "exclusion_criteria": req.exclusion_criteria or "",
            "prisma": prisma,
            "candidates": req.candidates,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        with open(screening_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return {"status": "saved", "prisma": prisma, "count": len(req.candidates)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/screening")
def get_screening_api():
    """Loads active or persisted screening results and PRISMA statistics."""
    screening_file = os.path.join(DATA_DIR, "screening_results.json")
    if os.path.exists(screening_file):
        try:
            with open(screening_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "status": "empty",
        "query": "",
        "inclusion_criteria": "",
        "exclusion_criteria": "",
        "candidates": [],
        "prisma": {"total_identified": 0, "screened": 0, "included": 0, "maybe": 0, "excluded": 0}
    }


@app.post("/api/download-papers")
def download_selected(req: DownloadRequest):
    """Downloads approved candidate papers to data/pdf_corpus/ and updates active review."""
    try:
        from src.ingestion.screening_agent import RelevanceScreeningAgent
        agent = RelevanceScreeningAgent()
        saved = agent.download_papers(req.papers, output_dir=PDF_DIR)

        # Track active paper IDs and review query
        active_file = os.path.join(DATA_DIR, "active_review.json")
        paper_ids = [p.get("arxiv_id") or p.get("id") for p in req.papers if (p.get("arxiv_id") or p.get("id"))]
        
        # Determine active topic / query
        active_query = (req.query or "").strip()
        if not active_query:
            scr_file = os.path.join(DATA_DIR, "screening_results.json")
            if os.path.exists(scr_file):
                try:
                    with open(scr_file, "r", encoding="utf-8") as sf:
                        active_query = json.load(sf).get("query", "").strip()
                except Exception:
                    pass

        try:
            with open(active_file, "w", encoding="utf-8") as f:
                json.dump({
                    "active_paper_ids": paper_ids,
                    "query": active_query,
                    "count": len(paper_ids),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }, f, indent=2)
        except Exception:
            pass

        if active_query:
            aq_file = os.path.join(DATA_DIR, "active_query.json")
            try:
                with open(aq_file, "w", encoding="utf-8") as aqf:
                    json.dump({"query": active_query}, aqf, indent=2)
            except Exception:
                pass

        return {"downloaded": saved, "count": len(saved), "active_ids": paper_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/load-benchmark")
def load_benchmark_corpus_api():
    """Downloads the default 12 benchmark Agentic AI papers."""
    try:
        from src.ingestion.downloader import download_benchmark_corpus
        download_benchmark_corpus()
        return {"status": "success", "message": "12 Benchmark Papers Downloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _get_active_corpus_ids() -> List[str]:
    """Returns the paper IDs for PDFs currently present in PDF_DIR."""
    if not os.path.exists(PDF_DIR):
        return []
    pdf_files = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    return [os.path.basename(p).split("_")[0] for p in pdf_files]


@app.post("/api/clear-corpus")
def clear_corpus_api():
    """Clears workspace session markers without wiping historical schemas or reports."""
    for fpath in [
        os.path.join(DATA_DIR, "active_review.json"),
        os.path.join(DATA_DIR, "active_query.json"),
        os.path.join(DATA_DIR, "screening_results.json"),
        os.path.join(DATA_DIR, "gap_analysis.json"),
        os.path.join(REPORTS_DIR, "diagram_data.json"),
        os.path.join(REPORTS_DIR, "literature_review_report.md"),
    ]:
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass

    return {"status": "cleared", "message": "Workspace initialized for fresh topic exploration"}


@app.post("/api/run-agent/extraction")
def run_extraction_agent_api():
    """Runs PDF Parser and Schema Extraction Agent on all corpus papers.
    Skips papers that already have a schema file to avoid redundant LLM calls
    (each NVIDIA NIM call takes ~270s).
    """
    from src.ingestion.parser import parse_pdf_file
    from src.agents.extraction import run_extraction_agent

    pdf_files = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    os.makedirs(SCHEMAS_DIR, exist_ok=True)
    extracted_schemas = []
    skipped = []

    if not pdf_files:
        existing_schemas = glob.glob(os.path.join(SCHEMAS_DIR, "*_schema.json"))
        if existing_schemas:
            for s_path in existing_schemas:
                try:
                    with open(s_path, "r", encoding="utf-8") as f:
                        extracted_schemas.append(json.load(f))
                except Exception:
                    pass
            return {
                "status": "completed",
                "count": len(extracted_schemas),
                "skipped": len(extracted_schemas),
                "schemas": extracted_schemas,
            }
        raise HTTPException(status_code=400, detail="No PDF files in corpus to extract")

    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        paper_id = pdf_name.split("_")[0]
        schema_file = os.path.join(SCHEMAS_DIR, f"{paper_id}_schema.json")

        # Skip if schema already exists — LLM call not needed
        if os.path.exists(schema_file):
            skipped.append(paper_id)
            try:
                with open(schema_file, "r", encoding="utf-8") as f:
                    extracted_schemas.append(json.load(f))
            except Exception:
                pass
            continue

        parsed_doc = parse_pdf_file(pdf_path)
        parsed_json_path = os.path.join(SCHEMAS_DIR, f"{paper_id}_parsed.json")
        with open(parsed_json_path, "w", encoding="utf-8") as f:
            json.dump(parsed_doc, f, indent=2, ensure_ascii=False)

        schema = run_extraction_agent(parsed_json_path, output_dir=SCHEMAS_DIR)
        extracted_schemas.append(schema.dict() if hasattr(schema, "dict") else schema)
        time.sleep(1)

    return {
        "status": "completed",
        "count": len(extracted_schemas),
        "skipped": len(skipped),
        "schemas": extracted_schemas,
    }


@app.post("/api/run-agent/summarizer")
def run_summarizer_agent_api():
    """Runs Single-Paper Summarizer Agent on active corpus papers."""
    from src.agents.summarizer import run_summarizer_agent

    active_ids = _get_active_corpus_ids()
    schema_files = [
        os.path.join(SCHEMAS_DIR, f"{pid}_schema.json")
        for pid in active_ids
        if os.path.exists(os.path.join(SCHEMAS_DIR, f"{pid}_schema.json"))
    ]
    if not schema_files:
        # Fallback to all schemas if no PDFs in corpus
        schema_files = sorted(glob.glob(os.path.join(SCHEMAS_DIR, "*_schema.json")))

    if not schema_files:
        raise HTTPException(status_code=400, detail="No schemas found. Run extraction first.")

    os.makedirs(SUMMARIES_DIR, exist_ok=True)
    summaries = []

    for s_file in schema_files:
        try:
            summary = run_summarizer_agent(s_file, output_dir=SUMMARIES_DIR, use_llm=False)
            summaries.append(summary.dict() if hasattr(summary, "dict") else summary)
        except Exception as e:
            print(f"Summarizer note: {e}")

    return {"status": "completed", "count": len(summaries), "summaries": summaries}


@app.post("/api/run-agent/comparative")
def run_comparative_agent_api():
    """Runs Cross-Paper Comparative Analysis Agent for active corpus papers."""
    from src.agents.comparative import run_comparative_analysis
    try:
        active_ids = _get_active_corpus_ids()
        comp_analysis = run_comparative_analysis(target_paper_ids=active_ids if active_ids else None)
        return {"status": "completed", "comparative": comp_analysis.dict() if hasattr(comp_analysis, "dict") else comp_analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/run-agent/research-gap")
def run_research_gap_agent_api():
    """Runs Research Gap Identification Agent on active corpus papers."""
    from src.agents.research_gap import run_research_gap_agent
    from src.agents.extraction import PaperSchema

    active_ids = _get_active_corpus_ids()
    if active_ids:
        schema_files = [
            os.path.join(SCHEMAS_DIR, f"{pid}_schema.json")
            for pid in active_ids
            if os.path.exists(os.path.join(SCHEMAS_DIR, f"{pid}_schema.json"))
        ]
    else:
        schema_files = sorted(glob.glob(os.path.join(SCHEMAS_DIR, "*_schema.json")))

    schemas = []
    for s_file in schema_files:
        with open(s_file, "r", encoding="utf-8") as f:
            schemas.append(PaperSchema(**json.load(f)))

    if not schemas:
        raise HTTPException(status_code=400, detail="No schemas found for research gap analysis.")

    gap_analysis = run_research_gap_agent(schemas)
    gaps_dict = gap_analysis.dict() if hasattr(gap_analysis, "dict") else (gap_analysis.model_dump() if hasattr(gap_analysis, "model_dump") else gap_analysis)
    try:
        with open(os.path.join(DATA_DIR, "gap_analysis.json"), "w", encoding="utf-8") as gf:
            json.dump(gaps_dict, gf, indent=2)
    except Exception:
        pass
    return {"status": "completed", "gaps": gaps_dict}


@app.post("/api/run-agent/report")
def run_report_agent_api():
    """Compiles Final Literature Review Report & Diagrams strictly for active corpus papers."""
    from src.agents.report_generator import generate_literature_review
    from src.agents.extraction import PaperSchema
    from src.agents.summarizer import ComprehensivePaperSummary
    from src.agents.comparative import run_comparative_analysis
    from src.agents.research_gap import run_research_gap_agent

    active_ids = _get_active_corpus_ids()
    if active_ids:
        schema_files = [
            os.path.join(SCHEMAS_DIR, f"{pid}_schema.json")
            for pid in active_ids
            if os.path.exists(os.path.join(SCHEMAS_DIR, f"{pid}_schema.json"))
        ]
        summary_files = [
            os.path.join(SUMMARIES_DIR, f"{pid}_summary.json")
            for pid in active_ids
            if os.path.exists(os.path.join(SUMMARIES_DIR, f"{pid}_summary.json"))
        ]
    else:
        schema_files = sorted(glob.glob(os.path.join(SCHEMAS_DIR, "*_schema.json")))
        summary_files = sorted(glob.glob(os.path.join(SUMMARIES_DIR, "*_summary.json")))

    schemas = []
    for s_file in schema_files:
        with open(s_file, "r", encoding="utf-8") as f:
            schemas.append(PaperSchema(**json.load(f)))

    summaries = []
    for sum_file in summary_files:
        with open(sum_file, "r", encoding="utf-8") as f:
            summaries.append(ComprehensivePaperSummary(**json.load(f)))

    comp = run_comparative_analysis(target_paper_ids=active_ids if active_ids else None)
    
    # Use cached gap analysis from Stage 4 if available to avoid duplicate LLM invocation
    gaps = None
    gap_cache = os.path.join(DATA_DIR, "gap_analysis.json")
    if os.path.exists(gap_cache):
        try:
            with open(gap_cache, "r", encoding="utf-8") as gf:
                gaps = json.load(gf)
        except Exception:
            pass
    if not gaps:
        gaps = run_research_gap_agent(schemas)

    report = generate_literature_review(schemas, summaries, comp, gaps)

    # Persist structured diagram_data alongside report
    diag_data = report.diagram_data if hasattr(report, "diagram_data") and report.diagram_data else {}
    if diag_data:
        diag_file = os.path.join(REPORTS_DIR, "diagram_data.json")
        try:
            with open(diag_file, "w", encoding="utf-8") as df:
                json.dump(diag_data, df, indent=2, ensure_ascii=False)
        except Exception:
            pass

    return {
        "status": "completed",
        "title": report.title,
        "full_markdown": report.full_markdown,
        "content": report.full_markdown,
        "report": report.full_markdown,
        "diagrams": report.diagrams if hasattr(report, "diagrams") else {},
        "diagram_data": diag_data
    }



@app.post("/api/upload-pdf")
async def upload_pdf(request: Request, filename: Optional[str] = Query(None)):
    """Accepts a PDF upload as raw binary or multipart and stores it in the corpus directory."""
    os.makedirs(PDF_DIR, exist_ok=True)
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded_file = form.get("file")
        if not uploaded_file:
            raise HTTPException(status_code=400, detail="No file field in multipart request")
        fname = filename or uploaded_file.filename
        content = await uploaded_file.read()
    else:
        fname = filename or request.headers.get("x-filename") or f"uploaded_{int(time.time())}.pdf"
        content = await request.body()

    if not fname.lower().endswith(".pdf"):
        fname += ".pdf"

    dest = os.path.join(PDF_DIR, os.path.basename(fname))
    with open(dest, "wb") as f:
        f.write(content)
    return {"status": "ok", "filename": os.path.basename(fname), "size": len(content)}


# Serve static built frontend if available
if os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print("  THE SCHOLAR'S DIGITAL BINDER: Fast API & Web Server")
    print("  Serving frontend & APIs at: http://127.0.0.1:8000")
    print("=" * 70 + "\n")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
