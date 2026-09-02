# Agentic AI for Automated Research Paper Analysis and Literature Review (NLP-05)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Agentic_Framework-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An autonomous multi-agent system designed to automate multi-paper literature synthesis, paper analysis, cross-paper comparison table generation, research gap identification, and citation-anchored survey drafting.

**Project ID:** NLP-05 (NLP Interest Group, AY: 2026-27)  
**Institution:** Department of Computer Science and Engineering, R.V. College of Engineering  
**Faculty Mentors:** Dr. Sagar B M (Faculty Mentor) | Prof. Kavi Narayana Murthy | Dr. Chethana R Murthy  

---

## 🌟 Key Features & Architecture

1. **Automated Paper Collection (Step 1):** Downloads and enriches research paper PDFs via `arXiv API`, `Semantic Scholar API`, and `Unpaywall API`.
2. **2-Stage Ingestion & Parsing Pipeline (Step 2):** Uses `GROBID TEI XML` layout parsing with fallback to `pypdf`/`Marker OCR` for 2-column layouts, LaTeX formulas, and tables.
3. **Structured Semantic Extraction:** Maps raw section text into validated Pydantic `PaperSchema` JSON files via `Gemini 2.0 Flash` / `Groq Llama-3.3 70B`.
4. **LangGraph Multi-Agent Orchestration:** 5 specialized agents (*Extraction, Summarizer, Comparison, Gap Identification, and Report Writer*) operating on a centralized `LiteratureReviewState`.
5. **Zero-Hallucination Citation Guardrail:** Page-level chunk vector matching (`cosine similarity >= 0.85`) to ensure 100% citation provenance.
6. **Interactive Streamlit Dashboard:** Split-screen UI with live agent progress indicators, comparison matrix viewer, and single-click `.md` / `.docx` export.

---

## 📁 Repository Structure

```
Agentic-AI-for-literature-review/
│── README.md                  # Project Documentation
│── requirements.txt           # Python Dependencies
│── app.py                     # Interactive Streamlit Web App
│── data/
│   ├── pdf_corpus/            # 16 Benchmark Research Paper PDFs
│   └── extracted_schemas/     # Extracted Pydantic Paper JSONs
└── src/
    ├── __init__.py
    ├── ingestion/             # Step 1: Automated Paper Collection & PDF Parser
    │   ├── __init__.py
    │   ├── downloader.py      # arXiv API automated fetcher
    │   └── parser.py          # 2-Stage PDF layout parser
    ├── agents/                # Step 2: Semantic Schema Extraction Agent
    │   ├── __init__.py
    │   └── extraction.py      # Pydantic schema extraction via Gemini Flash
    └── utils/                 # Vector & Cosine Similarity Helpers
        ├── __init__.py
        └── helpers.py
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Chaitanya-G41/Agentic-AI-for-literature-review.git
cd Agentic-AI-for-literature-review
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Set API Keys
Set your Gemini API key (Free Tier via Google AI Studio):
```bash
# On Windows PowerShell:
$env:GEMINI_API_KEY="your_api_key_here"
# On Linux/macOS:
export GEMINI_API_KEY="your_api_key_here"
```

### 4. Step 1: Download Benchmark Research Corpus
```bash
python src/ingestion/downloader.py
```

### 5. Step 2: Run PDF Parsing & Schema Extraction
```bash
python src/ingestion/parser.py
```

### 6. Run the Interactive Streamlit Web App
```bash
streamlit run app.py
```

---

## 📊 Benchmark Research Corpus (16 Papers)

The initial PoC dataset spans seminal papers in **Agentic AI & Multi-Agent Systems** across diverse publisher layout formats:
- **ACL Anthology (2-Column):** *ChatDev*
- **IEEE (2-Column):** *Med-Agents*
- **NeurIPS / ICLR (1-Column):** *ReAct, Reflexion, Toolformer, MetaGPT, Self-RAG, AgentBench, Tree of Thoughts, Voyager, SWE-bench*
- **Nature (2-Column):** *ChemCrow*
- **Technical Reports:** *AutoGen, CrewAI, LangGraph, Gorilla*

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
