# Agentic AI for Automated Research Paper Analysis and Literature Review (NLP-05)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Agentic_Framework-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Parser](https://img.shields.io/badge/PDF_Parser-GROBID_TEI_XML-green.svg)](https://github.com/kermitt2/grobid)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An autonomous multi-agent system designed to automate multi-paper literature synthesis, paper analysis, cross-paper comparison table generation, research gap identification, and citation-anchored survey drafting.

**Project ID:** NLP-05 (NLP Interest Group, AY: 2026-27)  
**Institution:** Department of Computer Science and Engineering, R.V. College of Engineering  
**Faculty Mentors:** Dr. Sagar B M (Faculty Mentor) | Prof. Kavi Narayana Murthy | Dr. Chethana R Murthy  

---

## 🌟 Key Features & Architecture

1. **Automated Paper Collection (Step 1):** Downloads and enriches research paper PDFs via `arXiv API`.
2. **GROBID TEI-XML & PyPDF Hybrid Parsing (Step 2 Stage 1):** Uses `GROBID TEI-XML` REST service (`src/ingestion/grobid_parser.py`) for extracting title, abstract, section hierarchy (`<div>`, `<head>`, `<p>`), and citations, with automatic fallback to PyPDF layout parsing.
3. **Structured Semantic Schema Extraction (Step 2 Stage 2):** Maps raw section text into validated Pydantic `PaperSchema` JSON files via `Gemini 2.0 Flash` or heuristic grounding.
4. **LangGraph Multi-Agent Orchestration:** Stateful multi-agent pipeline (*Extraction, Summarizer, Comparator, Gap Identification, and Report Writer*) operating on a centralized state graph.
5. **Zero-Hallucination Citation Guardrail:** Vector embedding cosine similarity matching (`similarity >= 0.85`) to guarantee page-level provenance.
6. **Command-Line & Web Dashboard Interface:** Includes both a lightweight terminal script (`demo.py`) and an interactive split-screen web interface (`app.py`).

---

## 📁 Repository Structure

```
Agentic-AI-for-literature-review/
│── README.md                  # Project Documentation
│── requirements.txt           # Python Dependencies
│── app.py                     # Streamlit Web Application
│── demo.py                    # Command-Line CLI Live Demo Script
│── data/
│   ├── pdf_corpus/            # Downloaded Benchmark Research Paper PDFs
│   └── extracted_schemas/     # Extracted Pydantic Paper JSONs & Parsed Layouts
└── src/
    ├── ingestion/             # Ingestion & PDF Parser Module
    │   ├── downloader.py      # Automated arXiv Paper Downloader (Step 1)
    │   ├── grobid_parser.py   # GROBID TEI-XML REST Service Client
    │   └── parser.py          # 2-Tier Hybrid Layout Parser (Step 2 Stage 1)
    └── agents/                # Semantic Extraction & Agent Workflows
        └── extraction.py      # Pydantic Schema Extraction Agent (Step 2 Stage 2)
```

---

## ⚙️ Setting Up GROBID (Optional but Recommended)

GROBID provides structural TEI-XML extraction for multi-column academic paper PDFs.

### Option A: Using Docker (Recommended)
```bash
docker run --rm --init -p 8070:8070 lfoppiano/grobid:0.8.0
```

### Option B: Local Java Service
```bash
./gradlew run
```
*Note: If the GROBID server (`http://localhost:8070`) is not running, the system automatically falls back to PyPDF layout parsing.*

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

### 3. Set API Keys (Optional for Gemini LLM Extraction)
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

### 5. Step 2: Run PDF Layout Parser & Schema Extraction
```bash
python src/ingestion/parser.py
```

### 6. Run the CLI Demo (No Web Server Needed)
```bash
python demo.py
```

### 7. Run the Streamlit Web Application
```bash
streamlit run app.py
```

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
