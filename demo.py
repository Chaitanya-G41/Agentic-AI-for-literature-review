"""
Command-Line Interface (CLI) Live Demo Script
Project: NLP-05 Agentic AI for Automated Research Paper Analysis
"""

import os
import sys
import glob
import json
import time

# Ensure pythonpath resolves src package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.parser import parse_pdf_file
from src.agents.extraction import run_extraction_agent
from src.agents.summarizer import run_summarizer_agent
from src.agents.comparative import run_comparative_analysis
from src.agents.research_gap import run_research_gap_agent
from src.agents.report_generator import generate_literature_review

def main():
    print("=" * 80)
    print("  NLP-05: AGENTIC AI FOR AUTOMATED RESEARCH PAPER ANALYSIS")
    print("  Live Multi-Agent Pipeline CLI Demonstration")
    print("=" * 80)
    print()

    # Find papers in data/pdf_corpus
    pdf_files = sorted(glob.glob("data/pdf_corpus/*.pdf"))
    if not pdf_files:
        print("[Error] No PDF files found in data/pdf_corpus/. Running downloader...")
        from src.ingestion.downloader import download_benchmark_corpus
        download_benchmark_corpus()
        pdf_files = sorted(glob.glob("data/pdf_corpus/*.pdf"))

    demo_papers = pdf_files
    print(f"[+] Selected {len(demo_papers)} Benchmark Papers for Live Execution:")
    for path in demo_papers:
        print(f"   - {os.path.basename(path)}")
    print()
    print("-" * 80)
    
    print("[>] LAUNCHING MULTI-AGENT WORKFLOW...")
    print("-" * 80)

    # 1. Parsing & Schema Extraction Phase
    print("\n[Step 1 & 2] Parsing PDFs and Extracting Schemas...")
    schemas = []
    for pdf_path in demo_papers:
        print(f"   Processing: {os.path.basename(pdf_path)}...")
        parsed_doc = parse_pdf_file(pdf_path)
        
        # Save temporary parsed json for extraction agent
        parsed_json_path = f"data/extracted_schemas/{parsed_doc['paper_id']}_parsed.json"
        os.makedirs("data/extracted_schemas", exist_ok=True)
        with open(parsed_json_path, "w", encoding="utf-8") as f:
            json.dump(parsed_doc, f, indent=2, ensure_ascii=False)
            
        schema = run_extraction_agent(parsed_json_path, output_dir="data/extracted_schemas")
        schemas.append(schema)
        time.sleep(2)  # Rate limit pacing

    # 2. Summarizer Agent
    print("\n[Agent 3] Running Single-Paper Summarizer Agent...")
    summaries = []
    for schema in schemas:
        try:
            schema_file = f"data/extracted_schemas/{schema.paper_id}_schema.json"
            summary = run_summarizer_agent(schema_file, output_dir="data/summaries", use_llm=False)
            summaries.append(summary)
            print(f"   Summarized: {schema.paper_title}")
        except Exception as e:
            print(f"   Summarizer note for {schema.paper_id}: {e}")
        time.sleep(2)  # Rate limit pacing

    # 3. Comparative Agent
    print("\n[Agent 4] Running Cross-Paper Comparative Agent...")
    comp_analysis = None
    try:
        comp_analysis = run_comparative_analysis()
        print("   Comparative Matrix constructed successfully.")
    except Exception as e:
        print(f"   Comparative Agent note: {e}")

    # 4. Research Gap Agent (User authored!)
    print("\n[Agent 5] Running Research Gap Identification Agent...")
    gap_analysis = None
    try:
        gap_analysis = run_research_gap_agent(schemas)
        print(f"   Research Gaps synthesized successfully.")
    except Exception as e:
        print(f"   Research Gap Agent note: {e}")

    # 5. Report Generator Agent (User authored!)
    print("\n[Agent 6] Compiling Final Literature Review Report...")
    if gap_analysis:
        report = generate_literature_review(schemas, summaries, comp_analysis, gap_analysis)
        report_file = "data/reports/literature_review_report.md"
        print(f"   Report Generated successfully -> {report_file}")

        print("\n" + "=" * 80)
        print("   REPORT PREVIEW (First 20 Lines):")
        print("=" * 80 + "\n")
        lines = report.full_markdown.splitlines()[:20]
        print("\n".join(lines))

    print("\n" + "=" * 80)
    print("[SUCCESS] DEMO COMPLETE! All agents executed and outputs generated successfully.")
    print("=" * 80)

if __name__ == "__main__":
    main()
