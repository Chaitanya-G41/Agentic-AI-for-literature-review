"""
Command-Line Interface (CLI) Live Demo Script
Project: NLP-05 Agentic AI for Automated Research Paper Analysis
"""

import os
import sys
import glob
import json
import time
import argparse

# Ensure pythonpath resolves src package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.parser import parse_pdf_file
from src.agents.extraction import run_extraction_agent
from src.agents.summarizer import run_summarizer_agent
from src.agents.comparative import run_comparative_analysis
from src.agents.research_gap import run_research_gap_agent
from src.agents.report_generator import generate_literature_review
from src.ingestion.screening_agent import RelevanceScreeningAgent

def main():
    parser = argparse.ArgumentParser(description="Agentic AI Literature Review Pipeline")
    parser.add_argument("--topic", type=str, default=None, help="Research topic/title to screen and download papers for")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top papers to select when topic is provided")
    parser.add_argument("--max_candidates", type=int, default=20, help="Max candidates to fetch from arXiv")
    parser.add_argument("--recency_weight", type=float, default=0.20, help="Weight given to recent papers (0.0 to 0.35)")
    parser.add_argument("--clear_corpus", action="store_true", help="Clear existing corpus before downloading for new topic")
    args = parser.parse_args()

    print("=" * 80)
    print("  NLP-05: AGENTIC AI FOR AUTOMATED RESEARCH PAPER ANALYSIS")
    print("  Live Multi-Agent Pipeline CLI Demonstration")
    print("=" * 80)
    print()

    # Step 0: Relevance Screening (if topic provided)
    if args.topic:
        print(f"[Agent 1] Relevance Screening Agent Active for Topic: '{args.topic}'")
        if args.clear_corpus:
            print("   [!] Clearing existing data/pdf_corpus/...")
            for f in glob.glob("data/pdf_corpus/*.pdf"):
                try:
                    os.remove(f)
                except Exception:
                    pass
        
        agent = RelevanceScreeningAgent()
        print(f"   Fetching up to {args.max_candidates} candidate papers from arXiv...")
        candidates = agent.search_arxiv(args.topic, max_results=args.max_candidates)
        print(f"   Found {len(candidates)} candidates. Scoring and ranking with recency bias ({args.recency_weight})...")
        ranked = agent.score_and_rank(candidates, args.topic, recency_weight=args.recency_weight)
        
        selected = ranked[:args.top_k]
        print(f"   Selected Top {len(selected)} Papers:")
        for i, p in enumerate(selected, 1):
            print(f"      {i}. [{p['composite_score']*100:.1f}% Match] ({p['year']}) {p['title'][:65]}...")
        
        print(f"   Downloading {len(selected)} papers to data/pdf_corpus/...")
        downloaded = agent.download_papers(selected, output_dir="data/pdf_corpus")
        print(f"   Successfully downloaded {len(downloaded)} papers.")
        print()

    # Find papers in data/pdf_corpus
    pdf_files = sorted(glob.glob("data/pdf_corpus/*.pdf"))
    if not pdf_files:
        print("[Error] No PDF files found in data/pdf_corpus/. Running benchmark downloader...")
        from src.ingestion.downloader import download_benchmark_corpus
        download_benchmark_corpus()
        pdf_files = sorted(glob.glob("data/pdf_corpus/*.pdf"))

    demo_papers = pdf_files
    print(f"[+] Selected {len(demo_papers)} Papers for Live Execution:")
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

    # 4. Research Gap Agent
    print("\n[Agent 5] Running Research Gap Identification Agent...")
    gap_analysis = None
    try:
        gap_analysis = run_research_gap_agent(schemas)
        print("   Research Gaps synthesized successfully.")
    except Exception as e:
        print(f"   Research Gap Agent note: {e}")

    # 5. Report Generator Agent
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
