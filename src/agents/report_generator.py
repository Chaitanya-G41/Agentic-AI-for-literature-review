"""
Agent 6: Report Generator Agent (Final Synthesis via [One Prompt ONLY])
Project: NLP-05 Agentic AI for Automated Research Paper Analysis
"""

import os
import json
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional

from src.agents.llm_manager import LLMManager

class LiteratureReviewReport(BaseModel):
    title: str = Field(description="Title of the report")
    generated_at: str = Field(description="Timestamp at which report was generated")
    paper_count: int = Field(default=0, description="Total number of papers reviewed")
    full_markdown: str = Field(description="Complete markdown text of the report")

def generate_literature_review(schemas, summaries, comp_analysis, gap_analysis, llm_manager=None) -> LiteratureReviewReport:
    """
    Compiles the final literature review report into a publication-ready markdown file.
    Executes exactly [ONE PROMPT ONLY] to the LLM to synthesize the complete review,
    grounded in the Pydantic comparison matrix, summaries, and research gaps.
    """
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    llm = llm_manager or LLMManager()

    # 1. Build Overview Table
    overview_table = "| Paper ID | Title | Venue | Mode |\n| :--- | :--- | :--- | :--- |\n"
    for p in schemas:
        pid = getattr(p, "paper_id", "")
        title = getattr(p, "paper_title", "")
        venue = getattr(p, "venue", "arXiv / Peer-Reviewed")
        mode = getattr(p, "generation_mode", "LLM")
        overview_table += f"| {pid} | {title} | {venue} | {mode} |\n"

    # 2. Extract Gaps summary
    gaps_text = ""
    if gap_analysis:
        for g in getattr(gap_analysis, "explicit_gaps", []):
            gaps_text += f"- **[{g.gap_id}] {g.title}** ({g.category}): {g.description}\n"
        for g in getattr(gap_analysis, "synthesized_gaps", []):
            gaps_text += f"- **[{g.gap_id}] {g.title}** ({g.category}): {g.description}\n"

    # 3. Extract Comparison Table
    comp_table = getattr(comp_analysis, "markdown_comparison_report", "") if comp_analysis else ""

    # [ONE PROMPT ONLY] Synthesis Call
    report_markdown = None
    if llm.keys:
        print("   [Agent 6] Synthesizing Final Literature Review Draft [ONE PROMPT ONLY] via LLM...")
        prompt = f"""You are a distinguished Senior AI Researcher.
Synthesize a comprehensive, publication-grade Literature Review Report titled:
'Automated Literature Review: Synthesis of Agentic AI Systems'

Analyze the following {len(schemas)} papers based on their structured extractions:

=== CORPUS OVERVIEW ===
{overview_table}

=== COMPARATIVE MATRIX & FINDINGS ===
{comp_table[:3000] if comp_table else "Comparative data across methods and benchmarks."}

=== IDENTIFIED RESEARCH GAPS ===
{gaps_text}

FORMAT REQUIREMENTS:
Draft a complete academic literature review in Markdown with these exact sections:
# Automated Literature Review: Synthesis of Agentic AI Systems
## Executive Summary
## 1. Corpus Overview & Scope
(Include the markdown table provided above)
## 2. Architectural Paradigms & Methodological Analysis
(Deep comparative narrative of the techniques, reasoning frameworks, and tool usage)
## 3. Empirical Benchmark Analysis
(Analysis of HotpotQA, HumanEval, ALFWorld, etc., comparing performance and trade-offs)
## 4. Cross-Paper Research Gaps & Unexplored Frontiers
(Synthesize systemic bottlenecks, conflicting evidence, and open problems)
## 5. Strategic Roadmap & Future Directions
## 6. References & Citation Anchors
"""
        sys_inst = "You are an expert NLP researcher. Write an exhaustive, rigorous, publication-quality literature survey in Markdown."
        llm_text = llm.generate_text(prompt, system_instruction=sys_inst)
        if llm_text and len(llm_text) > 400:
            report_markdown = llm_text
            print("   [GENUINE-LLM] Literature Review report synthesized successfully via [ONE PROMPT ONLY]!")

    # Fallback template if offline or LLM unavailable
    if not report_markdown:
        print("   [HEURISTIC-FALLBACK] Assembling Literature Review from Grounded Template...")
        report_markdown = f"""# Automated Literature Review: Synthesis of Agentic AI Systems

**Corpus Coverage:** {len(schemas)} Benchmark Papers  
**Generated At:** {generated_at}

## 1. Corpus Overview

{overview_table}

## 2. Comparative Analysis Matrix

{comp_table}

## 3. Synthesized Research Gaps

{gaps_text}

## 4. Conclusion & Future Directions
This literature review surveyed foundational frameworks for reasoning, tool use, and multi-agent coordination. Future work must address memory state persistence, multi-turn token overhead, and robust evaluation benchmarks.
"""

    output_dir = "data/reports"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "literature_review_report.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_markdown)

    return LiteratureReviewReport(
        title="Automated Literature Review: Synthesis of Agentic AI Systems",
        generated_at=generated_at,
        paper_count=len(schemas),
        full_markdown=report_markdown
    )
