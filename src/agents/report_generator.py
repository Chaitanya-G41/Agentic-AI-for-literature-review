from pydantic import BaseModel, Field
from typing import List
import os
import re
import json
import google.generativeai as genai
from datetime import datetime

class LiteratureReviewReport(BaseModel):
    title: str = Field(description="Title of the report")
    generated_at: str = Field(description="Timestamp at which report was generated")
    paper_count: int = Field(default=0, description="Total number of papers reviewed")
    full_markdown: str = Field(description="Complete markdown text of the report")

def generate_literature_review(schemas, summaries, comp_analysis, gap_analysis):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = ""
    md += "# Automated Literature Review: Synthesis of Agentic AI Systems\n\n"
    md += f"**Corpus Coverage:** {len(schemas)} Benchmark Papers\n\n"
    md += "## 1. Corpus Overview\n\n"
    md += " | Paper id | Title | Venue | \n"
    md += " | -------- | -----  | ---- |\n"
    for paper in schemas:
        md += f"| {paper.paper_id} | {paper.paper_title} | {paper.venue} | \n"  

    md += "\n## 2. Research Gaps Identified\n\n"
    if gap_analysis and hasattr(gap_analysis, "explicit_gaps"):
        for gap in gap_analysis.explicit_gaps:
            md += f"**{gap.title}:** {gap.description}\n\n"

    output_dir = "data/reports"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "literature_review_report.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md)

    return LiteratureReviewReport(
        title="Automated Literature Review: Synthesis of Agentic AI Systems",
        generated_at=generated_at,
        paper_count=len(schemas),
        full_markdown=md
    )
