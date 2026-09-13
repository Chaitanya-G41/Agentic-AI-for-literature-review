from pydantic import BaseModel, Field
from typing import List
import os
import re
import json
import google.generativeai as genai

class LiteratureReviewReport(BaseModel):
  title : str = Field(description ="Title of the report")
  generated_at : str = Field(description = "Timestamp at which report was generated")
  paper_Count : int = Field(default = 0, description = "Total number of papers reviewed")
  full_markdown : str = Field(description = "Complete markdown text of the report")

def generate_literature_review(schemas, summaries, comp_analysis, gap_analysis):
  md=""
  md += "# Automated Literature Review: Synthesis of Agentic AI Systems\n\n"
  md += f"**Corpus Coverage:** {len(schemas)} Benchmark Papers\n\n"
  
  
  
