from pydantic import BaseModel, Field
from typing import List
import os
import json
import google.generativeai as genai

class ResearchGap(BaseModel):
  gap_id: str = Field(description = "unique code")
  title : str = Field(description = "Title of the paper")
  category : str = Field(description = "Category of the gap")
  description : str = Field(description = "More on the content")
  affected_papers : List[str] = Field(default=[], description = "List of affected papers")

class ResearchGapAnalysis(BaseModel):
  explicit_gaps: List[ResearchGap] = Field(default=[], description = "List of author stated limitations")
  synthesized_gaps: List[ResearchGap] = Field(default=[], description = "List of cross paper synthesized research gaps")
  
def run_research_gap_agent(schemas):
  paper_context = ""
  for paper in schemas:
    paper_context += f"\nPaper Title: {paper.paper_title}\nLimitations: {paper.explicit_limitations}\n"
    
