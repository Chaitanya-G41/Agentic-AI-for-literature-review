from pydantic import BaseModel, Field
from typing import List
import os
import re
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
    paper_context += f"\nPaper Title: {paper.paper_title}\n Core Problem: {paper.core_problem}\n Methadology: {paper.methadology_summar}\nLimitations: {paper.explicit_limitations}\n"

  api_key = os.environ.get("GEMINI_API_KEY")
  if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""You are an AI Research Gap Agent. Here is the paper context {paper_context}. Synthesize 3 to 5 research gaps and return a JSON OBJECT WITH KEYS "explicit_gaps" and "synthesized_gaps". """
    response = model.generate_content(prompt)  
    clean_text = re.sub(r'```json|```', '', response.text).strip()
    data_dict = json.loads(clean_text)
    return ResearchGapAnalysis(**data_dict)
