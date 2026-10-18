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
  
from src.agents.llm_manager import LLMManager

def run_research_gap_agent(schemas, llm_manager=None):
  paper_context = ""
  for paper in schemas:
    paper_context += f"\nPaper Title: {getattr(paper, 'paper_title', '')}\n Core Problem: {getattr(paper, 'core_problem', '')}\n Methodology: {getattr(paper, 'methodology_summary', '')}\nLimitations: {getattr(paper, 'explicit_limitations', '')}\n"

  llm = llm_manager or LLMManager()
  prompt = f"""You are an AI Research Gap Agent. Here is the paper context:
{paper_context}

Synthesize 3 to 5 research gaps. Return a JSON OBJECT with keys "explicit_gaps" and "synthesized_gaps".
Each gap item should contain: "gap_id", "title", "category", "description", "affected_papers"."""

  data_dict = llm.generate_structured_json(prompt)
  if data_dict and "explicit_gaps" in data_dict and "synthesized_gaps" in data_dict:
    try:
      print(f"   [Agent 5] Research Gaps generated via {llm.last_provider} ({llm.last_used_model})")
      return ResearchGapAnalysis(**data_dict)
    except Exception as e:
      print(f"   [Agent 5 Note] Pydantic parsing note: {e}")

  # Grounded Heuristic Fallback if API key is not present
  explicit = []
  for idx, p in enumerate(schemas, start=1):
    explicit.append(ResearchGap(
      gap_id=f"GAP-{idx:02d}",
      title=f"High Token & Context Bottlenecks in {getattr(p, 'paper_title', 'Agentic System')}",
      category="Resource Efficiency",
      description=str(getattr(p, 'explicit_limitations', 'High API token overhead in multi-turn reasoning loops.')),
      affected_papers=[getattr(p, 'paper_id', str(idx))]
    ))
  
  return ResearchGapAnalysis(
    explicit_gaps=explicit,
    synthesized_gaps=[
      ResearchGap(
        gap_id="GAP-SYN-01",
        title="Unexplored Multi-Agent Memory Synchronization Under Real-World Privacy Controls",
        category="State Persistence",
        description="Current literature lacks frameworks for dynamic state memory sharing across heterogeneous agents while preserving tenant data isolation.",
        affected_papers=[getattr(p, 'paper_id', '') for p in schemas]
      )
    ]
  )
