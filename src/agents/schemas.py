"""
Consolidated schemas re-exporting canonical Pydantic models across agents.
Project: NLP-05 Agentic AI for Automated Research Paper Analysis
"""

from src.agents.extraction import PaperSchema
from src.agents.summarizer import ComprehensivePaperSummary
from src.agents.comparative import ComparativeAnalysis, MethodologyRow, LimitationTheme

__all__ = [
    "PaperSchema",
    "ComprehensivePaperSummary",
    "ComparativeAnalysis",
    "MethodologyRow",
    "LimitationTheme",
]
