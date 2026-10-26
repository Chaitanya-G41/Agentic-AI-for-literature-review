"""
Compatibility shim re-exporting LLMManager.
Project: NLP-05 Agentic AI for Automated Research Paper Analysis
"""

from src.agents.llm_manager import LLMManager, GeminiLLMManager

__all__ = ["LLMManager", "GeminiLLMManager"]
