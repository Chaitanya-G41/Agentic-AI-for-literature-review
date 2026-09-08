"""
Global LangGraph / Agentic Memory Store
Project: NLP-05 Agentic AI for Automated Research Paper Analysis
Provides persistent state and cross-agent data access for all paper schemas and summaries.
"""

import os
import json

class GlobalMemoryStore:
    """
    Persistent memory store for multi-agent literature review pipeline.
    Maintains schemas, grounded summaries, and paper metadata in data/global_memory_store.json.
    """
    def __init__(self, memory_file="data/global_memory_store.json"):
        self.memory_file = memory_file
        self.data = {
            "papers": {},      # paper_id -> {schema, summary, metadata}
            "last_updated": None
        }
        self.load()

    def load(self):
        """Loads memory state from disk if exists."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        self.data = json.loads(content)
            except Exception as e:
                print(f"   [WARN] Error loading memory store file: {e}")

    def save(self):
        """Saves memory state to disk."""
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        import datetime
        self.data["last_updated"] = datetime.datetime.now().isoformat()
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def register_paper_schema(self, paper_id, schema_dict):
        """Stores extracted PaperSchema for a paper."""
        if paper_id not in self.data["papers"]:
            self.data["papers"][paper_id] = {}
        self.data["papers"][paper_id]["schema"] = schema_dict
        self.save()

    def register_paper_summary(self, paper_id, summary_dict):
        """Stores comprehensive grounded summary for a paper."""
        if paper_id not in self.data["papers"]:
            self.data["papers"][paper_id] = {}
        self.data["papers"][paper_id]["summary"] = summary_dict
        self.save()

    def get_paper(self, paper_id):
        """Returns all registered data for a paper ID."""
        return self.data["papers"].get(paper_id, {})

    def get_summary(self, paper_id):
        """Returns summary dictionary for a paper ID."""
        paper = self.get_paper(paper_id)
        return paper.get("summary")

    def get_schema(self, paper_id):
        """Returns schema dictionary for a paper ID."""
        paper = self.get_paper(paper_id)
        return paper.get("schema")

    def get_all_paper_ids(self):
        """Returns list of all paper IDs in global memory store."""
        return list(self.data["papers"].keys())

    def get_all_summaries(self):
        """Returns dictionary of paper_id -> summary dict for downstream agents."""
        res = {}
        for pid, pdata in self.data["papers"].items():
            if "summary" in pdata:
                res[pid] = pdata["summary"]
        return res

    def get_all_schemas(self):
        """Returns dictionary of paper_id -> schema dict for downstream agents."""
        res = {}
        for pid, pdata in self.data["papers"].items():
            if "schema" in pdata:
                res[pid] = pdata["schema"]
        return res

    def register_comparative_analysis(self, analysis_dict):
        """Stores comparative analysis dictionary across papers."""
        self.data["comparative_analysis"] = analysis_dict
        self.save()

    def get_comparative_analysis(self):
        """Returns registered comparative analysis dictionary, if available."""
        return self.data.get("comparative_analysis")

# Global Singleton Instance Helper
_memory_instance = None

def get_global_memory_store():
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = GlobalMemoryStore()
    return _memory_instance
