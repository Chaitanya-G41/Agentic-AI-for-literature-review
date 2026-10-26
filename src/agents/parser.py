"""
PDF Parser Agent for Document Grounding and Layout Analysis.
Project: NLP-05 Agentic AI for Automated Research Paper Analysis
"""

import os
import json
from typing import Dict, Any, List, Optional
from src.ingestion.parser import parse_pdf_file


class ParsedSection:
    def __init__(self, title: str, page_number: int, content: str):
        self.title = title
        self.page_number = page_number
        self.content = content


class ParsedPDFData:
    """Wrapper around raw parsed PDF JSON providing object-attribute access."""
    def __init__(self, raw: Dict[str, Any]):
        self.raw_dict = raw
        self.paper_id = raw.get("paper_id", "")
        self.file_name = raw.get("file_name", "")
        self.parser_used = raw.get("parser_used", "PyMuPDF_Layout_Engine")
        self.total_pages = raw.get("total_pages", 1)
        self.full_text = raw.get("full_text", "")
        self.figures_count = raw.get("figures_count", 0)
        self.vector_graphs_count = raw.get("vector_graphs_count", 0)
        self.tables_count = raw.get("tables_count", 0)
        self.tables = raw.get("tables", [])

        # Reconstruct page-by-page text mapping (int page_number -> str)
        self.pages: Dict[int, str] = {}
        chunks = raw.get("chunks", [])
        sections_map = {}
        for c in chunks:
            p_num = c.get("page_number", 1)
            text = c.get("text", "")
            if p_num not in self.pages:
                self.pages[p_num] = ""
            self.pages[p_num] += text + "\n\n"

            sec_title = c.get("section", "General Content")
            key = (sec_title, p_num)
            if key not in sections_map:
                sections_map[key] = []
            sections_map[key].append(text)

        if not self.pages:
            self.pages[1] = self.full_text

        # Reconstruct list of section objects with .title, .page_number, .content
        self.sections: List[ParsedSection] = []
        for (sec_title, p_num), texts in sections_map.items():
            self.sections.append(ParsedSection(title=sec_title, page_number=p_num, content="\n".join(texts)))


class PDFParserAgent:
    """Agent interface for PDF layout and text parsing."""
    def __init__(self, output_dir: str = "data/extracted_schemas"):
        self.output_dir = output_dir

    def load_parsed_data(self, pdf_filename: str) -> Optional[ParsedPDFData]:
        pid = pdf_filename.split("_")[0].replace(".pdf", "")
        parsed_path = os.path.join(self.output_dir, f"{pid}_parsed.json")
        if not os.path.exists(parsed_path) and os.path.exists(self.output_dir):
            candidates = [os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir) if f.startswith(pid) and f.endswith("_parsed.json")]
            if candidates:
                parsed_path = candidates[0]

        if os.path.exists(parsed_path):
            try:
                with open(parsed_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return ParsedPDFData(data)
            except Exception:
                return None
        return None

    def parse(self, pdf_path: str) -> ParsedPDFData:
        raw_doc = parse_pdf_file(pdf_path)
        pid = raw_doc.get("paper_id", os.path.basename(pdf_path).split("_")[0])
        os.makedirs(self.output_dir, exist_ok=True)
        out_filepath = os.path.join(self.output_dir, f"{pid}_parsed.json")
        with open(out_filepath, "w", encoding="utf-8") as f:
            json.dump(raw_doc, f, indent=2, ensure_ascii=False)
        return ParsedPDFData(raw_doc)
