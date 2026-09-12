"""parser.py
Step 2 Stage 1: PDF Layout Parser & Section Chunk Indexer (GROBID + PyPDF Fallback)
Project: NLP-05 Agentic AI for Automated Research Paper Analysis
"""

import os
import json
import re
from pypdf import PdfReader
try:
    from src.ingestion.grobid_parser import check_grobid_server, parse_pdf_with_grobid
except ModuleNotFoundError:
    from grobid_parser import check_grobid_server, parse_pdf_with_grobid

def parse_pdf_file(filepath, use_grobid_if_available=True):
    """
    Parses a PDF file into structured paragraph chunks with page numbers and section labels.
    Attempts GROBID TEI-XML extraction first if service is active, otherwise uses PyPDF layout parsing fallback.
    """
    if use_grobid_if_available and check_grobid_server():
        print(f"[GROBID Parser] Parsing {os.path.basename(filepath)} via GROBID REST Service...")
        grobid_result = parse_pdf_with_grobid(filepath)
        if grobid_result:
            return grobid_result

    # Fallback layout parser using PyPDF
    reader = PdfReader(filepath)
    filename = os.path.basename(filepath)
    paper_id = filename.split("_")[0]
    
    extracted_chunks = []
    full_text_list = []
    
    current_section = "Abstract / Introduction"

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        lines = text.split("\n")

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Section Header Detection Heuristics
            if re.match(r'^(?:[0-9IVX]+\.|\b)(?:\s*)(Abstract|Introduction|Related Work|Methodology|Proposed Approach|System Architecture|Experiments|Results|Discussion|Conclusion|Limitations|Future Work)', line_str, re.IGNORECASE):
                current_section = line_str

            chunk_item = {
                "paper_id": paper_id,
                "file_name": filename,
                "page_number": page_num,
                "section": current_section,
                "text": line_str
            }
            extracted_chunks.append(chunk_item)
            full_text_list.append(line_str)

    parsed_document = {
        "paper_id": paper_id,
        "file_name": filename,
        "parser_used": "PyPDF_Layout_Fallback",
        "total_pages": len(reader.pages),
        "full_text": "\n".join(full_text_list),
        "chunks": extracted_chunks
    }
    return parsed_document

def process_pdf_directory(input_dir="data/pdf_corpus", output_dir="data/extracted_schemas"):
    """Processes all PDFs in input_dir and saves parsed JSON files to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} not found. Please run downloader.py first!")
        return

    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    print("==================================================")
    print("Step 2 Stage 1: Running PDF Layout Parser")
    print(f"Found {len(pdf_files)} PDF files in {input_dir}")
    print("==================================================\n")

    parsed_count = 0
    for idx, filename in enumerate(pdf_files, start=1):
        filepath = os.path.join(input_dir, filename)
        out_filename = filename.replace(".pdf", "_parsed.json")
        out_filepath = os.path.join(output_dir, out_filename)

        print(f"[{idx}/{len(pdf_files)}] Parsing: {filename}...")
        try:
            parsed_doc = parse_pdf_file(filepath)
            with open(out_filepath, "w", encoding="utf-8") as f:
                json.dump(parsed_doc, f, indent=2, ensure_ascii=False)
            print(f"   Saved Parsed JSON: {out_filename} ({parsed_doc['total_pages']} pages, {len(parsed_doc['chunks'])} chunks)")
            parsed_count += 1
        except Exception as e:
            print(f"   Failed to parse {filename}: {e}")

    print("\n==================================================")
    print(f"Stage 1 Parsing Complete: {parsed_count}/{len(pdf_files)} JSONs written to {output_dir}")
    print("==================================================")

if __name__ == "__main__":
    process_pdf_directory()
