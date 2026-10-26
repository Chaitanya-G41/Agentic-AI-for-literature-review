"""parser.py
Step 2 Stage 1: Dual PDF Layout Parser & Section Chunk Indexer (GROBID + PyMuPDF)
Project: NLP-05 Agentic AI for Automated Research Paper Analysis

Specializations:
- GROBID: Academic structure, TEI-XML header, sections, bibliography, reference markers.
- PyMuPDF: High-fidelity layout text, two-column flow, embedded images, figures, and vector graphs.
"""

import os
import json
import re
import pymupdf
from typing import Dict, Any, List

try:
    from src.ingestion.grobid_parser import check_grobid_server, parse_pdf_with_grobid
except ModuleNotFoundError:
    from grobid_parser import check_grobid_server, parse_pdf_with_grobid

FIGURES_BASE_DIR = "data/figures"

def extract_visuals_with_pymupdf(doc: pymupdf.Document, paper_id: str, save_images: bool = True) -> Dict[str, Any]:
    """
    Extracts embedded figures, raster images, and vector graph/chart data from PDF pages.
    """
    figures_info = []
    vector_graphs_info = []

    paper_fig_dir = os.path.join(FIGURES_BASE_DIR, paper_id)
    if save_images:
        os.makedirs(paper_fig_dir, exist_ok=True)

    for page_num, page in enumerate(doc, start=1):
        # 1. Raster Images & Diagrams
        image_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(image_list, start=1):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image")
                image_ext = base_image.get("ext", "png")
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # Filter out tiny icon artifacts (e.g. 1x1 pixels or decorative rules)
                if width >= 100 and height >= 100:
                    fig_filename = f"fig_p{page_num}_{img_idx}.{image_ext}"
                    fig_path = os.path.join(paper_fig_dir, fig_filename)
                    if save_images and image_bytes:
                        with open(fig_path, "wb") as f_img:
                            f_img.write(image_bytes)

                    figures_info.append({
                        "figure_id": f"FIG-{paper_id}-P{page_num}-{img_idx}",
                        "page_number": page_num,
                        "file_path": fig_path if save_images else "",
                        "dimensions": f"{width}x{height}",
                        "format": image_ext,
                        "type": "raster_figure_or_diagram"
                    })
            except Exception:
                continue

        # 2. Vector Drawings (Graphs, Charts, Flowcharts)
        drawings = page.get_drawings()
        if len(drawings) > 15:  # Clusters of vector paths typically represent statistical graphs or architectures
            vector_graphs_info.append({
                "page_number": page_num,
                "drawing_path_count": len(drawings),
                "type": "vector_chart_or_graph",
                "rect": [float(c) for c in page.rect]
            })

    return {
        "figures_count": len(figures_info),
        "figures": figures_info,
        "vector_graphs_count": len(vector_graphs_info),
        "vector_graphs": vector_graphs_info
    }

def parse_pdf_file(filepath: str, use_grobid_if_available: bool = True) -> Dict[str, Any]:
    """
    Parses a PDF using GROBID (for academic TEI-XML structure) and PyMuPDF (for layout, images, and graphs).
    Falls back gracefully to PyMuPDF when GROBID server is not active.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PDF file not found: {filepath}")

    filename = os.path.basename(filepath)
    paper_id = filename.split("_")[0]

    # Open with PyMuPDF for visual assets (figures, graphs, layout)
    doc = pymupdf.open(filepath)
    visuals = extract_visuals_with_pymupdf(doc, paper_id=paper_id, save_images=True)

    # 1. Try GROBID first if active
    if use_grobid_if_available and check_grobid_server():
        print(f"   [GROBID Parser] Parsing {filename} via GROBID REST Service...")
        grobid_result = parse_pdf_with_grobid(filepath)
        if grobid_result:
            grobid_result["figures"] = visuals["figures"]
            grobid_result["figures_count"] = visuals["figures_count"]
            grobid_result["vector_graphs"] = visuals["vector_graphs"]
            grobid_result["vector_graphs_count"] = visuals["vector_graphs_count"]
            grobid_result["parser_used"] = "GROBID_TEI_XML_with_PyMuPDF_Visuals"
            doc.close()
            return grobid_result

    # 2. PyMuPDF Native Layout Parsing Engine
    extracted_chunks = []
    extracted_tables = []
    full_text_list = []
    current_section = "Abstract / Introduction"

    for page_num, page in enumerate(doc, start=1):
        # Extract structured tables
        try:
            tabs = page.find_tables()
            if tabs.tables:
                for t_idx, tab in enumerate(tabs.tables, start=1):
                    raw_rows = tab.extract()
                    if raw_rows and len(raw_rows) >= 2:
                        clean_rows = [[str(cell).strip() if cell is not None else "" for cell in r] for r in raw_rows]
                        if any(any(c for c in r) for r in clean_rows):
                            extracted_tables.append({
                                "table_id": f"TAB-{paper_id}-P{page_num}-{t_idx}",
                                "page_number": page_num,
                                "rows_count": len(clean_rows),
                                "cols_count": len(clean_rows[0]),
                                "data": clean_rows[:20]
                            })
        except Exception:
            pass

        # Extract reading-order blocks (x0, y0, x1, y1, text, block_no, block_type)
        blocks = page.get_text("blocks")
        for b in blocks:
            # block_type 0 is text
            if len(b) >= 5 and b[4]:
                text = b[4].strip()
                if not text:
                    continue

                first_line = text.split("\n")[0].strip()
                # Section header regex detector
                if re.match(
                    r'^(?:[0-9IVX]+\.|\b)(?:\s*)(Abstract|Introduction|Related Work|Methodology|Proposed Approach|System Architecture|Experiments|Results|Discussion|Conclusion|Limitations|Future Work)',
                    first_line,
                    re.IGNORECASE
                ):
                    current_section = first_line

                extracted_chunks.append({
                    "paper_id": paper_id,
                    "file_name": filename,
                    "page_number": page_num,
                    "section": current_section,
                    "bbox": [round(coord, 1) for coord in b[:4]],
                    "text": text
                })
                full_text_list.append(text)

    total_pages = len(doc)
    doc.close()

    parsed_document = {
        "paper_id": paper_id,
        "file_name": filename,
        "parser_used": "PyMuPDF_Layout_Engine",
        "total_pages": total_pages,
        "figures_count": visuals["figures_count"],
        "figures": visuals["figures"],
        "vector_graphs_count": visuals["vector_graphs_count"],
        "vector_graphs": visuals["vector_graphs"],
        "tables_count": len(extracted_tables),
        "tables": extracted_tables,
        "full_text": "\n\n".join(full_text_list),
        "chunks": extracted_chunks
    }
    return parsed_document

def process_pdf_directory(input_dir: str = "data/pdf_corpus", output_dir: str = "data/extracted_schemas"):
    """Processes all PDFs in input_dir and saves parsed JSON files to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} not found. Please ensure PDFs are in {input_dir}")
        return

    pdf_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".pdf")])
    print("=" * 60)
    print("Step 2 Stage 1: Running Dual Layout & Visual Parser (GROBID + PyMuPDF)")
    print(f"Found {len(pdf_files)} PDF files in {input_dir}")
    print("=" * 60 + "\n")

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
            print(f"   [PARSED] {out_filename} ({parsed_doc['total_pages']} pages, {len(parsed_doc['chunks'])} text blocks, {parsed_doc['figures_count']} figures, {parsed_doc['vector_graphs_count']} graphs)")
            parsed_count += 1
        except Exception as e:
            print(f"   [ERROR] Failed to parse {filename}: {e}")

    print("\n" + "=" * 60)
    print(f"Stage 1 Parsing Complete: {parsed_count}/{len(pdf_files)} JSONs written to {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    process_pdf_directory()
