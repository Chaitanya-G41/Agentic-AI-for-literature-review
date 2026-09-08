"""
GROBID TEI-XML Parser Module
Project: NLP-05 Agentic AI for Automated Research Paper Analysis
"""

import os
import json
import requests
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional

GROBID_DEFAULT_URL = os.environ.get("GROBID_SERVER_URL", "http://localhost:8070")

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

def check_grobid_server(server_url: str = GROBID_DEFAULT_URL, timeout: int = 2) -> bool:
    """Checks if the GROBID REST service is alive."""
    try:
        url = f"{server_url.rstrip('/')}/api/isalive"
        resp = requests.get(url, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False

def parse_pdf_with_grobid(filepath: str, server_url: str = GROBID_DEFAULT_URL, timeout: int = 30) -> Optional[Dict[str, Any]]:
    """
    Sends PDF file to GROBID REST API and parses returned TEI-XML into structured paper JSON.
    Returns None if server is unreachable or fails.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PDF file not found: {filepath}")

    url = f"{server_url.rstrip('/')}/api/processFulltextDocument"
    filename = os.path.basename(filepath)
    paper_id = filename.split("_")[0]

    try:
        with open(filepath, "rb") as pdf_file:
            files = {"input": (filename, pdf_file, "application/pdf")}
            data = {
                "generateIDs": "1",
                "consolidateHeader": "1",
                "consolidateCitations": "0",
                "includeRawCitations": "1"
            }
            response = requests.post(url, files=files, data=data, timeout=timeout)

        if response.status_code != 200:
            print(f"[GROBID Warning] Server returned HTTP {response.status_code} for {filename}")
            return None

        # Parse TEI XML
        xml_root = ET.fromstring(response.content)

        # Extract Title
        title_node = xml_root.find(".//tei:titleStmt/tei:title", TEI_NS)
        paper_title = title_node.text.strip() if title_node is not None and title_node.text else filename

        # Extract Abstract
        abstract_chunks = []
        abstract_node = xml_root.find(".//tei:profileDesc/tei:abstract", TEI_NS)
        if abstract_node is not None:
            for p in abstract_node.findall(".//tei:p", TEI_NS):
                if p.text:
                    abstract_chunks.append(p.text.strip())
        abstract_text = "\n".join(abstract_chunks)

        # Extract Body Sections
        extracted_chunks = []
        full_text_list = []

        if abstract_text:
            extracted_chunks.append({
                "paper_id": paper_id,
                "file_name": filename,
                "page_number": 1,
                "section": "Abstract",
                "text": abstract_text
            })
            full_text_list.append(f"Abstract:\n{abstract_text}")

        div_nodes = xml_root.findall(".//tei:text/tei:body//tei:div", TEI_NS)
        for div in div_nodes:
            head_node = div.find("tei:head", TEI_NS)
            section_title = head_node.text.strip() if head_node is not None and head_node.text else "Body Paragraph"
            
            for p in div.findall("tei:p", TEI_NS):
                p_text = "".join(p.itertext()).strip()
                if p_text:
                    chunk_item = {
                        "paper_id": paper_id,
                        "file_name": filename,
                        "page_number": 1,  # GROBID TEI TEI XML paragraph block
                        "section": section_title,
                        "text": p_text
                    }
                    extracted_chunks.append(chunk_item)
                    full_text_list.append(f"[{section_title}] {p_text}")

        parsed_doc = {
            "paper_id": paper_id,
            "file_name": filename,
            "title": paper_title,
            "abstract": abstract_text,
            "parser_used": "GROBID_TEI_XML",
            "total_pages": 1,
            "full_text": "\n".join(full_text_list),
            "chunks": extracted_chunks
        }
        return parsed_doc

    except Exception as e:
        print(f"[GROBID Exception] Failed to parse {filename} via GROBID: {e}")
        return None
