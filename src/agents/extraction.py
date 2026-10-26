"""
Step 2 Stage 2: Extraction Agent (Semantic Schema Mapper via Gemini Flash / Heuristics)
Project: NLP-05 Agentic AI for Automated Research Paper Analysis
"""

import os
import json
import re
from pydantic import BaseModel, Field, field_validator

class PaperSchema(BaseModel):
    paper_id: str = Field(description="Unique paper ID or arXiv ID")
    paper_title: str = Field(description="Title of the research paper")
    authors: list[str] = Field(default=[], description="List of author names")
    venue: str = Field(default="arXiv / Conference", description="Publication venue e.g. ICLR, NeurIPS, ACL")
    core_problem: str = Field(description="Core problem or research question addressed")
    methodology_summary: str = Field(description="Summary of core architecture or technical method")
    datasets_eval: list[str] = Field(default=[], description="Datasets or benchmarks evaluated")
    empirical_results: str = Field(description="Key accuracy or benchmark results achieved")
    explicit_limitations: str = Field(description="Explicit limitations, drawbacks, or failure modes")
    numerical_metrics: list[dict] = Field(default_factory=list, description="Extracted numerical benchmark or evaluation metrics")
    geographical_data: dict = Field(default_factory=dict, description="Geographical or spatial entity data if present")
    pipeline_steps: list[str] = Field(default_factory=list, description="Key methodological pipeline steps")
    generation_mode: str = Field(default="HEURISTIC_FALLBACK", description="GENUINE_LLM or HEURISTIC_FALLBACK")
    model_name: str = Field(default="Rule-Based Heuristic Engine", description="Name of LLM or Heuristic Engine used")

    @field_validator("authors", "datasets_eval", "pipeline_steps", mode="before")
    @classmethod
    def _v_lists(cls, v):
        """Gemini sometimes sends null instead of omitting the field, or a
        single string instead of a one-item list. Normalize both."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)

    @field_validator("numerical_metrics", mode="before")
    @classmethod
    def _v_metrics(cls, v):
        if v is None:
            return []
        if isinstance(v, dict):
            return [v]
        return list(v)

    @field_validator("geographical_data", mode="before")
    @classmethod
    def _v_geo(cls, v):
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        return {"data": str(v)}

    @property
    def problem_statement(self) -> str:
        return self.core_problem

    @property
    def year(self) -> str:
        match = re.search(r'\b(20\d{2})\b', self.venue)
        return match.group(1) if match else "2023"

    @property
    def proposed_method(self):
        class _Method:
            def __init__(self, summary):
                self.architecture = (summary.split(".")[0][:60] if summary else "Specialized Architecture")
                self.description = summary
                self.key_innovations = [summary[:120]] if summary else []
        return _Method(self.methodology_summary)

    @property
    def evaluation(self):
        class _Eval:
            def __init__(self, datasets, results):
                self.datasets = datasets or ["Standard Benchmark"]
                self.baselines = ["State-of-the-Art Baselines"]
                self.quantitative_results = {"Benchmark Summary": results[:200]} if results else {}
        return _Eval(self.datasets_eval, self.empirical_results)

    @property
    def limitations(self) -> list:
        return [self.explicit_limitations] if self.explicit_limitations else []

    @property
    def future_work(self) -> list:
        return [self.explicit_limitations] if self.explicit_limitations else []

    @field_validator("venue", mode="before")
    @classmethod
    def _v_venue(cls, v):
        """A default= only applies when the key is missing entirely; an
        explicit null still overrides it, so catch that case here."""
        return v if v else "arXiv / Conference"

    @field_validator("core_problem", "methodology_summary", "empirical_results", "explicit_limitations", mode="before")
    @classmethod
    def _v_strs(cls, v):
        """Gemini sometimes returns these as a list of sentences instead of
        one string. Join them instead of failing validation."""
        if v is None:
            return ""
        if isinstance(v, list):
            return " ".join(str(x) for x in v if x)
        return v

def extract_candidate_blocks(parsed_json):
    """Keyword heuristic pre-filter to pull candidate blocks for Gemini LLM processing."""
    chunks = parsed_json.get("chunks", [])

    title = parsed_json.get("file_name", "Unknown Paper").replace(".pdf", "").replace("_parsed.json", "")
    method_chunks = []
    result_chunks = []
    limitation_chunks = []
    abstract_chunks = []

    for chunk in chunks:
        sec = chunk.get("section", "").lower()
        text = chunk.get("text", "")
        text_lower = text.lower()

        if any(k in sec or k in text_lower for k in ["abstract", "introduction", "background", "motivation"]):
            abstract_chunks.append(text)
        if any(k in sec or k in text_lower for k in ["method", "approach", "architecture", "framework", "model"]):
            method_chunks.append(text)
        if any(k in sec or k in text_lower for k in ["result", "evaluat", "experiment", "accuracy", "benchmark", "projection"]):
            result_chunks.append(text)
        if any(k in sec or k in text_lower for k in ["limitation", "however", "drawback", "fails to", "restricted to", "future work"]):
            limitation_chunks.append(text)

    # Grounded core problem deduction from abstract/intro chunks
    core_problem_candidate = ""
    full_snippet = parsed_json.get("full_text", "")[:3500]
    all_intro_text = " ".join(abstract_chunks[:4]) or full_snippet[:1200]

    problem_sentences = []
    for sent in re.split(r'(?<=[.!?])\s+', all_intro_text):
        sent_clean = sent.strip()
        sent_lower = sent_clean.lower()
        if any(k in sent_lower for k in ["we investigate", "this paper addresses", "we study", "we propose", "focuses on", "aims to", "challenge of", "problem of", "bottleneck", "limits"]):
            if len(sent_clean) > 25:
                problem_sentences.append(sent_clean)

    if problem_sentences:
        core_problem_candidate = " ".join(problem_sentences[:2])
    elif abstract_chunks:
        core_problem_candidate = abstract_chunks[0][:250]
    else:
        core_problem_candidate = f"Investigation and empirical analysis of {title.replace('_', ' ')}."

    # Dataset deduction
    dataset_candidates = []
    for tab in parsed_json.get("tables", []):
        tab_id = tab.get("table_id", "")
        if tab_id and tab_id not in dataset_candidates:
            dataset_candidates.append(f"Tabular Data ({tab_id})")

    for chunk in result_chunks[:8]:
        for match in re.finditer(r'\b([A-Z][a-zA-Z0-9_-]{2,20}(?:-(?:bench|QA|Eval|Corpus|Set))?)\b', chunk):
            word = match.group(1)
            if word not in ["Figure", "Table", "Section", "Model", "Paper", "The", "This", "Our", "Results", "Methods"] and word not in dataset_candidates:
                if any(x in chunk.lower() for x in ["dataset", "benchmark", "evaluation", "tested on"]):
                    dataset_candidates.append(word)

    candidate_summary = {
        "title": title,
        "abstract_context": " ".join(abstract_chunks[:6]),
        "method_context": " ".join(method_chunks[:15]),
        "results_context": " ".join(result_chunks[:15]),
        "limitation_context": " ".join(limitation_chunks[:15]),
        "core_problem": core_problem_candidate,
        "datasets": dataset_candidates[:4] or [f"{title.replace('_', ' ')} Evaluation Benchmark"],
        "full_snippet": full_snippet
    }
    return candidate_summary

def run_extraction_agent(parsed_json_path, output_dir="data/extracted_schemas", llm_manager=None):
    """
    Runs the Extraction Agent on a parsed JSON file.
    Uses Gemini LLM Manager if keys are available; falls back to structured heuristic extractor.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(parsed_json_path, "r", encoding="utf-8") as f:
        parsed_json = json.load(f)

    paper_id = parsed_json.get("paper_id", "unknown")
    candidates = extract_candidate_blocks(parsed_json)

    from src.agents.llm_manager import GeminiLLMManager
    llm = llm_manager or GeminiLLMManager()
    extracted_schema = None

    if llm.keys:
        print(f"   Running Gemini LLM Extraction for {paper_id}...")
        prompt = f"""
        You are an expert NLP Research Extraction Agent.
        Extract structured schema fields for this research paper:
        Title Snippet: {candidates['title']}
        Method Context: {candidates['method_context'][:1500]}
        Results Context: {candidates['results_context'][:1500]}
        Limitations Context: {candidates['limitation_context'][:1500]}

        Return a strict JSON object with these exact keys:
        paper_id, paper_title, authors, venue, core_problem, methodology_summary, datasets_eval, empirical_results, explicit_limitations
        """
        sys_inst = "You are an expert NLP Research Extraction Agent. Extract clean JSON matching the requested paper schema."
        schema_dict = llm.generate_structured_json(prompt, system_instruction=sys_inst, response_schema=PaperSchema)
        if schema_dict:
            try:
                if isinstance(schema_dict, list) and len(schema_dict) > 0 and isinstance(schema_dict[0], dict):
                    schema_dict = schema_dict[0]
                schema_dict["paper_id"] = schema_dict.get("paper_id") or paper_id
                schema_dict["generation_mode"] = "GENUINE_LLM"
                schema_dict["model_name"] = f"{llm.last_provider or 'LLM'} ({llm.last_used_model or 'default'})"
                extracted_schema = PaperSchema(**schema_dict)
                print(f"   [GENUINE-LLM] Extracted schema for {paper_id} using {extracted_schema.model_name}")
            except Exception as e:
                print(f"   Schema validation failed ({e}). Using Heuristic Fallback...")

    from src.agents.visual_engine import extract_numerical_benchmarks_from_paper, detect_geographical_relevance
    bench_data = extract_numerical_benchmarks_from_paper(parsed_json, None)
    geo_data = detect_geographical_relevance(parsed_json.get("full_text", ""), paper_id=paper_id)

    # Derive pipeline steps dynamically
    pipeline_steps = []
    if candidates.get("method_context"):
        for sent in re.split(r'(?<=[.!?])\s+', candidates["method_context"][:1000]):
            sent_str = sent.strip()
            if len(sent_str) > 20 and len(sent_str) < 120 and any(w in sent_str.lower() for w in ["first", "then", "step", "generat", "evaluat", "model", "extract"]):
                pipeline_steps.append(sent_str)
    if not pipeline_steps:
        pipeline_steps = ["Problem Formulation & Data Ingestion", "Method Execution & Architecture", "Empirical Evaluation & Benchmarking"]

    if not extracted_schema:
        # Structured Grounded Heuristic Fallback Extractor (Domain-Independent)
        extracted_schema = PaperSchema(
            paper_id=paper_id,
            paper_title=candidates['title'].replace("_", " "),
            authors=["Research Authors"],
            venue="arXiv / Peer-Reviewed Venue",
            core_problem=candidates['core_problem'],
            methodology_summary=candidates['method_context'][:350] if candidates['method_context'] else f"Proposed technical methodology and experimental framework for {candidates['title']}.",
            datasets_eval=candidates['datasets'],
            empirical_results=candidates['results_context'][:300] if candidates['results_context'] else f"Empirical evaluation and experimental performance metrics reported in {paper_id}.",
            explicit_limitations=candidates['limitation_context'][:300] if candidates['limitation_context'] else f"Experimental scope and domain constraints noted in {paper_id}.",
            numerical_metrics=[bench_data] if bench_data.get("chart_type") == "bar" else [],
            geographical_data=geo_data if geo_data.get("has_geography") else {},
            pipeline_steps=pipeline_steps[:5],
            generation_mode="HEURISTIC_FALLBACK",
            model_name="Rule-Based Heuristic Engine"
        )
        print(f"   [HEURISTIC-FALLBACK] Extracted schema for {paper_id} via Grounded Heuristic Engine")
    else:
        # Augment LLM schema with grounded benchmark and geographical metadata if empty
        if not extracted_schema.numerical_metrics and bench_data.get("chart_type") == "bar":
            extracted_schema.numerical_metrics = [bench_data]
        if not extracted_schema.geographical_data and geo_data.get("has_geography"):
            extracted_schema.geographical_data = geo_data
        if not extracted_schema.pipeline_steps:
            extracted_schema.pipeline_steps = pipeline_steps[:5]

    output_filename = f"{paper_id}_schema.json"
    output_filepath = os.path.join(output_dir, output_filename)
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(extracted_schema.model_dump(), f, indent=2, ensure_ascii=False)

    print(f"   Saved Extraction Schema: {output_filename}")
    return extracted_schema

def process_all_parsed_json(input_dir="data/extracted_schemas", api_keys=None):
    """Processes all parsed JSON files in input_dir."""
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} not found!")
        return

    parsed_files = [f for f in os.listdir(input_dir) if f.endswith("_parsed.json")]
    print("==================================================")
    print("Step 2 Stage 2: Running Semantic Extraction Agent")
    print(f"Found {len(parsed_files)} parsed files to extract")
    print("==================================================\n")

    from src.agents.llm_manager import GeminiLLMManager
    llm = GeminiLLMManager(api_keys=api_keys)

    for idx, filename in enumerate(parsed_files, start=1):
        filepath = os.path.join(input_dir, filename)
        print(f"[{idx}/{len(parsed_files)}] Extracting Schema: {filename}...")
        run_extraction_agent(filepath, output_dir=input_dir, llm_manager=llm)

    print("\n==================================================")
    print("Step 2 Complete: All paper schemas extracted successfully!")
    print("==================================================")



class SchemaExtractionAgent:
    """Class-based agent interface for extracting structured paper schemas."""
    def __init__(self, llm_manager=None, output_dir: str = "data/extracted_schemas"):
        self.llm_manager = llm_manager
        self.output_dir = output_dir

    def load_schema(self, paper_id: str):
        pid = paper_id.split("_")[0].replace(".pdf", "")
        schema_path = os.path.join(self.output_dir, f"{pid}_schema.json")
        if not os.path.exists(schema_path):
            candidates = [os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir) if f.startswith(pid) and f.endswith("_schema.json")]
            if candidates:
                schema_path = candidates[0]
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return PaperSchema(**data)
        return None

    def extract(self, parsed_data_or_path) -> PaperSchema:
        if isinstance(parsed_data_or_path, str) and os.path.exists(parsed_data_or_path):
            parsed_path = parsed_data_or_path
        elif hasattr(parsed_data_or_path, "raw_dict"):
            raw = parsed_data_or_path.raw_dict
            pid = raw.get("paper_id", "unknown")
            parsed_path = os.path.join(self.output_dir, f"{pid}_parsed.json")
            with open(parsed_path, "w", encoding="utf-8") as f:
                json.dump(raw, f, indent=2, ensure_ascii=False)
        elif isinstance(parsed_data_or_path, dict):
            pid = parsed_data_or_path.get("paper_id", "unknown")
            parsed_path = os.path.join(self.output_dir, f"{pid}_parsed.json")
            with open(parsed_path, "w", encoding="utf-8") as f:
                json.dump(parsed_data_or_path, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported parsed_data input: {type(parsed_data_or_path)}")

        return run_extraction_agent(parsed_path, output_dir=self.output_dir, llm_manager=self.llm_manager)

if __name__ == "__main__":
    process_all_parsed_json()
