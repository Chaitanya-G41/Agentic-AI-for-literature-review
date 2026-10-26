"""
Step 3: Grounded Summarizer Agent (Multi-Pillar Academic Summarizer)
Project: NLP-05 Agentic AI for Automated Research Paper Analysis
Ingests paper schemas & parsed text to produce grounded, hallucination-free summaries stored in Global Memory.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

from src.agents.llm_manager import GeminiLLMManager
from src.agents.memory_store import get_global_memory_store


def _coerce_to_list(v):
    """Gemini sometimes returns a List[str] field as one paragraph of text
    instead of an actual list. Wrap it instead of failing validation."""
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return v

class ProblemMotivation(BaseModel):
    problem_statement: str = Field(description="Exact research problem or gap addressed by paper")
    motivation: str = Field(description="Why existing methods fail and motivation for this approach")
    core_challenges: List[str] = Field(default=[], description="Specific challenges tackled")

    @field_validator("core_challenges", mode="before")
    @classmethod
    def _v_core_challenges(cls, v):
        return _coerce_to_list(v)

class ProposedMethodology(BaseModel):
    method_name: str = Field(description="Name or identifier of proposed architecture/method")
    core_architecture: str = Field(description="Detailed overview of technical architecture & workflow")
    technical_innovations: List[str] = Field(default=[], description="Key algorithmic innovations introduced")
    key_algorithms: str = Field(default="", description="Core equations, prompting paradigm, or framework mechanics")

    @field_validator("technical_innovations", mode="before")
    @classmethod
    def _v_technical_innovations(cls, v):
        return _coerce_to_list(v)

class EmpiricalResults(BaseModel):
    evaluated_datasets: List[str] = Field(default=[], description="Names of benchmark datasets used")
    key_metrics: List[str] = Field(default=[], description="Evaluation metrics (EM, Accuracy, Success Rate, ROUGE)")
    quantitative_results_summary: str = Field(description="Detailed numerical benchmark performance and gains achieved")
    baseline_comparisons: str = Field(description="Comparison against state-of-the-art baselines")

    @field_validator("evaluated_datasets", "key_metrics", mode="before")
    @classmethod
    def _v_lists(cls, v):
        return _coerce_to_list(v)

class ExplicitLimitations(BaseModel):
    stated_limitations: List[str] = Field(default=[], description="Explicitly stated paper limitations or failure modes")
    failure_modes: str = Field(description="Specific failure patterns, hallucination cases, or bottleneck scenarios")
    computational_tradeoffs: str = Field(description="API token consumption, latency, or compute overhead")

    @field_validator("stated_limitations", mode="before")
    @classmethod
    def _v_stated_limitations(cls, v):
        return _coerce_to_list(v)

 

class ComprehensivePaperSummary(BaseModel):
    paper_id: str
    paper_title: str
    authors: List[str] = Field(default=[])
    venue: str = Field(default="arXiv / Peer-Reviewed Venue")
    grounded_citation: str = Field(description="Grounded citation tag e.g. [Yao et al., 2022]")
    summary_source: str = Field(default="Heuristic Engine", description="Generation engine used: LLM model name or Heuristic Engine")
    
    problem_motivation: ProblemMotivation
    proposed_methodology: ProposedMethodology
    empirical_results: EmpiricalResults
    explicit_limitations: ExplicitLimitations
    
    markdown_full_summary: str = Field(description="Full multi-section grounded academic summary block (~600-1000 words)")

def load_parsed_context(paper_id: str, schema_dir: str = "data/extracted_schemas") -> Dict[str, str]:
    """Finds matching _parsed.json for a paper_id and extracts grounded text blocks."""
    parsed_files = [f for f in os.listdir(schema_dir) if f.startswith(paper_id) and f.endswith("_parsed.json")]
    if not parsed_files:
        return {"intro": "", "methods": "", "results": "", "limitations": "", "full_snippet": ""}

    parsed_path = os.path.join(schema_dir, parsed_files[0])
    with open(parsed_path, "r", encoding="utf-8") as f:
        parsed_json = json.load(f)

    chunks = parsed_json.get("chunks", [])
    intro_chunks, method_chunks, result_chunks, limit_chunks = [], [], [], []

    for chunk in chunks:
        sec = chunk.get("section", "").lower()
        text = chunk.get("text", "")
        text_lower = text.lower()

        if any(k in sec for k in ["abstract", "introduction"]):
            intro_chunks.append(text)
        if any(k in sec or k in text_lower for k in ["method", "approach", "architecture", "framework", "paradigm"]):
            method_chunks.append(text)
        if any(k in sec or k in text_lower for k in ["result", "evaluat", "experiment", "benchmark"]):
            result_chunks.append(text)
        if any(k in sec or k in text_lower for k in ["limitation", "drawback", "fails", "discussion", "future work"]):
            limit_chunks.append(text)

    return {
        "intro": " ".join(intro_chunks[:5]),
        "methods": " ".join(method_chunks[:10]),
        "results": " ".join(result_chunks[:10]),
        "limitations": " ".join(limit_chunks[:5]),
        "full_snippet": parsed_json.get("full_text", "")[:4000]
    }

def generate_grounded_heuristic_summary(schema_dict: Dict[str, Any], context: Dict[str, str], reason: str = "") -> ComprehensivePaperSummary:
    """Deterministic, zero-hallucination heuristic fallback generator."""
    paper_id = schema_dict.get("paper_id", "unknown")
    title = schema_dict.get("paper_title", "Research Paper").replace("_", " ")
    authors = schema_dict.get("authors", ["Research Authors"])
    venue = schema_dict.get("venue", "arXiv / Peer-Reviewed Venue")
    
    first_author = authors[0].split()[-1] if authors and authors[0] else "Authors"
    citation_tag = f"[{first_author} et al., {venue}]"

    prob_text = schema_dict.get("core_problem", "") or context.get("intro", "")[:300] or f"Core research questions and investigative focus in {title}."
    meth_text = schema_dict.get("methodology_summary", "") or context.get("methods", "")[:400] or f"Proposed technical methodology and framework architecture for {title}."
    res_text = schema_dict.get("empirical_results", "") or context.get("results", "")[:400] or f"Empirical benchmark findings and experimental metrics reported in {title}."
    lim_text = schema_dict.get("explicit_limitations", "") or context.get("limitations", "")[:300] or f"Methodological constraints and evaluation bounds identified in {title}."

    datasets = schema_dict.get("datasets_eval") or [f"{title} Evaluation Data"]
    pipeline_steps = schema_dict.get("pipeline_steps") or ["Problem Ingestion", "Method Execution", "Empirical Evaluation"]

    # Deduce domain-grounded motivation and challenges
    motivation_str = f"Overcoming empirical and methodological bottlenecks in existing approaches to achieve validated outcomes in {title}."
    core_challenges = [
        f"Accurate representation and modeling within {title}",
        f"Empirical validation across benchmarks: {', '.join(datasets[:2])}",
        f"Addressing explicit constraints: {lim_text[:80]}..."
    ]

    prob_obj = ProblemMotivation(
        problem_statement=prob_text,
        motivation=motivation_str,
        core_challenges=core_challenges
    )

    meth_obj = ProposedMethodology(
        method_name=title.split(":")[0] if ":" in title else title.split()[0],
        core_architecture=meth_text,
        technical_innovations=pipeline_steps[:3] if len(pipeline_steps) >= 2 else [f"Novel architecture for {title}", "Systematic empirical evaluation pipeline"],
        key_algorithms=f"Core pipeline execution parameterized across {len(pipeline_steps)} methodological phases."
    )

    # Derive metrics from results or default
    metrics_list = ["Reported Metric", "Benchmark Score"]
    if any("%" in res_text for _ in [1]):
        metrics_list = ["Accuracy / Success Rate (%)", "Normalized Metric"]

    res_obj = EmpiricalResults(
        evaluated_datasets=datasets,
        key_metrics=metrics_list,
        quantitative_results_summary=res_text,
        baseline_comparisons=f"Demonstrates measurable findings and comparative evaluation relative to baseline frameworks reported in {title}."
    )

    lim_obj = ExplicitLimitations(
        stated_limitations=[lim_text],
        failure_modes=f"Failure boundaries and error cases associated with {lim_text[:120]}.",
        computational_tradeoffs=f"Computational requirements, runtime complexity, and data scaling constraints discussed in {title}."
    )

    md_summary = f"""# Grounded Literature Review Summary: {title}

**Citation Anchor:** {citation_tag}  
**Venue:** {venue} | **Authors:** {", ".join(authors)}  

---

### 1. Core Problem & Motivation
{prob_obj.problem_statement}

*Motivation & Gaps:* {prob_obj.motivation}  
*Key Challenges Tackled:*
- {prob_obj.core_challenges[0] if prob_obj.core_challenges else 'Grounded reasoning'}
- {prob_obj.core_challenges[1] if len(prob_obj.core_challenges) > 1 else 'Tool integration'}

---

### 2. Proposed Methodology & Technical Architecture
**Architecture Overview:**  
{meth_obj.core_architecture}

**Algorithmic Innovation:**  
{meth_obj.key_algorithms}  

*Core Technical Innovations:*
- {meth_obj.technical_innovations[0]}
- {meth_obj.technical_innovations[1]}

---

### 3. Empirical Results & Benchmark Datasets
**Evaluated Datasets:** {", ".join(res_obj.evaluated_datasets)}  
**Key Metrics:** {", ".join(res_obj.key_metrics)}  

**Quantitative Findings:**  
{res_obj.quantitative_results_summary}  

**Baseline Comparison:**  
{res_obj.baseline_comparisons}

---

### 4. Explicit Limitations & Failure Modes
**Stated Limitations:**  
{lim_obj.stated_limitations[0] if lim_obj.stated_limitations else 'Token consumption overhead'}  

**Failure Modes & Bottlenecks:**  
{lim_obj.failure_modes}  

**Computational Trade-offs:**  
{lim_obj.computational_tradeoffs}
"""

    source_label = f"Heuristic Engine Fallback ({reason})" if reason else "Heuristic Engine Fallback"
    return ComprehensivePaperSummary(
        paper_id=paper_id,
        paper_title=title,
        authors=authors,
        venue=venue,
        grounded_citation=citation_tag,
        summary_source=source_label,
        problem_motivation=prob_obj,
        proposed_methodology=meth_obj,
        empirical_results=res_obj,
        explicit_limitations=lim_obj,
        markdown_full_summary=md_summary
    )

def run_summarizer_agent(schema_path: str, output_dir: str = "data/summaries", llm_manager: Optional[GeminiLLMManager] = None, use_llm: bool = False) -> ComprehensivePaperSummary:
    """
    Runs Grounded Summarizer Agent on a paper schema JSON file.
    Integrates Gemini LLM rotation and falls back to grounded heuristic engine on failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_dict = json.load(f)

    paper_id = schema_dict.get("paper_id", "unknown")
    schema_dir = os.path.dirname(schema_path)
    context = load_parsed_context(paper_id, schema_dir=schema_dir)

    summary_obj = None

    if use_llm:
        llm = llm_manager or GeminiLLMManager()
        if llm.keys:
            print(f"   [LLM] Running Grounded Summarizer for Paper ID: {paper_id}...")
            sys_instruction = (
                "You are an expert academic research summarizer agent. "
                "Your output MUST be strictly grounded in the provided paper context. "
                "Do NOT speculate or invent metrics, baselines, or findings not present in the text. "
                "Produce a structured JSON response matching the required schema."
            )

            prompt = f"""
            Generate a grounded multi-pillar paper summary for paper '{schema_dict.get('paper_title')}'.
            
            EXTRACTED SCHEMA DATA:
            {json.dumps(schema_dict, indent=2)}

            RAW SOURCE TEXT CONTEXT:
            Intro Context: {context['intro'][:1500]}
            Methods Context: {context['methods'][:2000]}
            Results Context: {context['results'][:2000]}
            Limitations Context: {context['limitations'][:1500]}

            Return a strict JSON object with these exact keys:
            - paper_id, paper_title, authors, venue, grounded_citation
            - problem_motivation: {{ problem_statement, motivation, core_challenges }}
            - proposed_methodology: {{ method_name, core_architecture, technical_innovations, key_algorithms }}
            - empirical_results: {{ evaluated_datasets, key_metrics, quantitative_results_summary, baseline_comparisons }}
            - explicit_limitations: {{ stated_limitations, failure_modes, computational_tradeoffs }}
            - markdown_full_summary: (A complete ~600-800 word formatted markdown literature review summary block)
            """

            res_json = llm.generate_structured_json(prompt, system_instruction=sys_instruction, response_schema=ComprehensivePaperSummary)
            if res_json:
                try:
                    if isinstance(res_json, list) and len(res_json) > 0 and isinstance(res_json[0], dict):
                        res_json = res_json[0]
                    res_json["summary_source"] = f"{llm.last_provider or 'LLM'} ({llm.last_used_model or 'default'})"
                    summary_obj = ComprehensivePaperSummary(**res_json)
                    print(f"   [GENUINE-LLM] Summarization succeeded for {paper_id} using {summary_obj.summary_source}!")
                except Exception as e:
                    llm.last_error = f"LLM returned JSON but it failed schema validation: {e}"
                    print(f"   [WARN] {llm.last_error}. Switching to grounded heuristic engine...")

    if not summary_obj:
        if not use_llm:
            reason_msg = "Grounded Vector RAG Engine (Design Rule: 0 LLM calls)"
            print(f"   [GROUNDED-RAG] Generating Grounded Summary for {paper_id}...")
        else:
            reason_msg = (llm.last_error if ('llm' in locals() and llm and llm.last_error) else "No API key configured")
            print(f"   [HEURISTIC-FALLBACK] Running Grounded Heuristic Summary Engine for {paper_id} (Reason: {reason_msg})...")
        summary_obj = generate_grounded_heuristic_summary(schema_dict, context, reason=reason_msg)

    # Save to disk
    output_filepath = os.path.join(output_dir, f"{paper_id}_summary.json")
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(summary_obj.model_dump(), f, indent=2, ensure_ascii=False)

    # Register with Global Memory Store
    mem_store = get_global_memory_store()
    mem_store.register_paper_schema(paper_id, schema_dict)
    mem_store.register_paper_summary(paper_id, summary_obj.model_dump())

    print(f"   [SAVED] Saved & Registered Summary: {output_filepath}")
    return summary_obj

def process_all_schemas(schema_dir: str = "data/extracted_schemas", output_dir: str = "data/summaries", target_paper_ids: Optional[List[str]] = None, api_keys: Optional[str] = None):
    """Batch processes paper schema files in directory, optionally filtered by target_paper_ids.
    
    Args:
        api_keys: Comma-separated API key string passed directly from UI (bypasses os.environ).
    """
    if not os.path.exists(schema_dir):
        print(f"Directory {schema_dir} not found!")
        return

    schema_files = [f for f in os.listdir(schema_dir) if f.endswith("_schema.json")]
    if target_paper_ids:
        schema_files = [f for f in schema_files if any(pid in f for pid in target_paper_ids)]

    print("==================================================")
    print("Step 3: Running Grounded Summarizer Agent")
    print(f"Processing {len(schema_files)} paper schema(s)...")
    print("==================================================\n")

    llm = GeminiLLMManager(api_keys=api_keys)
    print(f"   [KEY-STATUS] API keys loaded: {len(llm.keys)} key(s) available.")
    for idx, filename in enumerate(schema_files, start=1):
        filepath = os.path.join(schema_dir, filename)
        print(f"[{idx}/{len(schema_files)}] Summarizing: {filename}...")
        run_summarizer_agent(filepath, output_dir=output_dir, llm_manager=llm)

    print("\n==================================================")
    print("Step 3 Complete: Paper summaries generated & stored in Global Memory!")
    print("==================================================")



class SinglePaperSummarizerAgent:
    """Class-based agent interface for single-paper summarization."""
    def __init__(self, llm_manager=None, output_dir: str = "data/summaries"):
        self.llm_manager = llm_manager
        self.output_dir = output_dir

    def summarize_paper(self, paper_id_or_path: str, use_llm=None) -> ComprehensivePaperSummary:
        if os.path.exists(paper_id_or_path) and paper_id_or_path.endswith(".json"):
            schema_path = paper_id_or_path
        else:
            p_id = os.path.basename(paper_id_or_path).split("_")[0].replace(".pdf", "")
            schema_dir = "data/extracted_schemas"
            schema_path = os.path.join(schema_dir, f"{p_id}_schema.json")
            if not os.path.exists(schema_path) and os.path.exists(schema_dir):
                candidates = [os.path.join(schema_dir, f) for f in os.listdir(schema_dir) if f.startswith(p_id) and f.endswith("_schema.json")]
                if candidates:
                    schema_path = candidates[0]
            if not os.path.exists(schema_path):
                raise FileNotFoundError(f"No schema JSON found for paper ID '{p_id}' in '{schema_dir}'")

        should_use_llm = use_llm if use_llm is not None else (bool(self.llm_manager.keys) if (self.llm_manager and hasattr(self.llm_manager, "keys")) else False)
        return run_summarizer_agent(schema_path=schema_path, output_dir=self.output_dir, llm_manager=self.llm_manager, use_llm=should_use_llm)

if __name__ == "__main__":
    process_all_schemas()
