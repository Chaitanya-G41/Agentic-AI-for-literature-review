from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import re
import json

class ResearchGap(BaseModel):
    gap_id: str = Field(description="unique code")
    title: str = Field(description="Title of the gap")
    category: str = Field(description="Category of the gap")
    description: str = Field(description="More on the content")
    affected_papers: List[str] = Field(default=[], description="List of affected papers")
    confidence_level: str = Field(default="High", description="Confidence level grounded in evidence")
    reasoning: str = Field(default="", description="Reasoning establishing this gap from paper evidence")
    future_opportunity: str = Field(default="", description="Constructive actionable future opportunity")
    gap_type: str = Field(default="author_stated_limitation", description="author_stated_limitation, author_stated_future_work, synthesized_frontier_gap, or unavailable_evidence")
    evidence: str = Field(default="", description="Verbatim source excerpt(s) with paper IDs supporting this gap; empty string when no excerpt is available")

class ResearchGapAnalysis(BaseModel):
    explicit_gaps: List[ResearchGap] = Field(default=[], description="List of author stated limitations")
    synthesized_gaps: List[ResearchGap] = Field(default=[], description="List of cross paper synthesized research gaps")

from src.agents.llm_manager import LLMManager
from src.agents.report_generator import _classify_gap_excerpt, _derive_grounded_future_opportunity

def _truncate_snippet(text: str, max_words: int = 18) -> str:
    """Minimal local helper: truncate text to max_words words."""
    if not text:
        return ""
    words = str(text).replace("\n", " ").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."

def _get_val_local(obj, key: str, default=""):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def is_background_or_intro_statement(s: str, title: str = "") -> bool:
    """
    Topic-agnostic linguistic validator to identify introductory, background, or non-methodology statements
    without hardcoding specific benchmark publication phrases.
    """
    if not s or len(str(s).strip()) < 15:
        return True
    s_l = str(s).strip().lower()

    # Generic discourse starters & temporal background markers
    intro_regexes = [
        r'^\s*(?:in recent years|recently|historically|currently|traditionally|in the last decade|over the past|ever since|today|in the past)',
        r'^\s*(?:with the (?:recent|rapid|meteoric)?\s*(?:rise|advent|success|development|growth))',
        r'^\s*(?:due to the|owing to the|thanks to the|as a result of)',
        r'^\s*(?:despite (?:the|its)?\s*(?:success|recent|promising))',
        r'^\s*(?:in this (?:paper|work|study|article|review|investigation|survey))',
        r'^\s*(?:we (?:argue|show|demonstrate|find|conclude|review|discuss|outline|present|investigate|propose to study))',
        r'^\s*(?:the authors (?:argue|show|demonstrate|find|conclude|review|discuss|outline|present))',
        r'^\s*(?:against this backdrop|bearing the aforementioned|to account for this|taken together|on the one hand|on the other hand)',
        r'^\s*(?:it is (?:well known|widely acknowledged|clear|evident|vital|essential|crucial|imperative))',
        r'^\s*(?:people appear to think|there has been increasing|there are a large number of)',
        r'(?:is|are|has become)\s+an?\s+(?:essential|vital|crucial|promising|important|key|indispensable)\s+(?:tool|solution|task|component|field|domain|approach|area|instrument)',
        r'plays?\s+an?\s+(?:essential|vital|crucial|important|key)\s+role',
        r'has\s+received\s+(?:significant|considerable|widespread|growing)\s+attention'
    ]
    for pat in intro_regexes:
        if re.search(pat, s_l):
            return True

    # Title word overlap check: if string is primarily repeating the paper title, it is a title fragment
    if title:
        t_words = set(re.findall(r'\b[a-z]{3,}\b', title.lower()))
        s_words = set(re.findall(r'\b[a-z]{3,}\b', s_l))
        if len(t_words) > 2 and len(s_words & t_words) / max(1, len(t_words)) >= 0.70:
            return True

    return False

def run_research_gap_agent(schemas, llm_manager=None):
    paper_context = ""
    for paper in schemas:
        paper_context += f"\nPaper Title: {_get_val_local(paper, 'paper_title', '')}\n Core Problem: {_get_val_local(paper, 'core_problem', '')}\n Methodology: {_get_val_local(paper, 'methodology_summary', '')}\nLimitations: {_get_val_local(paper, 'explicit_limitations', '')}\n"

    llm = llm_manager or LLMManager()
    prompt = f"""You are an AI Research Gap Agent. Here is the paper context:
{paper_context}

Synthesize 3 to 5 research gaps. Return a JSON OBJECT with keys "explicit_gaps" and "synthesized_gaps".
Each gap item should contain: "gap_id", "title", "category", "description", "affected_papers"."""

    data_dict = llm.generate_structured_json(prompt)
    if isinstance(data_dict, list) and len(data_dict) > 0 and isinstance(data_dict[0], dict):
        data_dict = data_dict[0]
    if data_dict and isinstance(data_dict, dict) and "explicit_gaps" in data_dict and "synthesized_gaps" in data_dict:
        try:
            print(f"   [GENUINE-LLM] Research Gaps generated via {llm.last_provider or 'LLM'} ({llm.last_used_model or 'default'})")
            return ResearchGapAnalysis(**data_dict)
        except Exception as e:
            print(f"   [Agent 5 Note] Pydantic parsing note: {e}")

    # Grounded Heuristic Extraction & Dynamic Frontier Synthesis (Zero-LLM Fallback)
    print("   [HEURISTIC-FALLBACK] Synthesizing Evidence-Grounded Research Gaps (Reason: Offline or LLM unavailable)...")

    explicit = []
    parsed_dir = "data/extracted_schemas"
    paper_evidence = []

    for idx, p in enumerate(schemas, start=1):
        pid_val = str(_get_val_local(p, 'paper_id', str(idx))).strip()
        title_raw = str(_get_val_local(p, 'paper_title', f'Paper {idx}')).strip()
        clean_title = re.sub(r'^\s*' + re.escape(pid_val) + r'\s*[-_:]*\s*', '', title_raw, flags=re.IGNORECASE).replace('_', ' ').strip()

        lim_text = ""
        source_sec = "N/A"

        # Priority 1: Inspect parsed JSON on disk for explicit limitation / discussion chunks
        p_path = os.path.join(parsed_dir, f"{pid_val}_parsed.json")
        if os.path.exists(p_path):
            try:
                with open(p_path, "r", encoding="utf-8") as pf:
                    pj = json.load(pf)
                for c in pj.get("chunks", []):
                    sec_name = c.get("section", "")
                    sec_l = sec_name.lower()
                    if any(k in sec_l for k in ["limitation", "future work", "future research", "challenges and future", "discussion"]):
                        for sent in re.split(r'(?<=[.!?])\s+', c.get("text", "")):
                            s_str = sent.strip()
                            if len(s_str) > 25 and not is_background_or_intro_statement(s_str, clean_title):
                                if any(w in s_str.lower() for w in [
                                    "limitation", "limited by", "however", "rather than", "costly", "restricts",
                                    "rely on", "relies on", "subject to", "lack of", "trade-off", "bias", "overhead",
                                    "assumption", "unresolved", "degrades", "vulnerability", "bottleneck"
                                ]):
                                    lim_text = s_str
                                    source_sec = sec_name
                                    break
                        if lim_text:
                            break
            except Exception:
                pass

        # Priority 2: Use explicit_limitations from schema with strict background filtering
        if not lim_text:
            schema_lim = str(_get_val_local(p, 'explicit_limitations', '') or '').strip()
            if schema_lim and not is_background_or_intro_statement(schema_lim, clean_title) and not any(k in schema_lim.lower() for k in ["no explicit", "author constraints and domain"]):
                sentences = [s.strip() for s in re.split(r'[\.\n;]+', schema_lim) if len(s.strip()) > 20 and not is_background_or_intro_statement(s, clean_title)]
                if sentences:
                    lim_text = sentences[0]
                    source_sec = "Author Limitations Section"

        prob_val = str(_get_val_local(p, 'core_problem', '') or '')
        meth_val = str(_get_val_local(p, 'methodology_summary', '') or '')

        if lim_text:
            is_fw = _classify_gap_excerpt(lim_text) == "author_stated_future_work"
            g_type = "author_stated_future_work" if is_fw else "author_stated_limitation"
            type_label = "Author-Stated Future Work" if is_fw else "Author-Stated Limitation"
            gap_title = f"{type_label}: {lim_text[:50]}... ({pid_val})" if len(lim_text) > 50 else f"{type_label}: {lim_text} ({pid_val})"
            desc = (
                f"In [{pid_val}] (Section: {source_sec}), authors explicitly outline a future research avenue: \"{lim_text}\"."
                if is_fw else
                f"In [{pid_val}] (Section: {source_sec}), authors identify an operational constraint: \"{lim_text}\"."
            )
            conf = f"High (Direct {type_label.lower()} in Section: {source_sec})"
            reasoning = f"Directly extracted from the parsed primary text of [{pid_val}] where authors document {type_label.lower()} in Section '{source_sec}'."
            opp = _derive_grounded_future_opportunity(lim_text, is_future=is_fw, pid=pid_val)
        else:
            gap_title = f"Operational Boundary Unspecified in Inspected Sections ({pid_val})"
            desc = f"A research gap cannot be established from the available evidence in the inspected sections of [{pid_val}]."
            conf = "Unavailable Evidence (Not identified in inspected sections)"
            reasoning = f"Inspection of available parsed text for [{pid_val}] did not identify an explicit author-stated limitation or future work direction."
            opp = f"Empirical stress-testing is required to establish operational boundaries and formal research gaps for [{pid_val}]."
            g_type = "unavailable_evidence"

        lim_lower = (lim_text + " " + gap_title).lower()
        if any(k in lim_lower for k in ['compute', 'cost', 'latency', 'token', 'overhead', 'memory', 'gpu', 'throughput']):
            cat = "Computational Resources & Efficiency"
        elif any(k in lim_lower for k in ['error', 'noise', 'fidelity', 'drift', 'failure', 'bias', 'adversarial', 'robustness']):
            cat = "System Reliability & Robustness"
        elif any(k in lim_lower for k in ['generaliz', 'transfer', 'domain', 'out-of-distribution', 'unseen', 'field']):
            cat = "Generalization & Domain Adaptation"
        elif any(k in lim_lower for k in ['temporal', 'time', 'horizon', 'long-term', 'non-stationar']):
            cat = "Temporal Stability & Dynamic Horizon"
        elif any(k in lim_lower for k in ['scale', 'scalab', 'network', 'concurrency']):
            cat = "Scalability & Architecture"
        else:
            cat = "Empirical Evaluation"

        # Populate evidence with the verbatim extracted limitation excerpt and paper ID.
        # No placeholder text is used: if no excerpt was extracted, evidence stays empty.
        author_evidence = f"[{pid_val}]: \"{lim_text}\"" if lim_text else ""

        explicit.append(ResearchGap(
            gap_id=f"GAP-{idx:02d}",
            title=gap_title,
            category=cat,
            description=desc,
            affected_papers=[pid_val],
            confidence_level=conf,
            reasoning=reasoning,
            future_opportunity=opp,
            gap_type=g_type,
            evidence=author_evidence
        ))

        paper_evidence.append({
            "pid": pid_val,
            "title": clean_title,
            "limitation": lim_text,       # actual extracted limitation excerpt (may be empty)
            "problem": prob_val,           # introductory problem statement (NOT used for gap matching)
            "methodology": meth_val,       # methodology (NOT used for gap matching)
            "has_limitation": bool(lim_text),  # True only when an actual excerpt was extracted
        })

    # --- Dynamic Frontier Synthesized Gaps Discovery ---
    # Strictly discover cross-paper synthesized gaps ONLY when at least 2 active review papers
    # provide identifiable supporting evidence on shared empirical boundaries.
    synthesized = []
    syn_idx = 1

    # Standard empirical frontier dimensions across scientific & engineering literature
    FRONTIER_DIMENSIONS = [
        {
            "key": "generalization",
            "title": "Out-of-Distribution Generalization & Environmental Fluctuations",
            "category": "Generalization & Operational Stability",
            "keywords": [
                "generaliz", "domain adaptation", "transfer", "out-of-distribution", "distribution shift",
                "unseen", "robustness", "unmodeled", "drift", "field", "occlusion", "in-the-wild",
                "environmental", "weather", "disturbance", "meteorolog", "variance"
            ],
            "desc": "Cross-paper synthesis reveals that models evaluated under controlled or stationary environments suffer accuracy and policy stability degradation under out-of-distribution environmental shifts and unmodeled disturbances.",
            "opp": "Develop domain-adaptive representation learning, robust uncertainty quantification, and invariant objective formulations to guarantee operational stability under distribution shifts."
        },
        {
            "key": "complexity_latency",
            "title": "Computational Complexity & Real-Time Operational Latency Constraints",
            "category": "Computational Efficiency & Real-Time Systems",
            "keywords": [
                "latency", "real-time", "gpu", "memory", "computational overhead", "computational complexity",
                "throughput", "inference time", "cost", "hardware", "resource", "footprint", "sub-second", "microcontroller", "edge"
            ],
            "desc": "Cross-paper synthesis indicates that high-capacity neural architectures encounter inference latency and memory complexity bottlenecks that challenge real-time deployment under strict operational latency budgets.",
            "opp": "Investigate model compression, sparse attention mechanisms, and hardware-accelerated edge inference pipelines to satisfy real-time latency and computational budgets."
        },
        {
            "key": "temporal_horizon",
            "title": "Temporal Non-Stationarity & Multi-Horizon Prediction Degradation",
            "category": "Temporal Non-Stationarity & Forecasting",
            "keywords": [
                "temporal", "long-term", "multi-horizon", "compounding", "horizon", "time-series",
                "error propagation", "non-stationar", "dynamic", "delay", "sequential", "dispatch", "continual"
            ],
            "desc": "Cross-paper synthesis identifies compounding forecast error propagation and temporal non-stationarity over expanding prediction horizons, degrading policy reliability.",
            "opp": "Formulate multi-scale temporal alignment, closed-loop error correction, and online adaptive fine-tuning architectures to mitigate compounding degradation over extended horizons."
        },
        {
            "key": "data_scarcity_imbalance",
            "title": "Data Scarcity, Class Imbalance & Small-Sample Volatility",
            "category": "Data Scarcity & Robust Representation",
            "keywords": [
                "imbalance", "scarcity", "few-shot", "annotat", "labeling", "rare", "synthetic",
                "sample size", "small sample", "finite sample", "class-balanced", "data scarcity"
            ],
            "desc": "Cross-paper synthesis demonstrates that empirical performance degrades under severe class imbalance, label scarcity, or small-cohort sample sizes, increasing estimator variance.",
            "opp": "Establish non-parametric distribution metrics, contrastive self-supervised pre-training, and class-balanced re-weighting strategies to ensure stability under scarce or imbalanced observations."
        },
        {
            "key": "adversarial_security",
            "title": "Adversarial Vulnerability, Security Auditing & Trust Boundaries",
            "category": "Adversarial Robustness & Trustworthy AI",
            "keywords": [
                "adversarial", "backdoor", "poisoning", "evasion", "security", "trust",
                "contestability", "verification", "privacy", "federated", "defense", "vulnerability"
            ],
            "desc": "Cross-paper synthesis indicates an absence of certified defense protocols against adversarial evasion perturbations, data poisoning triggers, or unverified automated decision overrides.",
            "opp": "Formulate certified adversarial verification bounds, secure multiparty protocols, and contestable human-in-the-loop interfaces to audit and enforce system trustworthiness."
        },
        {
            "key": "sim_to_real",
            "title": "Simulation-to-Reality Transfer & Physical Deployment Disconnect",
            "category": "Deployment & Physical Transfer Verification",
            "keywords": [
                "simulation", "sim-to-real", "testbed", "contingency", "deployment",
                "operational verification", "physical", "offline", "surrogate"
            ],
            "desc": "Cross-paper synthesis highlights a persistent transfer gap between simulated benchmark testbeds and live physical environments with unmodeled operational contingencies.",
            "opp": "Establish hardware-in-the-loop testbeds and domain-randomized training protocols to bridge the simulation-to-reality transfer gap under live operational conditions."
        }
    ]

    for f_dim in FRONTIER_DIMENSIONS:
        # Match ONLY against lim_text — the actual extracted limitation excerpt.
        # problem and methodology fields are intentionally excluded so that
        # methodology-only or problem-description-only keyword matches cannot create a gap.
        # Papers without an actual limitation excerpt are excluded from the count.
        matching_pids = []
        matching_evidence = {}  # pid -> verbatim limitation excerpt
        for pe in paper_evidence:
            lim_text_pe = pe.get("limitation", "") or ""
            if not lim_text_pe or not pe.get("has_limitation", False):
                # No extractable evidence — this paper cannot contribute.
                continue
            lim_lower = lim_text_pe.lower()
            matched_kws = [k for k in f_dim["keywords"] if k in lim_lower]
            # Require at least 2 distinct keyword matches within the limitation excerpt.
            if len(matched_kws) >= 2:
                matching_pids.append(pe["pid"])
                matching_evidence[pe["pid"]] = lim_text_pe  # verbatim excerpt, no truncation yet

        # Strict Synthesis Rule: Only generate synthesized gap if at least 2 individually
        # evidenced papers support it. No placeholder evidence is ever emitted.
        if len(matching_pids) >= 2:
            evidence_parts = []
            evidenced_pids = []
            for p in matching_pids[:3]:
                ev = matching_evidence.get(p, "")
                if ev:  # only include papers that have an actual verbatim excerpt
                    evidence_parts.append(f"[{p}]: \"{_truncate_snippet(ev, 20)}\"")
                    evidenced_pids.append(p)
                # Papers with empty evidence are silently excluded — no placeholders.
            if len(evidenced_pids) < 2:
                continue  # Cannot generate gap without 2 individually evidenced papers.
            synthesized.append(ResearchGap(
                gap_id=f"GAP-SYN-{syn_idx:02d}",
                title=f"{f_dim['title']}",
                category=f_dim["category"],
                description=f"{f_dim['desc']} Author-stated limitations evidenced across [{', '.join(evidenced_pids)}].",
                affected_papers=evidenced_pids,
                confidence_level=f"Moderate (Keyword-matched across {len(evidenced_pids)} author-stated limitations; verify in primary text)",
                reasoning=(
                    f"Keyword analysis restricted to author-stated limitation excerpts of "
                    f"{', '.join(evidenced_pids)} identifies >= 2 keyword matches per paper "
                    f"for dimension '{f_dim['title']}'. Evidence is traceable to individual "
                    f"limitation sections, not inferred from methodology or introduction."
                ),
                future_opportunity=f"{f_dim['opp']} Tailored for [{', '.join(evidenced_pids)}].",
                gap_type="synthesized_frontier_gap",
                evidence="; ".join(evidence_parts)
            ))
            syn_idx += 1

    # NOTE: If no frontier dimension matches >= 2 papers, NO generic fallback is emitted.
    # Synthesized gaps are never fabricated without identifiable multi-paper evidence.

    return ResearchGapAnalysis(
        explicit_gaps=explicit,
        synthesized_gaps=synthesized
    )
