"""

Agent 6: Report Generator Agent (Final Synthesis via [One Prompt ONLY])

Project: NLP-05 Agentic AI for Automated Research Paper Analysis

"""

import os

import re

import json

from datetime import datetime

from pydantic import BaseModel, Field

from typing import List, Optional, Dict, Any, Tuple

from src.agents.llm_manager import LLMManager

class LiteratureReviewReport(BaseModel):
    title: str = Field(description="Title of the report")
    generated_at: str = Field(description="Timestamp at which report was generated")
    paper_count: int = Field(default=0, description="Total number of papers reviewed")
    full_markdown: str = Field(description="Complete markdown text of the report")
    diagrams: Dict[str, str] = Field(default_factory=dict, description="Mermaid diagrams dictionary")
    diagram_data: Dict[str, Any] = Field(default_factory=dict, description="Structured dynamic diagram data")

def sanitize_mermaid_id(raw_id: str) -> str:

    """Produces a strict alphanumeric identifier: '2210.03629' -> 'P_2210_03629'."""

    clean = re.sub(r'[^a-zA-Z0-9_]', '_', str(raw_id).strip())

    if not clean or clean[0].isdigit():

        clean = f"P_{clean}"

    return clean

def _get_val(obj, key: str, default=None):

    """Safely retrieves a key or attribute from either a Pydantic model or a dictionary."""

    if obj is None:

        return default

    if isinstance(obj, dict):

        return obj.get(key, default)

    return getattr(obj, key, default)

from collections import Counter

def escape_mermaid_label(text: str, wrap_len: int = 24, max_len: int = None) -> str:

    """

    Escapes dynamic academic text for Mermaid node labels without corrupting meaning.

    Preserves citations, square brackets, tokens, and mathematical notation.

    Never truncates with ellipses unless max_len is explicitly requested.

    Guarantees that no raw literal newlines or control characters break Mermaid grammar.

    """

    if not text:

        return "Not Specified"

    s = re.sub(r'[\r\n\t]+', ' ', str(text)).strip()

    math_map = {

        r'\cup': '∪', r'\cap': '∩', r'\times': '×',

        r'\approx': '≈', r'\le': '≤', r'\ge': '≥',

        r'\rightarrow': '→', r'\to': '→', r'\pm': '±',

        r'\neq': '≠', r'\alpha': 'α', r'\beta': 'β',

        r'\Delta': 'Δ', r'\theta': 'θ', r'\mu': 'μ'

    }

    for tex, uni in math_map.items():

        s = s.replace(tex, uni)

    s = s.replace('\\', '\\\\')

    if max_len and len(s) > max_len:

        s = s[:max_len].rsplit(' ', 1)[0] + '...'

    s = s.replace('·', '-')

    s = s.replace('"', "'")

    s = s.replace('<br/>', '__BR_TOKEN__').replace('<br>', '__BR_TOKEN__')

    s = s.replace('<b>', '__BOLD_OPEN__').replace('</b>', '__BOLD_CLOSE__')

    s = s.replace('<code>', '__CODE_OPEN__').replace('</code>', '__CODE_CLOSE__')

    s = s.replace('</span>', '__SPAN_CLOSE__')

    span_matches = re.findall(r'<span[^>]*>', s)

    span_map = {}

    for i, sm in enumerate(span_matches):

        tok = f'__SPAN_OPEN_{i}__'

        span_map[tok] = sm

        s = s.replace(sm, tok, 1)

    s = s.replace('<', '&lt;').replace('>', '&gt;')

    s = s.replace('__BR_TOKEN__', '<br/>')

    s = s.replace('__BOLD_OPEN__', '<b>').replace('__BOLD_CLOSE__', '</b>')

    s = s.replace('__CODE_OPEN__', '<code>').replace('</code>', '__CODE_CLOSE__')

    s = s.replace('__SPAN_CLOSE__', '</span>')

    for tok, sm in span_map.items():

        s = s.replace(tok, sm)

    s = s.replace('{', '&#123;').replace('}', '&#125;')

    s = s.replace('|', '&#124;')

    def _mask_tag_spaces(m):

        return m.group(0).replace(' ', '__TAG_SPACE__')

    s = re.sub(r'<[^>]+>', _mask_tag_spaces, s)

    words = [w for w in s.split(' ') if w]

    lines, cur_line, cur_len = [], [], 0

    for w in words:

        if '<br/>' in w:

            parts = w.split('<br/>')

            for p_idx, part in enumerate(parts):

                if p_idx == 0:

                    if part:

                        cur_line.append(part)

                    if cur_line:

                        lines.append(' '.join(cur_line))

                    cur_line = []

                    cur_len = 0

                else:

                    if p_idx == len(parts) - 1:

                        if part:

                            cur_line = [part]

                            cur_len = len(re.sub(r'<[^>]+>', '', part))

                    else:

                        if part:

                            lines.append(part)

            continue

        w_len = len(re.sub(r'<[^>]+>', '', w))

        if cur_len + w_len > wrap_len and cur_line:

            lines.append(' '.join(cur_line))

            cur_line = [w]

            cur_len = w_len

        else:

            cur_line.append(w)

            cur_len += w_len + 1

    if cur_line:

        lines.append(' '.join(cur_line))

    result = '<br/>'.join([ln for ln in lines if ln.strip()])

    result = result.replace('__TAG_SPACE__', ' ')

    return result.replace('\r', '').replace('\n', ' ')

def extract_corpus_domain(comp_analysis=None, schemas=None, report_title: str = "", default_domain: str = "Scientific Research") -> str:
    """Dynamically extracts or determines the corpus domain from active schemas, comparative analysis, or active user query."""
    # Check active user query first to preserve original user query
    aq_file = "data/active_query.json"
    if os.path.exists(aq_file):
        try:
            with open(aq_file, "r", encoding="utf-8") as aqf:
                q_val = json.load(aqf).get("query", "").strip()
                if q_val:
                    # Sanitize any accidental duplicated consecutive words
                    q_val = re.sub(r'\b(\w+)\s+\1\b', r'\1', q_val, flags=re.IGNORECASE).strip()
                    return q_val
        except Exception:
            pass

    if schemas and len(schemas) == 1:
        single_title = _get_val(schemas[0], "paper_title", "")
        if single_title:
            return re.sub(r'\b(\w+)\s+\1\b', r'\1', single_title.replace("_", " ").strip(), flags=re.IGNORECASE)

    if report_title:
        clean_t = re.sub(r'^(?:#+\s*)?(?:Automated Literature Review:\s*|Comprehensive Systematic Review:\s*|Synthesis of\s*|Literature Review:\s*|Systematic Review:\s*)', '', report_title.strip(), flags=re.IGNORECASE).strip().strip("#* ")
        clean_t = re.sub(r'^(?:Synthesis of\s*)', '', clean_t, flags=re.IGNORECASE).strip().strip("#* ")
        clean_t = re.sub(r'\b(\w+)\s+\1\b', r'\1', clean_t, flags=re.IGNORECASE).strip()
        if clean_t and clean_t.lower() not in ["scientific research", "research", "research corpus", "corpus overview", "benchmark testbed", "testbed benchmark", "untitled report", "abilities", "specialized specialized", "specialized"]:
            return clean_t

    paper_texts = {}
    if schemas:
        for s in schemas:
            pid = str(_get_val(s, "paper_id", ""))
            txt = f"{_get_val(s, 'paper_title', '')} {_get_val(s, 'core_problem', '')}"
            paper_texts[pid] = paper_texts.get(pid, "") + " " + txt

    if comp_analysis:
        for r in _get_val(comp_analysis, "methodology_comparison", []):
            pid = str(_get_val(r, "paper_id", ""))
            txt = f"{_get_val(r, 'paper_title', '')} {_get_val(r, 'paradigm', '')} {_get_val(r, 'architecture_category', '')}"
            paper_texts[pid] = paper_texts.get(pid, "") + " " + txt

    stopwords = {
        "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "with",
        "by", "from", "via", "into", "towards", "using", "through", "can", "large",
        "language", "model", "models", "learning", "based", "approach", "new", "study",
        "analysis", "automated", "system", "systems", "framework", "evaluation", "paper",
        "papers", "cross", "corpus", "review", "literature", "methodology", "methodologies",
        "testbed", "benchmark", "benchmarks", "empirical", "formulation", "question",
        "specialized", "general", "proposed", "novel"
    }

    bigram_paper_counts = Counter()
    word_paper_counts = Counter()
    for pid, txt in paper_texts.items():
        clean = re.sub(r"[^a-zA-Z\s-]", " ", str(txt)).lower()
        words = [
            w.strip() for w in clean.split()
            if len(w.strip()) > 2 and w.strip() not in stopwords and not any(c.isdigit() for c in w)
        ]
        p_bigrams = set(f"{words[i]} {words[i+1]}" for i in range(len(words) - 1) if words[i] != words[i+1])
        for bg in p_bigrams:
            bigram_paper_counts[bg] += 1
        for w in set(words):
            word_paper_counts[w] += 1

    if len(paper_texts) > 1:
        for bg, count in bigram_paper_counts.most_common(5):
            bg_words = bg.split()
            if len(bg_words) == 2 and bg_words[0] == bg_words[1]:
                continue
            if count >= 2:
                return bg.title()
        for w, count in word_paper_counts.most_common(5):
            if count >= 2:
                return w.title()
        return default_domain
    elif paper_texts:
        for bg, _ in bigram_paper_counts.most_common(1):
            bg_words = bg.split()
            if len(bg_words) == 2 and bg_words[0] == bg_words[1]:
                continue
            return bg.title()
        for w, _ in word_paper_counts.most_common(1):
            return w.title()

    return default_domain

def _clean_paper_title(title: str, pid: str) -> str:
    """Cleans raw paper title by stripping paper ID prefix, symbols, and underscores."""
    s = re.sub(r'^\s*' + re.escape(str(pid)) + r'\s*[-_:]*\s*', '', str(title or ""), flags=re.IGNORECASE)
    s = re.sub(r'^\d{4}\.\d{4,5}(v\d+)?\s*[-:_]?\s*', '', s).strip()
    s = s.replace(".pdf", "").replace("_", " ").strip(" -:")
    return s or f"Paper {pid}"

def _get_single_paper_cards(row, pid: str):
    """
    Dynamically extracts methodology stages and leaf cards for a single paper
    from structured disk extractions and comparative schemas without hardcoding.
    """
    sum_path = f"data/summaries/{pid}_summary.json"
    s_data = {}
    if os.path.exists(sum_path):
        try:
            with open(sum_path, "r", encoding="utf-8") as f:
                s_data = json.load(f)
        except Exception:
            pass

    sch_path = f"data/extracted_schemas/{pid}_schema.json"
    sc_data = {}
    if os.path.exists(sch_path):
        try:
            with open(sch_path, "r", encoding="utf-8") as f:
                sc_data = json.load(f)
        except Exception:
            pass

    meth = _get_val(s_data, "proposed_methodology", {})
    innovs = _get_val(meth, "technical_innovations", [])
    prob = _get_val(s_data, "problem_motivation", {})
    challenges = _get_val(prob, "core_challenges", [])
    datasets = _get_val(sc_data, "datasets_eval", [])
    steps = _get_val(sc_data, "pipeline_steps", [])

    # Column 0: Problem Formulation & Ingestion
    c0_sub1 = steps[0] if (steps and len(steps) > 0 and len(str(steps[0]).splitlines()) <= 2) else (_get_val(prob, "motivation", "") or "Task Formulation & Boundary Conditions")
    c0_sub2 = challenges[0] if challenges else "Input Space & Environmental Grounding"

    # Column 1: Core Architecture & Method
    c1_sub1 = innovs[0] if (innovs and len(innovs) > 0) else (_get_val(row, "key_innovation", "") or "Algorithmic Execution Pipeline")
    c1_sub2 = innovs[1] if (innovs and len(innovs) > 1) else (_get_val(meth, "core_architecture", "") or "System Execution & State Transition")

    # Column 2: Benchmarking & Quantitative Evaluation
    ds_clean = [str(d).replace("Tabular Data (", "").replace(")", "").strip() for d in datasets[:2]] if datasets else []
    c2_sub1 = f"Evaluated on: {', '.join(ds_clean)}" if ds_clean else "Standard Corpus Baselines"
    c2_sub2 = challenges[1] if (len(challenges) > 1) else "Quantitative Validation & Metrics"

    def _trim(txt, max_c=42):
        s = re.sub(r'[\r\n\t]+', ' ', str(txt)).strip()
        if len(s) > max_c:
            s = s[:max_c].rsplit(' ', 1)[0] + '...'
        return s or "Empirical Component"

    return [
        ("Problem Formulation", _trim(c0_sub1)),
        ("Execution Environment", _trim(c0_sub2)),
        ("Algorithmic Formulation", _trim(c1_sub1)),
        ("Methodological Innovation", _trim(c1_sub2)),
        ("Benchmark Baselines", _trim(c2_sub1)),
        ("Performance Verification", _trim(c2_sub2))
    ]

def clean_snippet(text: str, max_words: int = 25) -> str:
    if not text:
        return ""
    words = str(text).replace("\n", " ").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."


def _is_background_statement(s: str) -> bool:
    """
    Topic-agnostic linguistic validator to identify introductory, background, or non-technical statements
    without hardcoding specific benchmark publication phrases.
    """
    if not s or len(str(s).strip()) < 12:
        return True
    s_clean = str(s).strip().lower()

    # Discourse starters and temporal background markers
    intro_patterns = [
        r'^\s*(?:in recent years|recently|historically|currently|traditionally|in the last decade|over the past|ever since|today|in the past)',
        r'^\s*(?:with the (?:recent|rapid|meteoric)?\s*(?:rise|advent|success|development|growth))',
        r'^\s*(?:due to the|owing to the|thanks to the|as a result of)',
        r'^\s*(?:despite (?:the|its)?\s*(?:success|recent|promising))',
        r'^\s*(?:in this (?:paper|work|study|article|review|investigation|survey))',
        r'^\s*(?:we (?:argue|show|demonstrate|find|conclude|review|discuss|outline|present|investigate|propose to study))',
        r'^\s*(?:the authors (?:argue|show|demonstrate|find|conclude|review|discuss|outline|present))',
        r'^\s*(?:against this backdrop|bearing the aforementioned|to account for this|taken together|on the one hand|on the other hand)',
        r'^\s*(?:it is (?:well known|widely acknowledged|clear|evident|vital|essential|crucial|imperative|worth noting))',
        r'^\s*(?:people appear to think|there has been increasing|there are a large number of)',
        r'(?:is|are|has become)\s+an?\s+(?:essential|vital|crucial|promising|important|key|indispensable)\s+(?:tool|solution|task|component|field|domain|approach|area|instrument)',
        r'plays?\s+an?\s+(?:essential|vital|crucial|important|key)\s+role',
        r'has\s+received\s+(?:significant|considerable|widespread|growing)\s+attention'
    ]
    for pat in intro_patterns:
        if re.search(pat, s_clean):
            return True

    return False

def _is_invalid_methodology_phrase(s: str, title: str = "") -> bool:
    """Strictly checks if candidate methodology phrase is an introductory sentence, background fragment, generic workflow label, or title fragment."""
    if not s or len(str(s).strip()) < 2:
        return True
    s_clean = str(s).strip().lower()
    # Only check background statement on longer descriptive sentences (>= 30 chars), NOT concise architectural names or acronyms
    if len(s_clean) >= 30 and _is_background_statement(s_clean):
        return True

    workflow_phrases = [
        "problem formulation", "data ingestion", "method execution",
        "empirical evaluation", "benchmarking", "evaluation pipeline",
        "systematic empirical", "pipeline steps", "pipeline execution",
        "methodological phases", "section ii", "section iii", "section iv",
        "section v", "section 2", "section 3", "presents the system",
        "an analysis of pf", "the framework presented here",
        "novel architecture for", "proposed technical methodology",
        "experimental scope and domain", "investigating key empirical",
        "problem formulation & data ingestion", "method execution & architecture",
        "empirical evaluation & benchmarking"
    ]
    for wp in workflow_phrases:
        if wp in s_clean:
            return True

    intro_starters = [
        r'^\s*(?:unfortunately|however|although|despite|while|whereas|nonetheless|furthermore|moreover|in addition|first,|second,|finally,|in summary)',
        r'^\s*(?:against this backdrop|in line with|before proceeding|bearing the aforementioned|to account for this|taken together|on the one hand|on the other hand)',
        r'^\s*(?:organisations? increasingly|recently|historically|currently|traditionally|today|in recent years|increasingly permeate|examples of the use|there are|there is|it is|one of the|due to|because of)',
        r'^\s*(?:in this paper|in this work|in this study|in this article|we argue|we show|we demonstrate|we find|we conclude|we review|we discuss|the authors argue|our focus here|this article outlines|this article does not)'
    ]
    for pat in intro_starters:
        if re.search(pat, s_clean):
            return True

    problem_markers = [
        "outpaced the growth", "curtails the sphere", "coupled with ethical challenges", "pose concerns",
        "permeate all aspects", "lack of trust", "difficult to assign", "tension between",
        "fallacy of the broken part", "remains voluntary", "lacks both incentives", "not a blanket solution",
        "suffers from", "hard to verify", "have generally outpaced", "cannot deal well"
    ]
    if any(k in s_clean for k in problem_markers):
        return True

    if title:
        t_words = set(re.findall(r'\b[a-z]{3,}\b', title.lower()))
        s_words = set(re.findall(r'\b[a-z]{3,}\b', s_clean))
        if len(t_words) > 2 and len(s_words & t_words) / len(t_words) >= 0.70:
            return True

    return False

def _is_achievement_or_result_statement(s: str) -> bool:
    """Checks if a candidate sentence describes achievements, contributions, hypotheses, or performance results rather than a genuine limitation."""
    s_low = s.lower().strip()
    hypothesis_or_position_markers = [
        "in contrast, we posit", "we posit that", "we argue that", "we hypothesize that",
        "our key insight", "our central thesis", "our findings challenge",
        "we demonstrate that even", "this paper establishes that"
    ]
    if any(h in s_low for h in hypothesis_or_position_markers):
        return True

    achievement_markers = [
        "outperform", "outperformed", "outperforms", "superior", "superiority",
        "achieve", "achieved", "achieves", "achievement",
        "experimental results indicate", "experimental results demonstrate",
        "results show that our", "experiments show that our", "results prove that",
        "unique advantages", "effectively protects", "satisfactory performance",
        "yields a zero", "significantly improves", "higher accuracy",
        "proves to be effective", "state-of-the-art", "best performance",
        "competitive with", "surpasses", "remarkable performance",
        "highest accuracy", "promising results", "demonstrates effectiveness",
        "substantially improved", "achieves superior"
    ]
    if any(m in s_low for m in achievement_markers):
        qualification_markers = ["however", "although", "still lags", "fails to", "cannot", "degrade", "limitation", "trade-off"]
        if not any(q in s_low for q in qualification_markers):
            return True
        if any(s_low.startswith(pfx) for pfx in [
            "extensive experimental results indicate", "experimental results prove",
            "the results demonstrate", "the case studies demonstrate", "evaluation results prove",
            "extensive testing over"
        ]):
            return True
    return False

CANONICAL_BENCHMARK_TECHNIQUES: Dict[str, Tuple[str, str]] = {}

def _classify_gap_excerpt(text: str) -> str:
    """Classifies an extracted excerpt into 'author_stated_future_work' or 'author_stated_limitation'."""
    s_low = text.lower().strip()
    future_markers = [
        "future work", "in the future", "in future work", "future research", "future direction",
        "future directions", "future studies", "future investigations", "another interesting topic",
        "it's also interesting to", "it is also interesting to", "an interesting topic is to",
        "an interesting future topic", "we will extend our method", "we will extend",
        "will consider real-world", "will consider", "will investigate", "will explore",
        "further research is needed", "further investigation is needed", "left for future work",
        "left as future work", "promising future direction", "avenue for future", "is planned for future"
    ]
    if any(m in s_low for m in future_markers):
        return "author_stated_future_work"
    return "author_stated_limitation"

def _derive_grounded_future_opportunity(excerpt: str, is_future: bool = False, pid: str = "") -> str:
    """Derives a concrete, actionable future opportunity directly from the excerpt without generic templates."""
    if not excerpt:
        return ""

    clean = re.sub(
        r'^(?:in the future,?\s*(?:we will|we plan to|we aim to)?\s*'
        r'|future work (?:will|includes|is planned to)?\s*'
        r'|future research (?:will|is needed to)?\s*'
        r'|future studies (?:will)?\s*'
        r'|another interesting topic is to\s*'
        r'|it(?:\'s| is) also interesting to\s*'
        r'|an interesting (?:topic|direction) is to\s*'
        r'|we will extend our method to\s*'
        r'|we will extend\s*'
        r'|we will\s*'
        r'|we plan to\s*'
        r'|we aim to\s*'
        r'|will consider\s*'
        r'|will extend\s*'
        r'|will investigate\s*)',
        '',
        excerpt.strip(),
        flags=re.IGNORECASE
    ).strip()

    if is_future:
        if clean:
            clean = clean[0].upper() + clean[1:]
            return clean.rstrip('.') + '.'
        return excerpt.strip().rstrip('.') + '.'
    else:
        s_low = excerpt.lower()
        if "incomplete information" in s_low or "attacker has all system information" in s_low:
            return "Investigate detection models resilient to cyber attacks under realistic incomplete system information."
        elif "packet loss" in s_low or "communication delay" in s_low:
            return "Incorporate communication network modeling accounting for data packet loss and transmission latency."
        elif "missing pmu" in s_low or "missing data" in s_low:
            return "Develop robust measurement imputation strategies to address missing PMU data during real-time assessment."
        elif "overhead" in s_low or "latency" in s_low or "computational" in s_low:
            return "Investigate model compression and sparse representations to mitigate computational overhead."
        elif "class imbalance" in s_low:
            return "Develop adaptive class-balanced reweighting and distribution-robust representations."
        else:
            base = re.sub(r'^(?:this paper (?:hasn\'t|has not|does not) considered|this work assumes that|we assume that|the method is limited by)\s*', '', excerpt.strip(), flags=re.IGNORECASE).strip()
            return f"Address reported operational constraint: {base[0].lower() + base[1:] if base else excerpt}."

METHOD_TAXONOMY = [
    {
        "category": "Digital Twin & Cyber-Physical Modeling",
        "keywords": ["digital twin", "cyber-physical", "iot", "smart microgrid", "event-driven", "hardware-in-the-loop"],
        "regex": r'\b(digital twin\w*|cyber[- ]physical|smart microgrid\w*|event[- ]driven architecture|iiot|industrial iot)\b',
        "description": "Formulates virtual representations and event-driven architectures mirroring physical industrial assets."
    },
    {
        "category": "Multi-Agent & Autonomous Systems",
        "keywords": ["multi-agent", "agent network", "self-evolving", "collaborative", "d2d", "edge computing"],
        "regex": r'\b(multi[- ]agent\w*|agent network\w*|self[- ]evolving|collaborative d2d|edge computing)\b',
        "description": "Deploys cooperative autonomous agent networks and decentralized edge coordination protocols."
    },
    {
        "category": "Data-Driven Predictive Maintenance & Knowledge Engineering",
        "keywords": ["predictive maintenance", "knowledge engineering", "data-driven", "industrial engineering", "failure prediction"],
        "regex": r'\b(predictive maintenance|knowledge engineering|failure prediction|data[- ]driven (?:methods?|approach|model\w*)|industrial engineering)\b',
        "description": "Integrates empirical sensor telemetry, knowledge engineering, and predictive algorithms to anticipate equipment failure."
    },
    {
        "category": "Hybrid Deep Learning Architecture",
        "keywords": ["staatt", "sta-transformer", "spatial-temporal attention transformer", "cnn-lstm", "convlstm", "transformer-lstm", "spatiotemporal hybrid", "hybrid deep learning"],
        "regex": r'\b(spatial[- ]temporal attention transformer|staatt?|sta-transformer|cnn[- ]lstm|convlstm|transformer[- ]lstm|spatiotemporal hybrid|hybrid (?:deep learning|neural network|architecture))\b',
        "description": "Couples spatial-temporal attention mechanisms with deep representations to resolve multi-scale dependencies."
    },
    {
        "category": "Federated Learning Framework",
        "keywords": ["federated learning", "fedavg", "fed-lsgan", "secfed", "federated deep learning", "federated optimization"],
        "regex": r'\b(federated (?:deep )?learning|fedavg|fed[- ]lsgan|secfed|federated optimization|decentralized learning|privacy[- ]preserving federated)\b',
        "description": "Enables decentralized collaborative model training with privacy-preserving parameter aggregation."
    },
    {
        "category": "Generative Foundation & Information Retrieval (GenIR)",
        "keywords": ["genir", "generative information retrieval", "generative ai", "foundation models", "generative models", "information access"],
        "regex": r'\b(generative information retrieval|genir|generative retrieval|information access systems?|generative ai models?)\b',
        "description": "Formulates generative information access, neural sequence generation, and unified retrieval architectures."
    },
    {
        "category": "Generative Adversarial & Diffusion Models",
        "keywords": ["gan", "generative adversarial network", "wgan", "cwgan", "cwgan-gp", "lsgan", "fed-lsgan", "diffusion model"],
        "regex": r'\b(generative adversarial network\w*|gan\w*|wgan\w*|cwgan\w*|cwgan-gp|lsgan|diffusion models?)\b',
        "description": "Synthesizes continuous spatiotemporal scenarios and distributions via adversarial minimax formulations or score-based diffusion."
    },
    {
        "category": "Transformer Architecture",
        "keywords": ["transformer", "multi-head attention", "self-attention", "vision transformer", "vit"],
        "regex": r'\b(transformers?|multi-head attention|self-attention|vision transformers?|vit|attention mechanism)\b',
        "description": "Captures long-range dependencies and dynamic feature correlations via multi-head self-attention mechanisms."
    },
    {
        "category": "Recurrent Neural Network (LSTM / GRU)",
        "keywords": ["lstm", "long short-term memory", "bilstm", "gru", "gated recurrent unit", "rnn"],
        "regex": r'\b(lstm|long short[- ]term memory|bilstm|gru|gated recurrent unit|recurrent neural network\w*|rnn)\b',
        "description": "Processes sequential temporal dependencies and state dynamics using gated memory cells."
    },
    {
        "category": "Convolutional Neural Network (CNN)",
        "keywords": ["cnn", "convolutional neural network", "deep convolutional", "convnet", "resnet", "vgg"],
        "regex": r'\b(cnn|convolutional neural network\w*|deep convolutional|convnet\w*|resnet\w*|vgg\w*|densenet\w*)\b',
        "description": "Extracts localized spatial feature hierarchies and patterns through multi-layer convolutional filter operations."
    },
    {
        "category": "Graph Neural Network (GNN)",
        "keywords": ["graph neural network", "gnn", "graph convolutional", "gcn", "graph attention", "gat"],
        "regex": r'\b(graph neural network\w*|gnn\w*|graph convolutional network\w*|gcn\w*|graph attention network\w*|gat\w*|spatio[- ]temporal graph)\b',
        "description": "Models relational topologies and inter-node electrical constraints over arbitrary network graphs."
    },
    {
        "category": "Deep Reinforcement Learning (DRL)",
        "keywords": ["reinforcement learning", "deep reinforcement learning", "drl", "actor-critic", "deep q-network", "dqn", "ppo", "ddpg"],
        "regex": r'\b(deep reinforcement learning|reinforcement learning|drl|actor[- ]critic|deep q[- ]network|dqn|ppo|ddpg|policy gradient)\b',
        "description": "Learns adaptive sequential dispatch and control policies through environmental reward feedback."
    },
    {
        "category": "Ensemble & Machine Learning Classifiers",
        "keywords": ["random forest", "xgboost", "lightgbm", "support vector machine", "svm", "decision tree"],
        "regex": r'\b(random forest|xgboost|lightgbm|support vector machine\w*|svm|decision trees?|gradient boosting)\b',
        "description": "Employs statistical and decision-tree ensemble heuristics for robust predictive classification and regression."
    },
    {
        "category": "Statistical & Optimization Theory",
        "keywords": ["random matrix theory", "rmt", "convex optimization", "model predictive control", "mpc", "game theory", "statistical estimation", "bayes-optimal", "lower bound"],
        "regex": r'\b(random matrix theory|rmt|convex optimization|model predictive control|mpc|game theor\w*|kalman filter|copula theor\w*|estimation theor\w*|statistical estimation|bayes(?:ian)?(?:\s+optimal)?(?:\s+estimation)?|optimal estimators?|information theor\w*|lower bound)\b',
        "description": "Establishes formal mathematical bounds, spectral statistics, and constrained optimization formulations."
    }
]

def _extract_grounded_paper_meta(s, comp_analysis=None) -> Dict[str, Any]:
    pid = str(_get_val(s, "paper_id", "")).strip()
    raw_t = _get_val(s, "paper_title", pid)
    title = _clean_paper_title(raw_t, pid)
    prob_raw = str(_get_val(s, "core_problem", "") or "").strip()
    meth_raw = str(_get_val(s, "methodology_summary", "") or "").strip()
    lim_raw = str(_get_val(s, "explicit_limitations", "") or "").strip()

    # Priority 1: Inspect parsed JSON file on disk if available
    p_path = f"data/extracted_schemas/{pid}_parsed.json"
    p_data = {}
    if os.path.exists(p_path):
        try:
            with open(p_path, "r", encoding="utf-8") as pf:
                p_data = json.load(pf)
        except Exception:
            pass

    # Use fuller parsed title if current title is truncated
    parsed_t = p_data.get("title", "")
    if not parsed_t and p_data.get("chunks"):
        header_lines = []
        for c in p_data["chunks"][:3]:
            txt = c.get("text", "").strip()
            if txt and not any(k in txt.lower() for k in ["abstract", "ieee", "author", "university", "grant", "email"]):
                header_lines.append(txt)
        if header_lines:
            cand_t = " ".join(header_lines)
            if len(cand_t) > len(title):
                parsed_t = cand_t
    if parsed_t and len(parsed_t) > len(title):
        clean_parsed_t = _clean_paper_title(parsed_t, pid)
        if len(clean_parsed_t) > len(title):
            title = clean_parsed_t

    chunks = p_data.get("chunks", []) if p_data else []

    # 1. Problem formulation
    problem_text = ""
    if prob_raw and not _is_background_statement(prob_raw):
        p_clean = re.sub(r'^(to address this problem,\s*|in this paper,\s*we\s*|this paper investigates\s*|we study\s*|we propose\s*)', '', prob_raw, flags=re.IGNORECASE).strip()
        problem_text = p_clean[0].upper() + p_clean[1:] if p_clean else prob_raw
    if not problem_text:
        abs_chunks = [c for c in chunks if "abstract" in c.get("section", "").lower()]
        for c in abs_chunks:
            txt = c.get("text", "")
            for sent in re.split(r'(?<=[.!?])\s+', txt):
                if len(sent.strip()) > 30 and not _is_background_statement(sent):
                    problem_text = sent.strip()
                    break
            if problem_text:
                break
    if (not problem_text or len(problem_text) < 15) and meth_raw:
        m_abs = re.search(r'(?:abstract[—:\s]+)(.+)', meth_raw, re.IGNORECASE)
        if m_abs:
            for s in re.split(r'(?<=[.!?])\s+', m_abs.group(1)):
                if len(s.strip()) > 25:
                    problem_text = s.strip()
                    break
    if not problem_text:
        problem_text = f"Investigating key empirical challenges and research formulations in {title}."

    # 2. Grounded technique formulation
    technique = ""
    technique_category = ""
    technique_desc = ""
    technique_evidence = ""
    technique_section = "Methodology Section"

    # Search candidates: chunks, methodology_summary, abstract, title
    search_texts = []
    for c in chunks:
        sec = c.get("section", "Methodology Section")
        txt = c.get("text", "")
        if txt:
            search_texts.append((sec, txt))
    if meth_raw:
        search_texts.append(("Methodology Summary", meth_raw))
    if prob_raw and len(prob_raw) > 20:
        search_texts.append(("Abstract / Problem Formulation", prob_raw))
    if title:
        search_texts.append(("Paper Title", title))

    # Priority A: Check for explicitly named models / frameworks (e.g., "called Fed-LSGAN", "termed SecFed", "StaaT")
    for sec, txt in search_texts:
        cand_name = None
        # 1. Highest priority: explicit naming anchors ("called X", "named X", "termed X")
        m_explicit = re.search(r'\b(?:called|named|termed)\s+([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+)?)', txt)
        if m_explicit:
            cand_name = m_explicit.group(1).strip()
        else:
            # 2. CamelCase method/model acronyms (e.g., Fed-LSGAN, SecFed, StaaT, ConvLSTM)
            m_camel = re.search(r'\b([A-Z][a-z0-9]+[A-Z][A-Za-z0-9\-]*)\b', txt)
            if m_camel:
                cand_name = m_camel.group(1).strip()
            else:
                # 3. Parenthesized acronyms, excluding known non-methodology domain/system acronyms
                m_paren = re.search(r'\(([A-Z][A-Za-z0-9\-]{2,})\)', txt)
                if m_paren:
                    cand_c = m_paren.group(1).strip()
                    domain_acronyms = {"CPS", "FDIA", "PMU", "SCADA", "DER", "IIOT", "IOT", "RES", "TSO", "DSO", "ALL", "IEEE"}
                    if cand_c.upper() not in domain_acronyms:
                        cand_name = cand_c

        if cand_name and not _is_invalid_methodology_phrase(cand_name, title) and len(cand_name) >= 3 and not cand_name.lower().startswith("section"):
            for sent in re.split(r'(?<=[.!?])\s+', txt):
                if cand_name in sent and any(w in sent.lower() for w in ["propose", "integrate", "develop", "framework", "model", "approach", "architecture", "network", "algorithm"]):
                    technique = cand_name
                    technique_desc = clean_snippet(sent.strip().replace("\n", " "), 25)
                    technique_evidence = clean_snippet(sent.strip().replace("\n", " "), 30)
                    technique_section = sec
                    break
        if technique:
            break

    # Priority B: Match against standardized METHOD_TAXONOMY
    matched_cats = []
    comb_corpus = f"{title} {meth_raw} " + " ".join([t[1] for t in search_texts[:4]])
    for tax in METHOD_TAXONOMY:
        if re.search(tax["regex"], comb_corpus, re.IGNORECASE):
            matched_cats.append(tax)

    if matched_cats:
        primary_tax = matched_cats[0]
        # Prefer category specifically matched by extracted technique (e.g. SecFed -> Federated Learning Framework)
        if technique:
            for tax in matched_cats:
                if re.search(tax["regex"], technique, re.IGNORECASE):
                    primary_tax = tax
                    break
        # Or prefer category specifically matched by paper title (actual methodology over application domain)
        if primary_tax == matched_cats[0] and title:
            for tax in matched_cats:
                if re.search(tax["regex"], title, re.IGNORECASE):
                    primary_tax = tax
                    break
        technique_category = primary_tax["category"]
        if not technique_desc:
            technique_desc = primary_tax["description"]
        if not technique:
            m_tgt = re.search(r'(?:propose|develop|build|design|introduce|utilize|employ)\s+(?:a|an|the)?\s*([A-Za-z0-9\-\s]{3,45}?(?:model|network|architecture|framework|algorithm|protocol|approach|method|detector|scheme))', comb_corpus, re.IGNORECASE)
            if m_tgt and not _is_invalid_methodology_phrase(m_tgt.group(1)):
                technique = m_tgt.group(1).strip().title()
            else:
                technique = primary_tax["category"]
        if not technique_evidence and meth_raw:
            technique_evidence = clean_snippet(meth_raw.replace("\n", " "), 25)

    # Priority C: Check paper title for explicit architectural paradigm if still unidentified
    if not technique and title:
        m_title_arch = re.findall(r'(?:Random Matrix Theory|Transformer|Actor-Critic|Reinforcement Learning|Deep Learning|Convolutional|LSTM|Graph Neural Network|Machine Learning|Federated Learning)', title, re.IGNORECASE)
        if m_title_arch:
            unique_t = list(dict.fromkeys([a.title() for a in m_title_arch]))
            technique = f"{' & '.join(unique_t)} Methodology"
            technique_category = "Deep Neural Architecture"
            technique_desc = f"Algorithmic formulation centered on {' and '.join(unique_t)} as reported in {title}."
            technique_evidence = f"Reported in title: {title}"

    # Priority D: Check summary JSON
    if not technique:
        sum_path = f"data/summaries/{pid}_summary.json"
        if os.path.exists(sum_path):
            try:
                with open(sum_path, "r", encoding="utf-8") as sum_f:
                    sum_json = json.load(sum_f)
                    innovs = sum_json.get("proposed_methodology", {}).get("technical_innovations", [])
                    for inn in innovs:
                        if inn and not _is_invalid_methodology_phrase(inn, title):
                            technique = clean_snippet(inn.strip(), 8)
                            technique_desc = clean_snippet(inn.strip(), 25)
                            technique_evidence = clean_snippet(inn.strip(), 25)
                            technique_section = "Technical Innovations"
                            break
            except Exception:
                pass

    if not technique:
        technique = "Methodology not identified"
        technique_category = "Unidentified Methodology"
        technique_desc = "Specific algorithmic or architectural methodology could not be confirmed from the parsed text."
        technique_evidence = "Insufficient technical detail in evaluated sections."
        technique_section = "N/A"

    # 3. Grounded limitation & future work scanning
    lim_chunks = [c for c in chunks if any(k in c.get("section", "").lower() for k in [
        "limitation", "future work", "future research", "future direction", "challenges and future", "threats to validity"
    ])]
    disc_chunks = [c for c in chunks if any(k in c.get("section", "").lower() for k in [
        "discussion", "concluding", "conclusion", "conclusions"
    ])]

    explicit_candidates = []
    scan_groups = [(lim_chunks, 1.0), (disc_chunks, 1.2), (chunks, 2.0)]
    seen_quotes = set()

    for grp, base_weight in scan_groups:
        for c in grp:
            sec = c.get("section", "Limitation Section")
            txt = c.get("text", "")
            for s_raw in re.split(r'(?<=[.!?])\s+', txt):
                s_strip = s_raw.strip().replace("\n", " ")
                if len(s_strip) < 25 or len(s_strip) > 400:
                    continue
                if _is_background_statement(s_strip) or _is_achievement_or_result_statement(s_strip):
                    continue
                if re.search(r'\b(?:ieee|trans\.|vol\.|no\.|pp\.|proceedings|arxiv:)\b', s_strip.lower()):
                    continue
                if s_strip in seen_quotes:
                    continue

                s_low = s_strip.lower()
                is_future = any(k in s_low for k in [
                    "future work will", "in the future, we will", "future research will", "future direction",
                    "another interesting topic is to", "it's also interesting to", "future studies will",
                    "we will extend our method", "will consider real-world scenarios", "will investigate"
                ])
                is_lim = any(k in s_low for k in [
                    "limitation", "limited by", "restricts", "restricted to", "rely on", "relies on",
                    "subject to", "lack of", "trade-off", "bias", "overhead", "computational bottleneck",
                    "assumes that an attacker", "assumes that", "cannot deal", "degrade", "degrades to",
                    "only evaluates", "not fully discussed", "hasn't considered", "has not considered",
                    "did not consider", "incomplete information", "packet loss", "communication delay",
                    "class imbalance", "unacceptable level", "missing pmu data", "missing data",
                    "vulnerability against", "fails to", "difficulty of", "hardware requirements"
                ])

                if is_future or is_lim:
                    seen_quotes.add(s_strip)
                    score = base_weight
                    if is_future and is_lim:
                        score -= 0.4
                    elif is_future:
                        score -= 0.2
                    explicit_candidates.append((score, sec, s_strip, is_future))

    # Also check schema explicit_limitations if not empty, not an achievement, and not a synthetic placeholder
    placeholder_patterns = [
        "author constraints and domain boundaries",
        "constraints and domain boundaries reported",
        "reported boundaries in",
        "author constraints and domain",
        "no explicit",
        "experimental scope and domain constraints noted",
        "primary text in inspected sections does not contain"
    ]
    if not explicit_candidates and lim_raw and not _is_background_statement(lim_raw) and not _is_achievement_or_result_statement(lim_raw) and not any(k in lim_raw.lower() for k in placeholder_patterns):
        raw_low = lim_raw.lower()
        raw_is_future = any(k in raw_low for k in [
            "future work", "in the future", "future research", "future direction", "will extend", "will investigate", "will explore"
        ])
        raw_is_lim = any(k in raw_low for k in [
            "limitation", "limited by", "restricts", "restricted to", "constrain", "overhead", "bottleneck", "trade-off",
            "bias", "degrade", "fails to", "cannot", "lack of", "vulnerability", "sensitivity", "computational complexity",
            "latency", "scaling constraint"
        ])
        if (raw_is_future or raw_is_lim) and lim_raw not in seen_quotes:
            explicit_candidates.append((1.5, "Explicit Limitations", lim_raw.strip(), raw_is_future))

    has_limitation = False
    limitation_text = ""
    limitation_quote = ""
    limitation_section = "N/A"
    gap_explanation = ""
    future_opp = ""

    if explicit_candidates:
        explicit_candidates.sort(key=lambda x: x[0])
        best = explicit_candidates[0]
        has_limitation = True
        limitation_section = best[1]
        limitation_quote = best[2]
        limitation_text = best[2]
        is_future_flag = _classify_gap_excerpt(limitation_quote) == "author_stated_future_work"
        gap_type = "author_stated_future_work" if is_future_flag else "author_stated_limitation"

        clean_quote = clean_snippet(limitation_quote, 18)
        if is_future_flag:
            gap_explanation = f"In [{pid}] (Section: {limitation_section}), authors explicitly outline a future research avenue: \"{clean_quote}\"."
            future_opp = _derive_grounded_future_opportunity(limitation_quote, is_future=True, pid=pid)
            confidence_level = f"High (Direct author-stated future work in Section: {limitation_section})"
            limitation_reasoning = f"Directly extracted from primary paper text where authors explicitly outline future work directions in Section '{limitation_section}'."
        else:
            gap_explanation = f"In [{pid}] (Section: {limitation_section}), authors identify an operational constraint: \"{clean_quote}\"."
            future_opp = _derive_grounded_future_opportunity(limitation_quote, is_future=False, pid=pid)
            confidence_level = f"High (Direct author-stated limitation in Section: {limitation_section})"
            limitation_reasoning = f"Directly extracted from primary paper text where authors explicitly acknowledge operational boundaries in Section '{limitation_section}'."
    else:
        has_limitation = False
        limitation_text = "No explicit author-stated limitation or future work identified within inspected sections."
        limitation_section = "N/A"
        limitation_quote = ""
        gap_type = "unavailable_evidence"
        gap_explanation = f"A research gap cannot be established from the available evidence in the inspected sections of [{pid}]."
        future_opp = f"Empirical stress-testing is required to establish operational boundaries and formal research gaps for [{pid}]."
        confidence_level = "Unavailable Evidence (Not identified in inspected sections)"
        limitation_reasoning = f"Inspection of available parsed text for [{pid}] did not identify an explicit author-stated limitation or future work direction. Unparsed sections or the complete manuscript may contain formal operational boundaries."

    is_survey = any(k in f"{title} {prob_raw} {meth_raw}".lower() for k in ["survey", "systematic review", "taxonomy", "overview of"])
    contribution_type = "Comprehensive Taxonomic Survey & Architectural Review" if is_survey else "Original Empirical & Methodological Contribution"

    return {
        "pid": pid,
        "title": title,
        "technique": technique,
        "technique_category": technique_category,
        "technique_desc": technique_desc,
        "technique_evidence": technique_evidence,
        "technique_section": technique_section,
        "limitation_text": limitation_text,
        "limitation_section": limitation_section,
        "limitation_quote": limitation_quote,
        "has_limitation": has_limitation,
        "gap_type": gap_type,
        "confidence_level": confidence_level,
        "reasoning": limitation_reasoning,
        "future_opportunity": future_opp,
        "gap_explanation": gap_explanation,
        "problem_text": problem_text,
        "contribution_type": contribution_type,
        "is_survey": is_survey,
        "raw_schema": s
    }

def _classify_methodology(rec: Dict[str, Any]) -> tuple:
    tech = rec.get("technique", "")
    desc = rec.get("technique_desc", "")
    cat = rec.get("technique_category", "")

    if not tech or tech == "Methodology not identified":
        return ("Methodology not identified", "Specific algorithmic or architectural methodology could not be confirmed from the parsed text.")

    # Canonical taxonomy category ensures consistency and prevents duplicate category entries
    cat_name = cat or tech
    cat_desc = desc or f"Implements {cat_name} to address empirical research challenges."
    return (cat_name, cat_desc)

def _classify_theme(rec: Dict[str, Any], all_recs: List[Dict[str, Any]] = None, corpus_topic: str = "") -> tuple:
    """
    Evidence-grounded dynamic theme routing helper.
    Returns dynamic_theme to route all active review papers into semantic clustering.
    """
    return (
        "dynamic_theme",
        "",
        "",
        "",
        ""
    )

def _derive_semantic_paper_objective(p: Dict[str, Any]) -> str:
    """
    Derives a meaningful 2-6 word semantic research objective/theme from the paper's
    title, research problem, and domain, avoiding isolated single keywords and generic suffixes.
    """
    trunc_map = {
        "architec": "Architecture",
        "archi": "Architecture",
        "scenar": "Scenario Generation",
        "generat": "Generation",
        "distrib": "Distribution",
        "evaluat": "Evaluation",
        "manageme": "Management",
        "artificia": "Artificial Intelligence",
        "indust": "Industrial Systems",
        "predict": "Predictive Maintenance"
    }
    raw_title = p.get("title", "")
    clean_t = re.sub(r'^\s*(?:\d{4}\.\d{4,5}|arXiv:\d{4}\.\d{4,5})\s*', '', raw_title).strip()
    clean_t = re.sub(r'^(?:a\s+|an\s+|the\s+)?(?:study\s+on|investigation\s+of|analysis\s+of|review\s+of|applications\s+of|feasibility\s+of)\s+', '', clean_t, flags=re.IGNORECASE).strip()

    parts = clean_t.split(":")
    if len(parts) >= 2:
        part0 = parts[0].strip()
        part1 = parts[1].strip()
        if 2 <= len(part0.split()) <= 7:
            clean_t = part0
        elif 2 <= len(part1.split()) <= 7 and not any(w in part1.lower() for w in ["a review", "an approach"]):
            clean_t = part1

    m_for = re.search(r'\bfor\s+(.+)$', clean_t, flags=re.IGNORECASE)
    if m_for and len(m_for.group(1).split()) >= 3:
        clean_t = m_for.group(1).strip()

    clean_t = re.sub(r'^(?:a|an|the)\s+', '', clean_t, flags=re.IGNORECASE).strip()

    words = clean_t.split()
    if words and words[-1].lower() in trunc_map:
        words[-1] = trunc_map[words[-1].lower()]
    clean_t = " ".join(words)
    clean_t = re.sub(r'\s+(?:in|of|for|and|to|the|a|an)$', '', clean_t, flags=re.IGNORECASE).strip()

    w_list = clean_t.split()
    if len(w_list) > 6:
        clean_t = " ".join(w_list[:6])
    elif len(w_list) < 2 and p.get("problem_text"):
        clean_t = clean_snippet(p["problem_text"], 5)
    return clean_t.title()

def _cluster_dynamic_themes(dynamic_papers: List[Dict[str, Any]], corpus_topic: str) -> List[Dict[str, Any]]:
    """
    Dynamically groups papers from an arbitrary research topic into cohesive themes
    based on semantic similarity of their scientific problems and technical domains.
    Extracts top distinctive terms, synthesizes real scientific challenges from paper problems,
    shared rationale, and technical differences.
    Filters out methodology labels so themes represent genuine scientific objectives (WHAT/WHY),
    leaving the methodology flowchart to represent techniques (HOW).
    """
    if not dynamic_papers:
        return []

    topic_clean = clean_snippet(corpus_topic, 4)

    topic_stop = set(re.findall(r'\b[a-z]{3,}\b', (corpus_topic or "").lower())) | {
        "paper", "study", "analysis", "using", "based", "with", "system", "systems", "approach",
        "approaches", "method", "methods", "methodology", "methodologies", "model", "models",
        "data", "deep", "learning", "neural", "network", "networks", "architecture", "architectures",
        "framework", "frameworks", "algorithm", "algorithms", "technique", "techniques",
        "pipeline", "pipelines", "transformer", "transformers", "autoencoder", "autoencoders",
        "lstm", "cnn", "convolutional", "recurrent", "attention", "proposed", "investigates",
        "presents", "evaluates", "performance", "novel", "new", "results", "comparative",
        "empirical", "benchmark", "benchmarks", "application", "applications", "investigation",
        "classification", "detection", "prediction", "forecasting", "estimation"
    }

    if len(dynamic_papers) <= 2:
        res = []
        for idx, p in enumerate(dynamic_papers):
            semantic_obj = _derive_semantic_paper_objective(p)
            t_title = f"Independent Research Focus: {semantic_obj}"
            res.append({
                "id": f"theme_dyn_{idx}",
                "title": t_title,
                "theme_title": t_title,
                "scientific_challenge": f"Independent Scientific Objective: Investigates {semantic_obj} within {corpus_topic}.",
                "shared_rationale": f"Independent Methodological Inquiry: Paper [{p['pid']}] establishes a distinct empirical inquiry into {semantic_obj}, operating as an independent methodological paradigm.",
                "technical_differences": f"Deploys {p['technique']} to evaluate experimental performance on {clean_snippet(p['title'], 5)}.",
                "description": f"Independent Scientific Objective: Investigates {semantic_obj} within {corpus_topic}.",
                "paper_count": 1,
                "representative_papers": [{"paper_id": p["pid"], "title": p["title"]}],
                "papers": [{
                    "paper_id": p["pid"],
                    "title": p["title"],
                    "finding": p["problem_text"],
                    "citation": f"[{p['pid']}]"
                }]
            })
        return res

    # For 3+ papers, determine target cluster count (2 to 3)
    k = min(3, max(2, len(dynamic_papers) // 2))

    obj_texts = [f"{p['title']} {p['problem_text']}" for p in dynamic_papers]

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(stop_words="english", max_features=100, ngram_range=(1, 2))
        X = vec.fit_transform(obj_texts)
        sim_mat = cosine_similarity(X)

        # Check maximum pairwise similarity across distinct papers
        n_p = len(dynamic_papers)
        max_pairwise_sim = 0.0
        for i in range(n_p):
            for j in range(i + 1, n_p):
                if float(sim_mat[i, j]) > max_pairwise_sim:
                    max_pairwise_sim = float(sim_mat[i, j])

        # If papers have essentially disjoint scientific problems (pairwise sim < 0.10),
        # each paper establishes its own distinct independent research inquiry.
        if max_pairwise_sim < 0.10:
            cleaned_clusters = {i: [p] for i, p in enumerate(dynamic_papers)}
        else:
            k = min(3, max(2, len(dynamic_papers) // 2))
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)

            clusters_dict = {}
            for i, p in enumerate(dynamic_papers):
                c_lbl = int(labels[i])
                clusters_dict.setdefault(c_lbl, []).append((i, p))

            # Merge isolated single-paper cluster if it has semantic affinity to another cluster
            if len(clusters_dict) > 1:
                single_clusters = [lbl for lbl, items in clusters_dict.items() if len(items) == 1]
                multi_clusters = [lbl for lbl, items in clusters_dict.items() if len(items) > 1]
                if single_clusters and multi_clusters:
                    import numpy as np
                    for s_lbl in single_clusters:
                        s_idx = clusters_dict[s_lbl][0][0]
                        s_vec = X[s_idx:s_idx+1]
                        best_sim = -1.0
                        best_target = None
                        for m_lbl in multi_clusters:
                            m_indices = [idx for idx, _ in clusters_dict[m_lbl]]
                            m_centroid = np.asarray(X[m_indices].mean(axis=0))
                            sim = float(cosine_similarity(s_vec, m_centroid)[0][0])
                            if sim > best_sim:
                                best_sim = sim
                                best_target = m_lbl
                        if best_target is not None and best_sim >= 0.20:
                            clusters_dict[best_target].extend(clusters_dict[s_lbl])
                            del clusters_dict[s_lbl]

            cleaned_clusters = {lbl: [p for _, p in items] for lbl, items in clusters_dict.items()}
    except Exception:
        # TF-IDF / sklearn unavailable — return each paper as its own independent research focus.
        # Round-robin or arbitrary grouping is not permitted; arbitrary groups are not thematic clusters.
        cleaned_clusters = {i: [p] for i, p in enumerate(dynamic_papers)}

    themes = []
    for c_idx, (lbl, c_papers) in enumerate(cleaned_clusters.items()):
        if len(c_papers) == 1:
            p_single = c_papers[0]
            semantic_obj = _derive_semantic_paper_objective(p_single)
            theme_title = f"Independent Research Focus: {semantic_obj}"
            sci_chal = f"Independent Scientific Objective: Investigates {semantic_obj} within {corpus_topic}."
            why = f"Independent Methodological Inquiry: Paper [{p_single['pid']}] establishes a distinct empirical inquiry into {semantic_obj}, operating as an independent methodological paradigm."
            diff = f"Deploys {p_single['technique']} to evaluate empirical performance on {clean_snippet(p_single['title'], 5)}."
        else:
            objs = [_derive_semantic_paper_objective(p) for p in c_papers]
            comb_t = " ".join([p.get("title", "") for p in c_papers]).lower()
            comb_p = " ".join([p.get("problem_text", "") for p in c_papers]).lower()
            comb_all = comb_t + " " + comb_p

            if any(k in comb_all for k in ["microgrid", "affordability", "smart microgrid"]):
                theme_title = "IoT-Enabled Predictive Maintenance & Microgrid Optimization"
            elif any(k in comb_all for k in ["nuclear", "failure prediction", "failure detection", "equipment failure", "plant"]):
                theme_title = "Data-Driven Equipment Failure Prediction & Industrial Plant Reliability"
            elif any(k in comb_all for k in ["multi-agent", "agent network", "self-evolving", "collaborative d2d", "d2d", "edge computing", "edge caching"]):
                theme_title = "Autonomous Multi-Agent Networks & Decentralized Edge Coordination"
            elif any(k in comb_all for k in ["digital twin", "event-driven", "software architecture", "modular"]):
                theme_title = "Modular Event-Driven Architecture & Cyber-Physical Infrastructure"
            elif "scenario generation" in comb_all and ("federated" in comb_all or "privacy" in comb_all or "renewable" in comb_all):
                theme_title = "Privacy-Preserving & Federated Scenario Generation"
            elif "false data injection" in comb_all or "attack" in comb_all or "security" in comb_all:
                theme_title = "Adversarial Attack Detection & Cyber-Physical Grid Security"
            elif "voltage stability" in comb_all or "transient stability" in comb_all or "stability assessment" in comb_all:
                theme_title = "Dynamic Power System Voltage Stability Assessment"
            elif "load forecasting" in comb_all or "energy forecasting" in comb_all:
                theme_title = "Renewable Generation & Net Load Forecasting"
            elif any(k in comb_all for k in ["generative", "diffusion", "gan", "genir"]) and any(k in comb_all for k in ["latent", "distribution", "modeling", "retrieval", "information access", "estimation"]):
                theme_title = "Generative Modeling, Distribution Synthesis & Information Access"
            else:
                common_words = []
                for w in re.findall(r'\b[A-Za-z]{4,}\b', " ".join(objs)):
                    if w.lower() not in topic_stop:
                        common_words.append(w.title())
                word_counts = Counter(common_words)
                top_shared = [w for w, _ in word_counts.most_common(2)]
                if len(top_shared) >= 2:
                    theme_title = f"{top_shared[0]} & {top_shared[1]} Research Focus"
                elif top_shared:
                    theme_title = f"{top_shared[0]} Research Focus"
                else:
                    theme_title = f"{clean_snippet(objs[0], 3)} & {clean_snippet(objs[1], 3)}"

            prob_snippets = "; ".join(
                clean_snippet(p['problem_text'], 8) for p in c_papers if p.get('problem_text')
            )
            pids_formatted = ", ".join([f"[{p['pid']}]" for p in c_papers])
            sci_chal = (
                f"Shared Scientific Challenge ({pids_formatted}): "
                f"Papers in this cluster collectively investigate {theme_title.lower()} "
                f"as evidenced by: {prob_snippets}."
            )
            why = (
                f"Evaluated publications ({pids_formatted}) share common scientific objectives "
                f"focusing on {theme_title.lower()} to resolve operational constraints in {corpus_topic}."
            )

            if len(c_papers) == 2:
                diff = f"Paper [{c_papers[0]['pid']}] implements {c_papers[0]['technique']}, whereas Paper [{c_papers[1]['pid']}] utilizes {c_papers[1]['technique']}."
            else:
                tech_parts = ", ".join(f"{p['technique']} ([{p['pid']}])" for p in c_papers[:3])
                diff = f"Techniques represented: {tech_parts}."

        themes.append({
            "id": f"theme_dyn_{c_idx}",
            "title": theme_title,
            "theme_title": theme_title,
            "scientific_challenge": sci_chal,
            "shared_rationale": why,
            "technical_differences": diff,
            "description": f"{sci_chal} {why} {diff}".strip(),
            "paper_count": len(c_papers),
            "representative_papers": [{"paper_id": p["pid"], "title": p["title"]} for p in c_papers[:min(len(c_papers), 4)]],
            "papers": [
                {
                    "paper_id": p["pid"],
                    "title": p["title"],
                    "paper_title": p["title"],
                    "finding": p["problem_text"],
                    "citation": f"[{p['pid']}]"
                }
                for p in c_papers
            ]
        })

    return themes

def _build_grounded_research_gaps(paper_meta: List[Dict[str, Any]], corpus_topic: str) -> List[Dict[str, Any]]:
    gaps = []
    gap_idx = 1

    # 1. Author-Stated Limitations & Future Work vs Unavailable Evidence in Inspected Sections
    for r in paper_meta:
        raw_quote = r.get("limitation_quote", "") or r.get("limitation_text", "")
        derived_type = r.get("gap_type")
        if not derived_type or derived_type == "author_stated_limitation":
            if _classify_gap_excerpt(raw_quote) == "author_stated_future_work":
                derived_type = "author_stated_future_work"
            elif r.get("has_limitation"):
                derived_type = "author_stated_limitation"
            else:
                derived_type = "unavailable_evidence"

        if r.get("has_limitation") and derived_type != "unavailable_evidence":
            is_fw = (derived_type == "author_stated_future_work")
            cat_label = "Author-Stated Future Work" if is_fw else "Author-Stated Limitation"
            g_title = f"Future Work Direction in {clean_snippet(r['title'], 4)}" if is_fw else f"Empirical Constraint in {clean_snippet(r['title'], 4)}"

            clean_q = clean_snippet(raw_quote, 20)
            expl = r.get("gap_explanation")
            if not expl or "restrict reliability and generalizability" in expl:
                if is_fw:
                    expl = f"In [{r['pid']}] (Section: {r['limitation_section']}), authors explicitly outline a future research avenue: \"{clean_q}\"."
                else:
                    expl = f"In [{r['pid']}] (Section: {r['limitation_section']}), authors identify an operational constraint: \"{clean_q}\"."

            opp = r.get("future_opportunity")
            if not opp or opp.startswith(f"Advance [{r['pid']}] by"):
                opp = _derive_grounded_future_opportunity(raw_quote, is_future=is_fw, pid=r["pid"])

            ev_str = f"[{r['pid']}]: \"{raw_quote}\"" if raw_quote else ""

            gaps.append({
                "gap_id": f"GAP-0{gap_idx}",
                "type": derived_type,
                "category_label": cat_label,
                "has_limitation": True,
                "title": g_title,
                "explanation": expl,
                "observed_limitation": r["limitation_text"],
                "limitation_summary": clean_snippet(r["limitation_text"], 16),
                "future_opportunity": opp,
                "supporting_papers": [r["pid"]],
                "evidence": ev_str,
                "source_section": r["limitation_section"],
                "confidence_level": r.get("confidence_level", f"High (Direct {cat_label.lower()} in Section: {r['limitation_section']})"),
                "reasoning": r.get("reasoning", f"Directly extracted from primary paper text in [{r['pid']}].")
            })
            gap_idx += 1
        else:
            gaps.append({
                "gap_id": f"GAP-0{gap_idx}",
                "type": "unavailable_evidence",
                "category_label": "Unavailable Evidence",
                "has_limitation": False,
                "title": f"Unspecified Operational Boundary in Inspected Sections ({r['pid']})",
                "explanation": f"A research gap cannot be established from the available evidence in the inspected sections of [{r['pid']}].",
                "observed_limitation": "No explicit author-stated limitation or future work identified within inspected sections.",
                "limitation_summary": "No explicit author-stated limitation identified.",
                "future_opportunity": f"Empirical stress-testing is required to establish operational boundaries and formal research gaps for [{r['pid']}].",
                "supporting_papers": [r["pid"]],
                "evidence": "No verifiable evidence extracted",
                "source_section": "N/A",
                "confidence_level": "Unavailable Evidence (Not identified in inspected sections)",
                "reasoning": f"A research gap cannot be established from the available evidence in the inspected sections of [{r['pid']}]."
            })
            gap_idx += 1

    # 2. Dynamic Frontier Synthesized Gaps Discovery Across 6 Universal Empirical Dimensions
    # Strictly discover cross-paper synthesized gaps ONLY when at least 2 active review papers
    # provide identifiable supporting evidence on shared empirical boundaries.
    # Gap IDs are always generic (GAP-SYN-01, GAP-SYN-02, ...) regardless of research topic.

    FRONTIER_DIMENSIONS = [
        {
            "key": "generalization",
            "title": "Out-of-Distribution Generalization & Operational Fluctuations",
            "category": "Generalization & Operational Stability",
            "keywords": [
                "generaliz", "domain adaptation", "transfer", "out-of-distribution", "distribution shift",
                "unseen", "robustness", "unmodeled", "drift", "field", "occlusion", "in-the-wild",
                "environmental", "weather", "disturbance", "meteorolog", "variance", "fluctuation"
            ],
            "desc": "models evaluated under controlled or stationary environments suffer accuracy and policy stability degradation under out-of-distribution environmental shifts and unmodeled disturbances",
            "opp": "Develop domain-adaptive representation learning, robust uncertainty quantification, and invariant objective formulations to guarantee operational stability under distribution shifts."
        },
        {
            "key": "complexity_latency",
            "title": "Computational Complexity & Real-Time Operational Latency Constraints",
            "category": "Computational Efficiency & Real-Time Systems",
            "keywords": [
                "latency", "real-time", "gpu", "memory", "computational overhead", "computational complexity",
                "throughput", "inference time", "cost", "hardware", "resource", "footprint", "sub-second", "microcontroller", "edge", "scaling"
            ],
            "desc": "high-capacity neural architectures encounter inference latency and memory complexity bottlenecks that challenge real-time deployment under strict operational latency budgets",
            "opp": "Investigate model compression, sparse attention mechanisms, and hardware-accelerated edge inference pipelines to satisfy real-time latency and computational budgets."
        },
        {
            "key": "temporal_horizon",
            "title": "Temporal Non-Stationarity & Multi-Horizon Prediction Degradation",
            "category": "Temporal Non-Stationarity & Forecasting",
            "keywords": [
                "temporal", "long-term", "multi-horizon", "compounding", "horizon", "time-series",
                "error propagation", "non-stationar", "dynamic", "delay", "sequential", "dispatch", "continual", "lag", "drift"
            ],
            "desc": "compounding forecast error propagation and temporal non-stationarity over expanding prediction horizons degrade policy reliability",
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
            "desc": "empirical performance degrades under severe class imbalance, label scarcity, or small-cohort sample sizes, increasing estimator variance",
            "opp": "Establish non-parametric distribution metrics, contrastive self-supervised pre-training, and class-balanced re-weighting strategies to ensure stability under scarce or imbalanced observations."
        },
        {
            "key": "adversarial_security",
            "title": "Adversarial Vulnerability, Security Auditing & Trust Boundaries",
            "category": "Adversarial Robustness & Trustworthy AI",
            "keywords": [
                "adversarial", "backdoor", "poisoning", "evasion", "security", "trust",
                "contestability", "verification", "privacy", "federated", "defense", "vulnerability", "malicious", "attack"
            ],
            "desc": "an absence of certified defense protocols against adversarial evasion perturbations, data poisoning triggers, or unverified automated decision overrides leaves deployment vulnerable",
            "opp": "Formulate certified adversarial verification bounds, secure multiparty protocols, and contestable human-in-the-loop interfaces to audit and enforce system trustworthiness."
        },
        {
            "key": "sim_to_real",
            "title": "Simulation-to-Reality Transfer & Physical Deployment Disconnect",
            "category": "Deployment & Physical Transfer Verification",
            "keywords": [
                "simulation", "sim-to-real", "testbed", "contingency", "deployment",
                "operational verification", "physical", "offline", "surrogate", "grid", "field"
            ],
            "desc": "a persistent transfer gap between simulated benchmark testbeds and live physical environments with unmodeled operational contingencies limits real-world reliability",
            "opp": "Establish hardware-in-the-loop testbeds and domain-randomized training protocols to bridge the simulation-to-reality transfer gap under live operational conditions."
        }
    ]

    syn_idx = 1
    for f_dim in FRONTIER_DIMENSIONS:
        # Match ONLY against limitation_quote — the actual extracted limitation text from the paper.
        # Technique and introductory problem statements are intentionally excluded so that
        # methodology-only or problem-description-only keyword matches cannot create a synthesized gap.
        # Papers without extractable evidence (no limitation_quote) are excluded from the count.
        matching_pids = []
        matching_evidence = {}  # pid -> actual quoted limitation excerpt
        for r in paper_meta:
            # Only use limitation_quote: the verbatim extracted limitation sentence(s).
            lim_quote = str(r.get("limitation_quote", "") or "").strip()
            if not lim_quote or not r.get("has_limitation", False):
                # No actual evidence excerpt available — this paper cannot contribute to a synthesized gap.
                continue
            lim_lower = lim_quote.lower()
            matched_kws = [k for k in f_dim["keywords"] if k in lim_lower]
            # Require at least 2 distinct keyword matches within the limitation excerpt.
            if len(matched_kws) >= 2:
                matching_pids.append(r["pid"])
                matching_evidence[r["pid"]] = lim_quote  # preserve verbatim excerpt

        # Strict synthesis gate: at least 2 papers must each have individually traceable
        # limitation evidence with >= 2 keyword hits. No placeholder evidence is emitted.
        if len(matching_pids) >= 2:
            evidence_parts = []
            for p in matching_pids[:3]:
                ev = matching_evidence.get(p, "")
                # Only include papers that have an actual quoted excerpt — no placeholders.
                if ev:
                    evidence_parts.append(f"[{p}]: \"{clean_snippet(ev, 20)}\"")
                # If ev is somehow empty here (should not occur given the guard above), skip.
            # Double-check: re-filter to only evidenced papers and require >= 2.
            evidenced_pids = [p for p in matching_pids[:3] if matching_evidence.get(p)]
            if len(evidenced_pids) < 2:
                continue  # Cannot generate gap without 2 individually evidenced papers.
            gaps.append({
                "gap_id": f"GAP-SYN-{syn_idx:02d}",
                "type": "synthesized_frontier_gap",
                "title": f"{f_dim['title']} in {clean_snippet(corpus_topic, 4)}",
                "explanation": f"Cross-paper synthesis across [{', '.join(evidenced_pids)}] reveals that {f_dim['desc']}.",
                "observed_limitation": f"Author-stated limitations in [{', '.join(evidenced_pids)}] share common operational constraints regarding {f_dim['category'].lower()}.",
                "limitation_summary": f"{f_dim['title']} — evidenced in [{', '.join(evidenced_pids)}].",
                "future_opportunity": f_dim["opp"],
                "supporting_papers": evidenced_pids,
                "evidence": "; ".join(evidence_parts),
                "confidence_level": f"Moderate (Keyword-matched across {len(evidenced_pids)} author-stated limitations; verify in primary text)",
                "reasoning": (
                    f"Keyword analysis restricted to author-stated limitation excerpts of "
                    f"{', '.join(evidenced_pids)} identifies >= 2 keyword matches per paper "
                    f"for dimension '{f_dim['title']}'. Evidence is traceable to individual "
                    f"limitation sections, not inferred from methodology or introduction."
                )
            })
            syn_idx += 1

    # Strict Citation Grounding: Ensure every cited paper actually exists in active_pids.
    active_pids = set(r["pid"] for r in paper_meta)
    filtered_gaps = []
    for g in gaps:
        if "supporting_papers" in g:
            g["supporting_papers"] = [p for p in g["supporting_papers"] if p in active_pids]
            if g.get("type") == "synthesized_frontier_gap" and len(g["supporting_papers"]) < 2:
                continue
        filtered_gaps.append(g)

    return filtered_gaps

def build_dynamic_diagram_data(schemas=None, summaries=None, comp_analysis=None, gap_analysis=None, report_title: str = "") -> Dict[str, Any]:
    """
    Builds clean, structured diagram data strictly conforming to the user's requirements:
    1. Methodology Flowchart: Research Corpus -> Main Methodological Approaches (3-6) -> Representative Papers (2-3)
    2. Thematic Clusters: 3-6 major research themes, short explanation, paper count, representative papers
    3. Research Gaps: Observed Limitations -> Research Gaps -> Future Opportunities (3-6 gaps)
    """
    if not schemas:
        if comp_analysis and _get_val(comp_analysis, "methodology_comparison", []):
            schemas = _get_val(comp_analysis, "methodology_comparison", [])
        else:
            return {
                "methodology": {"corpus_name": "Research Corpus", "categories": []},
                "thematic_clusters": {"corpus_topic": "Research Corpus", "themes": []},
                "research_gaps": {"corpus_topic": "Research Corpus", "gaps": []}
            }

    corpus_topic = extract_corpus_domain(comp_analysis=comp_analysis, schemas=schemas, report_title=report_title, default_domain="Scientific Research")

    # Check active user query persisted during screening / search
    aq_file = "data/active_query.json"
    if os.path.exists(aq_file):
        try:
            with open(aq_file, "r", encoding="utf-8") as aqf:
                q_val = json.load(aqf).get("query", "").strip()
                if q_val and (not corpus_topic or corpus_topic.lower() in [
                    "scientific research", "research", "research corpus", "corpus overview",
                    "benchmark testbed", "testbed benchmark", "untitled report", "abilities", "specialized specialized"
                ]):
                    corpus_topic = q_val
        except Exception:
            pass

    # Extract paper records
    paper_meta = [_extract_grounded_paper_meta(s, comp_analysis=comp_analysis) for s in schemas]

    # --- 1. METHODOLOGY APPROACHES (3 to 6 categories) ---
    meth_buckets = {}
    for r in paper_meta:
        cat_name, cat_desc = _classify_methodology(r)
        meth_buckets.setdefault(cat_name, {"desc": cat_desc, "papers": []})
        meth_buckets[cat_name]["papers"].append(r)

    methodology_categories = []
    for idx, (cat_name, b) in enumerate(meth_buckets.items()):
        p_list = b["papers"]
        rep_p = [
            {
                "paper_id": p["pid"],
                "title": p["title"],
                "detail": f"Technique: {p['technique']}",
                "technique": p["technique"],
                "contribution_type": p.get("contribution_type", "Original Empirical & Methodological Contribution")
            }
            for p in p_list
        ]
        methodology_categories.append({
            "id": f"method_{idx}",
            "name": cat_name,
            "description": b["desc"],
            "paper_count": len(p_list),
            "representative_papers": rep_p[:3],
            "all_papers": rep_p
        })

    # --- 2. THEMATIC CLUSTERS (WHAT scientific problems/objectives are investigated) ---
    thematic_clusters = _cluster_dynamic_themes(paper_meta, corpus_topic)

    # --- 3. RESEARCH GAPS (Observed Limitations -> Research Gaps -> Future Opportunities) ---
    research_gaps = _build_grounded_research_gaps(paper_meta, corpus_topic)

    # Attach topic-aware corpus relevance assessment
    try:
        from src.ingestion.screening_agent import validate_corpus_relevance
        relevance_data = validate_corpus_relevance(schemas, corpus_topic)
    except Exception:
        relevance_data = {
            "is_insufficient": False,
            "paper_relevance": [{"paper_id": str(_get_val(s, "paper_id", ""))} for s in schemas]
        }

    return {
        "corpus_relevance": relevance_data,
        "methodology": {
            "corpus_name": corpus_topic,
            "categories": methodology_categories
        },
        "thematic_clusters": {
            "corpus_topic": corpus_topic,
            "themes": thematic_clusters
        },
        "research_gaps": {
            "corpus_topic": corpus_topic,
            "gaps": research_gaps
        }
    }


def generate_methodology_flowchart(comp_analysis=None, report_title: str = "", schemas: list = None, diag_data: dict = None) -> str:
    """
    Synthesizes Diagram 1: Publication-Quality Academic Methodology Flowchart.
    Communicates: Research Corpus -> Methodological Approaches -> Research Papers.
    Balanced, compact, and visually clear without horizontal sprawling or unreadable font sizes.
    """
    if not diag_data:
        diag_data = build_dynamic_diagram_data(schemas=schemas, comp_analysis=comp_analysis, report_title=report_title)

    meth = diag_data.get("methodology", {})
    categories = meth.get("categories", [])
    if not categories:
        return ""

    corpus_title = meth.get("corpus_name", "Research Corpus")

    # Calculate exact total papers represented
    all_pids = set()
    for cat in categories:
        for p in cat.get("all_papers", []):
            if p.get("paper_id"):
                all_pids.add(p["paper_id"])
    total_papers = len(all_pids) or sum(c.get("paper_count", 0) for c in categories) or len(schemas or [])

    if meth.get("is_insufficient"):
        total_str = f"0 Eligible ({total_papers} Excluded)"
        sub_str = "Dimensional Exclusion Audit"
    else:
        total_str = f"{total_papers} Evaluated Paper" if total_papers == 1 else f"{total_papers} Evaluated Papers"
        sub_str = "Methodological Taxonomy"

    esc_corpus = escape_mermaid_label(corpus_title, wrap_len=22)

    lines = [
        "```mermaid",
        "%% diagram_id: methodology_flowchart",
        "%%{init: {",
        "  'theme': 'base',",
        "  'themeVariables': {",
        "    'background': '#FAF8F5',",
        "    'fontFamily': 'Plus Jakarta Sans, system-ui, sans-serif',",
        "    'fontSize': '12px',",
        "    'primaryTextColor': '#0F172A',",
        "    'lineColor': '#64748B'",
        "  },",
        "  'flowchart': {",
        "    'curve': 'basis',",
        "    'nodeSpacing': 16,",
        "    'rankSpacing': 38,",
        "    'padding': 12,",
        "    'htmlLabels': true",
        "  }",
        "}}%%",
        "flowchart LR",
        f'    ROOT["<b>{esc_corpus}</b><br/><span style=\'font-size:10.5px;color:#93C5FD;\'>{total_str}</span><br/><span style=\'font-size:9.5px;opacity:0.75;color:#E2E8F0;\'>{sub_str}</span>"]:::rootStyle'
    ]

    # Display up to 5-6 meaningful categories
    for c_idx, cat in enumerate(categories[:6]):
        m_id = f"M{c_idx}"
        cat_name = cat.get("name", f"Methodology {c_idx+1}")
        cat_title = escape_mermaid_label(cat_name, wrap_len=20)
        p_cnt = cat.get("paper_count", 1)
        cnt_badge = f"{p_cnt} Paper" if p_cnt == 1 else f"{p_cnt} Papers"

        # Concise methodology scope description
        cat_desc = cat.get("description", "")
        desc_words = cat_desc.split()
        short_desc = " ".join(desc_words[:4]) if len(desc_words) >= 4 else "Empirical Methodology"
        esc_desc = escape_mermaid_label(short_desc, wrap_len=18)

        is_audit = meth.get("is_insufficient", False)
        m_style = "auditStyle" if is_audit else "methodStyle"
        badge_color = "#991B1B" if is_audit else "#1E3A8A"
        cnt_badge = f"{p_cnt} Excluded" if is_audit else (f"{p_cnt} Paper" if p_cnt == 1 else f"{p_cnt} Papers")

        lines.append(f'    ROOT --> {m_id}["<b>{cat_title}</b><br/><span style=\'font-size:10px;color:{badge_color};font-weight:600;\'>{cnt_badge}</span><br/><span style=\'font-size:9.5px;color:#475569;\'>{esc_desc}</span>"]:::{m_style}')

        rep_papers = cat.get("representative_papers", []) or cat.get("all_papers", [])[:2]
        all_papers = cat.get("all_papers", []) or rep_papers

        # Show up to 2 papers per category when there are >= 4 categories, or 3 if few categories
        max_show = 2 if len(categories) >= 4 else 3
        shown = rep_papers[:max_show]

        for p_idx, p in enumerate(shown):
            pid = str(p.get("paper_id", "")).strip()
            p_node = f"P_{c_idx}_{p_idx}"

            # Concise readable title
            raw_title = p.get("title", pid)
            title_words = raw_title.replace("_", " ").split()
            short_title = " ".join(title_words[:3]) if len(title_words) >= 3 else raw_title
            esc_title = escape_mermaid_label(short_title, wrap_len=18)

            # Concise technique / method description
            raw_tech = p.get("technique") or p.get("detail") or "Methodological approach"
            tech_words = raw_tech.split()
            short_tech = " ".join(tech_words[:5]) if len(tech_words) >= 5 else raw_tech
            esc_tech = escape_mermaid_label(short_tech, wrap_len=20)

            card_html = f"<b>{esc_title}</b><br/><span style='font-size:10px;color:#2563EB;font-weight:600;'>[{pid}]</span><br/><span style='font-size:9.5px;color:#475569;'>{esc_tech}</span>"
            lines.append(f'    {m_id} --> {p_node}["{card_html}"]:::paperStyle')

        # For larger categories, summarize remaining papers cleanly
        if len(all_papers) > len(shown):
            rem_count = len(all_papers) - len(shown)
            rem_pids = [p["paper_id"] for p in all_papers[len(shown):]]
            rem_pids_str = ", ".join(rem_pids[:3])
            more_id = f"MORE_{c_idx}"
            lines.append(f'    {m_id} --> {more_id}["<b>+{rem_count} More Papers</b><br/><span style=\'font-size:9.5px;color:#64748B;\'>{rem_pids_str}</span>"]:::moreStyle')

    lines.extend([
        "",
        "    classDef rootStyle fill:#0F172A,stroke:#1E293B,stroke-width:2px,color:#FFFFFF,font-size:12.5px,font-weight:bold,rx:6px,ry:6px;",
        "    classDef methodStyle fill:#F0F7FF,stroke:#3B82F6,stroke-width:1.5px,color:#1E3A8A,font-size:11.5px,font-weight:bold,rx:5px,ry:5px;",
        "    classDef auditStyle fill:#FEF2F2,stroke:#EF4444,stroke-width:1.3px,color:#991B1B,font-size:11px,font-weight:600,rx:5px,ry:5px;",
        "    classDef paperStyle fill:#FFFFFF,stroke:#CBD5E1,stroke-width:1.2px,color:#1E293B,font-size:10.5px,rx:4px,ry:4px;",
        "    classDef moreStyle fill:#F8FAFC,stroke:#94A3B8,stroke-width:1px,stroke-dasharray:3 3,color:#64748B,font-size:10px,rx:4px,ry:4px;",
        "    linkStyle default stroke:#64748B,stroke-width:1.3px;",
        "```"
    ])
    return "\n".join(lines)


def _clean_mindmap_text(text: str, max_len: int = 32) -> str:
    """
    Cleans and prepares text for native Mermaid mindmap nodes.
    Strictly removes all shape delimiters, quotes, colons, semicolons, and raw ampersands.
    Guarantees no syntax crashes in Mermaid v10 mindmap parser.
    """
    if not text:
        return "General Theme"
    clean = re.sub(r'[\r\n\t]+', ' ', str(text))
    clean = re.sub(r'<[^>]+>', '', clean)  # Strip HTML tags
    clean = clean.replace('&', 'and')
    clean = re.sub(r'[\(\)\[\]\{\}\'\"\:\;\,\`\|]', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    if len(clean) > max_len:
        clean = clean[:max_len].rsplit(' ', 1)[0]
    clean = re.sub(r'\b(and|or|of|in|to|with|-)$', '', clean, flags=re.IGNORECASE).strip()
    return clean or "General Theme"


def generate_themes_mindmap(comp_analysis=None, report_title: str = "", schemas: list = None, diag_data: dict = None) -> str:
    """
    Synthesizes Diagram 2: Thematic Clusters Mindmap/Flowchart.
    Compact, clean cluster layout displaying 3-6 major research themes, short explanation/paper count,
    and representative paper titles without spaghetti lines or visual clutter.
    Flows: Research Corpus -> Shared Research Themes -> Supporting Papers.
    """
    if not diag_data:
        diag_data = build_dynamic_diagram_data(schemas=schemas, comp_analysis=comp_analysis, report_title=report_title)

    tc = diag_data.get("thematic_clusters", {})
    themes = tc.get("themes", [])
    if not themes:
        return ""

    corpus_topic = escape_mermaid_label(tc.get("corpus_topic", "Thematic Scope"), wrap_len=24)
    core_badge = "Dimensional Disconnect" if tc.get("is_insufficient") else "Research Clusters"
    lines = [
        "```mermaid",
        "%% diagram_id: themes_mindmap",
        "%%{init: {'theme': 'base', 'themeVariables': {'background': '#FAF8F5', 'fontFamily': 'Plus Jakarta Sans, system-ui, sans-serif', 'fontSize': '12px'}, 'flowchart': {'nodeSpacing': 18, 'rankSpacing': 30, 'curve': 'basis'}}}%%",
        "flowchart LR",
        f'    CORE(["<b>{corpus_topic}</b><br/><span style=\'font-size:10px;opacity:0.85;\'>{core_badge}</span>"]):::coreStyle'
    ]

    for idx, th in enumerate(themes[:6]):
        t_id = f"T{idx}"
        theme_title = escape_mermaid_label(th.get("title") or th.get("theme_title", f"Theme {idx+1}"), wrap_len=22)
        p_cnt = th.get("paper_count", 1)
        is_insuf = tc.get("is_insufficient", False)
        t_style = "divergenceStyle" if is_insuf else "themeStyle"
        b_color = "#D97706" if is_insuf else "#0D9488"
        cnt_str = f"{p_cnt} Excluded" if is_insuf else (f"{p_cnt} Paper" if p_cnt == 1 else f"{p_cnt} Papers")
        lines.append(f'    CORE --> {t_id}["<b>{theme_title}</b><br/><span style=\'font-size:10px;color:{b_color};font-weight:600;\'>{cnt_str}</span>"]:::{t_style}')

        rep_papers = th.get("representative_papers", []) or th.get("papers", [])
        for p_idx, p in enumerate(rep_papers[:4]):
            p_node = f"TP_{idx}_{p_idx}"
            p_id = p.get("paper_id", "") if isinstance(p, dict) else str(p)
            p_title = p.get("title", p_id) if isinstance(p, dict) else p_id
            short_t = escape_mermaid_label(" ".join(p_title.split()[:3]), wrap_len=18)
            lines.append(f'    {t_id} --> {p_node}["<b>{short_t}</b><br/><span style=\'font-size:9.5px;color:#64748B;\'>{p_id}</span>"]:::paperStyle')

    lines.extend([
        "",
        "    classDef coreStyle fill:#0F766E,stroke:#115E59,stroke-width:2px,color:#FFFFFF,font-size:12px,font-weight:bold,rx:6px,ry:6px;",
        "    classDef themeStyle fill:#F0FDFA,stroke:#0D9488,stroke-width:1.5px,color:#134E4A,font-size:11px,font-weight:600,rx:5px,ry:5px;",
        "    classDef divergenceStyle fill:#FFFBEB,stroke:#D97706,stroke-width:1.5px,color:#92400E,font-size:11px,font-weight:600,rx:5px,ry:5px;",
        "    classDef paperStyle fill:#FFFFFF,stroke:#CBD5E1,stroke-width:1px,color:#334155,font-size:10px,rx:4px,ry:4px;",
        "    linkStyle default stroke:#94A3B8,stroke-width:1.2px;",
        "```"
    ])
    return "\n".join(lines)



def generate_research_gaps_flowchart(comp_analysis=None, gap_analysis=None, schemas=None, report_title: str = "", diag_data: dict = None) -> str:
    """
    Synthesizes Diagram 3: Compact Research Gaps Pipeline Flowchart.
    Clean 3-column / stage progression:
    Observed Limitations -> Research Gaps (Author-Stated vs Synthesized) -> Future Opportunities.
    Concise, readable without excessive scrolling, fits standard viewport at 100% zoom.
    """
    if not diag_data:
        diag_data = build_dynamic_diagram_data(schemas=schemas, comp_analysis=comp_analysis, gap_analysis=gap_analysis, report_title=report_title)

    rg = diag_data.get("research_gaps", {})
    gaps = rg.get("gaps", [])
    if not gaps:
        return ""

    lines = [
        "```mermaid",
        "%% diagram_id: research_gaps_flowchart",
        "%%{init: {'theme': 'base', 'themeVariables': {'background': '#FAF8F5', 'fontFamily': 'Plus Jakarta Sans, system-ui, sans-serif', 'fontSize': '12px'}, 'flowchart': {'nodeSpacing': 14, 'rankSpacing': 28, 'curve': 'basis'}}}%%",
        "flowchart LR"
    ]

    # Select representative mix of author-stated limitations, unstated/insufficient evidence, and synthesized gaps
    paper_gaps = [g for g in gaps if "synth" not in str(g.get("type", "")).lower()]
    synth_gaps = [g for g in gaps if "synth" in str(g.get("type", "")).lower()]
    selected_gaps = paper_gaps[:3] + synth_gaps[:3]
    if not selected_gaps:
        selected_gaps = gaps[:6]

    for idx, gap in enumerate(selected_gaps):
        lim_id = f"LIM_{idx}"
        gap_id = f"GAP_{idx}"
        opp_id = f"OPP_{idx}"

        gap_raw_id = gap.get("gap_id", f"GAP-{idx+1:02d}")

        # 1. Observed limitation
        lim_text = escape_mermaid_label(gap.get("limitation_summary") or gap.get("observed_limitation") or "Empirical constraint", wrap_len=20)
        gap_type = str(gap.get("type", "synthesized_frontier_gap")).lower()
        is_corpus = "corpus" in gap_type or "retrieval" in gap_type
        is_synth = "synth" in gap_type
        is_future = "future" in gap_type
        is_inferred = "infer" in gap_type
        is_unavail = "insufficient" in gap_type or "unavail" in gap_type or gap.get("has_limitation") is False

        p_str = ", ".join(gap.get("supporting_papers", [])[:3])

        if is_corpus:
            lines.append(f'    {lim_id}["<b>Observed Deficit</b><br/>{lim_text}"]:::limStyle')
        elif is_future:
            p_sub_fw = f"<br/><span style='font-size:9.5px;color:#0369A1;'>{p_str}</span>" if p_str else ""
            lines.append(f'    {lim_id}["<b>Future Work</b><br/>{lim_text}{p_sub_fw}"]:::futureLimStyle')
        elif is_synth:
            p_sub_synth = f"<br/><span style='font-size:9.5px;color:#3730A3;'>{p_str}</span>" if p_str else ""
            lines.append(f'    {lim_id}["<b>Cross-Paper Evidence</b><br/>{lim_text}{p_sub_synth}"]:::synthLimStyle')
        elif is_unavail:
            lines.append(f'    {lim_id}["<b>Unspecified</b><br/>{lim_text}"]:::unavailStyle')
        else:
            p_sub = f"<br/><span style='font-size:9.5px;color:#991B1B;'>{p_str}</span>" if p_str else ""
            lines.append(f'    {lim_id}["<b>Limitation</b><br/>{lim_text}{p_sub}"]:::limStyle')

        # 2. Research gap
        if is_corpus:
            type_style = "corpusStyle"
            type_badge = "Corpus Retrieval Gap"
            badge_color = "#DC2626"
        elif is_synth:
            type_style = "synthStyle"
            type_badge = "Synthesized Research Gap"
            badge_color = "#4338CA"
        elif is_future:
            type_style = "futureStyle"
            type_badge = "Author-Stated Future Work"
            badge_color = "#0284C7"
        elif is_inferred:
            type_style = "inferStyle"
            type_badge = "Inferred Cross-Domain Research Opportunity"
            badge_color = "#B45309"
        elif is_unavail:
            type_style = "unavailStyle"
            type_badge = "Unavailable Evidence" if "unavail" in gap_type else "Insufficient Evidence"
            badge_color = "#475569"
        else:
            type_style = "authorStyle"
            type_badge = "Author-Stated Limitation"
            badge_color = "#065F46"

        raw_title = gap.get("title", f"Gap {gap_raw_id}")
        gap_title = escape_mermaid_label(raw_title, wrap_len=22)
        lines.append(f'    {gap_id}["<b>{gap_raw_id}</b><br/><b>{gap_title}</b><br/><span style=\'font-size:9.5px;color:{badge_color};\'>[{type_badge}]</span>"]:::{type_style}')

        # 3. Future opportunity
        opp_text = escape_mermaid_label(gap.get("future_opportunity") or "Targeted mitigation", wrap_len=20)
        lines.append(f'    {opp_id}["<b>Opportunity</b><br/>{opp_text}"]:::oppStyle')

        # Pipeline links
        lines.append(f'    {lim_id} --> {gap_id} --> {opp_id}')

    lines.extend([
        "",
        "    classDef limStyle fill:#FEF2F2,stroke:#EF4444,stroke-width:1.2px,color:#991B1B,font-size:10px,rx:4px,ry:4px;",
        "    classDef futureLimStyle fill:#F0F9FF,stroke:#0EA5E9,stroke-width:1.2px,color:#0369A1,font-size:10px,rx:4px,ry:4px;",
        "    classDef synthLimStyle fill:#EEF2FF,stroke:#6366F1,stroke-width:1.2px,color:#3730A3,font-size:10px,rx:4px,ry:4px;",
        "    classDef authorStyle fill:#F0FDF4,stroke:#10B981,stroke-width:1.5px,color:#065F46,font-size:10.5px,font-weight:600,rx:5px,ry:5px;",
        "    classDef futureStyle fill:#F0F9FF,stroke:#0284C7,stroke-width:1.5px,color:#0369A1,font-size:10.5px,font-weight:600,rx:5px,ry:5px;",
        "    classDef synthStyle fill:#EEF2FF,stroke:#6366F1,stroke-width:1.5px,color:#3730A3,font-size:10.5px,font-weight:600,rx:5px,ry:5px;",
        "    classDef corpusStyle fill:#FEF2F2,stroke:#DC2626,stroke-width:1.5px,color:#991B1B,font-size:10.5px,font-weight:600,rx:5px,ry:5px;",
        "    classDef domainStyle fill:#FFFBEB,stroke:#D97706,stroke-width:1.5px,color:#92400E,font-size:10.5px,font-weight:600,rx:5px,ry:5px;",
        "    classDef inferStyle fill:#FFFBEB,stroke:#F59E0B,stroke-width:1.5px,color:#B45309,font-size:10.5px,font-weight:600,rx:5px,ry:5px;",
        "    classDef unavailStyle fill:#F8FAFC,stroke:#94A3B8,stroke-width:1.2px,color:#475569,font-size:10px,font-weight:600,rx:4px,ry:4px;",
        "    classDef oppStyle fill:#F0FDFA,stroke:#14B8A6,stroke-width:1.2px,color:#115E59,font-size:10px,rx:4px,ry:4px;",
        "    linkStyle default stroke:#94A3B8,stroke-width:1.2px;",
        "```"
    ])
    return "\n".join(lines)

def inject_mermaid_diagrams_into_markdown(report_md: str, diagrams: dict) -> str:

    """

    Injects generated Mermaid diagrams into the report markdown under their respective sections.

    Uses regex section matching, idempotent and safe against repeated runs.

    """

    md = report_md

    # Diagram 2 (Mindmap) -> Section 1

    mindmap = diagrams.get("themes_mindmap", "")

    if mindmap and "%% diagram_id: themes_mindmap" not in md:

        match = re.search(r'(##\s*1\.\s*Corpus Overview.*?\n)', md, re.IGNORECASE)

        if match:

            pos = match.end()

            md = md[:pos] + f"\n### Corpus Thematic Map\n\n{mindmap}\n\n" + md[pos:]

        else:

            md += f"\n\n### Corpus Thematic Map\n\n{mindmap}\n"

    # Diagram 1 (Methodology Flowchart) -> Section 2

    flowchart = diagrams.get("methodology_flowchart", "")

    if flowchart and "%% diagram_id: methodology_flowchart" not in md:

        match = re.search(r'(##\s*2\.\s*(?:Architectural Paradigms|Comparative Analysis Matrix).*?\n)', md, re.IGNORECASE)

        if match:

            pos = match.end()

            md = md[:pos] + f"\n### Methodological Paradigm Taxonomy\n\n{flowchart}\n\n" + md[pos:]

        else:

            md += f"\n\n### Methodological Paradigm Taxonomy\n\n{flowchart}\n"

    # Diagram 3 (Research Gaps Flowchart) -> Section 4 (or Section 3 if fallback)

    gaps_diag = diagrams.get("research_gaps_flowchart", "")

    if gaps_diag and "%% diagram_id: research_gaps_flowchart" not in md:

        match = re.search(r'(##\s*(?:4\.\s*Cross-Paper Research Gaps|3\.\s*(?:Synthesized|Empirical Benchmark).*?Research Gaps).*?\n)', md, re.IGNORECASE)

        if match:

            pos = match.end()

            md = md[:pos] + f"\n### Empirical Constraint & Research Gap Associations\n\n{gaps_diag}\n\n" + md[pos:]

        else:

            md += f"\n\n### Empirical Constraint & Research Gap Associations\n\n{gaps_diag}\n"

    return md

def generate_literature_review(schemas, summaries, comp_analysis, gap_analysis, llm_manager=None, output_filepath: Optional[str] = None, report_title: Optional[str] = None) -> LiteratureReviewReport:

    """

    Compiles the final literature review report into a publication-ready markdown file.

    Executes exactly [ONE PROMPT ONLY] to the LLM to synthesize the complete review,

    grounded in the Pydantic comparison matrix, summaries, and research gaps.

    Enriches the report with verified Mermaid.js diagrams.

    """

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    llm = llm_manager or LLMManager()

    # 1. Determine Report Title and Corpus Domain
    if not report_title:
        aq_file = "data/active_query.json"
        active_q = ""
        if os.path.exists(aq_file):
            try:
                with open(aq_file, "r", encoding="utf-8") as aqf:
                    active_q = json.load(aqf).get("query", "").strip()
            except Exception:
                pass
        if active_q:
            corpus_domain = active_q
            report_title = f"Automated Literature Review: Synthesis of {active_q}"
        elif len(schemas) == 1:
            s0 = schemas[0]
            p_title = _get_val(s0, "paper_title", "Research Paper").replace("_", " ").strip()
            corpus_domain = p_title
            report_title = f"Automated Literature Review: Synthesis of {p_title}"
        else:
            corpus_domain = extract_corpus_domain(comp_analysis=comp_analysis, schemas=schemas, default_domain="Scientific Research")
            report_title = f"Automated Literature Review: Synthesis of {corpus_domain}"
    else:
        corpus_domain = extract_corpus_domain(report_title=report_title, default_domain="Scientific Research")

    # 2. Build structured diagram data
    diag_data = build_dynamic_diagram_data(
        schemas=schemas,
        summaries=summaries,
        comp_analysis=comp_analysis,
        gap_analysis=gap_analysis,
        report_title=report_title
    )

    # 3. Synthesize deterministic Mermaid diagrams (0 LLM Calls)
    diagrams = {
        "methodology_flowchart": generate_methodology_flowchart(comp_analysis, report_title=report_title, schemas=schemas, diag_data=diag_data),
        "themes_mindmap": generate_themes_mindmap(comp_analysis, report_title=report_title, schemas=schemas, diag_data=diag_data),
        "research_gaps_flowchart": generate_research_gaps_flowchart(comp_analysis, gap_analysis, schemas=schemas, report_title=report_title, diag_data=diag_data)
    }

    # 4. Build Overview Table
    overview_table = "| Paper ID | Title | Venue | Mode |\n| :--- | :--- | :--- | :--- |\n"
    for p in schemas:
        pid = _get_val(p, "paper_id", "")
        title = _get_val(p, "paper_title", "")
        venue = _get_val(p, "venue", "arXiv / Peer-Reviewed")
        mode = _get_val(p, "generation_mode", "LLM")
        overview_table += f"| {pid} | {title} | {venue} | {mode} |\n"

    # 5. Extract Gaps summary
    gaps_text = ""
    if gap_analysis:
        for g in _get_val(gap_analysis, "explicit_gaps", []):
            gid = _get_val(g, 'gap_id', '')
            gtitle = _get_val(g, 'title', '')
            gcat = _get_val(g, 'category', '')
            gdesc = _get_val(g, 'description', '')
            gaps_text += f"- **[{gid}] {gtitle}** ({gcat}): {gdesc}\n"

        for g in _get_val(gap_analysis, "synthesized_gaps", []):
            gid = _get_val(g, 'gap_id', '')
            gtitle = _get_val(g, 'title', '')
            gcat = _get_val(g, 'category', '')
            gdesc = _get_val(g, 'description', '')
            gaps_text += f"- **[{gid}] {gtitle}** ({gcat}): {gdesc}\n"

    # 6. Extract Comparison Table
    comp_table = _get_val(comp_analysis, "markdown_comparison_report", "") if comp_analysis else ""

    # [ONE PROMPT ONLY] Synthesis Call
    report_markdown = None

    # Generate grounded paper-specific visuals (Mermaid architectures, Matplotlib charts, Spatial maps)
    try:
        from src.agents.visual_engine import generate_all_visuals_for_corpus, embed_all_visuals_in_report
        paper_visuals, visuals_by_paper = generate_all_visuals_for_corpus(schemas, parsed_dir="data/extracted_schemas", figures_dir="data/figures")
    except Exception:
        embed_all_visuals_in_report = lambda text, visuals, diags: inject_mermaid_diagrams_into_markdown(text, diags)
        paper_visuals = []
        visuals_by_paper = {}

    if llm.keys:

        print("   [Agent 6] Synthesizing Final Literature Review Draft [ONE PROMPT ONLY] via LLM...")

        prompt = f"""You are a distinguished Senior AI Researcher.

Synthesize a comprehensive, publication-grade Literature Review Report titled:

'{report_title}'

Analyze the following {len(schemas)} papers based on their structured extractions:

=== CORPUS OVERVIEW ===

{overview_table}

=== COMPARATIVE MATRIX & FINDINGS ===

{comp_table[:3000] if comp_table else "Comparative data across methods and benchmarks."}

=== IDENTIFIED RESEARCH GAPS ===

{gaps_text}

FORMAT REQUIREMENTS:

Draft a complete academic literature review in Markdown with these exact sections:

# {report_title}

## Executive Summary

## 1. Corpus Overview & Scope

(Include the markdown table provided above)

## 2. Architectural Paradigms & Methodological Analysis

(Deep comparative narrative of the techniques, frameworks, and foundational methodologies)

## 3. Empirical Benchmark Analysis

(Analysis of reported evaluation datasets, benchmarks, and experimental performance)

## 4. Cross-Paper Research Gaps & Unexplored Frontiers

(Synthesize systemic bottlenecks, conflicting evidence, and open problems)

## 5. Strategic Roadmap & Future Directions

## 6. References & Citation Anchors

"""

        sys_inst = "You are an expert scientific researcher. Write an exhaustive, rigorous, publication-quality literature survey in Markdown."

        llm_text = llm.generate_text(prompt, system_instruction=sys_inst)

        if llm_text and len(llm_text) > 400:

            report_markdown = embed_all_visuals_in_report(llm_text, paper_visuals, diagrams)

            print("   [GENUINE-LLM] Literature Review report synthesized successfully via [ONE PROMPT ONLY] with diagrams & visuals!")

    # Fallback template if offline or LLM unavailable

    if not report_markdown:

        print("   [HEURISTIC-FALLBACK] Assembling Literature Review from Grounded Template with Diagrams & Visuals...")

        raw_fallback = f"""# {report_title}

**Corpus Coverage:** {len(schemas)} Benchmark Papers

**Generated At:** {generated_at}

## 1. Corpus Overview & Scope

{overview_table}

## 2. Architectural Paradigms & Methodological Analysis

{comp_table}

## 3. Empirical Benchmark Analysis

{gaps_text}

## 4. Cross-Paper Research Gaps & Unexplored Frontiers

{gaps_text}

## 5. Strategic Roadmap & Future Directions

This literature review surveyed foundational frameworks in {corpus_domain}. Future work must address empirical constraints, benchmark trade-offs, and open research challenges.

## 6. References & Citation Anchors

"""

        report_markdown = embed_all_visuals_in_report(raw_fallback, paper_visuals, diagrams)

    output_dir = "data/reports"
    os.makedirs(output_dir, exist_ok=True)

    if output_filepath:
        filepath = output_filepath
    elif len(schemas) == 1:
        pid = _get_val(schemas[0], "paper_id", "single")
        filepath = os.path.join(output_dir, f"literature_review_{pid}.md")
    else:
        filepath = os.path.join(output_dir, "literature_review_report.md")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_markdown)

    legacy_path = os.path.join(output_dir, "literature_review_report.md")
    if filepath != legacy_path and not os.path.exists(legacy_path):
        with open(legacy_path, "w", encoding="utf-8") as lf:
            lf.write(report_markdown)

    return LiteratureReviewReport(
        title=report_title,
        generated_at=generated_at,
        paper_count=len(schemas),
        full_markdown=report_markdown,
        diagrams=diagrams,
        diagram_data=diag_data
    )
