from pydantic import BaseModel, Field
from typing import List, Dict, Any, Set
import os
import re
import json
from collections import Counter

class ResearchGap(BaseModel):
  gap_id: str = Field(description = "unique code")
  title : str = Field(description = "Title of the paper")
  category : str = Field(description = "Category of the gap")
  description : str = Field(description = "More on the content")
  affected_papers : List[str] = Field(default=[], description = "List of affected papers")

class ResearchGapAnalysis(BaseModel):
  explicit_gaps: List[ResearchGap] = Field(default=[], description = "List of author stated limitations")
  synthesized_gaps: List[ResearchGap] = Field(default=[], description = "List of cross paper synthesized research gaps")

from src.agents.llm_manager import LLMManager

def _get_val_local(obj, key: str, default=""):
  if obj is None:
    return default
  if isinstance(obj, dict):
    return obj.get(key, default)
  return getattr(obj, key, default)

ACADEMIC_STOPWORDS = {
    'the', 'and', 'for', 'with', 'that', 'this', 'are', 'from', 'can', 'has', 'have',
    'which', 'been', 'were', 'our', 'their', 'such', 'more', 'paper', 'method', 'study',
    'system', 'systems', 'based', 'using', 'proposed', 'approach', 'model', 'models',
    'results', 'performance', 'data', 'learning', 'evaluation', 'analysis', 'paper_id',
    'title', 'abstract', 'introduction', 'conclusion', 'section', 'also', 'into',
    'both', 'other', 'these', 'those', 'than', 'then', 'each', 'work', 'future',
    'present', 'presents', 'presented', 'framework', 'provide', 'provides', 'novel', 'new',
    'state', 'art', 'across', 'within', 'between', 'through', 'under', 'over', 'well',
    'experimental', 'scope', 'domain', 'noted', 'validated', 'explicit', 'evidential',
    'empirical', 'various', 'different', 'aims', 'clarify', 'especially', 'exploring',
    'about', 'after', 'against', 'among', 'because', 'before', 'being', 'between', 'during',
    'whether', 'neutral', 'fundamental', 'existing', 'tasks', 'task', 'step', 'steps',
    'execute', 'execution', 'multi', 'potential', 'high', 'large', 'small', 'many',
    'most', 'some', 'only', 'same', 'first', 'second', 'third', 'while', 'where',
    'when', 'what', 'which', 'who', 'whom', 'whose', 'why', 'how', 'here', 'there'
}

def _extract_author_limitation_sentence(text: str, pid: str = "") -> str:
  if not text:
    return ""
  clean = re.sub(
    r'^(the remainder of this paper|section \d|in this section|this paper is organized)[^\.\n]*[\.\n]',
    '', text, flags=re.I
  ).strip()
  sentences = [s.strip() for s in re.split(r'[\.\n;]+', clean) if len(s.strip()) > 15]
  contrast_re = re.compile(
    r'\b(however|but|yet|nevertheless|nonetheless|although|while|despite|limitation|bottleneck|drawback|suffer|struggles?|fails?|inability|challenge|trade[- ]offs?|expensive|overhead|cost|leakage|attack|drift|uncertainty|untested|degrade)\b',
    re.I
  )
  solution_re = re.compile(
    r'\b(?:we|our\s+method|our\s+framework|our\s+approach|to)\s+'
    r'(?:overcome|overcomes|address|addresses|resolve|resolves|mitigate|mitigates|tackle|tackles|solve|solves|propose|proposes)\s+'
    r'(?:this|these|the|such)?\s*(?:limitation|limitations|drawback|drawbacks|challenge|challenges|bottleneck|bottlenecks)\b'
    r'|\b(?:this|these)\s+limitations?\s+motivates?\s+(?:our|consideration|the)\b',
    re.I
  )
  for s in sentences:
    if contrast_re.search(s):
      if solution_re.search(s):
        continue  # Reject author contribution / solution statement
      clean_s = re.sub(r'^(however|but|yet|nevertheless|nonetheless|although|while|despite),?\s*', '', s, flags=re.I).strip()
      return clean_s
  for s in sentences:
    if not solution_re.search(s):
      return s
  return ""

def _extract_domain_keywords(text: str) -> List[str]:
  words = re.findall(r'\b[a-z]{4,}\b', text.lower())
  return [w for w in words if w not in ACADEMIC_STOPWORDS]

def _categorize_limitation(text: str) -> str:
  low = text.lower()
  if any(k in low for k in ['compute', 'cost', 'latency', 'token', 'overhead', 'memory', 'gpu', 'parameter', 'complexity']):
    return "Computational Resources & Efficiency"
  if any(k in low for k in ['error', 'noise', 'fidelity', 'hallucinat', 'drift', 'failure', 'leak', 'attack', 'security', 'trust', 'vulnerab', 'privacy', 'authorization']):
    return "System Reliability & Security"
  if any(k in low for k in ['grid', 'power', 'battery', 'storage', 'energy', 'cryogenic', 'dispatch', 'converter', 'mimo']):
    return "Operational & Physical Constraints"
  if any(k in low for k in ['generaliz', 'transfer', 'out-of-distribution', 'robustness', 'tabular', 'bias']):
    return "Generalization & Domain Adaptation"
  if any(k in low for k in ['scale', 'scalab', 'coordination', 'concurrency', 'throughput', 'multi-agent', 'deadlock']):
    return "Scalability & Architecture"
  return "Empirical Limitations"

def run_research_gap_agent(schemas, llm_manager=None):
  paper_context = ""
  for paper in schemas:
    paper_context += (
      f"\nPaper ID: {_get_val_local(paper, 'paper_id', '')}\n"
      f"Paper Title: {_get_val_local(paper, 'paper_title', '')}\n"
      f"Core Problem: {_get_val_local(paper, 'core_problem', '')}\n"
      f"Methodology: {_get_val_local(paper, 'methodology_summary', '')}\n"
      f"Limitations: {_get_val_local(paper, 'explicit_limitations', '')}\n"
    )

  llm = llm_manager or LLMManager()
  prompt = f"""You are an AI Research Gap Agent. Analyze the following research papers:
{paper_context}

Synthesize empirical research gaps grounded strictly in the provided paper context.
Requirements:
1. "explicit_gaps": One per paper containing its author-stated limitations.
2. "synthesized_gaps": Cross-paper research gaps that are shared across AT LEAST TWO papers.
Do NOT invent shared gaps if papers belong to unrelated domains.
If evidence is insufficient for cross-paper gaps, return an empty list for "synthesized_gaps".

Return a JSON OBJECT with keys "explicit_gaps" and "synthesized_gaps".
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

  # Grounded Dynamic Heuristic Engine (zero hardcoded paper IDs or predefined descriptions)
  print("   [DYNAMIC-HEURISTIC] Synthesizing Evidence-Grounded Research Gaps from Active Corpus...")

  explicit: List[ResearchGap] = []
  paper_profiles: Dict[str, Dict[str, Any]] = {}

  for idx, p in enumerate(schemas, start=1):
    pid = str(_get_val_local(p, 'paper_id', str(idx))).strip()
    title = str(_get_val_local(p, 'paper_title', f'Paper {idx}')).strip()
    lim_raw = str(_get_val_local(p, 'explicit_limitations', '') or '').strip()
    core_prob = str(_get_val_local(p, 'core_problem', '') or '').strip()
    meth_summary = str(_get_val_local(p, 'methodology_summary', '') or '').strip()

    # Extract genuine author limitation sentence
    lim_sentence = _extract_author_limitation_sentence(lim_raw, pid)
    if not lim_sentence:
      lim_sentence = _extract_author_limitation_sentence(core_prob, pid)

    if lim_sentence:
      short_lim = lim_sentence[:55].rstrip()
      if len(lim_sentence) > 55:
        short_lim += "..."
      gap_title = f"{short_lim} ({pid})"
      cat = _categorize_limitation(f"{lim_sentence} {title}")
      desc = lim_sentence
    else:
      gap_title = f"Insufficient author-stated limitation evidence ({pid})"
      cat = "Empirical Limitations"
      desc = f"No explicit operational constraint or failure mode documented in extracted schema for {title}."

    explicit.append(ResearchGap(
      gap_id=f"GAP-{idx:02d}",
      title=gap_title,
      category=cat,
      description=desc,
      affected_papers=[pid]
    ))

    # Compile domain tokens for cross-paper problem clustering
    combined_text = f"{title} {core_prob} {meth_summary}"
    domain_toks = set(_extract_domain_keywords(combined_text))
    paper_profiles[pid] = {
      "title": title,
      "limitation": desc,
      "category": cat,
      "tokens": domain_toks,
      "has_real_limitation": bool(lim_sentence)
    }

  # Dynamic Problem Overlap Clustering
  pids = list(paper_profiles.keys())
  adj: Dict[str, Set[str]] = {pid: set() for pid in pids}

  for i in range(len(pids)):
    for j in range(i + 1, len(pids)):
      p1, p2 = pids[i], pids[j]
      shared = paper_profiles[p1]["tokens"] & paper_profiles[p2]["tokens"]
      # Require at least 2 distinct domain-specific concepts shared between problems
      if len(shared) >= 2:
        adj[p1].add(p2)
        adj[p2].add(p1)

  # Form problem clusters (only clusters with >= 2 papers form synthesized gaps)
  visited: Set[str] = set()
  clusters: List[List[str]] = []

  for pid in pids:
    if pid not in visited:
      comp = []
      queue = [pid]
      visited.add(pid)
      while queue:
        curr = queue.pop(0)
        comp.append(curr)
        for nbr in adj[curr]:
          if nbr not in visited:
            visited.add(nbr)
            queue.append(nbr)
      if len(comp) >= 2:
        clusters.append(comp)

  synthesized: List[ResearchGap] = []
  for c_idx, comp_pids in enumerate(clusters, start=1):
    # Find top shared domain concepts
    tok_counts = Counter()
    for pid in comp_pids:
      tok_counts.update(paper_profiles[pid]["tokens"])
    top_concepts = [w.title() for w, cnt in tok_counts.most_common(4) if cnt >= 2]
    concept_str = " & ".join(top_concepts[:2]) if top_concepts else "Empirical Methodology"

    # Assemble supporting limitations from the clustered papers
    lim_excerpts = [
      paper_profiles[pid]["limitation"][:90].rstrip()
      for pid in comp_pids if paper_profiles[pid]["has_real_limitation"]
    ]
    if not lim_excerpts:
      lim_excerpts = [paper_profiles[pid]["limitation"][:90].rstrip() for pid in comp_pids]

    cats = [paper_profiles[pid]["category"] for pid in comp_pids]
    maj_cat = Counter(cats).most_common(1)[0][0]

    evidence_summary = "; ".join(lim_excerpts[:3])
    synthesized.append(ResearchGap(
      gap_id=f"GAP-SYN-{c_idx:02d}",
      title=f"Cross-Paper Bottlenecks in {concept_str}",
      category=maj_cat,
      description=f"Across {len(comp_pids)} evaluated papers ({', '.join(sorted(comp_pids))}), empirical findings demonstrate shared systemic challenges in {concept_str}: {evidence_summary}.",
      affected_papers=sorted(comp_pids)
    ))

  # If evidence is insufficient (no 2+ paper overlap), explicitly state insufficient evidence
  if not synthesized:
    synthesized.append(ResearchGap(
      gap_id="GAP-SYN-01",
      title="Insufficient cross-paper evidence for shared systemic research gap",
      category="Empirical Evaluation",
      description="Cross-paper synthesis requires explicit supporting evidence from at least two relevant papers sharing the same research problem. No multi-paper overlap identified across active corpus.",
      affected_papers=[]
    ))

  return ResearchGapAnalysis(
    explicit_gaps=explicit,
    synthesized_gaps=synthesized
  )
