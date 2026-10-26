"""
Relevance Screening Agent
Project: NLP-05 Agentic AI for Automated Research Paper Analysis

Capabilities:
1. Queries arXiv API for candidate papers based on arbitrary research project titles/topics.
2. Performs vector-based semantic similarity scoring (composite Title + Abstract).
3. Applies a recency decay bonus to prioritize newer publications.
4. Ranks candidates and downloads selected top-X papers directly into data/pdf_corpus/.
"""

import os
import re
import datetime
import requests
import arxiv
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RelevanceScreeningAgent:
    """
    Screens and filters candidate research papers from arXiv
    based on semantic similarity to a user's project title/topic
    with recency biasing.
    """
    def __init__(self, recency_weight: float = 0.15):
        """
        Args:
            recency_weight: Weight (0.0 to 0.35) given to publication recency bonus.
                           Default 0.15 gives recent papers an intentional boost
                           without completely overriding semantic relevance.
        """
        self.recency_weight = recency_weight
        self.client = arxiv.Client()

    def search_arxiv(self, query: str, max_candidates: int = 25, **kwargs) -> List[Dict[str, Any]]:
        """
        Queries arXiv for candidate papers using the provided query string.
        Returns parsed lightweight metadata dictionaries.
        """
        limit = kwargs.get("max_results", max_candidates)
        clean_query = query.strip()
        if not clean_query:
            return []

        search = arxiv.Search(
            query=clean_query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance
        )

        candidates = []
        try:
            for result in self.client.results(search):
                arxiv_id = result.get_short_id()
                clean_id = re.sub(r'v\d+$', '', arxiv_id)

                pub_year = result.published.year if result.published else datetime.datetime.now().year
                pub_date_str = result.published.strftime("%Y-%m-%d") if result.published else "Unknown"

                authors = [a.name for a in result.authors] if result.authors else ["Unknown"]

                candidates.append({
                    "arxiv_id": clean_id,
                    "raw_id": arxiv_id,
                    "title": result.title.replace("\n", " ").strip(),
                    "authors": authors,
                    "abstract": result.summary.replace("\n", " ").strip(),
                    "published_date": pub_date_str,
                    "published_year": pub_year,
                    "pdf_url": result.pdf_url,
                    "categories": result.categories or [],
                    "entry_id": result.entry_id
                })
        except Exception as e:
            print(f"   [WARN] arXiv search error: {e}")

        return candidates

    def score_and_rank(
        self,
        arg1: Any,
        arg2: Any = None,
        recency_weight: Optional[float] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Calculates composite semantic relevance + recency bonus scores (0-100%).
        Ranks candidates descending by composite score.
        Accepts (user_query, candidates) or (candidates, user_query).
        """
        if isinstance(arg1, str):
            user_query = arg1
            candidates = arg2 or []
        elif isinstance(arg2, str):
            user_query = arg2
            candidates = arg1 or []
        else:
            user_query = kwargs.get("user_query") or kwargs.get("query") or ""
            candidates = arg1 if isinstance(arg1, list) else (arg2 if isinstance(arg2, list) else [])

        if not candidates:
            return []

        active_weight = recency_weight if recency_weight is not None else kwargs.get("recency_weight", self.recency_weight)

        # 1. Build text corpus for vectorization: query + (title * 2 + abstract) for each paper
        corpus = [user_query]
        for c in candidates:
            paper_text = f"{c['title']} {c['title']} {c['abstract']}"
            corpus.append(paper_text)

        # 2. Vector similarity using TF-IDF with sublinear tf and character/word ngrams
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True
            )
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            query_vec = tfidf_matrix[0:1]
            paper_vecs = tfidf_matrix[1:]
            sim_scores = cosine_similarity(query_vec, paper_vecs)[0]
        except Exception as e:
            print(f"   [WARN] TF-IDF computation fallback ({e}). Using uniform similarity.")
            sim_scores = np.array([0.5] * len(candidates))

        # 3. Calculate Recency Bonus
        current_year = datetime.datetime.now().year
        scored_candidates = []

        for idx, c in enumerate(candidates):
            raw_semantic_score = float(sim_scores[idx])

            years_old = max(0, current_year - c["published_year"])
            recency_factor = max(0.0, 1.0 - (years_old * 0.15))

            composite_score = ((1.0 - active_weight) * raw_semantic_score) + (active_weight * recency_factor)
            pct_score = round(composite_score * 100, 1)

            c_scored = dict(c)
            c_scored["semantic_score"] = round(raw_semantic_score * 100, 1)
            c_scored["recency_factor"] = round(recency_factor * 100, 1)
            c_scored["relevance_score"] = pct_score
            c_scored["composite_score"] = composite_score
            c_scored["year"] = c["published_year"]
            scored_candidates.append(c_scored)

        # 4. Sort descending by composite relevance score
        scored_candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_candidates

    def download_papers(self, selected_candidates: List[Dict[str, Any]], output_dir: str = "data/pdf_corpus") -> List[str]:
        """
        Downloads the approved candidate PDFs into output_dir with standardized naming.
        Returns list of saved file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        print("==================================================")
        print("Relevance Screening Agent: Downloading Approved Papers")
        print(f"Target Directory: {output_dir}")
        print("==================================================")

        headers = {
            "User-Agent": "AgenticAI-LiteratureReview/1.0 (RVCE NLP-05 Project)"
        }

        for idx, paper in enumerate(selected_candidates, 1):
            arxiv_id = paper.get("arxiv_id", "")
            title = paper.get("title", "paper")
            # Normalize title for safe filename: alphanumeric + underscore
            clean_title = re.sub(r'[^a-zA-Z0-9_\- ]', '', title)
            clean_title = re.sub(r'\s+', '_', clean_title)[:40]

            filename = f"{arxiv_id}_{clean_title}.pdf" if arxiv_id else f"{clean_title}.pdf"
            filepath = os.path.join(output_dir, filename)

            if os.path.exists(filepath):
                print(f"[{idx}/{len(selected_candidates)}] Already exists: {filename}")
                saved_paths.append(filepath)
                continue

            pdf_url = paper.get("pdf_url")
            if not pdf_url and arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            print(f"[{idx}/{len(selected_candidates)}] Downloading: {arxiv_id} - {title[:50]}...")
            try:
                resp = requests.get(pdf_url, headers=headers, timeout=45)
                if resp.status_code == 200:
                    with open(filepath, "wb") as f:
                        f.write(resp.content)
                    print(f"   Saved to: {filepath} ({len(resp.content)//1024} KB)")
                    saved_paths.append(filepath)
                else:
                    print(f"   [FAIL] HTTP {resp.status_code} for {pdf_url}")
            except Exception as e:
                print(f"   [ERROR] Failed to download {pdf_url}: {e}")

        print("==================================================")
        print(f"Completed: Downloaded {len(saved_paths)}/{len(selected_candidates)} papers.")
        print("==================================================")
        return saved_paths


def screen_and_download_topic_papers(
    topic: str,
    top_k: int = 5,
    max_candidates: int = 20,
    recency_weight: float = 0.20,
    output_dir: str = "data/pdf_corpus"
) -> List[Dict[str, Any]]:
    """
    Convenience function: Given a topic, queries arXiv, screens candidates,
    ranks them by semantic similarity + recency bonus, and downloads top-k.
    """
    agent = RelevanceScreeningAgent(recency_weight=recency_weight)
    print(f"\n[Screening Agent] Fetching candidates for topic: '{topic}'...")
    candidates = agent.search_arxiv(topic, max_candidates=max_candidates)
    if not candidates:
        print("[Screening Agent] No candidates found on arXiv.")
        return []

    print(f"[Screening Agent] Scoring {len(candidates)} candidates against '{topic}'...")
    ranked = agent.score_and_rank(topic, candidates)
    selected = ranked[:top_k]

    print(f"[Screening Agent] Top {len(selected)} Selected Papers:")
    for idx, p in enumerate(selected, 1):
        print(f"   {idx}. [{p['relevance_score']}% Match, {p['published_year']}] {p['title'][:70]}")

    agent.download_papers(selected, output_dir=output_dir)
    return selected


if __name__ == "__main__":
    test_topic = "Multi-Agent LLM Coordination and Memory"
    print(f"Testing RelevanceScreeningAgent with topic: '{test_topic}'")
    screen_and_download_topic_papers(test_topic, top_k=2, max_candidates=5)
