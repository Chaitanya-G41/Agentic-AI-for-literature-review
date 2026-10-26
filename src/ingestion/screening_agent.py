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
            recency_weight: Weight (0.0 to 0.3) given to publication recency bonus.
                           Default 0.15 gives recent papers an intentional boost
                           without completely overriding semantic relevance.
        """
        self.recency_weight = recency_weight
        self.client = arxiv.Client()

    def search_arxiv(self, query: str, max_candidates: int = 25) -> List[Dict[str, Any]]:
        """
        Queries arXiv for candidate papers using the provided query string.
        Returns parsed lightweight metadata dictionaries.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        search = arxiv.Search(
            query=clean_query,
            max_results=max_candidates,
            sort_by=arxiv.SortCriterion.Relevance
        )

        candidates = []
        try:
            for result in self.client.results(search):
                arxiv_id = result.get_short_id()
                # Clean arxiv_id (remove version suffix e.g. v1, v2 for standard IDs)
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

    def score_and_rank(self, user_query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculates composite semantic relevance + recency bonus scores (0-100%).
        Ranks candidates descending by composite score.
        """
        if not candidates:
            return []

        # 1. Build text corpus for vectorization: query + (title * 2 + abstract) for each paper
        corpus = [user_query]
        for c in candidates:
            # Emphasize title matches by duplicating title in composite string
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
            
            # Cosine similarity of user query (row 0) against all candidate papers (rows 1..)
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

            # Recency factor: papers from current year get 1.0, decreasing by 0.15 per year older, floor at 0.0
            years_old = max(0, current_year - c["published_year"])
            recency_factor = max(0.0, 1.0 - (years_old * 0.15))

            # Composite formula: (1 - weight) * semantic + (weight * recency)
            # Scaled to 0-100%
            composite_score = ((1.0 - self.recency_weight) * raw_semantic_score) + (self.recency_weight * recency_factor)
            pct_score = round(composite_score * 100, 1)

            c_scored = dict(c)
            c_scored["semantic_score"] = round(raw_semantic_score * 100, 1)
            c_scored["recency_factor"] = round(recency_factor * 100, 1)
            c_scored["relevance_score"] = pct_score
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
        print(f"Saving {len(selected_candidates)} papers to {output_dir}")
        print("==================================================\n")

        for idx, paper in enumerate(selected_candidates, start=1):
            arxiv_id = paper.get("arxiv_id") or "paper"
            safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', paper.get("title", "")[:35]).strip('_')
            filename = f"{arxiv_id}_{safe_title}.pdf"
            filepath = os.path.join(output_dir, filename)

            if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
                print(f"[{idx}/{len(selected_candidates)}] Already exists: {filename}")
                saved_paths.append(filepath)
                continue

            print(f"[{idx}/{len(selected_candidates)}] Downloading: {paper.get('title')}...")
            pdf_url = paper.get("pdf_url")
            if not pdf_url:
                pdf_url = f"https://arxiv.org/pdf/{paper['arxiv_id']}.pdf"

            try:
                res = requests.get(pdf_url, headers={"User-Agent": "AgenticAI-LitReview/1.0"}, timeout=25)
                if res.status_code == 200 and len(res.content) > 1000:
                    with open(filepath, "wb") as f:
                        f.write(res.content)
                    print(f"   Saved: {filename} ({len(res.content) // 1024} KB)")
                    saved_paths.append(filepath)
                else:
                    print(f"   [WARN] Direct URL returned status {res.status_code}. Attempting arxiv package download...")
                    search = arxiv.Search(id_list=[paper['arxiv_id']])
                    paper_obj = next(self.client.results(search))
                    paper_obj.download_pdf(dirpath=output_dir, filename=filename)
                    saved_paths.append(filepath)
                    print(f"   Saved via arXiv client: {filename}")
            except Exception as e:
                print(f"   [ERROR] Failed to download {arxiv_id}: {e}")

        return saved_paths


def screen_and_download_topic_papers(
    topic_query: str,
    top_k: int = 5,
    max_search_candidates: int = 25,
    output_dir: str = "data/pdf_corpus",
    recency_weight: float = 0.15
) -> List[Dict[str, Any]]:
    """
    Convenience end-to-end function:
    Searches arXiv -> scores & ranks with recency bias -> returns top_k candidate metadata.
    """
    agent = RelevanceScreeningAgent(recency_weight=recency_weight)
    candidates = agent.search_arxiv(topic_query, max_candidates=max_search_candidates)
    ranked = agent.score_and_rank(topic_query, candidates)
    return ranked[:top_k]
