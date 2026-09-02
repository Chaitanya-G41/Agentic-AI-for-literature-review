"""
Step 1: Automated Paper Collection & Ingestion Script
Project: NLP-05 Agentic AI for Automated Research Paper Analysis
"""

import os
import requests
import arxiv

BENCHMARK_PAPERS = [
    {"title": "ReAct: Synergizing Reasoning and Acting in Language Models", "arxiv_id": "2210.03629"},
    {"title": "Reflexion: Language Agents with Verbal Reinforcement Learning", "arxiv_id": "2303.11366"},
    {"title": "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation", "arxiv_id": "2308.08155"},
    {"title": "Toolformer: Language Models Can Teach Themselves to Use Tools", "arxiv_id": "2302.04761"},
    {"title": "Self-RAG: Learning to Retrieve, Generate, and Critique via Self-Reflection", "arxiv_id": "2310.11511"},
    {"title": "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework", "arxiv_id": "2308.00352"},
    {"title": "Gorilla: Large Language Model Connected with Massive APIs", "arxiv_id": "2305.15334"},
    {"title": "AgentBench: Evaluating LLMs as Agents", "arxiv_id": "2308.03688"},
    {"title": "ChatDev: Communicative Agents for Software Development", "arxiv_id": "2307.07924"},
    {"title": "Tree of Thoughts: Deliberate Problem Solving with Large Language Models", "arxiv_id": "2305.10601"},
    {"title": "Voyager: An Open-Ended Embodied Agent with Large Language Models", "arxiv_id": "2305.16291"},
    {"title": "SWE-bench: Can Language Models Resolve Real-World GitHub Issues", "arxiv_id": "2310.06770"},
]

def download_benchmark_corpus(output_dir="data/pdf_corpus"):
    """Downloads the benchmark research papers into output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    print("==================================================")
    print("Step 1: Starting Automated Paper Corpus Collection")
    print(f"Target Directory: {os.path.abspath(output_dir)}")
    print("==================================================\n")

    client = arxiv.Client()
    downloaded_count = 0

    for idx, paper in enumerate(BENCHMARK_PAPERS, start=1):
        filename = f"{paper['arxiv_id']}_{paper['title'][:25].replace(' ', '_').replace(':', '')}.pdf"
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            print(f"[{idx}/{len(BENCHMARK_PAPERS)}] Already Exists: {paper['title']}")
            downloaded_count += 1
            continue

        print(f"[{idx}/{len(BENCHMARK_PAPERS)}] Downloading via arXiv API: {paper['title']}...")
        try:
            search = arxiv.Search(id_list=[paper['arxiv_id']])
            paper_result = next(client.results(search))
            paper_result.download_pdf(dirpath=output_dir, filename=filename)
            print(f"   Saved: {filename}")
            downloaded_count += 1
        except Exception as e:
            print(f"   Direct download failed ({e}). Fetching via HTTPS fallback...")
            pdf_url = f"https://arxiv.org/pdf/{paper['arxiv_id']}.pdf"
            try:
                res = requests.get(pdf_url, timeout=15)
                if res.status_code == 200:
                    with open(filepath, "wb") as f:
                        f.write(res.content)
                    print(f"   Saved via Fallback: {filename}")
                    downloaded_count += 1
                else:
                    print(f"   HTTP Error {res.status_code} for {pdf_url}")
            except Exception as err:
                print(f"   Failed to download {paper['title']}: {err}")

    print("\n==================================================")
    print(f"Step 1 Complete: {downloaded_count}/{len(BENCHMARK_PAPERS)} Papers Ready in {output_dir}")
    print("==================================================")

if __name__ == "__main__":
    download_benchmark_corpus()
