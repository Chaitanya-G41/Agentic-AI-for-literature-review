"""Live synthesis smoke test using the current comparative-analysis snapshot.

Run from repository root: python -B tests/smoke_synthesis.py.
Does not modify the current report. Exits nonzero for incomplete output.
"""
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.agents.llm_manager import LLMManager
from src.agents.report_generator import generate_literature_review


def main():
    comp = json.loads((ROOT / 'data/comparative_analysis.json').read_text(encoding='utf-8'))
    schemas, summaries = [], []
    for pid in comp['paper_ids']:
        schemas.append(json.loads((ROOT / f'data/extracted_schemas/{pid}_schema.json').read_text(encoding='utf-8')))
        path = ROOT / f'data/summaries/{pid}_summary.json'
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding='utf-8')))
    gaps = {'explicit_gaps': [dict(gap_id=f'G{i+1:02}', title='Reported limitation',
        description=p['explicit_limitations'], affected_papers=[p['paper_id']])
        for i, p in enumerate(schemas) if p.get('explicit_limitations')], 'synthesized_gaps': []}
    # Bound provider attempts during smoke testing; retain the production cascade.
    class SmokeLLM(LLMManager):
        def generate_text(self, prompt, **kwargs):
            return super().generate_text(prompt, max_retries=1, **kwargs)
    llm = SmokeLLM(preferred_models=['nvidia/nemotron-3.5-bolt-e4b'])
    if not llm.keys:
        raise SystemExit('Live synthesis unavailable: no configured API keys')
    output = ROOT / 'data/reports/literature_review_validation.md'
    report = generate_literature_review(schemas, summaries, comp, gaps, llm, str(output))
    print(f'Smoke result: {len(report.full_markdown)} characters; mode={llm.generation_mode}; model={llm.last_used_model}')
    if llm.generation_mode != 'GENUINE_LLM':
        error = (llm.last_error or '').lower()
        print('Provider diagnostic:', next((v for v in ['timeout', 'timed out', 'connection', '401', '403', '404', '429', '500', '503', 'token'] if v in error), 'unclassified provider failure'))
        raise SystemExit('Live provider unavailable; output is the labeled fallback inventory')
    prose = re.sub(r'```[\s\S]*?```', '', report.full_markdown)
    if len(prose) <= 25000:
        raise SystemExit(f'Incomplete survey: {len(prose)} non-diagram characters; target >25,000')
    from eval_benchmark import evaluate
    audit = evaluate(report.full_markdown, ROOT / 'data')
    if len(audit['sections_present']) != 6 or audit['invalid_ids']:
        raise SystemExit('Incomplete survey: missing numbered sections or invalid citation IDs')


if __name__ == '__main__':
    main()
