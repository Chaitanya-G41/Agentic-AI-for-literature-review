# Literature Review Evaluation Benchmark

| Automated measure | Result |
| --- | --- |
| Active PDF papers | 0 |
| Coverage: Section 1 | 0/0 |
| Coverage: Section 2 | 0/0 |
| Coverage score (papers in BOTH sections) | N/A |
| Citation integrity (valid tag occurrences / all ID tag occurrences) | 100.0% |
| Citation tags / unique IDs | 6 / 6 |
| Numerical candidates reproduced beside the same paper citation | 0/11 |
| Required numbered sections present | 6/6 |
| Report characters (target >25,000) | 16,427 |

## Interpretation

Coverage uses current PDFs only. No active PDFs or no citation tags yields N/A, never a perfect score.
Citation integrity checks IDs against PDF or extracted-schema files; it does not measure factual hallucination.
Quantitative scope: cited schema snapshot (no active PDFs).
Quantitative grounding is a conservative lexical proxy: the extracted number and paper citation must occur in the same paragraph/table row. Narrative candidates can include dataset years or sample counts. Units, semantic equivalence and experimental fairness require human review.

## Findings

- Invalid citation IDs: None detected
- Missing in Section 1: None / no active corpus
- Missing in Section 2: None / no active corpus
- Cited IDs outside active PDF corpus: 1909.06684, 2009.12188, 2011.02840, 2011.11052, 2411.01896, 2510.08617

## Four-pillar review rubric

| Pillar | Human / judge criteria |
| --- | --- |
| Coverage | Every active paper and benchmark represented; inspect omissions above |
| Logical Structure | Coherent scope → taxonomy → evidence → gaps → roadmap (1–5) |
| Synthesis Quality | Mechanistic comparisons, conflicting evidence and testable proposals (1–5) |
| Hallucination Rate | Audit claims against source passages; ID validity alone is insufficient |

## Optional LLM-as-a-Judge

Not run. Use `python eval_benchmark.py --judge` to enable external evaluation.
