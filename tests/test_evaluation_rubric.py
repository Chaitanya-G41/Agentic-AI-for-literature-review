import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval_benchmark import evaluate, citation_ids, judge_report
from src.agents.report_generator import generate_literature_review, generate_research_gaps_flowchart


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'pdf_corpus').mkdir()
        (self.root / 'extracted_schemas').mkdir()
        self.schema = dict(paper_id='2011.02840', paper_title='Example segmentation',
                           datasets_eval=['Dataset A'], empirical_results='Dice 0.8862 on validation.',
                           numerical_metrics=[], generation_mode='GENUINE_LLM')
        (self.root / 'extracted_schemas/2011.02840_schema.json').write_text(json.dumps(self.schema))

    def test_no_active_corpus_is_not_perfect_coverage(self):
        result = evaluate('## 1. Corpus\n[2011.02840]', self.root)
        self.assertIsNone(result['coverage_score'])
        self.assertEqual(result['citation_integrity'], 1)

    def test_both_sections_required_and_unknown_citations_fail(self):
        (self.root / 'pdf_corpus/2011.02840_example.pdf').touch()
        report = '## 1. Corpus\n[2011.02840]\n## 2. Methods\n[9999.12345]\n## 3. Results\n[2011.02840] Dice 0.8862.'
        result = evaluate(report, self.root)
        self.assertEqual(result['coverage_score'], 0)
        self.assertEqual(result['invalid_ids'], ['9999.12345'])
        self.assertEqual(result['quantitative_matches'], 1)

    def test_unrelated_number_and_diagram_do_not_count(self):
        result = evaluate('0.8862\n\n[2011.02840]\n```mermaid\n[9999.12345]\n```', self.root)
        self.assertEqual(result['quantitative_matches'], 0)
        self.assertEqual(result['invalid_ids'], [])
        self.assertIsNone(evaluate('', self.root)['citation_integrity'])

    def test_grouped_versioned_and_local_citations(self):
        self.assertEqual(citation_ids('[2011.02840v2; 1909.06684] [upload-abc] [GAP-01]'),
                         ['2011.02840', '1909.06684', 'upload-abc'])

    def test_corpus_api_exposes_real_extraction_mode(self):
        import server
        (self.root / 'pdf_corpus/2011.02840_example.pdf').touch()
        (self.root / 'pdf_corpus/unknown_example.pdf').touch()
        with patch.object(server, 'PDF_DIR', str(self.root / 'pdf_corpus')), \
             patch.object(server, 'SCHEMAS_DIR', str(self.root / 'extracted_schemas')):
            papers = server.get_corpus()['papers']
        self.assertEqual(papers[0]['generation_mode'], 'GENUINE_LLM')
        self.assertEqual(papers[1]['generation_mode'], 'UNKNOWN')

    def test_optional_judge_validates_scores_and_handles_no_key(self):
        with patch('src.agents.llm_manager.LLMManager') as manager:
            manager.return_value.keys = []
            self.assertIn('Skipped', judge_report('', self.root)['status'])
            manager.return_value.keys = ['test-only']
            manager.return_value.generate_structured_json.return_value = {'logical_coherence': {'score': 9}}
            self.assertIn('Unavailable', judge_report('', self.root)['status'])
            scores = {k: {'score': 4, 'rationale': 'Grounded comparisons'} for k in
                      ['logical_coherence', 'comparative_synthesis_depth', 'actionable_future_work']}
            manager.return_value.generate_structured_json.return_value = scores
            self.assertEqual(judge_report('', self.root)['scores'], scores)

    def test_gap_edges_only_share_evidence(self):
        gaps = {'explicit_gaps': [{'gap_id': 'G01', 'title': 'Resolution', 'affected_papers': ['a']}],
                'synthesized_gaps': [{'gap_id': 'S01', 'title': 'Efficiency', 'affected_papers': ['a']},
                                     {'gap_id': 'S02', 'title': 'Unrelated', 'affected_papers': ['b']}]}
        diagram = generate_research_gaps_flowchart({}, gaps)
        self.assertIn('EG_0_G01 -. "Shared paper evidence" .-> SG_0_S01', diagram)
        self.assertNotIn('.-> SG_1_S02', diagram)
        self.assertNotIn('ReAct', diagram)
        self.assertEqual(generate_research_gaps_flowchart({}, {}), '')

    def test_synthesis_receives_full_evidence_and_single_request(self):
        class FakeLLM:
            keys = ['fake']
            calls = []
            def generate_text(self, prompt, **kwargs):
                self.calls.append((prompt, kwargs))
                return '# Survey\n\n' + 'Grounded discussion. ' * 1300
        fake = FakeLLM()
        with patch('src.agents.report_generator.generate_methodology_flowchart', return_value=''), \
             patch('src.agents.report_generator.generate_themes_mindmap', return_value=''):
            result = generate_literature_review([self.schema], [],
                {'markdown_comparison_report': 'x' * 4000 + 'END_OF_EVIDENCE'}, {}, fake,
                str(self.root / 'report.md'))
        self.assertEqual(len(fake.calls), 1)
        self.assertIn('END_OF_EVIDENCE', fake.calls[0][0])
        self.assertIn('0.8862', fake.calls[0][0])
        self.assertEqual(fake.calls[0][1]['max_tokens'], 16384)
        self.assertGreater(len(result.full_markdown), 25000)

    def test_offline_fallback_preserves_results_and_references(self):
        class Offline:
            keys = []
        result = generate_literature_review([self.schema], [], {}, {}, Offline(), str(self.root / 'offline.md'))
        self.assertIn('0.8862', result.full_markdown)
        self.assertIn('## 6. References & Grounded Citation Anchors', result.full_markdown)
        self.assertIn('below the 25,000-character', result.full_markdown)


if __name__ == '__main__':
    unittest.main()
