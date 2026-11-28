import unittest
import json
import os
import re
from unittest.mock import patch, MagicMock

from src.agents.extraction import PaperSchema
from src.agents.report_generator import (
    extract_corpus_domain,
    build_dynamic_diagram_data,
    generate_methodology_flowchart,
    generate_themes_mindmap,
    generate_research_gaps_flowchart,
    METHOD_TAXONOMY
)


def _create_schema(
    paper_id: str,
    paper_title: str,
    core_problem: str,
    methodology_summary: str,
    explicit_limitations: str,
    empirical_results: str = "Empirical validation demonstrated benchmark results.",
    dataset_names: list = None
) -> PaperSchema:
    return PaperSchema(
        paper_id=paper_id,
        paper_title=paper_title,
        core_problem=core_problem,
        methodology_summary=methodology_summary,
        explicit_limitations=explicit_limitations,
        empirical_results=empirical_results,
        datasets_eval=dataset_names or ["Standard Benchmark"]
    )


class TestArchitectureDiagramsRegression(unittest.TestCase):
    """
    Comprehensive regression test suite for Architecture Diagrams:
    1. Single Source of Truth for Research Gaps (Mermaid & Cards 100% synchronized)
    2. Pre-Generation Relevance Gating (Exclusion of Irrelevant Papers)
    3. Topic Preservation & Resolution (Elimination of 'Method Computational' corruption)
    4. Stale Cache Invalidation in API
    5. Multi-Domain Generality across 5 Distinct Scientific Domains
    """

    def setUp(self):
        self.maxDiff = None

    # =========================================================================
    # Test 1: Single Source of Truth for Research Gaps
    # =========================================================================
    def test_single_source_of_truth_research_gaps(self):
        """
        Verifies that diag_data['research_gaps']['gaps'] is the single source of truth:
        Card data and Mermaid flowchart consume identical gap IDs, titles, limitations, and supporting papers.
        """
        schemas = [
            _create_schema(
                paper_id="2301.00001",
                paper_title="Multimodal Transformer for Alzheimer's Stage Classification",
                core_problem="Early diagnosis of Alzheimer's disease using combined MRI and PET scans.",
                methodology_summary="Vision Transformer architecture fusing T1-MRI and FDG-PET volumetric representations.",
                explicit_limitations="High computational complexity and inference latency on clinical 3D scans.",
                dataset_names=["ADNI-1", "ADNI-2"]
            ),
            _create_schema(
                paper_id="2301.00002",
                paper_title="Deep Residual Networks for Mild Cognitive Impairment Detection",
                core_problem="Accurate detection of mild cognitive impairment from longitudinal MRI.",
                methodology_summary="3D ResNet-50 trained on structural brain MRI scans.",
                explicit_limitations="Performance degrades significantly under severe clinical scanner distribution shift.",
                dataset_names=["OASIS-3"]
            )
        ]
        topic = "Deep Learning for Early Detection of Alzheimer's Disease Using MRI and PET Brain Imaging"

        diag_data = build_dynamic_diagram_data(schemas=schemas, report_title=topic)
        gaps_list = diag_data.get("research_gaps", {}).get("gaps", [])

        self.assertGreater(len(gaps_list), 0, "Should generate structured research gaps from grounded schemas")

        # Generate Mermaid diagram passing diag_data
        mermaid_code = generate_research_gaps_flowchart(schemas=schemas, report_title=topic, diag_data=diag_data)

        self.assertIn("flowchart LR", mermaid_code)
        self.assertIn("STAGE 1: AUTHOR-STATED LIMITATIONS", mermaid_code)
        self.assertIn("STAGE 2: SYNTHESIZED RESEARCH GAPS", mermaid_code)
        self.assertIn("STAGE 3: FUTURE OPPORTUNITIES", mermaid_code)

        # Verify that EVERY gap in diag_data is represented in Mermaid with exact matching IDs
        for gap in gaps_list:
            gid = gap["gap_id"]
            self.assertIn(f"[{gid}]", mermaid_code, f"Mermaid diagram must contain exact gap ID {gid} matching card data")

            # Observed limitation or title must be present in Mermaid
            if gap.get("observed_limitation"):
                words = [w for w in re.findall(r'\b[A-Za-z]{4,}\b', gap["observed_limitation"]) if w.lower() not in ["this", "that", "with", "from"]]
                if words:
                    # At least one key term from the card limitation quote must be in Mermaid
                    self.assertTrue(
                        any(w.lower() in mermaid_code.lower() for w in words[:3]),
                        f"Mermaid must be grounded in card's limitation quote: {gap['observed_limitation']}"
                    )

            # Supporting papers must match
            for pid in gap.get("supporting_papers", []):
                # PID or clean title must be traceable in diagram
                self.assertTrue(
                    pid in mermaid_code or "Multimodal" in mermaid_code or "Deep Residual" in mermaid_code,
                    f"Supporting paper {pid} must be referenced in Mermaid"
                )

    # =========================================================================
    # Test 2: Pre-Generation Relevance Gating
    # =========================================================================
    def test_pre_generation_relevance_gating_excludes_irrelevant_papers(self):
        """
        Verifies that irrelevant papers (e.g. black hole physics in an Alzheimer's study)
        are excluded BEFORE methodology classification, thematic clustering, and research gaps.
        """
        schemas = [
            # Relevant Paper 1
            _create_schema(
                paper_id="2301.00010",
                paper_title="Convolutional Neural Networks for Brain MRI Alzheimer's Detection",
                core_problem="Classification of Alzheimer's disease stages from structural MRI brain scans.",
                methodology_summary="Deep convolutional neural network for hippocampal feature extraction.",
                explicit_limitations="Limited generalization across different clinical MRI scanner manufacturers.",
                dataset_names=["ADNI"]
            ),
            # Relevant Paper 2
            _create_schema(
                paper_id="2301.00011",
                paper_title="Multimodal Fusion of PET and MRI for Neurodegenerative Diagnosis",
                core_problem="Multimodal neuroimaging fusion for early Alzheimer's disease diagnosis.",
                methodology_summary="Dual-branch cross-attention transformer combining PET metabolic and MRI structural data.",
                explicit_limitations="High latency in multimodal co-registration and fusion pipelines.",
                dataset_names=["ADNI", "AIBL"]
            ),
            # Irrelevant Paper 1: Neutron star / black hole physics
            _create_schema(
                paper_id="2111.00099",
                paper_title="Gravitational-Wave Signatures of Neutron Star-Black Hole Binaries",
                core_problem="Observational constraints on equation of state from gravitational-wave merger events.",
                methodology_summary="Numerical relativity simulations and Bayesian parameter estimation using LIGO/Virgo data.",
                explicit_limitations="High computational cost of full numerical relativity wave-form generation.",
                dataset_names=["LIGO-O3"]
            ),
            # Irrelevant Paper 2: Power electronics and microgrids
            _create_schema(
                paper_id="2205.00088",
                paper_title="Stability Analysis of Power Electronics Based Renewable Energy Storage",
                core_problem="Small-signal stability and voltage regulation in islanded microgrids.",
                methodology_summary="MIMO state-space modeling and Lyapunov stability analysis.",
                explicit_limitations="Neglects higher-order harmonics in inverter switching.",
                dataset_names=["IEEE-14-Bus"]
            )
        ]
        topic = "Deep Learning for Early Detection of Alzheimer's Disease Using MRI and PET Brain Imaging: A Comparative Study of CNNs, Transformers, and Multimodal Learning"

        diag_data = build_dynamic_diagram_data(schemas=schemas, report_title=topic)

        # Check relevance validation was performed
        rel_info = diag_data.get("corpus_relevance", {})
        self.assertEqual(len(rel_info.get("accepted_papers", [])), 2, "Expected 2 accepted (relevant) papers")
        self.assertEqual(rel_info.get("irrelevant_count"), 2, "Expected 2 rejected (irrelevant) papers")

        # Verify that Irrelevant papers NEVER appear in Methodology Categories
        method_papers = []
        for cat in diag_data.get("methodology", {}).get("categories", []):
            for p in cat.get("all_papers", []):
                method_papers.append(p.get("paper_id"))
        self.assertIn("2301.00010", method_papers)
        self.assertIn("2301.00011", method_papers)
        self.assertNotIn("2111.00099", method_papers, "Black hole physics paper must NOT appear in Alzheimer's methodology")
        self.assertNotIn("2205.00088", method_papers, "Power electronics paper must NOT appear in Alzheimer's methodology")

        # Verify that Irrelevant papers NEVER appear in Thematic Clusters
        theme_papers = []
        for th in diag_data.get("thematic_clusters", {}).get("themes", []):
            for p in th.get("papers", []):
                theme_papers.append(p.get("paper_id"))
        self.assertNotIn("2111.00099", theme_papers)
        self.assertNotIn("2205.00088", theme_papers)

        # Verify that Irrelevant papers NEVER generate Research Gaps
        gap_papers = []
        for g in diag_data.get("research_gaps", {}).get("gaps", []):
            gap_papers.extend(g.get("supporting_papers", []))
        self.assertNotIn("2111.00099", gap_papers)
        self.assertNotIn("2205.00088", gap_papers)

    # =========================================================================
    # Test 3: Topic Resolution & Topic Preservation
    # =========================================================================
    def test_topic_resolution_never_falls_back_to_method_computational(self):
        """
        Verifies that extract_corpus_domain never extracts 'Method Computational'
        or guesses from paper paradigms.
        """
        # Test explicit topic preservation
        topic = "Federated Learning for Privacy-Preserving Medical Image Analysis"
        domain = extract_corpus_domain(report_title=topic)
        self.assertEqual(domain, topic)

        # Test resolution when report_title has markdown prefix
        prefix_topic = "### Automated Literature Review: Quantum Computing for Molecular Simulation"
        domain2 = extract_corpus_domain(report_title=prefix_topic)
        self.assertEqual(domain2, "Quantum Computing for Molecular Simulation")

        # Test resolution with mock comparative analysis containing 'Computational Method'
        comp_mock = {
            "methodology_comparison": [
                {"paper_id": "P1", "paradigm": "Computational Method", "architecture_category": "Method Computational"},
                {"paper_id": "P2", "paradigm": "Computational Method", "architecture_category": "Method Computational"}
            ]
        }
        # With no topic anywhere, should fall back to default_domain ("Scientific Research"), NEVER "Method Computational"
        with patch("os.path.exists", return_value=False):
            fallback_domain = extract_corpus_domain(comp_analysis=comp_mock, report_title="")
            self.assertNotEqual(fallback_domain.lower(), "method computational")
            self.assertNotEqual(fallback_domain.lower(), "computational method")
            self.assertEqual(fallback_domain, "Scientific Research")

    # =========================================================================
    # Test 4: Methodology Taxonomy Cleanliness (No Paper Acronyms)
    # =========================================================================
    def test_methodology_taxonomy_clean_of_paper_specific_acronyms(self):
        """
        Verifies that METHOD_TAXONOMY does not contain test-paper-specific acronyms:
        hdp, ogx1, polaris, monbm, mbdres-u-net, kits, pees, peess, staatt, tierkreis, lsst.
        """
        prohibited_acronyms = [
            r'\bhdp\b', r'\bogx1\b', r'\bpolaris\b', r'\bmonbm\b',
            r'\bmbdres[- ]u[- ]net\b', r'\bkits\b', r'\bpees\b',
            r'\bpeess\b', r'\bstaatt\b', r'\btierkreis\b', r'\blsst\b'
        ]

        for item in METHOD_TAXONOMY:
            regex_str = item.get("regex", "")
            cat_name = item.get("category", "")
            keywords = [k.lower() for k in item.get("keywords", [])]

            for prohibited in prohibited_acronyms:
                # Check regex
                self.assertIsNone(
                    re.search(prohibited, regex_str, re.IGNORECASE),
                    f"Taxonomy regex for '{cat_name}' contains hardcoded acronym pattern '{prohibited}': {regex_str}"
                )
                # Check keywords
                clean_acronym = prohibited.replace(r'\b', '').replace('\\', '')
                self.assertNotIn(
                    clean_acronym,
                    keywords,
                    f"Taxonomy keywords for '{cat_name}' contains hardcoded acronym '{clean_acronym}'"
                )

    # =========================================================================
    # Test 5: Multi-Domain Generality Across 5 Diverse Scientific Fields
    # =========================================================================
    def test_multi_domain_generality_five_scientific_fields(self):
        """
        Validates that the architecture diagrams pipeline dynamically works
        without hardcoding across 5 diverse scientific domains:
        1. Medical Neuroimaging
        2. Cybersecurity & IoT Zero-Trust
        3. Robotics Locomotion
        4. Quantum Chemistry
        5. Climate Modeling & Extreme Weather
        """
        test_domains = [
            {
                "topic": "Deep Learning for Early Detection of Alzheimer's Disease Using MRI and PET Brain Imaging",
                "schemas": [
                    _create_schema(
                        paper_id="MED-01",
                        paper_title="Volumetric CNN for Hippocampal Atrophy Detection in Structural MRI",
                        core_problem="Early Alzheimer's biomarker detection using volumetric T1 brain scans.",
                        methodology_summary="3D Convolutional Neural Network trained on structural MRI volumetric features.",
                        explicit_limitations="Limited clinical cross-scanner generalization due to varying magnetic field strengths."
                    ),
                    _create_schema(
                        paper_id="MED-02",
                        paper_title="Multimodal Transformer for Joint PET and MRI Neurodegeneration Classification",
                        core_problem="Integrating metabolic PET and anatomical MRI for mild cognitive impairment prediction.",
                        methodology_summary="Cross-attention vision transformer fusing PET and MRI tokens.",
                        explicit_limitations="Requires paired multimodal imaging, restricting applicability when PET is unavailable."
                    )
                ]
            },
            {
                "topic": "Zero-Trust Architecture and Graph Neural Networks for IoT Intrusion Detection",
                "schemas": [
                    _create_schema(
                        paper_id="SEC-01",
                        paper_title="Spatiotemporal Graph Neural Networks for Microsegmentation in Industrial IoT",
                        core_problem="Detecting lateral movement attacks across heterogeneous zero-trust IoT networks.",
                        methodology_summary="Spatiotemporal Graph Convolutional Network modeling device communication graphs.",
                        explicit_limitations="Inference latency scales quadratically with graph node count during DDoS bursts."
                    ),
                    _create_schema(
                        paper_id="SEC-02",
                        paper_title="Decentralized Identity and Cryptographic Verification for Sensor Nodes",
                        core_problem="Continuous device authentication without centralized perimeter trust.",
                        methodology_summary="Zero-knowledge proof verification protocol for resource-constrained edge microcontrollers.",
                        explicit_limitations="High cryptographic verification latency on 8-bit embedded sensor nodes."
                    )
                ]
            },
            {
                "topic": "Deep Reinforcement Learning for Dynamic Quadrupedal Locomotion over Rough Terrain",
                "schemas": [
                    _create_schema(
                        paper_id="ROB-01",
                        paper_title="Model-Free Actor-Critic Reinforcement Learning for Blind Quadruped Locomotion",
                        core_problem="Robust legged locomotion over slippery and unpredictable obstacle fields.",
                        methodology_summary="Proximal Policy Optimization (PPO) with privileged height-map critic teacher.",
                        explicit_limitations="Sim-to-real gap causes foot slip and torque saturation under unmodeled terrain compliance."
                    ),
                    _create_schema(
                        paper_id="ROB-02",
                        paper_title="Proprioceptive Terrain Adaptation via Transformer State History",
                        core_problem="Estimating ground contact friction purely from motor encoder history.",
                        methodology_summary="Causal Transformer encoder processing sequential joint torque telemetry.",
                        explicit_limitations="Compounding attention history requires significant edge GPU memory on mobile robots."
                    )
                ]
            },
            {
                "topic": "Quantum Machine Learning for Molecular Ground-State Energy Estimation in Drug Discovery",
                "schemas": [
                    _create_schema(
                        paper_id="QNT-01",
                        paper_title="Variational Quantum Eigensolver with Symmetry-Preserving Ansätze for Molecular Hamiltonians",
                        core_problem="Accurate calculation of electronic ground state energies for transition metal complexes.",
                        methodology_summary="Hardware-efficient Variational Quantum Eigensolver (VQE) with active-space reduction.",
                        explicit_limitations="Barren plateau phenomenon impedes gradient descent on circuits exceeding 20 qubits."
                    ),
                    _create_schema(
                        paper_id="QNT-02",
                        paper_title="Equivariant Graph Neural Networks for Quantum Chemical Property Prediction",
                        core_problem="Predicting molecular dipole moments and HOMO-LUMO energy gaps.",
                        methodology_summary="E(3)-equivariant Graph Neural Network with spherical harmonics representations.",
                        explicit_limitations="Struggles to model long-range dispersion forces and non-covalent intermolecular binding."
                    )
                ]
            },
            {
                "topic": "Deep Learning and Numerical Physics for Extreme Weather Event Forecasting under Climate Change",
                "schemas": [
                    _create_schema(
                        paper_id="CLM-01",
                        paper_title="Fourier Neural Operators for Sub-Seasonal Atmospheric Circulation Modeling",
                        core_problem="High-resolution global weather forecasting under computational resource limits.",
                        methodology_summary="Fourier Neural Operator (FNO) solving 3D Navier-Stokes atmospheric equations.",
                        explicit_limitations="Smooths out extreme tail-risk precipitation events and hurricane eye-wall peaks."
                    ),
                    _create_schema(
                        paper_id="CLM-02",
                        paper_title="Spatiotemporal Diffusion Models for Tropical Cyclone Trajectory Prediction",
                        core_problem="Probabilistic ensemble forecasting of cyclone tracks and landfall intensity.",
                        methodology_summary="Conditional Denoising Diffusion Probabilistic Model guided by satellite telemetry.",
                        explicit_limitations="Diffusion sampling steps are too computationally slow for real-time evacuation warnings."
                    )
                ]
            }
        ]

        for dom in test_domains:
            topic = dom["topic"]
            schemas = dom["schemas"]

            diag_data = build_dynamic_diagram_data(schemas=schemas, report_title=topic)

            # 1. Corpus name should match the topic
            corpus_name = diag_data.get("methodology", {}).get("corpus_name", "")
            self.assertEqual(corpus_name, topic, f"Corpus name should equal active topic for '{topic}'")

            # 2. Methodology categories must be populated
            categories = diag_data.get("methodology", {}).get("categories", [])
            self.assertGreater(len(categories), 0, f"Must have methodology categories for '{topic}'")
            for cat in categories:
                # No methodology category should be a paper title
                cat_name = cat.get("name", "")
                self.assertFalse(
                    any(cat_name == s.paper_title for s in schemas),
                    f"Category name '{cat_name}' must not equal any paper title"
                )

            # 3. Thematic clusters: Valid themes or clean diagnostic assessment (FIX 2)
            themes = diag_data.get("thematic_clusters", {}).get("themes", [])
            for th in themes:
                self.assertNotEqual(th.get("title"), "Insufficient Evidence for Thematic Clustering")

            # 4. Research gaps must be populated and contain actionable opportunities
            gaps = diag_data.get("research_gaps", {}).get("gaps", [])
            self.assertGreater(len(gaps), 0, f"Must have research gaps for '{topic}'")
            for g in gaps:
                self.assertTrue(g.get("future_opportunity"), f"Gap {g.get('gap_id')} must have a future opportunity")

            # 5. Flowchart generation must succeed and contain all 3 stages
            flowchart = generate_methodology_flowchart(schemas=schemas, report_title=topic, diag_data=diag_data)
            self.assertTrue("flowchart" in flowchart or "graph" in flowchart)
            clean_chart = flowchart.replace("<br/>", " ").replace("<br>", " ")
            self.assertIn(topic.split()[0], clean_chart)

            mindmap = generate_themes_mindmap(schemas=schemas, report_title=topic, diag_data=diag_data)
            self.assertTrue("mindmap" in mindmap or "flowchart" in mindmap)

            gaps_chart = generate_research_gaps_flowchart(schemas=schemas, report_title=topic, diag_data=diag_data)
            self.assertIn("flowchart LR", gaps_chart)
            self.assertIn("STAGE 1", gaps_chart)
            self.assertIn("STAGE 2", gaps_chart)
            self.assertIn("STAGE 3", gaps_chart)


    # =========================================================================
    # Test 6: Stale Cache Invalidation in Server API
    # =========================================================================
    def test_cache_invalidation_when_topic_mismatches(self):
        """
        Verifies that server.get_report() detects stale or corrupted cached diagram_data.json
        (e.g. topic is 'Method Computational' or an old topic) and invalidates it.
        """
        import server
        from server import get_report

        stale_cache = {
            "corpus_relevance": {"topic": "Method Computational"},
            "methodology": {"corpus_name": "Method Computational", "categories": []},
            "thematic_clusters": {"corpus_topic": "Method Computational", "themes": []},
            "research_gaps": {"corpus_topic": "Method Computational", "gaps": []}
        }

        active_topic = "Deep Learning for Early Detection of Alzheimer's Disease Using MRI and PET Brain Imaging"

        # Mock os.path.exists and file reading to simulate stale cache on disk and active Alzheimer's session
        with patch("server._get_active_topic", return_value=active_topic), \
             patch("server._get_active_corpus_ids", return_value=["MED-01"]), \
             patch("os.path.exists", return_value=True), \
             patch("builtins.open", unittest.mock.mock_open(read_data=json.dumps(stale_cache))), \
             patch("src.agents.report_generator.build_dynamic_diagram_data") as mock_build, \
             patch("src.agents.report_generator.generate_methodology_flowchart", return_value="flowchart TD"), \
             patch("src.agents.report_generator.generate_themes_mindmap", return_value="mindmap"), \
             patch("src.agents.report_generator.generate_research_gaps_flowchart", return_value="flowchart LR"):

            mock_build.return_value = {
                "corpus_relevance": {"topic": active_topic},
                "methodology": {"corpus_name": active_topic, "categories": []},
                "thematic_clusters": {"corpus_topic": active_topic, "themes": []},
                "research_gaps": {"corpus_topic": active_topic, "gaps": []}
            }

            resp = get_report()
            # Must have called build_dynamic_diagram_data to regenerate fresh data
            mock_build.assert_called_once()
            self.assertEqual(resp["diagram_data"]["methodology"]["corpus_name"], active_topic)

    def test_server_guarantees_all_three_diagrams_even_if_markdown_has_missing_blocks(self):
        """
        Verifies that server.get_report() guarantees all 3 diagram keys
        (methodology_flowchart, themes_mindmap, research_gaps_flowchart) are present and non-empty
        in the returned response even if the report markdown contains only 1 or 0 mermaid blocks.
        """
        import server
        from server import get_report

        active_topic = "Deep Learning for Early Detection of Alzheimer's Disease Using MRI and PET Brain Imaging"
        dummy_schema = _create_schema(
            paper_id="2301.00001",
            paper_title="Multimodal Transformer for Alzheimer's",
            core_problem="Early diagnosis of Alzheimer's",
            methodology_summary="Vision Transformer",
            explicit_limitations="High latency"
        )

        # Markdown that only has research_gaps_flowchart (missing methodology_flowchart and themes_mindmap)
        partial_md = (
            "# Literature Review\n\n"
            "```mermaid\n"
            "%% research_gaps_flowchart\n"
            "flowchart LR\n"
            "    G1[Gap 1]\n"
            "```\n"
        )

        valid_diag_data = {
            "corpus_relevance": {"topic": active_topic},
            "methodology": {"corpus_name": active_topic, "categories": [{"name": "Deep Learning", "paper_count": 1}]},
            "thematic_clusters": {"corpus_topic": active_topic, "themes": [{"cluster_name": "Cluster 1", "paper_count": 1}]},
            "research_gaps": {"corpus_topic": active_topic, "gaps": [{"gap_id": "GAP-01", "name": "Gap 1", "future_opportunity": "Explore transformers"}]}
        }

        with patch("server._get_active_topic", return_value=active_topic), \
             patch("server._get_active_corpus_ids", return_value=["2301.00001"]), \
             patch("server._load_paper_schemas", return_value=[dummy_schema]), \
             patch("os.path.exists", return_value=True), \
             patch("builtins.open", unittest.mock.mock_open(read_data=partial_md)), \
             patch("json.load", return_value=valid_diag_data), \
             patch("src.agents.report_generator.generate_methodology_flowchart", return_value="flowchart TD\n    A[Methodology]"), \
             patch("src.agents.report_generator.generate_themes_mindmap", return_value="mindmap\n    root((Themes))"):

            resp = get_report()
            diagrams = resp.get("diagrams", {})
            self.assertIn("methodology_flowchart", diagrams)
            self.assertIn("themes_mindmap", diagrams)
            self.assertIn("research_gaps_flowchart", diagrams)

            self.assertTrue(len(diagrams["methodology_flowchart"]) > 0, "methodology_flowchart must be non-empty")
            self.assertTrue(len(diagrams["themes_mindmap"]) > 0, "themes_mindmap must be non-empty")
            self.assertTrue(len(diagrams["research_gaps_flowchart"]) > 0, "research_gaps_flowchart must be non-empty")

            self.assertIn("flowchart", diagrams["methodology_flowchart"])
            self.assertIn("mindmap", diagrams["themes_mindmap"])
            self.assertIn("flowchart", diagrams["research_gaps_flowchart"])

    def test_cache_invalidation_when_categories_empty_with_active_schemas(self):
        """
        Verifies that if cached diagram_data.json has categories: [] while active schemas exist,
        server.get_report() treats it as corrupted and rebuilds fresh diagram data.
        """
        import server
        from server import get_report

        active_topic = "Federated Learning for Medical Imaging"
        dummy_schema = _create_schema(
            paper_id="FL-01",
            paper_title="Federated Learning in Healthcare",
            core_problem="Privacy preservation",
            methodology_summary="FedAvg on CNN",
            explicit_limitations="Communication bottleneck"
        )

        empty_categories_cache = {
            "corpus_relevance": {"topic": active_topic},
            "methodology": {"corpus_name": active_topic, "categories": []},
            "thematic_clusters": {"corpus_topic": active_topic, "themes": []},
            "research_gaps": {"corpus_topic": active_topic, "gaps": []}
        }

        fresh_diag_data = {
            "corpus_relevance": {"topic": active_topic},
            "methodology": {"corpus_name": active_topic, "categories": [{"name": "Federated Learning", "paper_count": 1}]},
            "thematic_clusters": {"corpus_topic": active_topic, "themes": [{"cluster_name": "Privacy", "paper_count": 1}]},
            "research_gaps": {"corpus_topic": active_topic, "gaps": [{"gap_id": "GAP-01", "name": "Bandwidth Gap", "future_opportunity": "Quantization"}]}
        }

        with patch("server._get_active_topic", return_value=active_topic), \
             patch("server._get_active_corpus_ids", return_value=["FL-01"]), \
             patch("server._load_paper_schemas", return_value=[dummy_schema]), \
             patch("os.path.exists", return_value=True), \
             patch("builtins.open", unittest.mock.mock_open(read_data="# Dummy Report")), \
             patch("json.load", return_value=empty_categories_cache), \
             patch("src.agents.report_generator.build_dynamic_diagram_data", return_value=fresh_diag_data) as mock_build, \
             patch("src.agents.report_generator.generate_methodology_flowchart", return_value="flowchart TD"), \
             patch("src.agents.report_generator.generate_themes_mindmap", return_value="mindmap"), \
             patch("src.agents.report_generator.generate_research_gaps_flowchart", return_value="flowchart LR"):

            resp = get_report()
            # Must have invalidated the empty categories cache and rebuilt fresh data
            mock_build.assert_called_once()
            self.assertEqual(len(resp["diagram_data"]["methodology"]["categories"]), 1)


if __name__ == "__main__":
    unittest.main()
