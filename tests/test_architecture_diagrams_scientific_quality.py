import pytest
import re
from unittest.mock import patch
from src.agents.report_generator import (
    _is_invalid_methodology_phrase,
    _is_valid_limitation_statement,
    _sanitize_for_thematic_clustering,
    _extract_5_methodology_dimensions,
    _synthesize_thematic_cluster_title,
    _cluster_dynamic_themes,
    _build_grounded_research_gaps,
    build_dynamic_diagram_data,
    generate_methodology_flowchart,
    generate_themes_mindmap,
    generate_research_gaps_flowchart,
    extract_corpus_domain
)

# ---------------------------------------------------------------------------
# Requirement 1: Paradigm, domain, puzzle concepts never become methodologies
# ---------------------------------------------------------------------------
def test_req1_semantic_rules_reject_philosophical_and_domain_concepts():
    """Verify general semantic rules veto philosophical dilemmas, broad application domains, and metadata."""
    # Broad philosophical / puzzle concepts
    assert _is_invalid_methodology_phrase("AGI") is True
    assert _is_invalid_methodology_phrase("Artificial General Intelligence") is True
    assert _is_invalid_methodology_phrase("Fermi Paradox") is True
    assert _is_invalid_methodology_phrase("Creative Problem Solving") is True
    assert _is_invalid_methodology_phrase("Dilemma in Machine Autonomy") is True
    assert _is_invalid_methodology_phrase("Drake Equation") is True

    # Application domains (WHAT/WHERE, not algorithmic HOW)
    assert _is_invalid_methodology_phrase("Smart Manufacturing") is True
    assert _is_invalid_methodology_phrase("Predictive Maintenance") is True
    assert _is_invalid_methodology_phrase("Astrophysics") is True
    assert _is_invalid_methodology_phrase("Healthcare") is True

    # Temporal & Bibliographic metadata
    assert _is_invalid_methodology_phrase("October 2021") is True
    assert _is_invalid_methodology_phrase("arXiv:2110.01835") is True
    assert _is_invalid_methodology_phrase("Springer Proceedings") is True

    # Valid algorithmic techniques must pass
    assert _is_invalid_methodology_phrase("Random Forest") is False
    assert _is_invalid_methodology_phrase("Gradient Boosting") is False
    assert _is_invalid_methodology_phrase("Deep Neural Network") is False
    assert _is_invalid_methodology_phrase("Convolutional Neural Network") is False
    assert _is_invalid_methodology_phrase("SHAP") is False

# ---------------------------------------------------------------------------
# Requirement 2: Strict evidence limitation validator rejects authors & section pointers
# ---------------------------------------------------------------------------
def test_req2_limitation_validator_rejects_authors_affiliations_and_forward_ptrs():
    """Verify limitation validator strictly rejects author lists, affiliations, and forward references."""
    assert _is_valid_limitation_statement("John Smith, Jane Doe and Bob Johnson from University of Cambridge") is False
    assert _is_valid_limitation_statement("Department of Computer Science, University of California, Berkeley") is False
    assert _is_valid_limitation_statement("We discuss this limitation in greater detail in Section 7.") is False
    assert _is_valid_limitation_statement("See Section 4 for further discussion of computational limitations.") is False
    assert _is_valid_limitation_statement("Further research is required.") is False
    assert _is_valid_limitation_statement("Experimental scope and domain constraints noted.") is False

    # Valid technical limitations must pass
    valid_lim1 = "The proposed tree-based model incurs high inference latency on high-frequency streaming sensor telemetry."
    assert _is_valid_limitation_statement(valid_lim1) is True
    valid_lim2 = "Our approach assumes that all sensor nodes have synchronized clock times, which fails under network packet loss."
    assert _is_valid_limitation_statement(valid_lim2) is True

# ---------------------------------------------------------------------------
# Requirement 3: Thematic clustering sanitizer removes dates, years, arXiv IDs
# ---------------------------------------------------------------------------
def test_req3_thematic_clustering_sanitizer_removes_dates_and_arxiv():
    """Verify arXiv identifiers, publication dates, and months are stripped before clustering."""
    raw = "arXiv:2110.01835v1 [cs.AI] 5 Oct 2021 Vibration sensor telemetry analysis"
    clean = _sanitize_for_thematic_clustering(raw)
    assert "2110.01835" not in clean
    assert "2021" not in clean
    assert "Oct" not in clean
    assert "telemetry" in clean.lower()

# ---------------------------------------------------------------------------
# Requirement 4: Thematic cluster title synthesis rejects dates
# ---------------------------------------------------------------------------
def test_req4_thematic_cluster_title_rejects_dates():
    """Verify cluster title synthesis never outputs dates like 'Oct 2021'."""
    papers = [
        {"pid": "p1", "title": "Predictive Maintenance in Industrial Sensors", "problem_text": "Evaluating vibration data on machine bearings."},
        {"pid": "p2", "title": "Anomaly Detection for Turbines", "problem_text": "Detecting early mechanical degradation from acoustic telemetry."}
    ]
    title = _synthesize_thematic_cluster_title(papers, "Predictive Maintenance")
    assert not re.search(r'\b(?:19\d\d|20\d\d|oct|nov|dec|jan|feb)\b', title, re.I)
    assert "Oct 2021" not in title

# ---------------------------------------------------------------------------
# Requirement 5: Thematic clustering enforces minimum two papers
# ---------------------------------------------------------------------------
def test_req5_thematic_clustering_enforces_minimum_two_papers():
    """Verify that thematic clusters require >= 2 papers and never output single-paper clusters."""
    papers = [
        {"pid": "p1", "title": "Predictive Maintenance A", "problem_text": "Bearing wear analysis in turbines.", "technique": "Random Forest"},
        {"pid": "p2", "title": "Predictive Maintenance B", "problem_text": "Vibration monitoring for industrial turbines.", "technique": "Gradient Boosting"}
    ]
    clusters = _cluster_dynamic_themes(papers, "Predictive Maintenance")
    for c in clusters:
        assert c["paper_count"] >= 2
        assert len(c["paper_ids"]) >= 2

# ---------------------------------------------------------------------------
# Requirement 6: 5 orthogonal methodology dimensions with textual evidence
# ---------------------------------------------------------------------------
def test_req6_five_dimension_extraction():
    """Verify extraction of 5 orthogonal methodology dimensions with textual evidence."""
    text = (
        "We evaluate a Random Forest classifier and Gradient Boosting against Deep Neural Networks "
        "using supervised learning on industrial sensor and vibration telemetry records. "
        "We perform a comparative study and apply SHAP (Shapley Additive Explanations) for model interpretability."
    )
    dims = _extract_5_methodology_dimensions(text, "Comparative Maintenance Study")
    assert dims["model_architecture"] in ["Random Forest", "Gradient Boosting", "Deep Neural Network (DNN)"]
    assert dims["learning_paradigm"] == "Supervised Learning"
    assert dims["data_modality"] == "Sensor & Telemetry Data"
    assert dims["research_design"] == "Comparative Study"
    assert dims["explainability_technique"] == "SHAP (Shapley Additive Explanations)"
    assert dims["explainability_technique_evidence"] != "Insufficient evidence extracted for this dimension"

# ---------------------------------------------------------------------------
# Requirement 7: Research gap evidence gating and uniform IDs
# ---------------------------------------------------------------------------
def test_req7_gap_evidence_gating_and_uniform_ids():
    """Verify research gaps require verifiable evidence and produce uniform GAP-01..GAP-xx IDs."""
    papers = [
        {
            "pid": "P1",
            "title": "Study on Bearing Degradation",
            "limitation_quote": "The proposed tree ensemble incurs high inference latency on high-frequency streaming telemetry.",
            "limitation_text": "The proposed tree ensemble incurs high inference latency on high-frequency streaming telemetry.",
            "limitation_section": "Section 5.2",
            "has_limitation": True,
            "gap_type": "author_stated_limitation"
        },
        {
            "pid": "P2",
            "title": "Unrelated Study",
            "limitation_quote": "We discuss this limitation in Section 7.",  # Invalid limitation
            "limitation_text": "We discuss this limitation in Section 7.",
            "limitation_section": "N/A",
            "has_limitation": False,
            "gap_type": "unavailable_evidence"
        }
    ]
    gaps = _build_grounded_research_gaps(papers, "Predictive Maintenance")
    # Only P1 should produce a valid gap; P2 must be excluded (no fake unavailable_evidence cards)
    assert len(gaps) == 1
    assert gaps[0]["gap_id"] == "GAP-01"
    assert gaps[0]["supporting_papers"] == ["P1"]
    assert "inference latency" in gaps[0]["observed_limitation"]

# ---------------------------------------------------------------------------
# Requirement 8: Relevance pre-gating excludes irrelevant papers
# ---------------------------------------------------------------------------
def test_req8_corpus_relevance_gating_zero_relevant_papers():
    """Verify when 0 papers are directly relevant to the topic, system declares is_insufficient and empty primary diagrams."""
    schemas = [
        {
            "paper_id": "2110.01835",
            "paper_title": "Interstellar Communication and the Fermi Paradox",
            "abstract": "We explore radio transmissions and astronomical constraints on intelligent extraterrestrial civilizations.",
            "core_problem": "Fermi paradox constraints on extraterrestrial intelligence."
        },
        {
            "paper_id": "2204.10358",
            "paper_title": "Creative Problem Solving and the MacGyver Paradigm",
            "abstract": "Robotic improvisation of non-standard tools for solving physical puzzles.",
            "core_problem": "Improvising non-standard tools for physical puzzle tasks."
        }
    ]
    diag = build_dynamic_diagram_data(
        schemas=schemas,
        report_title="Explainable Artificial Intelligence for Predictive Maintenance in Smart Manufacturing"
    )
    assert diag["is_insufficient"] is True
    assert len(diag["methodology"]["categories"]) == 0
    assert len(diag["thematic_clusters"]["themes"]) == 0
    assert len(diag["research_gaps"]["gaps"]) == 0

# ---------------------------------------------------------------------------
# Requirement 9: Mermaid diagnostic state on insufficient corpus
# ---------------------------------------------------------------------------
def test_req9_mermaid_diagnostic_state_on_insufficient_corpus():
    """Verify Mermaid generators produce valid diagnostic subgraphs instead of empty strings when insufficient."""
    diag = {
        "is_insufficient": True,
        "methodology": {"corpus_name": "Predictive Maintenance", "categories": [], "is_insufficient": True},
        "thematic_clusters": {"corpus_topic": "Predictive Maintenance", "themes": [], "is_insufficient": True},
        "research_gaps": {"corpus_topic": "Predictive Maintenance", "gaps": [], "is_insufficient": True}
    }
    m1 = generate_methodology_flowchart(diag_data=diag)
    assert "Diagnostic Assessment" in m1
    assert "flowchart" in m1

    m2 = generate_themes_mindmap(diag_data=diag)
    assert "Diagnostic Assessment" in m2
    assert "flowchart" in m2

    m3 = generate_research_gaps_flowchart(diag_data=diag)
    assert "Diagnostic Assessment" in m3
    assert "flowchart" in m3

# ---------------------------------------------------------------------------
# Requirement 10: Topic resolution preserves full user topic
# ---------------------------------------------------------------------------
def test_req10_topic_resolution_preserves_full_user_topic():
    """Verify extract_corpus_domain preserves active research topic instead of generic 'Literature Review'."""
    full_topic = "Explainable Artificial Intelligence for Predictive Maintenance in Smart Manufacturing: A Comparative Study of Random Forests, Gradient Boosting, Deep Neural Networks, and SHAP-Based Interpretability"
    resolved = extract_corpus_domain(report_title=f"# Automated Literature Review: {full_topic}")
    assert "Predictive Maintenance" in resolved
    assert resolved != "Literature Review"
    assert resolved != "Topic Not Identified"

# ---------------------------------------------------------------------------
# Requirement 11: Partially relevant papers are never presented as directly relevant
# ---------------------------------------------------------------------------
def test_req11_partially_relevant_papers_isolated():
    """Verify partially relevant papers are tracked separately and not labeled as Directly Relevant."""
    schemas = [
        {
            "paper_id": "P_DIR_1",
            "paper_title": "Deep Learning for Predictive Maintenance in Industrial Smart Manufacturing",
            "abstract": "We evaluate random forest and deep neural networks on sensor telemetry for predictive maintenance in manufacturing.",
            "core_problem": "Predictive maintenance on industrial machine tool telemetry."
        },
        {
            "paper_id": "P_PART_1",
            "paper_title": "Overview of Deep Neural Network Architectures in General Computing",
            "abstract": "A survey of neural network layers in general computing benchmarks.",
            "core_problem": "General computing performance across benchmark datasets."
        }
    ]
    diag = build_dynamic_diagram_data(schemas=schemas, report_title="Predictive Maintenance in Smart Manufacturing")
    eval_map = {p["paper_id"]: p.get("relevance_level") for p in diag.get("corpus_relevance", {}).get("paper_relevance", [])}
    assert eval_map.get("P_DIR_1") == "Directly Relevant"
    assert eval_map.get("P_PART_1") != "Directly Relevant"

# ---------------------------------------------------------------------------
# Requirement 12: Methodology cards and Mermaid contain identical structured data
# ---------------------------------------------------------------------------
def test_req12_methodology_cards_and_mermaid_match():
    """Verify categories and representative papers in interactive cards match Mermaid flowchart."""
    schemas = [
        {
            "paper_id": "M1",
            "paper_title": "Random Forest for Predictive Maintenance Telemetry",
            "abstract": "We evaluate Random Forest on sensor vibration data for predictive maintenance.",
            "core_problem": "Vibration analysis for bearing fault detection."
        },
        {
            "paper_id": "M2",
            "paper_title": "Gradient Boosting for Industrial Sensor Prediction",
            "abstract": "We apply Gradient Boosting on acoustic sensor telemetry for predictive maintenance.",
            "core_problem": "Acoustic sensor fault detection in turbines."
        }
    ]
    diag = build_dynamic_diagram_data(schemas=schemas, report_title="Predictive Maintenance")
    flowchart = generate_methodology_flowchart(diag_data=diag, schemas=schemas, report_title="Predictive Maintenance")
    categories = diag["methodology"]["categories"]
    assert len(categories) > 0
    for cat in categories:
        cat_name = cat["name"]
        # Category name must appear in the Mermaid code
        assert cat_name.split()[0] in flowchart

# ---------------------------------------------------------------------------
# Requirement 13: Thematic cards and Mermaid contain identical structured data
# ---------------------------------------------------------------------------
def test_req13_thematic_cards_and_mermaid_match():
    """Verify themes in interactive cards match themes rendered in Mermaid mindmap."""
    schemas = [
        {
            "paper_id": "T1",
            "paper_title": "Predictive Maintenance for Turbine Bearings",
            "abstract": "Investigating vibration sensor telemetry for predictive maintenance in industrial manufacturing.",
            "core_problem": "Bearing degradation detection in manufacturing."
        },
        {
            "paper_id": "T2",
            "paper_title": "Predictive Maintenance for Industrial Motors",
            "abstract": "Investigating acoustic sensor telemetry for predictive maintenance in industrial manufacturing.",
            "core_problem": "Motor degradation detection in manufacturing."
        }
    ]
    diag = build_dynamic_diagram_data(schemas=schemas, report_title="Predictive Maintenance")
    mindmap = generate_themes_mindmap(diag_data=diag, schemas=schemas, report_title="Predictive Maintenance")
    themes = diag["thematic_clusters"]["themes"]
    assert len(themes) > 0
    for th in themes:
        t_title = th["title"]
        assert t_title.split()[0] in mindmap

# ---------------------------------------------------------------------------
# Requirement 14: Research gap cards and Mermaid contain identical IDs and evidence
# ---------------------------------------------------------------------------
def test_req14_research_gap_cards_and_mermaid_single_source_of_truth():
    """Verify research gap cards and Mermaid flowchart consume identical GAP IDs and evidence."""
    papers = [
        {
            "pid": "G1",
            "title": "Predictive Maintenance Study",
            "limitation_quote": "The model suffers from high latency when processing streaming high-frequency sensor telemetry.",
            "limitation_text": "The model suffers from high latency when processing streaming high-frequency sensor telemetry.",
            "limitation_section": "Section 5.1",
            "has_limitation": True,
            "gap_type": "author_stated_limitation"
        }
    ]
    gaps = _build_grounded_research_gaps(papers, "Predictive Maintenance")
    diag = {
        "research_gaps": {
            "corpus_topic": "Predictive Maintenance",
            "gaps": gaps
        }
    }
    flowchart = generate_research_gaps_flowchart(diag_data=diag, report_title="Predictive Maintenance")
    assert gaps[0]["gap_id"] in flowchart
    assert "high latency" in flowchart.lower()

# ---------------------------------------------------------------------------
# Requirement 15: Relevant papers not rejected solely because titles lack topic keywords
# ---------------------------------------------------------------------------
def test_req15_relevant_papers_accepted_via_abstract_and_problem():
    """Verify a paper whose title is technical (without topic keywords) is accepted if abstract/problem matches."""
    schemas = [
        {
            "paper_id": "P_TECH_1",
            "paper_title": "Acoustic Telemetry Feature Extraction via Wavelet Ensembles",
            "abstract": "We evaluate wavelet ensembles on sensor telemetry specifically for predictive maintenance in industrial manufacturing applications.",
            "core_problem": "Bearing failure prediction for predictive maintenance in smart manufacturing."
        }
    ]
    diag = build_dynamic_diagram_data(schemas=schemas, report_title="Predictive Maintenance in Smart Manufacturing")
    relevance = diag.get("corpus_relevance", {})
    p_status = next((p.get("status") for p in relevance.get("paper_relevance", []) if p.get("paper_id") == "P_TECH_1"), None)
    assert p_status in ["Directly Relevant", "Partially Relevant"]

# ---------------------------------------------------------------------------
# Requirement 16: Navigation statements strictly rejected as research gaps
# ---------------------------------------------------------------------------
def test_navigation_statements_rejected_as_gaps():
    """Verify retrospective section-navigation statements and discourse markers are rejected as research gaps."""
    nav1 = "In this section, we have discussed the future directions of IoT and ML-based smart healthcare challenges."
    nav2 = "This section is divided into three subsections, with each subsection individually representing the three domain-specific smart healthcare open challenges."
    nav3 = "Finally, section 10.4 concludes by outlining domain-specific potential future directions based on the survey."
    nav4 = "In the next section, we provide an in-depth survey summarizing the open challenges."
    nav5 = "The aim is to assist future contributors in understanding and addressing these challenges."

    assert _is_valid_limitation_statement(nav1) is False
    assert _is_valid_limitation_statement(nav2) is False
    assert _is_valid_limitation_statement(nav3) is False
    assert _is_valid_limitation_statement(nav4) is False
    assert _is_valid_limitation_statement(nav5) is False

# ---------------------------------------------------------------------------
# Requirement 17: Vague limitation boilerplate statements strictly rejected
# ---------------------------------------------------------------------------
def test_vague_limitation_statements_rejected():
    """Verify broad, non-actionable boilerplate acknowledgments lacking technical constraints are rejected."""
    vague1 = "Because of this, even while XAI has the potential to contribute  to reliable AI, it is not without its limitations [167]."
    vague2 = "The proposed study has several limitations [12]."
    vague3 = "Our methodology is not without its limitations."
    vague4 = "The limitations of this approach are discussed."
    vague5 = "The system is subject to certain limitations."

    assert _is_valid_limitation_statement(vague1) is False
    assert _is_valid_limitation_statement(vague2) is False
    assert _is_valid_limitation_statement(vague3) is False
    assert _is_valid_limitation_statement(vague4) is False
    assert _is_valid_limitation_statement(vague5) is False

# ---------------------------------------------------------------------------
# Requirement 18: IoT alone does not imply Sensor & Telemetry Data modality
# ---------------------------------------------------------------------------
def test_iot_alone_does_not_imply_sensor_telemetry():
    """Verify mention of IoT or smart systems without physical sensor telemetry does not match Sensor & Telemetry Data."""
    text = (
        "We discuss an architectural framework for IoT-based smart healthcare systems and cloud integration. "
        "The distributed platform coordinates network communication protocols and edge server requests."
    )
    dims = _extract_5_methodology_dimensions(text, "IoT Healthcare Framework")
    assert dims["data_modality"] != "Sensor & Telemetry Data"
    assert dims["data_modality"] == "Not identified"

# ---------------------------------------------------------------------------
# Requirement 19: Casual literature review mention does not imply Systematic Review & Survey
# ---------------------------------------------------------------------------
def test_mention_of_literature_review_alone_does_not_imply_systematic_review():
    """Verify casual phrases like 'the literature review revealed' do not classify a paper as a Systematic Review."""
    text = (
        "The literature review revealed a recurring gap: a lack of explicit application of service design principles. "
        "We propose an empirical user evaluation of healthcare digital twins with visual analytics interfaces."
    )
    dims = _extract_5_methodology_dimensions(text, "Digital Twins Evaluation")
    assert dims["research_design"] != "Systematic Review & Survey"

# ---------------------------------------------------------------------------
# Requirement 20: Healthcare overlap alone does not force shared XAI theme
# ---------------------------------------------------------------------------
def test_healthcare_overlap_alone_does_not_force_shared_xai_theme():
    """Verify papers sharing domain words like 'healthcare' are not forced into an XAI cluster or truncated titles."""
    papers = [
        {
            "pid": "XAI_1",
            "title": "A Review on Explainable Artificial Intelligence in Healthcare",
            "problem_text": "Investigating post-hoc explainability and feature importance in clinical diagnostics.",
            "technique": "SHAP (Shapley Additive Explanations)"
        },
        {
            "pid": "XAI_2",
            "title": "Study on the Helpfulness of Explainable Artificial Intelligence",
            "problem_text": "Evaluating human user trust and understanding of explainable AI decision support.",
            "technique": "Deep Neural Network (DNN) / MLP"
        },
        {
            "pid": "IOT_1",
            "title": "Smart Healthcare Challenges and Resource Bottlenecks in IoT",
            "problem_text": "Addressing energy consumption and fog computing constraints in wearable devices.",
            "technique": "Distributed Cloud Architecture"
        }
    ]
    clusters = _cluster_dynamic_themes(papers, "Explainable Artificial Intelligence in Healthcare")
    xai_cluster = next((c for c in clusters if "Explainable" in c["title"]), None)
    assert xai_cluster is not None
    # Must NOT contain the unrelated IoT paper
    assert "IOT_1" not in xai_cluster["paper_ids"]
    # Must NOT have truncated title 'Explainable Artificial'
    assert xai_cluster["title"] != "Explainable Artificial"
    assert "Intelligence" in xai_cluster["title"] or "Systems" in xai_cluster["title"]

# ---------------------------------------------------------------------------
# Requirement 21: Rejected evidence disappears from both cards and Mermaid
# ---------------------------------------------------------------------------
def test_rejected_evidence_disappears_from_both_cards_and_mermaid():
    """Verify when navigation or vague limitation statements are rejected, they disappear from cards and Mermaid."""
    papers = [
        {
            "pid": "P_VAGUE",
            "title": "Vague Study",
            "limitation_quote": "Because of this, even while XAI has the potential to contribute to reliable AI, it is not without its limitations [167].",
            "limitation_text": "Because of this, even while XAI has the potential to contribute to reliable AI, it is not without its limitations [167].",
            "limitation_section": "Section 6",
            "has_limitation": True
        },
        {
            "pid": "P_NAV",
            "title": "Navigation Study",
            "limitation_quote": "In this section, we have discussed the future directions of IoT and ML-based smart healthcare challenges.",
            "limitation_text": "In this section, we have discussed the future directions of IoT and ML-based smart healthcare challenges.",
            "limitation_section": "Section 10.4",
            "has_limitation": True
        }
    ]
    gaps = _build_grounded_research_gaps(papers, "Explainable AI")
    # Both invalid statements must be rejected -> 0 gaps
    assert len(gaps) == 0

    diag = {"research_gaps": {"corpus_topic": "Explainable AI", "gaps": gaps}}
    flowchart = generate_research_gaps_flowchart(diag_data=diag, report_title="Explainable AI")
    assert "not without its limitations" not in flowchart.lower()
    assert "in this section, we have discussed" not in flowchart.lower()

# ---------------------------------------------------------------------------
# Requirement 22: Existing valid limitations remain intact
# ---------------------------------------------------------------------------
def test_existing_valid_limitations_remain_intact():
    """Verify valid technical limitations and future directions are accepted and present in both views."""
    papers = [
        {
            "pid": "P_VALID_1",
            "title": "Clinical XAI Study",
            "limitation_quote": "Most of these studies focus on usability aspects such as satisfaction which are biased by user preferences and opinion.",
            "limitation_text": "Most of these studies focus on usability aspects such as satisfaction which are biased by user preferences and opinion.",
            "limitation_section": "Section 4.3",
            "has_limitation": True,
            "gap_type": "author_stated_limitation"
        },
        {
            "pid": "P_VALID_2",
            "title": "IoT Healthcare Study",
            "limitation_quote": "In the domain of IoT-based smart healthcare, the integration of small wearable devices and sensors introduces a critical concern: power usage.",
            "limitation_text": "In the domain of IoT-based smart healthcare, the integration of small wearable devices and sensors introduces a critical concern: power usage.",
            "limitation_section": "Section 10.3",
            "has_limitation": True,
            "gap_type": "author_stated_limitation"
        }
    ]
    gaps = _build_grounded_research_gaps(papers, "Healthcare AI")
    assert len(gaps) == 2
    assert gaps[0]["gap_id"] == "GAP-01"
    assert "biased by user preferences" in gaps[0]["observed_limitation"]
    assert gaps[1]["gap_id"] == "GAP-02"
    assert "power usage" in gaps[1]["observed_limitation"]

    diag = {"research_gaps": {"corpus_topic": "Healthcare AI", "gaps": gaps}}
    flowchart = generate_research_gaps_flowchart(diag_data=diag, report_title="Healthcare AI")
    assert "GAP-01" in flowchart
    assert "GAP-02" in flowchart
    assert "usability aspects" in flowchart.lower()
    assert "iot-based smart" in flowchart.lower()

# ---------------------------------------------------------------------------
# Requirement 23: Thematic cluster title generated from shared evidence without hardcoded mappings
# ---------------------------------------------------------------------------
def test_cluster_title_pure_grounded_no_appended_words():
    """Verify cluster titles are generated strictly from shared evidence across all cluster papers, without appending artificial words."""
    papers = [
        {
            "pid": "2304.04780",
            "title": "A Review on Explainable Artificial Intelligence in Healthcare",
            "problem_text": "Investigating post-hoc explainability and feature attribution in clinical decisions."
        },
        {
            "pid": "2410.11896",
            "title": "Study on the Helpfulness of Explainable Artificial Intelligence",
            "problem_text": "Evaluating human user trust and empirical helpfulness of explainable artificial intelligence."
        }
    ]
    title = _synthesize_thematic_cluster_title(papers, "Explainable AI")
    # Must be grounded strictly in both papers: "Explainable Artificial Intelligence"
    assert title == "Explainable Artificial Intelligence"
    # Never artificially appended with "Systems" or other non-existent nouns
    assert not title.endswith(" Systems")
    assert not title.endswith(" Intelligence Systems")

# ---------------------------------------------------------------------------
# Requirement 24: Thematic cluster title returns insufficient-evidence state if ungrounded
# ---------------------------------------------------------------------------
def test_cluster_title_insufficient_evidence_state():
    """Verify when papers share no defensible research objective phrase, an explicit insufficient-evidence state is returned."""
    papers = [
        {
            "pid": "P_A",
            "title": "Quantum Computing Annealing for Material Science",
            "problem_text": "Simulating crystal lattice vibrations under superconducting conditions."
        },
        {
            "pid": "P_B",
            "title": "Microbial Fermentation Dynamics in Agricultural Soil",
            "problem_text": "Tracking nitrogen-fixing bacterial populations across crop rotation cycles."
        }
    ]
    title = _synthesize_thematic_cluster_title(papers, "Interdisciplinary Studies")
    assert title == "Insufficient Evidence for Thematic Clustering"

# ---------------------------------------------------------------------------
# Requirement 25: Background definition rejected as research limitation (GAP-03 reassessment)
# ---------------------------------------------------------------------------
def test_background_definition_rejected_as_research_limitation():
    """Verify textbook definitions (such as GAP-03 replacement evidence) are rejected as empirical limitations."""
    textbook_def = "Imbalanced Data Class imbalance is a prevalent factor in ML, occurring when the distribution of classes within a dataset is uneven."
    assert _is_valid_limitation_statement(textbook_def) is False

    generic_def2 = "Deep learning is characterized by multi-layered neural representations traditionally used for feature extraction."
    assert _is_valid_limitation_statement(generic_def2) is False

# ---------------------------------------------------------------------------
# Requirement 26: Third-party tool reviews and domain importance statements rejected
# ---------------------------------------------------------------------------
def test_third_party_tool_description_and_importance_rejected():
    """Verify descriptions of third-party tools (FallCare) and broad domain importance statements are rejected."""
    fallcare_desc = "FallCare faces three prominent issues: (1) constrained computational resources on the Raspberry Pi 2 (RPi 2); (2) a lack of live streaming capability."
    assert _is_valid_limitation_statement(fallcare_desc) is False

    domain_importance = "Developing Lightweight and Energy Efficient Solution Smart healthcare issues, such as high energy consumption and the lack of an effective solution for resource management, are of the utmost importance."
    assert _is_valid_limitation_statement(domain_importance) is False

    chapter_nav = "Finally, the chapter concludes by outlining domain-specific potential future directions in Section (10.4), as based on the survey."
    assert _is_valid_limitation_statement(chapter_nav) is False

# ---------------------------------------------------------------------------
# Requirement 27: Rejected limitation disappears completely from cards and Mermaid
# ---------------------------------------------------------------------------
def test_rejected_limitation_disappears_completely_from_cards_and_mermaid():
    """Verify that when a paper has no grounded author-stated limitation, it emits zero gap cards and zero Mermaid nodes."""
    papers = [
        {
            "pid": "2312.05530",
            "title": "Towards Smart Healthcare: Challenges and Opportunities in IoT and ML",
            "limitation_quote": "Imbalanced Data Class imbalance is a prevalent factor in ML, occurring when the distribution of classes within a dataset is uneven.",
            "limitation_text": "Imbalanced Data Class imbalance is a prevalent factor in ML, occurring when the distribution of classes within a dataset is uneven.",
            "limitation_section": "Section 10",
            "has_limitation": True
        }
    ]
    gaps = _build_grounded_research_gaps(papers, "Smart Healthcare")
    # The rejected definition must produce ZERO gaps
    assert len(gaps) == 0

    diag = {"research_gaps": {"corpus_topic": "Smart Healthcare", "gaps": gaps}}
    flowchart = generate_research_gaps_flowchart(diag_data=diag, report_title="Smart Healthcare")
    assert "2312.05530" not in flowchart
    assert "class imbalance" not in flowchart.lower()
    assert "LIM_GAP-01" not in flowchart

# ---------------------------------------------------------------------------
# Requirement 28: Distinct evidence classifications for research gaps
# ---------------------------------------------------------------------------
def test_distinct_evidence_classifications_for_research_gaps():
    """Verify that GAP-01, GAP-02, and GAP-03 receive their distinct semantic classifications:
    - Data-Quality Challenge & Research Consideration (GAP-01)
    - Methodological Limitation of Prior Research (GAP-02)
    - Literature-Derived Research Gap (GAP-03)
    """
    papers = [
        {
            "pid": "2304.04780",
            "title": "A Review on Explainable Artificial Intelligence",
            "limitation_quote": "Real-world  data may include biases, inaccuracies, or be incomplete since it  was not gathered for study reasons.",
            "limitation_section": "Section IV.B",
            "has_limitation": True
        },
        {
            "pid": "2410.11896",
            "title": "Study on the Helpfulness of Explainable Artificial Intelligence",
            "limitation_quote": "Most of these studies focus on usability aspects such as satisfaction which are biased by user preferences and opinion.",
            "limitation_section": "Section 1",
            "has_limitation": True
        },
        {
            "pid": "2506.24104",
            "title": "Bridging Service Design, Visualizations, and Visual Analytics",
            "limitation_quote": "The literature review revealed a recur- ring gap: a lack of explicit application of service design principles.",
            "limitation_section": "Section 2",
            "has_limitation": True
        }
    ]
    gaps = _build_grounded_research_gaps(papers, "Explainable AI in Healthcare")
    assert len(gaps) == 3

    # GAP-01: Data-Quality Challenge
    g1 = gaps[0]
    assert g1["gap_id"] == "GAP-01"
    assert g1["type"] == "data_quality_challenge"
    assert g1["category_label"] == "Data-Quality Challenge"
    assert "Data-Quality Challenge in" in g1["title"]
    assert "data-quality challenge and research consideration" in g1["explanation"]
    assert "experimentally observed limitation of the paper's own study" not in g1["explanation"]

    # GAP-02: Methodological Limitation of Prior Research
    g2 = gaps[1]
    assert g2["gap_id"] == "GAP-02"
    assert g2["type"] == "prior_work_methodological_limitation"
    assert g2["category_label"] == "Methodological Limitation of Prior Research"
    assert "Methodological Limitation of Prior Research in" in g2["title"]
    assert "methodological limitation of previous research motivating their evaluation approach" in g2["explanation"]

    # GAP-03: Literature-Derived Research Gap
    g3 = gaps[2]
    assert g3["gap_id"] == "GAP-03"
    assert g3["type"] == "literature_derived_research_gap"
    assert g3["category_label"] == "Literature-Derived Research Gap"
    assert "Literature-Derived Research Gap in" in g3["title"]
    assert "explicit literature-derived research gap from their targeted literature review" in g3["explanation"]
    assert "systematic review" not in g3["explanation"]

    # Mermaid synchronization check
    diag = {"research_gaps": {"corpus_topic": "Explainable AI in Healthcare", "gaps": gaps}}
    flowchart = generate_research_gaps_flowchart(diag_data=diag, report_title="Explainable AI in Healthcare")
    assert "Data-Quality Challenge" in flowchart
    assert "Methodological" in flowchart and "Prior" in flowchart
    assert "Literature-Derived" in flowchart

# ---------------------------------------------------------------------------
# Requirement 29: Classification fallback on ambiguous/uncertain passages
# ---------------------------------------------------------------------------
def test_classification_fallback_on_ambiguous_passage():
    """Verify that ambiguous or keyword-only passages without operational constraint context
    fall back to 'unclassified' rather than being forced into author_stated_limitation."""
    from src.agents.report_generator import _classify_gap_excerpt

    # Negative test cases across all classification dimensions:
    # 1. Literature gap negatives: keywords present ("literature review", "state of the art", "gap") without literature deficit finding
    assert _classify_gap_excerpt("The literature review includes many papers on this topic.") == "unclassified"
    assert _classify_gap_excerpt("We present a review of the state of the art in healthcare.") == "unclassified"
    assert _classify_gap_excerpt("A gap exists between clinical practice and hospital guidelines.") == "unclassified"

    # 2. Prior work critique negatives: mentions prior work or studies without critique
    assert _classify_gap_excerpt("Previous studies focus on deep learning algorithms.") == "unclassified"
    assert _classify_gap_excerpt("Existing studies use various benchmark datasets.") == "unclassified"
    assert _classify_gap_excerpt("Most of these studies were conducted in 2023.") == "unclassified"
    assert _classify_gap_excerpt("Prior work established foundational principles.") == "unclassified"

    # 3. Data quality negatives: mentions data/EHR/cohort without observational bias or deficit context
    assert _classify_gap_excerpt("The data was collected over a three-month period.") == "unclassified"
    assert _classify_gap_excerpt("High quality of data is important for all machine learning models.") == "unclassified"
    assert _classify_gap_excerpt("We analyzed EHR data using random forest models.") == "unclassified"

    # 4. Future work negatives: mentions future without explicit author intent
    assert _classify_gap_excerpt("In the future, technology will continue to advance rapidly.") == "unclassified"

    # 5. Operational limitation negatives: generic keyword without technical boundary
    assert _classify_gap_excerpt("There are several general limitations in contemporary healthcare technologies.") == "unclassified"
    assert _classify_gap_excerpt("Data is subject to observational characteristics across standard medical domains.") == "unclassified"
    assert _classify_gap_excerpt("The authors acknowledge that the problem is difficult.") == "unclassified"
    assert _classify_gap_excerpt("") == "unclassified"

    # Valid positive cases:
    assert _classify_gap_excerpt("Our literature review revealed a recurring gap: lack of explicit validation.") == "literature_derived_research_gap"
    assert _classify_gap_excerpt("Most of these studies focus on usability aspects such as satisfaction which are biased by user preferences.") == "prior_work_methodological_limitation"
    assert _classify_gap_excerpt("Real-world data may include biases, inaccuracies, or be incomplete.") == "data_quality_challenge"
    assert _classify_gap_excerpt("In the future, we will extend our method to 3D volumetric images.") == "author_stated_future_work"
    assert _classify_gap_excerpt("Our deep learning model is limited by high computational latency and degrades under packet loss.") == "author_stated_limitation"

# ---------------------------------------------------------------------------
# Requirement 30: Grounded source section metadata resolution
# ---------------------------------------------------------------------------
def test_grounded_source_section_metadata_resolution():
    """Use verified PDF metadata where available; otherwise preserve the extracted heading
    rather than inventing a section title or page citation."""
    from src.agents.report_generator import _resolve_grounded_section_metadata

    # GAP-01: 2304.04780
    # The supplied PDF's extracted heading is Section VI; no verified page citation is available.
    sec_gap01 = _resolve_grounded_section_metadata("2304.04780", fallback_sec="VI. DISCUSSION ON DIFFERENT ASPECTS OF XAI")
    assert sec_gap01 == "VI. DISCUSSION ON DIFFERENT ASPECTS OF XAI"

    # GAP-02: 2410.11896
    sec_gap02 = _resolve_grounded_section_metadata("2410.11896", fallback_sec="Abstract. Explainable Artificial Intelligence")
    assert sec_gap02 == "Section 1: Introduction (p. 2)"

    # GAP-03: 2506.24104
    sec_gap03 = _resolve_grounded_section_metadata("2506.24104", fallback_sec="ABSTRACT")
    assert sec_gap03 == "Section 2: State of the Art (p. 1)"

# ---------------------------------------------------------------------------
# Requirement 31: Matching category labels in Interactive Cards and Mermaid
# ---------------------------------------------------------------------------
def test_matching_category_labels_in_cards_and_mermaid():
    """Verify that every gap in diagram_data has category_label synchronized 1:1 with Mermaid stage 1 nodes."""
    import os, json
    from src.agents.report_generator import generate_research_gaps_flowchart, build_dynamic_diagram_data

    diag_data = {}
    if os.path.exists("data/reports/diagram_data.json"):
        with open("data/reports/diagram_data.json", "r", encoding="utf-8") as f:
            diag_data = json.load(f)

    gaps = diag_data.get("research_gaps", {}).get("gaps", [])
    if len(gaps) < 3:
        schemas = []
        for pid in ['2304.04780', '2410.11896', '2506.24104']:
            with open(f"data/extracted_schemas/{pid}_schema.json", "r", encoding="utf-8") as sf:
                schemas.append(json.load(sf))
        diag_data = build_dynamic_diagram_data(schemas=schemas, report_title="Explainable Artificial Intelligence in Healthcare")
        gaps = diag_data.get("research_gaps", {}).get("gaps", [])

    assert len(gaps) >= 3

    flowchart = generate_research_gaps_flowchart(diag_data=diag_data)

    for g in gaps:
        cat_lbl = g.get("category_label")
        assert cat_lbl is not None
        assert cat_lbl != ""
        # The category label (or its key terms) must appear in the Mermaid node definition for this gap
        gid = g.get("gap_id")
        assert f"LIM_{gid.replace('-', '_')}" in flowchart
        # Check that the category label words are in the Mermaid flowchart
        for word in cat_lbl.split():
            if len(word) > 4:
                assert word in flowchart

# ---------------------------------------------------------------------------
# Requirement 32: Unclassified evidence never generates future opportunity or gap
# ---------------------------------------------------------------------------
def test_unclassified_evidence_never_generates_future_opportunity_or_gap():
    """Verify the exact forensic defect:
    1. Unclassified evidence (definitions, notation, technical context without operational constraint)
       does NOT set has_limitation = True.
    2. Does NOT call _derive_grounded_future_opportunity().
    3. Produces future_opportunity == '' (never 'Address reported operational constraint: ...').
    4. Does NOT generate a gap card in _build_grounded_research_gaps().
    5. Does NOT appear in Mermaid diagram.
    """
    from src.agents.report_generator import (
        _extract_grounded_paper_meta,
        _build_grounded_research_gaps,
        generate_research_gaps_flowchart
    )

    # Recreate the exact forensic failure cases from 2306.04338 and 2105.04026
    schema_2306_drift = {
        "paper_id": "2306.04338",
        "paper_title": "CHANGING DATA SOURCES IN THE AGE OF MACHINE LEARNING",
        "explicit_limitations": "These include technical effects such as concept drift, bias, availability, validity, accuracy and completeness, but also the neutrality and potential discontinuation of the statistical offering."
    }
    schema_2105_notation = {
        "paper_id": "2105.04026",
        "paper_title": "The Modern Mathematics of Deep Learning",
        "explicit_limitations": "We refer to W (ℓ) ∈RNℓ×Nℓ−1 and b(ℓ) ∈RNℓas the weight matrices and bias vectors, and to ¯Φ(ℓ) and Φ(ℓ) as the activations and pre-activations of the Nℓneurons in the ℓ-th layer."
    }

    meta_2306 = _extract_grounded_paper_meta(schema_2306_drift)
    meta_2105 = _extract_grounded_paper_meta(schema_2105_notation)

    # 1. Verify has_limitation is consistent with classification: must be False
    assert meta_2306["has_limitation"] is False
    assert meta_2105["has_limitation"] is False
    assert meta_2306["gap_type"] == "unclassified"
    assert meta_2105["gap_type"] == "unclassified"

    # 2. Verify future_opportunity is strictly empty (never synthesized from unclassified text)
    assert meta_2306["future_opportunity"] == ""
    assert meta_2105["future_opportunity"] == ""
    assert "Address reported operational constraint" not in meta_2306["future_opportunity"]
    assert "Address reported operational constraint" not in meta_2105["future_opportunity"]

    # 3. Verify _build_grounded_research_gaps strictly rejects unclassified evidence
    gaps = _build_grounded_research_gaps([meta_2306, meta_2105], "Scientific Research")
    assert len(gaps) == 0  # Zero gap cards emitted!

    # 4. Verify Mermaid flowchart reflects diagnostic empty state and contains no cards/nodes for unclassified evidence
    diag = {"research_gaps": {"corpus_topic": "Scientific Research", "gaps": gaps}}
    flowchart = generate_research_gaps_flowchart(diag_data=diag, report_title="Scientific Research")
    assert "concept drift" not in flowchart
    assert "weight matrices" not in flowchart
    assert "Address reported operational constraint" not in flowchart

# ---------------------------------------------------------------------------
# Requirement 33: Empty structured gaps cannot trigger legacy Mermaid fallback
# ---------------------------------------------------------------------------
def test_empty_structured_gaps_never_triggers_legacy_fallback():
    """Verify FIX 1: When structured research_gaps is empty, generate_research_gaps_flowchart()
    strictly returns the clean diagnostic state and NEVER triggers the legacy 3-column fallback
    (EG_0, SG_0, OPP_0)."""
    from src.agents.report_generator import generate_research_gaps_flowchart

    # Comp analysis containing legacy limitation themes
    mock_comp = {
        "limitation_themes": [
            {"theme": "Empirical Constraints in [P1]", "affected_papers": ["P1"]},
            {"theme": "Empirical Constraints in [P2]", "affected_papers": ["P2"]}
        ]
    }
    mock_gap_analysis = {
        "explicit_gaps": [{"gap_id": "G01", "title": "Old Raw Gap", "affected_papers": ["P1"]}],
        "synthesized_gaps": [{"gap_id": "SYN01", "title": "Old Raw Synthesis", "affected_papers": ["P1", "P2"]}]
    }
    diag_empty = {
        "research_gaps": {
            "corpus_topic": "Phishing Website Detection",
            "is_insufficient": True,
            "gaps": []
        }
    }

    flowchart = generate_research_gaps_flowchart(
        comp_analysis=mock_comp,
        gap_analysis=mock_gap_analysis,
        report_title="Phishing Website Detection",
        diag_data=diag_empty
    )

    # Must be diagnostic assessment only: no Stage 1/2/3, no raw gap IDs
    assert "Diagnostic Assessment: Phishing Website Detection" in flowchart
    assert "STAGE 1: AUTHOR-STATED LIMITATIONS" not in flowchart
    assert "STAGE 2: SYNTHESIZED RESEARCH GAPS" not in flowchart
    assert "STAGE 3: FUTURE OPPORTUNITIES" not in flowchart
    assert "EG_0" not in flowchart
    assert "SG_0" not in flowchart
    assert "OPP_0" not in flowchart

# ---------------------------------------------------------------------------
# Requirement 34: Insufficient thematic evidence never creates pseudo-cluster node
# ---------------------------------------------------------------------------
def test_insufficient_thematic_evidence_never_creates_pseudo_cluster_node():
    """Verify FIX 2: When papers do not share a defensible multi-paper title,
    _cluster_dynamic_themes() returns an empty list, and generate_themes_mindmap()
    renders the diagnostic assessment without drawing an 'Insufficient Evidence' cluster node."""
    from src.agents.report_generator import _cluster_dynamic_themes, generate_themes_mindmap

    # Two papers with unrelated topics that share no defensible noun phrase
    papers = [
        {"pid": "P_A", "title": "IJCSI International Journal of Computer Science", "problem_text": "Online banking security in India"},
        {"pid": "P_B", "title": "Quantum Mechanics and Superconductivity", "problem_text": "Cooper pair condensation at zero Kelvin"}
    ]

    themes = _cluster_dynamic_themes(papers, "Synthesis of Scientific Research")
    # Must NOT contain a theme named 'Insufficient Evidence for Thematic Clustering'
    assert len(themes) == 0
    assert not any(t.get("title") == "Insufficient Evidence for Thematic Clustering" for t in themes)

    diag = {
        "thematic_clusters": {
            "corpus_topic": "Synthesis of Scientific Research",
            "is_insufficient": True,
            "themes": themes
        }
    }
    mindmap = generate_themes_mindmap(report_title="Synthesis of Scientific Research", diag_data=diag)
    assert "Diagnostic Assessment: Synthesis of Scientific" in mindmap
    assert "CORE --> T0" not in mindmap
    assert "Insufficient Evidence<br/>for Thematic Clustering" not in mindmap

# ---------------------------------------------------------------------------
# Requirement 35: Single source of truth consistency between Cards and Mermaid
# ---------------------------------------------------------------------------
def test_cards_and_mermaid_consistent_empty_and_valid_states():
    """Verify FIX 3: Empty and valid states between Interactive Cards (diag_data) and
    Mermaid diagrams are 100% consistent across all three views."""
    from src.agents.report_generator import (
        build_dynamic_diagram_data,
        generate_methodology_flowchart,
        generate_themes_mindmap,
        generate_research_gaps_flowchart
    )

    # Empty/irrelevant corpus state
    empty_diag = {
        "corpus_relevance": {"is_insufficient": True, "paper_relevance": []},
        "is_insufficient": True,
        "methodology": {"corpus_name": "Test Scope", "is_insufficient": True, "categories": []},
        "thematic_clusters": {"corpus_topic": "Test Scope", "is_insufficient": True, "themes": []},
        "research_gaps": {"corpus_topic": "Test Scope", "is_insufficient": True, "gaps": []}
    }

    m1 = generate_methodology_flowchart(report_title="Test Scope", diag_data=empty_diag)
    m2 = generate_themes_mindmap(report_title="Test Scope", diag_data=empty_diag)
    m3 = generate_research_gaps_flowchart(report_title="Test Scope", diag_data=empty_diag)

    # In empty state, all 3 Mermaid diagrams render diagnostic states matching the 3 cards
    assert "Diagnostic Assessment: Test Scope" in m1
    assert "Diagnostic Assessment: Test Scope" in m2
    assert "Diagnostic Assessment: Test Scope" in m3
    assert len(empty_diag["methodology"]["categories"]) == 0
    assert len(empty_diag["thematic_clusters"]["themes"]) == 0
    assert len(empty_diag["research_gaps"]["gaps"]) == 0
