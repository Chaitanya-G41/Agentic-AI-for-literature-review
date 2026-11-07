import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Sparkles, 
  Search, 
  Download, 
  Cpu, 
  FileText, 
  Table2, 
  AlertCircle, 
  BookOpen, 
  CheckCircle2, 
  ArrowRight, 
  RotateCcw,
  Loader2,
  FolderPlus,
  Play,
  Layers,
  Check
} from 'lucide-react';

export default function InteractiveAgentPipeline({
  papers,
  onRefreshCorpus,
  onSwitchToDeskView,
  fieldLogs,
  setFieldLogs
}) {
  const [currentStep, setCurrentStep] = useState(1);
  const [topic, setTopic] = useState("Agentic AI with Multi-Agent Systems and Tool Use");
  const [recencyWeight, setRecencyWeight] = useState(0.20);
  const [maxCandidates, setMaxCandidates] = useState(25);
  const [targetCount, setTargetCount] = useState(6);

  // Agent execution states
  const [screeningLoading, setScreeningLoading] = useState(false);
  const [candidates, setCandidates] = useState([]);
  const [selectedArxivIds, setSelectedArxivIds] = useState([]);
  const [downloading, setDownloading] = useState(false);

  const [extracting, setExtracting] = useState(false);
  const [summarizing, setSummarizing] = useState(false);
  const [comparing, setComparing] = useState(false);
  const [gapping, setGapping] = useState(false);
  const [reporting, setReporting] = useState(false);

  const [stepCompleted, setStepCompleted] = useState({
    1: papers.length > 0,
    2: papers.some(p => p.is_extracted),
    3: papers.some(p => p.is_summarized),
    4: false,
    5: false,
    6: false,
  });

  const addLog = (msg) => {
    setFieldLogs(prev => [
      `[${new Date().toLocaleTimeString()}] ${msg}`,
      ...prev.slice(0, 20)
    ]);
  };

  // Step 1: Run Relevance Screening Agent
  const handleRunScreening = async () => {
    if (!topic.trim()) return;
    setScreeningLoading(true);
    addLog(`Agent 1 (Relevance Screener): Querying arXiv for "${topic}"...`);

    try {
      const res = await fetch('/api/screen', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: topic,
          max_candidates: maxCandidates,
          recency_weight: recencyWeight
        })
      });
      const data = await res.json();
      const list = data.candidates || [];
      setCandidates(list);
      setSelectedArxivIds(list.slice(0, targetCount).map(c => c.arxiv_id));
      addLog(`Agent 1: Screened ${list.length} candidates with TF-IDF cosine similarity and ${Math.round(recencyWeight*100)}% recency bias.`);
    } catch (err) {
      addLog(`Agent 1 Error: ${err.message}`);
    } finally {
      setScreeningLoading(false);
    }
  };

  // Download approved candidates into corpus
  const handleDownloadSelected = async () => {
    if (selectedArxivIds.length === 0) return;
    setDownloading(true);
    addLog(`Agent 1: Downloading ${selectedArxivIds.length} approved PDFs into data/pdf_corpus...`);

    try {
      const selected = candidates.filter(c => selectedArxivIds.includes(c.arxiv_id));
      const res = await fetch('/api/download-papers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ papers: selected })
      });
      const data = await res.json();
      addLog(`Agent 1: Download complete. ${data.count || selected.length} papers ingested into corpus.`);
      setStepCompleted(prev => ({ ...prev, 1: true }));
      await onRefreshCorpus();
      setCurrentStep(2);
    } catch (err) {
      addLog(`Download Error: ${err.message}`);
    } finally {
      setDownloading(false);
    }
  };

  // Fallback: Load 12 Default Benchmark Papers
  const handleLoadBenchmark = async () => {
    addLog("Loading 12 standard Agentic AI benchmark preprints...");
    try {
      await fetch('/api/load-benchmark', { method: 'POST' });
      await onRefreshCorpus();
      setStepCompleted(prev => ({ ...prev, 1: true }));
      addLog("12 benchmark papers ready in corpus.");
      setCurrentStep(2);
    } catch (err) {
      addLog(`Error loading benchmark: ${err.message}`);
    }
  };

  // Clear Corpus / Reset
  const handleResetCorpus = async () => {
    addLog("Resetting corpus and working memory for fresh research topic...");
    try {
      await fetch('/api/clear-corpus', { method: 'POST' });
      setCandidates([]);
      setSelectedArxivIds([]);
      setStepCompleted({ 1: false, 2: false, 3: false, 4: false, 5: false, 6: false });
      setCurrentStep(1);
      await onRefreshCorpus();
      addLog("Ready for fresh topic exploration.");
    } catch (err) {
      addLog(`Reset error: ${err.message}`);
    }
  };

  // Step 2: Run Extraction Agent
  const handleRunExtraction = async () => {
    setExtracting(true);
    addLog("Agent 2 (Schema Extractor): Parsing layouts and extracting schemas...");
    try {
      const res = await fetch('/api/run-agent/extraction', { method: 'POST' });
      const data = await res.json();
      addLog(`Agent 2: Extracted schemas for ${data.count} papers.`);
      setStepCompleted(prev => ({ ...prev, 2: true }));
      await onRefreshCorpus();
      setCurrentStep(3);
    } catch (err) {
      addLog(`Extraction Error: ${err.message}`);
    } finally {
      setExtracting(false);
    }
  };

  // Step 3: Run Summarizer Agent
  const handleRunSummarizer = async () => {
    setSummarizing(true);
    addLog("Agent 3 (Summarizer): Constructing structured 4-part paper summaries...");
    try {
      const res = await fetch('/api/run-agent/summarizer', { method: 'POST' });
      const data = await res.json();
      addLog(`Agent 3: Generated summaries for ${data.count} papers.`);
      setStepCompleted(prev => ({ ...prev, 3: true }));
      await onRefreshCorpus();
      setCurrentStep(4);
    } catch (err) {
      addLog(`Summarizer Error: ${err.message}`);
    } finally {
      setSummarizing(false);
    }
  };

  // Step 4: Run Comparative Matrix Agent
  const handleRunComparative = async () => {
    setComparing(true);
    addLog("Agent 4 (Comparative Analysis): Synthesizing architectural feature matrix...");
    try {
      const res = await fetch('/api/run-agent/comparative', { method: 'POST' });
      const data = await res.json();
      addLog("Agent 4: Cross-paper comparative matrix constructed successfully.");
      setStepCompleted(prev => ({ ...prev, 4: true }));
      setCurrentStep(5);
    } catch (err) {
      addLog(`Comparative Agent Error: ${err.message}`);
    } finally {
      setComparing(false);
    }
  };

  // Step 5: Run Research Gap Agent
  const handleRunResearchGap = async () => {
    setGapping(true);
    addLog("Agent 5 (Research Gap Agent): Synthesizing author limitations and systemic bottlenecks...");
    try {
      const res = await fetch('/api/run-agent/research-gap', { method: 'POST' });
      const data = await res.json();
      addLog("Agent 5: Synthesized explicit and cross-paper research gap taxonomy.");
      setStepCompleted(prev => ({ ...prev, 5: true }));
      setCurrentStep(6);
    } catch (err) {
      addLog(`Research Gap Agent Error: ${err.message}`);
    } finally {
      setGapping(false);
    }
  };

  // Step 6: Run Report Generator Agent
  const handleRunReport = async () => {
    setReporting(true);
    addLog("Agent 6 (Report Generator): Synthesizing One-Prompt literature review and Mermaid diagrams...");
    try {
      const res = await fetch('/api/run-agent/report', { method: 'POST' });
      const data = await res.json();
      addLog("Agent 6: Final literature review report compiled with verified structural diagrams.");
      setStepCompleted(prev => ({ ...prev, 6: true }));
      await onRefreshCorpus();
      // Prompt switch to desk view
      onSwitchToDeskView();
    } catch (err) {
      addLog(`Report Generator Error: ${err.message}`);
    } finally {
      setReporting(false);
    }
  };

  const steps = [
    { num: 1, name: "Relevance Screening", agent: "Agent 1: arXiv & TF-IDF Cosine", desc: "Topic Discovery & Screening" },
    { num: 2, name: "Schema Extraction", agent: "Agent 2: PyMuPDF & Schema", desc: "Grounding & Structural Extraction" },
    { num: 3, name: "Paper Summaries", agent: "Agent 3: Structured Summarizer", desc: "4-Part Structured Synthesis" },
    { num: 4, name: "Comparative Matrix", agent: "Agent 4: Cross-Paper Matrix", desc: "Feature Alignment & Comparison" },
    { num: 5, name: "Research Gaps", agent: "Agent 5: Gap Synthesis", desc: "Systemic Bottleneck Lineage" },
    { num: 6, name: "Final Report", agent: "Agent 6: Monograph & Visuals", desc: "Editorial Report & Mermaid Visuals" },
  ];

  return (
    <div className="flex-1 overflow-y-auto bg-stone-50/70 p-6 lg:p-8">
      <div className="max-w-5xl mx-auto space-y-6">
        
        {/* Header Hero Banner */}
        <div className="bg-white rounded-2xl border border-stone-200/90 shadow-binder p-6 relative overflow-hidden">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 mb-1.5">
                <span className="text-[11px] font-mono uppercase tracking-widest text-amber-800 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
                  Interactive Pipeline Runner
                </span>
                <span className="text-[11px] font-mono text-stone-500">
                  Step {currentStep} of 6 Active
                </span>
              </div>
              <h2 className="font-serif text-2xl font-bold text-stone-900">
                Experience the Multi-Agent Literature Review Pipeline
              </h2>
              <p className="text-xs text-stone-600 font-sans mt-1">
                Follow each specialized agent sequentially or start fresh with your own research topic query.
              </p>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={handleResetCorpus}
                className="text-xs font-medium text-stone-600 hover:text-stone-900 bg-stone-100 hover:bg-stone-200 px-3 py-2 rounded-xl border border-stone-300 transition flex items-center gap-1.5"
                title="Clear current papers and start completely fresh"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Reset / Fresh Topic</span>
              </button>

              <button
                onClick={onSwitchToDeskView}
                className="text-xs font-semibold text-stone-900 bg-amber-100 hover:bg-amber-200 px-3.5 py-2 rounded-xl border border-amber-300 transition flex items-center gap-1.5 shadow-sm"
              >
                <BookOpen className="w-3.5 h-3.5 text-amber-800" />
                <span>View Full Binder Desk</span>
              </button>
            </div>
          </div>

          {/* Stepper Progress Bar */}
          <div className="grid grid-cols-2 md:grid-cols-6 gap-2 mt-6 pt-5 border-t border-stone-100">
            {steps.map((s) => {
              const isCurrent = currentStep === s.num;
              const isDone = stepCompleted[s.num];
              return (
                <div
                  key={s.num}
                  onClick={() => setCurrentStep(s.num)}
                  className={`p-2.5 rounded-xl border cursor-pointer transition-all ${
                    isCurrent 
                      ? 'bg-amber-50/70 border-amber-400 shadow-sm ring-1 ring-amber-400/40' 
                      : isDone 
                      ? 'bg-emerald-50/50 border-emerald-200 hover:bg-emerald-50' 
                      : 'bg-stone-50/70 border-stone-200 opacity-70 hover:opacity-100'
                  }`}
                >
                  <div className="flex items-center justify-between text-[10px] font-mono mb-1">
                    <span className="font-bold text-stone-700">Step {s.num}</span>
                    {isDone ? (
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                    ) : isCurrent ? (
                      <span className="w-2 h-2 rounded-full bg-amber-500 animate-ping" />
                    ) : null}
                  </div>
                  <h4 className="font-serif font-bold text-xs text-stone-900 truncate">
                    {s.name}
                  </h4>
                  <p className="text-[10px] text-stone-500 truncate mt-0.5">
                    {s.desc}
                  </p>
                </div>
              );
            })}
          </div>
        </div>

        {/* STEP 1: USER INPUT & RELEVANCE SCREENING AGENT */}
        {currentStep === 1 && (
          <div className="bg-white rounded-2xl border border-stone-200 p-6 shadow-binder space-y-5">
            <div className="flex items-start justify-between">
              <div>
                <span className="font-mono text-xs font-semibold text-amber-800 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
                  Agent 1: Relevance Screening
                </span>
                <h3 className="font-serif text-xl font-bold text-stone-900 mt-1">
                  Step 1: Enter Research Project Title & Screen arXiv
                </h3>
                <p className="text-xs text-stone-600 mt-0.5">
                  The Relevance Screening Agent autonomously queries arXiv, computes TF-IDF semantic similarities, and applies recency weighting to filter the highest-value preprints.
                </p>
              </div>

              <button
                onClick={handleLoadBenchmark}
                className="text-xs font-mono font-medium text-stone-600 hover:text-stone-900 bg-stone-100 hover:bg-stone-200 px-3 py-1.5 rounded-lg border border-stone-300 transition"
              >
                Load 12 Default Benchmark Papers
              </button>
            </div>

            {/* Topic Input */}
            <div>
              <label className="block font-serif font-bold text-xs text-stone-800 mb-1.5">
                Research Project Title or Topic:
              </label>
              <input
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="e.g. Multi-Modal Vision-Language Agents for Autonomous Web Navigation"
                className="w-full bg-[#FAF9F6] border border-stone-300 rounded-xl px-4 py-2.5 text-xs text-stone-900 focus:outline-none focus:ring-2 focus:ring-amber-500 font-sans shadow-inner"
              />
              {/* Presets */}
              <div className="flex flex-wrap items-center gap-1.5 mt-2">
                <span className="text-[10px] font-mono text-stone-400">Suggestions:</span>
                {[
                  "Agentic AI with Multi-Agent Systems and Tool Use",
                  "Large Language Model Self-Reflection & Tree Search",
                  "Embodied Decision Making in Open-Ended Environments",
                  "Automated Software Engineering with Multi-Agent Systems"
                ].map((s, idx) => (
                  <button
                    key={idx}
                    onClick={() => setTopic(s)}
                    className="text-[10px] font-mono bg-stone-100 hover:bg-amber-50 hover:text-amber-900 text-stone-600 px-2 py-0.5 rounded border border-stone-200 transition"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>

            {/* Screening Sliders */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 bg-stone-50 p-4 rounded-xl border border-stone-200 text-xs">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="font-medium text-stone-700">Target Papers (Top X):</span>
                  <span className="font-mono font-bold text-stone-900">{targetCount}</span>
                </div>
                <input
                  type="range"
                  min="2"
                  max="12"
                  value={targetCount}
                  onChange={(e) => setTargetCount(parseInt(e.target.value))}
                  className="w-full accent-amber-600"
                />
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <span className="font-medium text-stone-700">Recency Preference:</span>
                  <span className="font-mono font-bold text-amber-800">{Math.round(recencyWeight * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="0.35"
                  step="0.05"
                  value={recencyWeight}
                  onChange={(e) => setRecencyWeight(parseFloat(e.target.value))}
                  className="w-full accent-amber-600"
                />
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <span className="font-medium text-stone-700">Candidate Pool:</span>
                  <span className="font-mono font-bold text-stone-900">{maxCandidates}</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="40"
                  step="5"
                  value={maxCandidates}
                  onChange={(e) => setMaxCandidates(parseInt(e.target.value))}
                  className="w-full accent-amber-600"
                />
              </div>
            </div>

            {/* Run Button */}
            <div className="flex justify-end">
              <button
                onClick={handleRunScreening}
                disabled={screeningLoading || !topic.trim()}
                className="bg-stone-900 hover:bg-stone-800 text-stone-100 font-semibold text-xs px-5 py-2.5 rounded-xl shadow flex items-center gap-2 transition disabled:opacity-50"
              >
                {screeningLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin text-amber-300" />
                    <span>Querying arXiv & Scoring Candidates...</span>
                  </>
                ) : (
                  <>
                    <Search className="w-4 h-4 text-amber-300" />
                    <span>Run Relevance Screening Agent</span>
                  </>
                )}
              </button>
            </div>

            {/* Candidate Papers List & Ingestion */}
            {candidates.length > 0 && (
              <div className="space-y-3 pt-3 border-t border-stone-200">
                <div className="flex items-center justify-between">
                  <span className="font-serif font-bold text-sm text-stone-900">
                    Candidate Preprints Ranked by Semantic Match & Recency ({candidates.length})
                  </span>
                  <span className="text-xs font-mono text-stone-500">
                    {selectedArxivIds.length} Papers Selected
                  </span>
                </div>

                <div className="divide-y divide-stone-200 border border-stone-200 rounded-xl max-h-72 overflow-y-auto bg-[#FAF9F6] shadow-inner">
                  {candidates.map((c) => {
                    const isChecked = selectedArxivIds.includes(c.arxiv_id);
                    return (
                      <div
                        key={c.arxiv_id}
                        onClick={() => {
                          setSelectedArxivIds(prev => 
                            prev.includes(c.arxiv_id) ? prev.filter(x => x !== c.arxiv_id) : [...prev, c.arxiv_id]
                          );
                        }}
                        className={`p-3 text-xs flex items-start gap-3 cursor-pointer transition ${
                          isChecked ? 'bg-amber-50/60' : 'hover:bg-white'
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={() => {}}
                          className="mt-1 accent-amber-600"
                        />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-mono text-[10px] font-bold text-amber-900 bg-amber-100 px-2 py-0.5 rounded">
                              {c.relevance_score}% Match
                            </span>
                            <span className="text-[10px] font-mono text-stone-500">
                              Published: {c.published_year} &bull; arXiv: {c.arxiv_id}
                            </span>
                          </div>
                          <h4 className="font-serif font-bold text-xs text-stone-900">
                            {c.title}
                          </h4>
                          <p className="text-[11px] text-stone-600 line-clamp-2 mt-0.5">
                            {c.abstract}
                          </p>
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="flex items-center justify-between pt-2">
                  <span className="text-xs font-mono text-stone-500">
                    PDFs will be stored in <code>data/pdf_corpus/</code>
                  </span>
                  <button
                    onClick={handleDownloadSelected}
                    disabled={downloading || selectedArxivIds.length === 0}
                    className="bg-amber-700 hover:bg-amber-800 text-white font-semibold text-xs px-5 py-2.5 rounded-xl shadow flex items-center gap-2 transition disabled:opacity-50"
                  >
                    {downloading ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Downloading Selected PDFs...</span>
                      </>
                    ) : (
                      <>
                        <Download className="w-4 h-4" />
                        <span>Download & Ingest Selected ({selectedArxivIds.length}) &rarr; Step 2</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* STEP 2: SCHEMA EXTRACTION AGENT */}
        {currentStep === 2 && (
          <div className="bg-white rounded-2xl border border-stone-200 p-6 shadow-binder space-y-5">
            <div>
              <span className="font-mono text-xs font-semibold text-indigo-800 bg-indigo-50 px-2 py-0.5 rounded border border-indigo-200">
                Agent 2: Document Grounding & Schema Extraction
              </span>
              <h3 className="font-serif text-xl font-bold text-stone-900 mt-1">
                Step 2: Parse PDF Layout & Extract Structured Schemas
              </h3>
              <p className="text-xs text-stone-600 mt-0.5">
                Utilizes PyMuPDF to parse two-column academic layouts and extracts Title, Core Problem, Proposed Methodology, Quantitative Evaluation, and Limitations.
              </p>
            </div>

            <div className="bg-stone-50 p-4 rounded-xl border border-stone-200">
              <span className="text-xs font-mono text-stone-500 block mb-2">
                Available Papers in Corpus ({papers.length}):
              </span>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {papers.map(p => (
                  <div key={p.paper_id} className="p-2 bg-white rounded-lg border border-stone-200 text-xs">
                    <span className="font-mono text-[10px] text-stone-400 block">{p.paper_id}</span>
                    <span className="font-serif font-bold text-stone-800 line-clamp-1">{p.title}</span>
                    <span className="text-[10px] font-mono text-emerald-700 mt-1 block">
                      {p.is_extracted ? "✓ Extracted" : "Pending Extraction"}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="flex justify-end gap-3">
              <button
                onClick={() => setCurrentStep(1)}
                className="text-xs font-medium text-stone-600 hover:text-stone-900 px-4 py-2"
              >
                &larr; Back to Step 1
              </button>

              <button
                onClick={handleRunExtraction}
                disabled={extracting || papers.length === 0}
                className="bg-indigo-700 hover:bg-indigo-800 text-white font-semibold text-xs px-5 py-2.5 rounded-xl shadow flex items-center gap-2 transition disabled:opacity-50"
              >
                {extracting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Extracting Schemas with PyMuPDF...</span>
                  </>
                ) : (
                  <>
                    <Cpu className="w-4 h-4" />
                    <span>Run Schema Extraction Agent &rarr; Step 3</span>
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* STEP 3: SUMMARIZER AGENT */}
        {currentStep === 3 && (
          <div className="bg-white rounded-2xl border border-stone-200 p-6 shadow-binder space-y-5">
            <div>
              <span className="font-mono text-xs font-semibold text-emerald-800 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                Agent 3: Structured Paper Summarizer
              </span>
              <h3 className="font-serif text-xl font-bold text-stone-900 mt-1">
                Step 3: Generate 4-Part Structured Paper Summaries
              </h3>
              <p className="text-xs text-stone-600 mt-0.5">
                Generates modular summaries covering: 1. Core Problem & Motivation, 2. Proposed Architecture, 3. Quantitative Evaluation Benchmarks, and 4. Critical Failure Modes.
              </p>
            </div>

            <div className="flex justify-end gap-3">
              <button
                onClick={() => setCurrentStep(2)}
                className="text-xs font-medium text-stone-600 hover:text-stone-900 px-4 py-2"
              >
                &larr; Back to Step 2
              </button>

              <button
                onClick={handleRunSummarizer}
                disabled={summarizing}
                className="bg-emerald-700 hover:bg-emerald-800 text-white font-semibold text-xs px-5 py-2.5 rounded-xl shadow flex items-center gap-2 transition disabled:opacity-50"
              >
                {summarizing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Summarizing Papers...</span>
                  </>
                ) : (
                  <>
                    <FileText className="w-4 h-4" />
                    <span>Run Summarizer Agent &rarr; Step 4</span>
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* STEP 4: COMPARATIVE ANALYSIS AGENT */}
        {currentStep === 4 && (
          <div className="bg-white rounded-2xl border border-stone-200 p-6 shadow-binder space-y-5">
            <div>
              <span className="font-mono text-xs font-semibold text-amber-800 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
                Agent 4: Cross-Paper Comparative Analysis
              </span>
              <h3 className="font-serif text-xl font-bold text-stone-900 mt-1">
                Step 4: Build Cross-Paper Comparative Matrix
              </h3>
              <p className="text-xs text-stone-600 mt-0.5">
                Maps architectural paradigms, action spaces, reflection mechanisms, and benchmark environments into a deterministic Pydantic matrix.
              </p>
            </div>

            <div className="flex justify-end gap-3">
              <button
                onClick={() => setCurrentStep(3)}
                className="text-xs font-medium text-stone-600 hover:text-stone-900 px-4 py-2"
              >
                &larr; Back to Step 3
              </button>

              <button
                onClick={handleRunComparative}
                disabled={comparing}
                className="bg-amber-700 hover:bg-amber-800 text-white font-semibold text-xs px-5 py-2.5 rounded-xl shadow flex items-center gap-2 transition disabled:opacity-50"
              >
                {comparing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Constructing Comparative Matrix...</span>
                  </>
                ) : (
                  <>
                    <Table2 className="w-4 h-4" />
                    <span>Run Comparative Matrix Agent &rarr; Step 5</span>
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* STEP 5: RESEARCH GAP AGENT */}
        {currentStep === 5 && (
          <div className="bg-white rounded-2xl border border-stone-200 p-6 shadow-binder space-y-5">
            <div>
              <span className="font-mono text-xs font-semibold text-rose-800 bg-rose-50 px-2 py-0.5 rounded border border-rose-200">
                Agent 5: Research Gap Identification
              </span>
              <h3 className="font-serif text-xl font-bold text-stone-900 mt-1">
                Step 5: Synthesize Systemic Bottlenecks & Research Gaps
              </h3>
              <p className="text-xs text-stone-600 mt-0.5">
                Analyzes author-explicit disclosures across papers to extract cross-cutting limitations (quadratic token expansion, multi-agent hallucination drift, benchmark saturation).
              </p>
            </div>

            <div className="flex justify-end gap-3">
              <button
                onClick={() => setCurrentStep(4)}
                className="text-xs font-medium text-stone-600 hover:text-stone-900 px-4 py-2"
              >
                &larr; Back to Step 4
              </button>

              <button
                onClick={handleRunResearchGap}
                disabled={gapping}
                className="bg-rose-700 hover:bg-rose-800 text-white font-semibold text-xs px-5 py-2.5 rounded-xl shadow flex items-center gap-2 transition disabled:opacity-50"
              >
                {gapping ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Synthesizing Research Gaps...</span>
                  </>
                ) : (
                  <>
                    <AlertCircle className="w-4 h-4" />
                    <span>Run Research Gap Agent &rarr; Step 6</span>
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* STEP 6: REPORT GENERATOR AGENT */}
        {currentStep === 6 && (
          <div className="bg-white rounded-2xl border border-stone-200 p-6 shadow-binder space-y-5">
            <div>
              <span className="font-mono text-xs font-semibold text-stone-800 bg-stone-100 px-2 py-0.5 rounded border border-stone-200">
                Agent 6: Final Synthesis & Visualization
              </span>
              <h3 className="font-serif text-xl font-bold text-stone-900 mt-1">
                Step 6: Compile Literature Review Monograph & Mermaid Visuals
              </h3>
              <p className="text-xs text-stone-600 mt-0.5">
                Executes the final synthesis pass via LLMManager to generate the full academic monograph with embedded, verified Mermaid.js mindmaps and flowcharts.
              </p>
            </div>

            <div className="flex justify-end gap-3">
              <button
                onClick={() => setCurrentStep(5)}
                className="text-xs font-medium text-stone-600 hover:text-stone-900 px-4 py-2"
              >
                &larr; Back to Step 5
              </button>

              <button
                onClick={handleRunReport}
                disabled={reporting}
                className="bg-stone-900 hover:bg-stone-800 text-stone-100 font-semibold text-xs px-6 py-2.5 rounded-xl shadow flex items-center gap-2 transition disabled:opacity-50"
              >
                {reporting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin text-amber-300" />
                    <span>Compiling Monograph & Structural Diagrams...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 text-amber-300" />
                    <span>Compile Final Literature Review & Open Binder Desk &rarr;</span>
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Live Lab Field Log Terminal */}
        <div className="bg-[#18191B] rounded-2xl p-4 text-[#E2E8F0] font-mono text-xs leading-relaxed border border-stone-800 shadow-lifted">
          <div className="flex items-center justify-between pb-2 mb-2 border-b border-stone-800 text-[10px] text-stone-400 uppercase tracking-widest">
            <span className="flex items-center gap-1.5 text-emerald-400">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              Live Multi-Agent Pipeline Activity Ticker
            </span>
            <span>DOT-MATRIX LOG</span>
          </div>

          <div className="space-y-1 max-h-36 overflow-y-auto">
            {fieldLogs.map((log, idx) => (
              <div key={idx} className="text-stone-300 font-mono text-[11px]">
                {log}
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
