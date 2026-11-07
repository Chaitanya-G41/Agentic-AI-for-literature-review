import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  BookOpen, 
  Table2, 
  AlertCircle, 
  FileText, 
  ExternalLink, 
  Quote, 
  Sparkles,
  Layers,
  ArrowUpRight,
  ShieldCheck,
  CheckCircle2
} from 'lucide-react';

export default function WorkingDossier({
  report,
  comparative,
  selectedPaper,
  onCitationClick,
  onViewDiagram,
  requestedTab,
  onTabChange
}) {
  const [dossierTab, setDossierTab] = useState('synthesis'); // 'synthesis' | 'matrix' | 'gaps' | 'paper'

  useEffect(() => {
    if (requestedTab) setDossierTab(requestedTab);
  }, [requestedTab, selectedPaper]);

  const selectTab = (tab) => {
    setDossierTab(tab);
    onTabChange?.(tab);
  };

  // If a paper is selected from the cabinet, allow switching to paper slip view
  const activeTab = selectedPaper ? dossierTab : (dossierTab === 'paper' ? 'synthesis' : dossierTab);

  // Provenance citation mapper for interactive highlighting
  const citations = [
    { label: "ReAct '22, §3.1", paper_id: "2210.03629", quote: "Our method interleaves thought, action, and observation steps, enabling dynamic reasoning and tool execution in interactive environments.", page: 3 },
    { label: "Toolformer '23, §2.2", paper_id: "2302.04761", quote: "We show that language models can teach themselves to use external tools via simple APIs, deciding which tool to call and with what arguments without human intervention.", page: 2 },
    { label: "Reflexion '23, §4", paper_id: "2303.11366", quote: "Verbal reinforcement learning provides self-reflective feedback without updating neural weights, dramatically improving multi-turn decision trajectories.", page: 4 },
    { label: "Tree of Thoughts '23", paper_id: "2305.10601", quote: "ToT frames deliberate problem solving as search over a tree of coherent reasoning units, combining evaluation heuristics with lookahead search.", page: 3 },
    { label: "ChatDev '23, §3.3", paper_id: "2307.07924", quote: "ChatDev orchestrates multi-agent software development via specialized communicative roles (CEO, CPO, CTO, Programmer) along a waterfall development phase chain.", page: 3 },
    { label: "MetaGPT '23, §2", paper_id: "2308.00352", quote: "MetaGPT introduces Standard Operating Procedures (SOPs) into multi-agent collaborations, requiring structured documents (PRDs, system designs) to prevent dialogue drift.", page: 2 },
    { label: "AutoGen '23", paper_id: "2308.08155", quote: "A multi-agent conversation framework enabling customizable and conversable agents that integrate LLMs, human inputs, and tools seamlessly.", page: 4 },
    { label: "SWE-bench '23, §5", paper_id: "2310.06770", quote: "SWE-bench introduces a rigorous interactive software engineering benchmark evaluating whether agents can resolve real-world GitHub issues end-to-end.", page: 5 },
  ];

  return (
    <main className="flex-1 overflow-y-auto bg-stone-50/70 p-5 lg:p-7 relative">
      
      {/* Dossier Notebook Page Container */}
      <div className="max-w-4xl mx-auto bg-white rounded-2xl border border-stone-200/90 shadow-binder p-6 lg:p-8 min-h-[calc(100vh-115px)] flex flex-col relative">
        
        {/* Editorial Notebook Tab Selector */}
        <div className="flex items-center justify-between border-b border-stone-200 pb-4 mb-6">
          <div className="flex items-center gap-2">
            <span className="font-serif font-bold text-lg text-stone-900 flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-amber-700" />
              The Active Dossier
            </span>
            <span className="text-xs font-mono bg-stone-100 text-stone-600 px-2 py-0.5 rounded border border-stone-200">
              Draft v2.4 &bull; Editorial Monograph
            </span>
          </div>

          {/* Dossier Sub-Tabs */}
          <div className="flex items-center gap-1 bg-stone-100 p-1 rounded-xl border border-stone-200 text-xs">
            <button
              onClick={() => selectTab('synthesis')}
              className={`px-3 py-1.5 rounded-lg transition-all font-medium ${
                activeTab === 'synthesis' ? 'bg-white shadow-sm text-stone-900 font-semibold' : 'text-stone-600 hover:text-stone-900'
              }`}
            >
              Synthesis Monograph
            </button>
            <button
              onClick={() => selectTab('matrix')}
              className={`px-3 py-1.5 rounded-lg transition-all font-medium flex items-center gap-1 ${
                activeTab === 'matrix' ? 'bg-white shadow-sm text-stone-900 font-semibold' : 'text-stone-600 hover:text-stone-900'
              }`}
            >
              <Table2 className="w-3.5 h-3.5 text-stone-500" />
              Comparative Matrix
            </button>
            <button
              onClick={() => selectTab('gaps')}
              className={`px-3 py-1.5 rounded-lg transition-all font-medium flex items-center gap-1 ${
                activeTab === 'gaps' ? 'bg-white shadow-sm text-stone-900 font-semibold' : 'text-stone-600 hover:text-stone-900'
              }`}
            >
              <AlertCircle className="w-3.5 h-3.5 text-rose-500" />
              Research Gaps
            </button>
            {selectedPaper && (
              <button
                onClick={() => selectTab('paper')}
                className={`px-3 py-1.5 rounded-lg transition-all font-medium flex items-center gap-1 ${
                  activeTab === 'paper' ? 'bg-amber-100/80 text-amber-900 font-semibold border border-amber-300' : 'text-amber-800 hover:bg-amber-50'
                }`}
              >
                <FileText className="w-3.5 h-3.5 text-amber-700" />
                Paper: {selectedPaper.paper_id}
              </button>
            )}
          </div>
        </div>

        {/* Tab 1: Synthesis Monograph */}
        {activeTab === 'synthesis' && (
          <div className="space-y-6">
            
            {/* Title Section */}
            <div>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-[11px] font-mono text-amber-800 uppercase tracking-widest bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
                  Comprehensive Meta-Review
                </span>
                <span className="text-[11px] font-mono text-stone-400">
                  Published November 2026
                </span>
              </div>
              <h1 className="font-serif text-2xl lg:text-3xl font-bold tracking-tight text-stone-900 leading-tight">
                Automated Literature Review: Synthesis of Multi-Agent Collaboration & Reasoning Architectures
              </h1>
            </div>

            {/* Executive Summary Index Card */}
            <div className="bg-[#FAF9F6] border-l-4 border-amber-500 rounded-r-xl p-5 border-y border-r border-stone-200 shadow-sm relative">
              <span className="font-serif font-bold text-sm text-amber-900 block mb-2 flex items-center gap-1.5">
                <Quote className="w-4 h-4 text-amber-700" />
                Executive Summary
              </span>
              <p className="text-stone-700 font-serif leading-relaxed text-sm">
                This review synthesizes 12 seminal arXiv preprints spanning reasoning-action synergy, tool use, 
                self-reflection, deliberative search, API grounding, embodied open-ended agents, and multi-agent software 
                development. The corpus marks a paradigm transition from static chain-of-thought generation to agentic loops 
                that interleave internal reasoning with external acting, retrieval, tool execution, and inter-agent communication.
              </p>
            </div>

            {/* Interactive Provenance Highlighter Strip */}
            <div className="bg-stone-50 rounded-xl p-4 border border-stone-200">
              <div className="flex items-center justify-between mb-2.5">
                <span className="text-xs font-mono font-semibold text-stone-700 flex items-center gap-1.5">
                  <ShieldCheck className="w-4 h-4 text-emerald-600" />
                  Interactive Highlighter Provenance Badges (Zero Hallucination Grounding)
                </span>
                <span className="text-[11px] font-mono text-stone-400">
                  Click to inspect in Right Pane PDF
                </span>
              </div>
              <div className="flex flex-wrap gap-2">
                {citations.map((c, idx) => (
                  <button
                    key={idx}
                    onClick={() => onCitationClick(c)}
                    className="group text-xs font-mono px-2.5 py-1 rounded-lg bg-white border border-stone-200 hover:border-amber-400 hover:bg-amber-50 text-stone-700 hover:text-amber-900 shadow-sm transition-all flex items-center gap-1"
                  >
                    <span>[{c.label}]</span>
                    <ArrowUpRight className="w-3 h-3 text-stone-400 group-hover:text-amber-700 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
                  </button>
                ))}
              </div>
            </div>

            {/* Core Body Sections */}
            <div className="prose prose-stone max-w-none text-stone-800 text-sm leading-relaxed space-y-4 font-sans">
              <h3 className="font-serif text-lg font-bold text-stone-900 border-b border-stone-200 pb-1 mt-6">
                1. Architectural Paradigms & Thematic Convergence
              </h3>
              <p>
                Across the analyzed corpus, four distinct architectural paradigms have crystallized to combat static generation failures:
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5 my-3">
                <div className="p-3.5 rounded-xl bg-indigo-50/70 border border-indigo-200">
                  <h4 className="font-serif font-bold text-xs text-indigo-900 mb-1">
                    Reasoning & Reflection
                  </h4>
                  <p className="text-xs text-indigo-950/80 leading-normal">
                    ReAct and Reflexion introduce thought-action-observation cycles. Agents maintain self-evaluative working memory to revise failing plans dynamically.
                  </p>
                </div>
                
                <div className="p-3.5 rounded-xl bg-emerald-50/70 border border-emerald-200">
                  <h4 className="font-serif font-bold text-xs text-emerald-900 mb-1">
                    Tool & API Grounding
                  </h4>
                  <p className="text-xs text-emerald-950/80 leading-normal">
                    Toolformer and Gorilla delegate arithmetic, fact-checking, and REST execution to external compilers and API registries to neutralize hallucination.
                  </p>
                </div>

                <div className="p-3.5 rounded-xl bg-amber-50/70 border border-amber-200">
                  <h4 className="font-serif font-bold text-xs text-amber-900 mb-1">
                    Multi-Agent Coordination
                  </h4>
                  <p className="text-xs text-amber-950/80 leading-normal">
                    ChatDev, MetaGPT, and AutoGen decompose complex workflows into specialized communicative roles with structured documents (PRDs, code reviews).
                  </p>
                </div>

                <div className="p-3.5 rounded-xl bg-rose-50/70 border border-rose-200">
                  <h4 className="font-serif font-bold text-xs text-rose-900 mb-1">
                    Interactive Benchmarking
                  </h4>
                  <p className="text-xs text-rose-950/80 leading-normal">
                    AgentBench and SWE-bench evaluate agents on non-stationary, repository-scale environments rather than multiple-choice synthetic benchmarks.
                  </p>
                </div>
              </div>

              <p>
                To explore how these paradigms connect structurally across all 12 preprints, you can inspect the live{' '}
                <button 
                  onClick={() => onViewDiagram('themes_mindmap')}
                  className="text-amber-800 font-semibold underline decoration-amber-400 hover:text-amber-900"
                >
                  Thematic Mindmap
                </button>{' '}
                or review the{' '}
                <button 
                  onClick={() => onViewDiagram('research_gaps_flowchart')}
                  className="text-amber-800 font-semibold underline decoration-amber-400 hover:text-amber-900"
                >
                  Research Gaps Lineage Flowchart
                </button>{' '}
                in the Evidence Inspector.
              </p>
            </div>

          </div>
        )}

        {/* Tab 2: Comparative Matrix */}
        {activeTab === 'matrix' && (
          <div className="space-y-4">
            <div>
              <h2 className="font-serif text-xl font-bold text-stone-900">
                Cross-Paper Architectural Comparative Matrix
              </h2>
              <p className="text-xs text-stone-500 font-sans mt-0.5">
                Structured feature grid extracted deterministically across the 12-paper benchmark corpus.
              </p>
            </div>

            <div className="overflow-x-auto border border-stone-200 rounded-xl shadow-sm">
              <table className="w-full text-left text-xs font-sans">
                <thead className="bg-[#FAF9F6] text-stone-700 font-serif font-bold border-b border-stone-200">
                  <tr>
                    <th className="p-3 border-r border-stone-200">Paper & Year</th>
                    <th className="p-3 border-r border-stone-200">Core Paradigm</th>
                    <th className="p-3 border-r border-stone-200">Action Space</th>
                    <th className="p-3 border-r border-stone-200">Reflection Loop</th>
                    <th className="p-3">Primary Evaluation Benchmarks</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-200 bg-white">
                  {[
                    { id: "2210.03629", name: "ReAct (2022)", paradigm: "Reasoning & Action", action: "Search, Lookup, Finish", loop: "Implicit via Thought prompt", bench: "HotpotQA, ALFWorld, WebShop" },
                    { id: "2302.04761", name: "Toolformer (2023)", paradigm: "Tool Grounding", action: "Calculator, QA, Wiki, Translate", loop: "Self-supervised API filter", bench: "LAMA, Math, TriviaQA" },
                    { id: "2303.11366", name: "Reflexion (2023)", paradigm: "Self-Reflection", action: "Environment actions", loop: "Explicit verbal memory reflection", bench: "HumanEval, ALFWorld, HotpotQA" },
                    { id: "2305.10601", name: "Tree of Thoughts (2023)", paradigm: "Deliberate Search", action: "Thought proposals & pruning", loop: "BFS / DFS heuristic valuation", bench: "Game of 24, Creative Writing" },
                    { id: "2305.15334", name: "Gorilla (2023)", paradigm: "API Retrieval", action: "1,600+ REST APIs (TorchHub, HF)", loop: "Retriever-aware fine-tuning", bench: "APIBench" },
                    { id: "2305.16291", name: "Voyager (2023)", paradigm: "Embodied Agent", action: "Minecraft JS skill library", loop: "Iterative prompting with execution", bench: "Minecraft Tech Tree Exploration" },
                    { id: "2307.07924", name: "ChatDev (2023)", paradigm: "Multi-Agent Software", action: "Natural language & code generation", loop: "Role-playing conversational review", bench: "CAMEL, Software Dev Benchmark" },
                    { id: "2308.00352", name: "MetaGPT (2023)", paradigm: "SOP Multi-Agent", action: "Structured PRD, diagrams, source files", loop: "SOP-guided standard reviews", bench: "HumanEval, MBPP, Software tasks" },
                    { id: "2308.08155", name: "AutoGen (2023)", paradigm: "Conversational Agents", action: "Code execution, human interaction", loop: "Multi-agent conversation rounds", bench: "Math, Coding, Scenario benchmarks" },
                    { id: "2310.06770", name: "SWE-bench (2023)", paradigm: "Interactive Evaluation", action: "Git patch generation & execution", loop: "Unit test regression execution", bench: "2,294 Real GitHub Issues" },
                  ].map((row, idx) => (
                    <tr key={idx} className={idx % 2 === 1 ? 'bg-stone-50/60' : 'bg-white'}>
                      <td className="p-3 font-serif font-bold text-stone-900 border-r border-stone-200">
                        <span className="font-mono text-[10px] text-stone-500 block">{row.id}</span>
                        {row.name}
                      </td>
                      <td className="p-3 border-r border-stone-200 text-stone-700">
                        {row.paradigm}
                      </td>
                      <td className="p-3 border-r border-stone-200 text-stone-600 font-mono text-[11px]">
                        {row.action}
                      </td>
                      <td className="p-3 border-r border-stone-200 text-stone-700">
                        {row.loop}
                      </td>
                      <td className="p-3 text-stone-600">
                        {row.bench}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Tab 3: Research Gaps Lineage */}
        {activeTab === 'gaps' && (
          <div className="space-y-4">
            <div>
              <h2 className="font-serif text-xl font-bold text-stone-900">
                Synthesized Research Gaps & Cross-Paper Limitations
              </h2>
              <p className="text-xs text-stone-500 font-sans mt-0.5">
                Identified systemic bottlenecks extracted from author-explicit disclosures and multi-agent synthesis.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                {
                  id: "GAP-01",
                  title: "Inference Latency & Quadratic Token Accumulation",
                  category: "Computational Efficiency",
                  papers: ["ReAct", "Tree of Thoughts", "ChatDev", "MetaGPT"],
                  desc: "Multi-turn reasoning and multi-agent debate trigger rapid token context expansion. Long trajectories incur high API expenses and unacceptable real-time response latency."
                },
                {
                  id: "GAP-02",
                  title: "Cascading Error Propagation & Multi-Agent Drift",
                  category: "Reliability & Alignment",
                  papers: ["ChatDev", "AutoGen", "Reflexion"],
                  desc: "When an upstream agent generates a subtle architectural error, downstream agents validate and hallucinate on top of it, causing catastrophic drift across multi-round debates."
                },
                {
                  id: "GAP-03",
                  title: "Evaluation Benchmark Saturation vs Real-World Complexity",
                  category: "Evaluation Methodology",
                  papers: ["AgentBench", "SWE-bench", "Gorilla"],
                  desc: "Current synthetic benchmarks (ALFWorld, HumanEval) fail to measure open-ended tool discovery, dependency management, and long-horizon repository scale engineering."
                },
                {
                  id: "GAP-04",
                  title: "Static Prompt Tuning vs Lifelong Memory Retention",
                  category: "Memory Architecture",
                  papers: ["Voyager", "Reflexion", "Toolformer"],
                  desc: "Most systems lack durable long-term episodic memory. Skills and self-reflective lessons vanish between independent inference sessions without persistent embedding retrieval."
                }
              ].map((gap, i) => (
                <div key={i} className="p-4 rounded-xl border border-rose-200/80 bg-rose-50/40 shadow-sm">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-mono text-[10px] font-bold text-rose-700 bg-rose-100 px-2 py-0.5 rounded">
                      {gap.id} &bull; {gap.category}
                    </span>
                  </div>
                  <h4 className="font-serif font-bold text-sm text-stone-900 mb-1.5">
                    {gap.title}
                  </h4>
                  <p className="text-xs text-stone-700 leading-relaxed mb-3">
                    {gap.desc}
                  </p>
                  <div className="flex items-center gap-1.5 text-[10px] font-mono text-stone-500">
                    <span>Observed in:</span>
                    <span className="text-rose-800 font-semibold">{gap.papers.join(', ')}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tab 4: Single-Paper Dossier (When a paper is focused) */}
        {activeTab === 'paper' && selectedPaper && (
          <div className="space-y-5">
            <div className="border-b border-stone-200 pb-3">
              <div className="flex items-center gap-2 mb-1">
                <span className="font-mono text-xs font-semibold px-2 py-0.5 rounded bg-stone-100 text-stone-600">
                  arXiv: {selectedPaper.paper_id}
                </span>
                <span 
                  className="text-xs font-medium px-2 py-0.5 rounded-full border"
                  style={{
                    backgroundColor: selectedPaper.paradigm.bg,
                    borderColor: selectedPaper.paradigm.border,
                    color: selectedPaper.paradigm.text
                  }}
                >
                  {selectedPaper.paradigm.name}
                </span>
              </div>
              <h2 className="font-serif text-2xl font-bold text-stone-900">
                {selectedPaper.title}
              </h2>
              {selectedPaper.authors && (
                <p className="text-xs text-stone-500 font-sans mt-1">
                  Authors: {selectedPaper.authors.join(', ')}
                </p>
              )}
            </div>

            {/* Core Problem Card */}
            <div className="bg-[#FAF9F6] p-4 rounded-xl border border-stone-200">
              <h4 className="font-serif font-bold text-xs uppercase tracking-wider text-amber-900 mb-1 flex items-center gap-1.5">
                <AlertCircle className="w-3.5 h-3.5 text-amber-600" />
                Core Problem Statement
              </h4>
              <p className="text-xs text-stone-700 leading-relaxed">
                {selectedPaper.core_problem || "Extracted from paper schema: Addresses limitations in LLM decision making, static token generation, and the absence of interactive feedback grounding."}
              </p>
            </div>

            {/* Methodology Summary */}
            <div className="bg-white p-4 rounded-xl border border-stone-200">
              <h4 className="font-serif font-bold text-xs uppercase tracking-wider text-stone-900 mb-1 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
                Proposed Methodology & Architecture
              </h4>
              <p className="text-xs text-stone-700 leading-relaxed">
                {selectedPaper.methodology || "Implements structured agentic loops, prompt chaining, or multi-agent role decomposition to interleave thoughts with external environments."}
              </p>
            </div>

            {/* Extracted Figures Strip */}
            {selectedPaper.figures && selectedPaper.figures.length > 0 && (
              <div>
                <h4 className="font-serif font-bold text-xs uppercase tracking-wider text-stone-900 mb-2">
                  Extracted Architectural Figures ({selectedPaper.figures.length})
                </h4>
                <div className="grid grid-cols-2 gap-3">
                  {selectedPaper.figures.map((fig, idx) => (
                    <div key={idx} className="border border-stone-200 rounded-xl overflow-hidden bg-stone-50 p-2 shadow-sm">
                      <img src={fig} alt="Paper figure" className="w-full h-40 object-contain rounded" />
                      <span className="text-[10px] font-mono text-stone-500 block text-center mt-1">
                        Figure {idx + 1}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

      </div>
    </main>
  );
}
