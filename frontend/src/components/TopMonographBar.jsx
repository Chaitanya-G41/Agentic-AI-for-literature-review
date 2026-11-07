import React, { useState } from 'react';
import { 
  BookOpen, 
  Sparkles, 
  FileDown, 
  Cpu, 
  ChevronDown,
  Workflow,
  RotateCcw
} from 'lucide-react';

export default function TopMonographBar({ 
  status, 
  viewMode,
  setViewMode,
  paradigmFilter, 
  setParadigmFilter, 
  onOpenScreening,
  onResetCorpus,
  onExport,
  totalPapers
}) {
  const [exportOpen, setExportOpen] = useState(false);

  const paradigms = [
    { id: 'all', label: `All (${totalPapers})`, color: 'bg-stone-100 text-stone-700 border-stone-300' },
    { id: 'reasoning', label: '🪻 Reasoning (4)', color: 'bg-indigo-50 text-indigo-700 border-indigo-200' },
    { id: 'tools', label: '🍵 Tools (2)', color: 'bg-emerald-50 text-emerald-700 border-emerald-200' },
    { id: 'multi-agent', label: '🍑 Multi-Agent (3)', color: 'bg-amber-50 text-amber-800 border-amber-200' },
    { id: 'benchmarks', label: '🪸 Benchmarks (2)', color: 'bg-rose-50 text-rose-700 border-rose-200' },
  ];

  return (
    <header className="border-b border-stone-200 bg-[#FAF9F6] sticky top-0 z-30 px-6 py-3 transition-all w-full">
      <div className="flex items-center justify-between gap-4 w-full">
        
        {/* Left: Branding & Metaphor */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-amber-100 to-amber-200 border border-stone-300 shadow-sm flex items-center justify-center text-stone-800 font-serif font-bold text-lg">
            §
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-serif text-base font-bold tracking-tight text-stone-900">
                The Scholar's Digital Binder
              </h1>
              <span className="text-[10px] font-mono bg-stone-200/80 text-stone-700 px-2 py-0.5 rounded-full border border-stone-300">
                NLP-05
              </span>
            </div>
            <p className="text-[11px] text-stone-500 font-sans">
              Autonomous Multi-Agent Literature Review System &bull; RVCE CSE
            </p>
          </div>
        </div>

        {/* Center: View Switcher (Pipeline Stepper vs Desk View) */}
        <div className="flex items-center gap-2">
          <div className="flex items-center bg-stone-200/60 p-1 rounded-xl border border-stone-300/80 text-xs shadow-inner">
            <button
              onClick={() => setViewMode('pipeline')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-semibold transition-all ${
                viewMode === 'pipeline'
                  ? 'bg-white text-stone-900 shadow-sm'
                  : 'text-stone-600 hover:text-stone-900'
              }`}
            >
              <Workflow className="w-3.5 h-3.5 text-amber-600" />
              <span>1. Agent Pipeline Runner</span>
            </button>

            <button
              onClick={() => setViewMode('desk')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-semibold transition-all ${
                viewMode === 'desk'
                  ? 'bg-white text-stone-900 shadow-sm'
                  : 'text-stone-600 hover:text-stone-900'
              }`}
            >
              <BookOpen className="w-3.5 h-3.5 text-indigo-600" />
              <span>2. 3-Pane Research Desk</span>
            </button>
          </div>

          {/* Paradigm Filter Bookmark Tabs (When in Desk Mode) */}
          {viewMode === 'desk' && (
            <div className="hidden lg:flex items-center gap-1 bg-stone-100 p-1 rounded-xl border border-stone-200">
              {paradigms.map(p => (
                <button
                  key={p.id}
                  onClick={() => setParadigmFilter(p.id)}
                  className={`text-[11px] font-medium px-2.5 py-1 rounded-lg border transition-all ${
                    paradigmFilter === p.id
                      ? 'bg-white shadow-sm border-stone-300 text-stone-900 font-semibold'
                      : `${p.color} border-transparent opacity-80 hover:opacity-100`
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Right: Actions, Key Status & Export */}
        <div className="flex items-center gap-2">
          
          {/* Start Fresh Topic */}
          <button
            onClick={onResetCorpus}
            className="flex items-center gap-1.5 text-xs font-medium text-stone-600 hover:text-stone-900 bg-stone-100 hover:bg-stone-200 border border-stone-300 px-3 py-1.5 rounded-lg transition"
            title="Start fresh with a new research topic"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Fresh Topic</span>
          </button>

          {/* Screening Trigger */}
          <button
            onClick={onOpenScreening}
            className="flex items-center gap-1.5 text-xs font-semibold bg-amber-50 hover:bg-amber-100 text-amber-900 border border-amber-300 px-3 py-1.5 rounded-lg shadow-sm transition"
          >
            <Sparkles className="w-3.5 h-3.5 text-amber-600" />
            <span>Screen arXiv</span>
          </button>

          {/* Model Pill */}
          <div className="hidden sm:flex items-center gap-1.5 text-[11px] font-mono px-2 py-1 rounded-lg bg-stone-100 border border-stone-200 text-stone-600">
            <Cpu className="w-3 h-3 text-emerald-600 animate-pulse" />
            <span>NVIDIA NIM</span>
          </div>

          {/* Export as Academic Dossier */}
          <div className="relative">
            <button
              onClick={() => setExportOpen(!exportOpen)}
              className="flex items-center gap-1.5 text-xs font-semibold bg-stone-900 hover:bg-stone-800 text-stone-100 px-3 py-1.5 rounded-lg shadow transition"
            >
              <FileDown className="w-3.5 h-3.5 text-amber-300" />
              <span>Export</span>
              <ChevronDown className="w-3 h-3 text-stone-400" />
            </button>

            {exportOpen && (
              <div 
                className="absolute right-0 mt-1.5 w-48 bg-white rounded-xl shadow-xl border border-stone-200 py-1.5 z-50 text-xs font-sans"
                onMouseLeave={() => setExportOpen(false)}
              >
                <div className="px-3 py-1 border-b border-stone-100 font-serif font-bold text-stone-700">
                  Export Literature Review
                </div>
                <button
                  onClick={() => { onExport('markdown'); setExportOpen(false); }}
                  className="w-full text-left px-3 py-1.5 hover:bg-stone-50 flex items-center justify-between text-stone-800"
                >
                  <span>Markdown (.md)</span>
                  <span className="text-[10px] font-mono text-stone-400">Full</span>
                </button>
                <button
                  onClick={() => { onExport('latex'); setExportOpen(false); }}
                  className="w-full text-left px-3 py-1.5 hover:bg-stone-50 flex items-center justify-between text-stone-800"
                >
                  <span>LaTeX (.tex)</span>
                  <span className="text-[10px] font-mono text-stone-400">IEEE/ACM</span>
                </button>
                <button
                  onClick={() => { onExport('print'); setExportOpen(false); }}
                  className="w-full text-left px-3 py-1.5 hover:bg-stone-50 flex items-center justify-between text-stone-800"
                >
                  <span>Print Monograph</span>
                  <span className="text-[10px] font-mono text-stone-400">PDF</span>
                </button>
              </div>
            )}
          </div>

        </div>

      </div>
    </header>
  );
}
