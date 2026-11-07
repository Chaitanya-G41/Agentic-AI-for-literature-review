import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Folder, 
  Search, 
  FileText, 
  CheckCircle2, 
  Clock, 
  Image as ImageIcon,
  Layers,
  Sparkles,
  ArrowRight
} from 'lucide-react';

export default function FilingCabinet({
  papers,
  selectedPaper,
  onSelectPaper,
  searchQuery,
  setSearchQuery,
  paradigmFilter,
  workspace = false
}) {
  // Filter papers based on search query & paradigm filter
  const filteredPapers = papers.filter(p => {
    const matchesSearch = 
      p.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      p.paper_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (p.authors && p.authors.some(a => a.toLowerCase().includes(searchQuery.toLowerCase())));

    if (paradigmFilter === 'all') return matchesSearch;
    return matchesSearch && p.paradigm.tag === paradigmFilter;
  });

  return (
    <aside className={`${workspace ? 'w-full h-full' : 'w-80 lg:w-96 h-[calc(100vh-61px)] border-r'} flex-shrink-0 border-stone-200 bg-[#F5F2EB]/60 flex flex-col`}>
      
      {/* Filing Cabinet Manila Folder Header */}
      <div className="p-4 border-b border-stone-200 bg-[#F5F2EB] relative">
        <div className="flex items-center justify-between mb-2.5">
          <div className="flex items-center gap-2">
            <span className="text-base font-serif font-bold text-stone-800 flex items-center gap-1.5">
              <Folder className="w-4 h-4 text-amber-700/80 fill-amber-200/50" />
              The Filing Cabinet
            </span>
            <span className="text-[10px] font-mono px-2 py-0.5 bg-stone-200/80 rounded-full text-stone-700">
              {filteredPapers.length} / {papers.length} Slips
            </span>
          </div>
          <span className="text-[11px] font-mono text-stone-500">
            Corpus 2022-2023
          </span>
        </div>

        {/* Search input with slip styling */}
        <div className="relative">
          <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-stone-400" />
          <input
            type="text"
            placeholder="Search papers, authors, arxiv id..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-white/90 border border-stone-300 rounded-lg pl-8 pr-3 py-1.5 text-xs text-stone-800 placeholder-stone-400 focus:outline-none focus:ring-1 focus:ring-amber-500 focus:bg-white transition-all shadow-inner"
          />
        </div>
      </div>

      {/* Laboratory Slips & Paper Cards List */}
      <div className={`flex-1 overflow-y-auto p-4 ${workspace ? 'grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3 content-start max-w-7xl mx-auto w-full' : 'space-y-2.5'}`}>
        {filteredPapers.length === 0 ? (
          <div className="text-center py-12 px-4">
            <Layers className="w-8 h-8 mx-auto text-stone-300 mb-2" />
            <p className="text-xs font-serif text-stone-600">No matching laboratory slips in cabinet.</p>
            <p className="text-[11px] text-stone-400 mt-1">Try adjusting your paradigm filter or search terms.</p>
          </div>
        ) : (
          filteredPapers.map((paper) => {
            const isSelected = selectedPaper?.paper_id === paper.paper_id;
            const pColor = paper.paradigm;

            return (
              <motion.div
                key={paper.paper_id}
                layoutId={`paper-${paper.paper_id}`}
                whileHover={{ y: -2, scale: 1.008 }}
                transition={{ type: "spring", stiffness: 350, damping: 25 }}
                onClick={() => onSelectPaper(paper)}
                className={`cursor-pointer rounded-xl p-3.5 transition-all relative border ${
                  isSelected 
                    ? 'bg-white shadow-lifted border-stone-400 ring-2 ring-stone-900/10' 
                    : 'bg-white/90 hover:bg-white shadow-slip border-stone-200/90'
                }`}
              >
                {/* Manila Sticky Tab Edge on selected card */}
                {isSelected && (
                  <div className="absolute -left-1 top-3 bottom-3 w-1.5 bg-amber-600 rounded-r-md" />
                )}

                {/* Top Metas: Monospace arXiv ID & Paradigm Pill */}
                <div className="flex items-center justify-between gap-2 mb-2">
                  <div className="flex items-center gap-1.5">
                    <span className="font-mono text-[10px] font-semibold text-stone-500 bg-stone-100 px-1.5 py-0.5 rounded border border-stone-200">
                      {paper.paper_id}
                    </span>
                    <span className="text-[10px] font-mono text-stone-400">
                      {paper.size_kb} KB
                    </span>
                  </div>

                  {/* Pastel Paradigm Badge */}
                  <span 
                    className="text-[10px] font-medium px-2 py-0.5 rounded-full border flex items-center gap-1"
                    style={{ 
                      backgroundColor: pColor.bg, 
                      borderColor: pColor.border, 
                      color: pColor.text 
                    }}
                  >
                    {pColor.name}
                  </span>
                </div>

                {/* Paper Title (Editorial Serif feel) */}
                <h3 className="font-serif text-xs font-bold text-stone-900 line-clamp-2 leading-snug mb-1.5">
                  {paper.title}
                </h3>

                {/* Author Monogram Stamp */}
                {paper.authors && paper.authors.length > 0 && (
                  <p className="text-[11px] text-stone-500 font-sans line-clamp-1 mb-2">
                    {paper.authors.join(', ')}
                  </p>
                )}

                {/* Extracted Figures Miniature Strip */}
                {paper.figures && paper.figures.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-stone-100 flex items-center gap-1.5 overflow-hidden">
                    <span className="text-[9px] font-mono text-stone-400 flex items-center gap-0.5">
                      <ImageIcon className="w-2.5 h-2.5" />
                      {paper.figures.length} Figs:
                    </span>
                    <div className="flex items-center gap-1">
                      {paper.figures.slice(0, 3).map((figUrl, i) => (
                        <img
                          key={i}
                          src={figUrl}
                          alt="figure thumbnail"
                          className="w-7 h-5 object-cover rounded border border-stone-200 bg-stone-50"
                        />
                      ))}
                    </div>
                  </div>
                )}

                {/* Status Badges: Extracted, Summarized, Grounded */}
                <div className="mt-2.5 flex items-center justify-between text-[10px] font-mono pt-1 text-stone-400">
                  <div className="flex items-center gap-2">
                    {paper.is_extracted ? (
                      <span className="text-emerald-700 flex items-center gap-0.5">
                        <CheckCircle2 className="w-3 h-3 text-emerald-600" /> Schema
                      </span>
                    ) : (
                      <span className="text-stone-400">Parsed</span>
                    )}

                    {paper.is_summarized && (
                      <span className="text-amber-700 flex items-center gap-0.5">
                        <CheckCircle2 className="w-3 h-3 text-amber-600" /> Summary
                      </span>
                    )}
                  </div>

                  <span className="text-stone-400 group-hover:text-stone-600 flex items-center gap-0.5">
                    View <ArrowRight className="w-2.5 h-2.5" />
                  </span>
                </div>

              </motion.div>
            );
          })
        )}
      </div>

    </aside>
  );
}
