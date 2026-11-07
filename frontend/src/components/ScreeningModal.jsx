import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  X, 
  Sparkles, 
  Search, 
  Download, 
  Clock, 
  CheckCircle2, 
  FileText,
  AlertCircle,
  Loader2
} from 'lucide-react';

export default function ScreeningModal({ isOpen, onClose, onRefreshCorpus }) {
  const [topic, setTopic] = useState("Agentic AI with Multi-Agent Systems and Tool Use");
  const [recencyWeight, setRecencyWeight] = useState(0.20);
  const [maxCandidates, setMaxCandidates] = useState(25);
  const [loading, setLoading] = useState(false);
  const [candidates, setCandidates] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);
  const [downloading, setDownloading] = useState(false);
  const [downloadSuccess, setDownloadSuccess] = useState(null);

  if (!isOpen) return null;

  const handleScreen = async () => {
    if (!topic.trim()) return;
    setLoading(true);
    setCandidates([]);
    setDownloadSuccess(null);

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
      // Pre-select top 6
      setSelectedIds(list.slice(0, 6).map(c => c.arxiv_id));
    } catch (err) {
      console.error('Screening error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    if (selectedIds.length === 0) return;
    setDownloading(true);
    setDownloadSuccess(null);

    try {
      const papersToDownload = candidates.filter(c => selectedIds.includes(c.arxiv_id));
      const res = await fetch('/api/download-papers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ papers: papersToDownload })
      });
      const data = await res.json();
      setDownloadSuccess(`Successfully downloaded ${data.count || papersToDownload.length} paper(s) to data/pdf_corpus!`);
      if (onRefreshCorpus) onRefreshCorpus();
    } catch (err) {
      console.error('Download error:', err);
    } finally {
      setDownloading(false);
    }
  };

  const toggleSelect = (id) => {
    setSelectedIds(prev => 
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    );
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-stone-900/60 backdrop-blur-sm">
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 10 }}
        className="bg-white rounded-2xl border border-stone-300 shadow-2xl w-full max-w-3xl max-h-[85vh] flex flex-col overflow-hidden font-sans"
      >
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-stone-200 bg-[#FAF9F6] flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-amber-100 border border-amber-300 flex items-center justify-center text-amber-800">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-serif font-bold text-base text-stone-900">
                Autonomous Relevance Screening Agent
              </h3>
              <p className="text-xs text-stone-500">
                arXiv API Candidate Ingestion &bull; TF-IDF Vector Cosine &bull; Recency Bias
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 hover:bg-stone-200 rounded-lg text-stone-500 hover:text-stone-800 transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-4 flex-1">
          
          {/* Topic Input */}
          <div>
            <label className="block font-serif font-bold text-xs text-stone-800 mb-1.5">
              Research Project Title / Query
            </label>
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g. Multi-Modal Vision-Language Agents for Autonomous Web Navigation"
              className="w-full bg-[#FAF9F6] border border-stone-300 rounded-xl px-4 py-2.5 text-xs text-stone-900 focus:outline-none focus:ring-2 focus:ring-amber-500 font-sans shadow-inner"
            />
          </div>

          {/* Sliders Grid */}
          <div className="grid grid-cols-2 gap-4 bg-stone-50 p-4 rounded-xl border border-stone-200">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="font-medium text-stone-700">Recency Preference Bonus:</span>
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
              <span className="text-[10px] text-stone-400 block mt-0.5">
                Boosts newer 2024-2026 preprints
              </span>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="font-medium text-stone-700">Candidates to Screen:</span>
                <span className="font-mono font-bold text-stone-800">{maxCandidates}</span>
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
              <span className="text-[10px] text-stone-400 block mt-0.5">
                Initial arXiv candidate pool
              </span>
            </div>
          </div>

          {/* Trigger Button */}
          <div className="flex justify-end">
            <button
              onClick={handleScreen}
              disabled={loading || !topic.trim()}
              className="bg-stone-900 hover:bg-stone-800 text-stone-100 font-medium text-xs px-4 py-2.5 rounded-xl shadow flex items-center gap-2 transition disabled:opacity-50"
            >
              {loading ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-amber-300" />
                  <span>Querying arXiv & Screening...</span>
                </>
              ) : (
                <>
                  <Search className="w-3.5 h-3.5 text-amber-300" />
                  <span>Run Relevance Screening Agent</span>
                </>
              )}
            </button>
          </div>

          {/* Download Success Notice */}
          {downloadSuccess && (
            <div className="bg-emerald-50 border border-emerald-300 text-emerald-800 p-3 rounded-xl text-xs flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
              <span>{downloadSuccess}</span>
            </div>
          )}

          {/* Results Table */}
          {candidates.length > 0 && (
            <div className="space-y-3 pt-2">
              <div className="flex items-center justify-between">
                <span className="font-serif font-bold text-xs text-stone-800">
                  Screened Candidates ({candidates.length} Papers Ranked by Relevance)
                </span>
                <span className="text-[11px] font-mono text-stone-500">
                  {selectedIds.length} Selected for Download
                </span>
              </div>

              <div className="border border-stone-200 rounded-xl overflow-hidden max-h-60 overflow-y-auto divide-y divide-stone-200 bg-white shadow-inner">
                {candidates.map((c) => {
                  const isChecked = selectedIds.includes(c.arxiv_id);
                  return (
                    <div
                      key={c.arxiv_id}
                      onClick={() => toggleSelect(c.arxiv_id)}
                      className={`p-3 text-xs flex items-start gap-3 cursor-pointer transition ${
                        isChecked ? 'bg-amber-50/50' : 'hover:bg-stone-50'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={isChecked}
                        onChange={() => {}} // handled by parent div
                        className="mt-0.5 accent-amber-600"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-mono text-[10px] font-bold text-amber-900 bg-amber-100 px-1.5 py-0.5 rounded">
                            {c.relevance_score}% Match
                          </span>
                          <span className="font-mono text-[10px] text-stone-400">
                            Year: {c.published_year} &bull; arXiv: {c.arxiv_id}
                          </span>
                        </div>
                        <h5 className="font-serif font-bold text-xs text-stone-900 line-clamp-1">
                          {c.title}
                        </h5>
                        <p className="text-[11px] text-stone-500 line-clamp-2 mt-0.5">
                          {c.abstract}
                        </p>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Download Action */}
              <div className="flex items-center justify-between pt-2">
                <span className="text-xs text-stone-500 font-mono">
                  Target directory: <code>data/pdf_corpus/</code>
                </span>
                <button
                  onClick={handleDownload}
                  disabled={downloading || selectedIds.length === 0}
                  className="bg-amber-700 hover:bg-amber-800 text-white font-medium text-xs px-4 py-2 rounded-xl shadow flex items-center gap-1.5 transition disabled:opacity-50"
                >
                  {downloading ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      <span>Downloading PDFs...</span>
                    </>
                  ) : (
                    <>
                      <Download className="w-3.5 h-3.5" />
                      <span>Download Selected ({selectedIds.length})</span>
                    </>
                  )}
                </button>
              </div>

            </div>
          )}

        </div>

      </motion.div>
    </div>
  );
}
