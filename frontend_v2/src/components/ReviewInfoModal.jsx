import { useEffect, useRef, useState } from 'react'
import { X } from 'lucide-react'
import MermaidDiagrams from './MermaidDiagrams'
import { extractionStats, citationStats } from './reviewStats'
import styles from './ReviewInfoModal.module.css'

const PIPELINE = `flowchart LR
 A1["Agent 1<br/>Ingestion<br/>arXiv / local PDF"] --> A2["Agent 2<br/>GROBID / PyMuPDF<br/>NVIDIA NIM"]
 A2 --> A3["Agent 3<br/>Summarizer<br/>0 LLM calls"]
 A3 --> A4["Agent 4<br/>Comparative<br/>0 LLM calls"]
 A4 --> A5["Agent 5<br/>Research Gaps<br/>1 LLM call"]
 A5 --> A6["Agent 6<br/>Monograph<br/>1 unified call"]
 classDef default fill:ivory,stroke:black,color:black;
 `

const AGENTS = [
  ['1 · Ingestion', 'Screen arXiv results or receive local PDFs', 'PDF + parsed document JSON', 'Local PDF bypasses arXiv search'],
  ['2 · Extraction', 'Parse structure and extract methods, results and limitations', 'PaperSchema', 'GROBID → PyMuPDF; NIM → Gemini → heuristics'],
  ['3 · Summarizer', 'Compile structured single-paper summaries; 0 LLM calls', 'ComprehensivePaperSummary', 'Deterministic compilation'],
  ['4 · Comparative', 'Align methods, benchmarks and limitations; 0 LLM calls', 'ComparativeAnalysis', 'Deterministic compilation'],
  ['5 · Research gaps', 'Separate explicit limitations and synthesized gaps; 1 request', 'ResearchGapAnalysis', 'NIM → Gemini → heuristics'],
  ['6 · Monograph', 'Synthesize one survey and insert deterministic diagrams', 'LiteratureReviewReport', 'NIM → Gemini → evidence inventory'],
]
const RUBRIC = [
  ['Coverage', '1: major omissions · 3: most papers · 5: every paper, method and benchmark represented.'],
  ['Logical Structure', '1: disconnected lists · 3: readable sequence · 5: coherent scope, taxonomy, evidence, gaps and roadmap.'],
  ['Synthesis Quality', '1: abstract repetition · 3: useful comparisons · 5: evidence-based trade-offs, conflicts and testable future work.'],
  ['Hallucination Rate', '1: unsupported claims · 3: unresolved evidence · 5: audited claims and citations. ID validity alone cannot measure factual hallucination.'],
]

export default function ReviewInfoModal({ kind, onClose, corpus, content }) {
  const dialog = useRef(null)
  const stats = extractionStats(corpus)
  const citations = citationStats(content, corpus)
  const architecture = kind === 'architecture'

  // Fetch live evaluation report from backend
  const [evalData, setEvalData] = useState(null)
  const [evalLoading, setEvalLoading] = useState(false)
  useEffect(() => {
    if (kind !== 'evaluation') return
    setEvalLoading(true)
    fetch('/api/evaluation-report')
      .then(r => r.json())
      .then(d => { setEvalData(d); setEvalLoading(false) })
      .catch(() => setEvalLoading(false))
  }, [kind])
  useEffect(() => {
    const previous = document.activeElement
    const el = dialog.current
    el.showModal()
    return () => { el.close(); previous?.focus() }
  }, [])
  return (
    <dialog ref={dialog} className={styles.dialog} aria-labelledby="review-info-title"
      onKeyDown={e => {
        if (e.key !== 'Tab') return
        const items = [...dialog.current.querySelectorAll('button, a[href], input, select, textarea, [tabindex="0"]')].filter(el => !el.disabled)
        const first = items[0], last = items[items.length - 1]
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last?.focus() }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first?.focus() }
      }}
      onCancel={e => { e.preventDefault(); onClose() }}
      onClick={e => { if (e.target === dialog.current) { const r = dialog.current.getBoundingClientRect(); if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) onClose() } }}>
      <div className={styles.heading}>
        <div><p className={styles.eyebrow}>NLP-05 · DELIVERABLE {architecture ? '1' : '5'}</p>
          <h2 id="review-info-title">{architecture ? 'System Architecture' : 'Evaluation Rubric'}</h2></div>
        <button autoFocus onClick={onClose} aria-label="Close dialog"><X size={20} /></button>
      </div>
      {architecture ? <>        <p>Six specialized agents communicate through structured JSON and Pydantic models. NVIDIA NIM uses <code>nvidia/nemotron-3.5-bolt-e4b</code>; Gemini 2.5 Flash is the secondary provider.</p>
        <MermaidDiagrams diagrams={{ 'Six-agent pipeline': PIPELINE }} />
        <div className={styles.tableWrap}><table><thead><tr><th>Agent</th><th>Responsibility</th><th>Communication schema</th><th>Fallback</th></tr></thead>
          <tbody>{AGENTS.map(row => <tr key={row[0]}>{row.map(cell => <td key={cell}>{cell}</td>)}</tr>)}</tbody></table></div>
        <p>Call counts describe logical requests per stage; provider retries and fallback attempts can add API calls. Typed validation checks structure, not scientific truth.</p>
      </> : <>
        {/* ── LIVE BENCHMARK SCOREBOARD ── */}
        {evalLoading && <p style={{ fontFamily: 'var(--font-mono)', fontSize: 11, opacity: 0.5 }}>Loading evaluation scores…</p>}
        {evalData?.available && (
          <div className={styles.stats} style={{ marginBottom: '1rem' }}>
            <div>
              <strong>{evalData.section_coverage_pct}%</strong>
              <span>Section Coverage ({evalData.sections_found?.length}/{evalData.sections_required?.length} required sections)</span>
            </div>
            <div>
              <strong>{evalData.citation_integrity_pct}%</strong>
              <span>Citation Integrity ({evalData.valid_citations}/{evalData.total_citations} grounded citations)</span>
            </div>
            <div>
              <strong>{evalData.word_count?.toLocaleString()}</strong>
              <span>Words in report · {evalData.report_chars?.toLocaleString()} chars</span>
            </div>
            <div>
              <strong>{evalData.diagram_count}</strong>
              <span>Mermaid diagrams embedded · {evalData.paper_count} corpus papers</span>
            </div>
          </div>
        )}
        {evalData?.available === false && (
          <p style={{ fontFamily: 'var(--font-mono)', fontSize: 11, opacity: 0.6 }}>⚠ {evalData.message}</p>
        )}
        <div className={styles.stats}>
          <div><strong>{stats.total}</strong><span>Active papers</span></div>
          <div><strong>{stats.percent === null ? 'N/A' : `${stats.percent}%`}</strong><span>Genuine LLM extractions ({stats.llm}/{stats.total})</span></div>
          <div><strong>{stats.heuristic}</strong><span>Heuristic extractions · {stats.unknown} unknown</span></div>
          <div><strong>{citations.percent === null ? 'N/A' : `${citations.percent}%`}</strong><span>Citation-ID provenance · {citations.valid}/{citations.total} tags in active corpus</span></div>
        </div>
        <p>Grounded Provenance Score above measures citation-ID membership only. Claim-level embedding verification and the factual hallucination rate are not measured by this statistic.</p>
        <div className={styles.cards}>{RUBRIC.map(([title, description]) => <section key={title}><h3>{title}</h3><p>{description}</p></section>)}</div>
        <h3>LLM-as-a-Judge readiness</h3>
        <ul><li>{content?.trim() ? 'Report available' : 'Report missing'}; {stats.total ? 'active corpus available' : 'active corpus missing'}.</li>
          <li>Require all six sections, traceable citations, quantitative evidence and three actionable directions.</li>
          <li>Run <code>python eval_benchmark.py --judge</code> with a configured NVIDIA or Gemini key. Scores are advisory (1–5); human source review remains required.</li></ul>
        <p>Offline evaluation: <code>python eval_benchmark.py</code>. Results are saved to <code>data/reports/evaluation_benchmark_report.md</code>.</p>
      </>}
    </dialog>
  )
}
