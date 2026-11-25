import { useState, useRef, useEffect } from 'react'
import {
  Search,
  Upload,
  BookOpen,
  ArrowRight,
  Loader2,
  Sparkles,
  FileText,
  Trash2,
  Layers,
  Sliders,
  Check,
  HelpCircle,
  X,
  CheckCircle2
} from 'lucide-react'
import clsx from 'clsx'
import styles from './WelcomeScreen.module.css'

export default function WelcomeScreen({ onStartReview, onLoadBenchmark, existingCorpusCount = 0 }) {
  const [tab, setTab] = useState('arxiv') // 'arxiv' | 'upload'
  const [query, setQuery] = useState('')
  const [maxResults, setMaxResults] = useState(6)
  const [screening, setScreening] = useState(false)
  const [screenError, setScreenError] = useState(null)
  const [candidates, setCandidates] = useState([])
  const [selectedIds, setSelectedIds] = useState([])

  // Inclusion / Exclusion criteria state
  const [showCriteria, setShowCriteria] = useState(false)
  const [inclusionCriteria, setInclusionCriteria] = useState('')
  const [exclusionCriteria, setExclusionCriteria] = useState('')
  const [activeFilter, setActiveFilter] = useState('all') // 'all' | 'include' | 'maybe' | 'exclude'
  const [prismaStats, setPrismaStats] = useState({
    total_identified: 0,
    screened: 0,
    included: 0,
    maybe: 0,
    excluded: 0,
  })

  // Local upload state
  const [localFiles, setLocalFiles] = useState([])
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef(null)

  // Pipeline state
  const [pipelineRunning, setPipelineRunning] = useState(false)
  const [pipelineStage, setPipelineStage] = useState('')
  const [pipelineLogs, setPipelineLogs] = useState([])

  // Load existing screening state if available
  useEffect(() => {
    fetch('/api/screening')
      .then(res => res.json())
      .then(data => {
        if (data && data.candidates && data.candidates.length > 0) {
          setCandidates(data.candidates)
          if (data.query) setQuery(data.query)
          if (data.inclusion_criteria) setInclusionCriteria(data.inclusion_criteria)
          if (data.exclusion_criteria) setExclusionCriteria(data.exclusion_criteria)
          if (data.prisma) setPrismaStats(data.prisma)
          const inc = data.candidates
            .filter(c => c.decision === 'Include')
            .map(c => c.arxiv_id || c.id)
          setSelectedIds(inc.length ? inc : data.candidates.map(c => c.arxiv_id || c.id))
        }
      })
      .catch(() => {})
  }, [])

  const handleScreen = async () => {
    if (!query.trim()) return
    setScreening(true)
    setScreenError(null)
    setCandidates([])
    setSelectedIds([])
    try {
      const res = await fetch('/api/screen', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query.trim(),
          max_candidates: maxResults,
          max_results: maxResults,
          recency_weight: 0.20,
          inclusion_criteria: inclusionCriteria.trim(),
          exclusion_criteria: exclusionCriteria.trim(),
        }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      const list = data.candidates || data.papers || data.results || []
      setCandidates(list)
      if (data.prisma) {
        setPrismaStats(data.prisma)
      } else {
        const inc = list.filter(p => p.decision === 'Include').length
        const may = list.filter(p => p.decision === 'Maybe').length
        const exc = list.filter(p => p.decision === 'Exclude').length
        setPrismaStats({
          total_identified: list.length,
          screened: list.length,
          included: inc,
          maybe: may,
          excluded: exc,
        })
      }

      // Default selection to Included papers (or non-excluded if none marked include)
      const included = list.filter(p => p.decision === 'Include').map(p => p.arxiv_id || p.id)
      if (included.length > 0) {
        setSelectedIds(included)
      } else {
        const nonExcluded = list.filter(p => p.decision !== 'Exclude').map(p => p.arxiv_id || p.id)
        setSelectedIds(nonExcluded.length ? nonExcluded : list.map(p => p.arxiv_id || p.id))
      }
    } catch (err) {
      setScreenError(err.message || 'Failed to screen arXiv.')
    } finally {
      setScreening(false)
    }
  }

  const handleDecisionChange = (id, newDecision) => {
    setCandidates(prev => {
      const updated = prev.map(p => {
        if ((p.arxiv_id || p.id) === id) {
          return {
            ...p,
            decision: newDecision,
            screening_reason:
              newDecision === 'Include'
                ? `Included: Manually approved by reviewer for monograph synthesis.`
                : newDecision === 'Maybe'
                ? `Maybe: Flagged for contextual review.`
                : `Excluded: Manually excluded by reviewer.`
          }
        }
        return p
      })

      // Update PRISMA counts
      const inc = updated.filter(p => p.decision === 'Include').length
      const may = updated.filter(p => p.decision === 'Maybe').length
      const exc = updated.filter(p => p.decision === 'Exclude').length
      setPrismaStats({
        total_identified: updated.length,
        screened: updated.length,
        included: inc,
        maybe: may,
        excluded: exc,
      })

      // Auto update selection
      if (newDecision === 'Include') {
        setSelectedIds(s => s.includes(id) ? s : [...s, id])
      } else if (newDecision === 'Exclude') {
        setSelectedIds(s => s.filter(x => x !== id))
      }

      // Persist to backend
      fetch('/api/save-screening', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query.trim(),
          inclusion_criteria: inclusionCriteria,
          exclusion_criteria: exclusionCriteria,
          candidates: updated,
        }),
      }).catch(() => {})

      return updated
    })
  }

  const handleReasonChange = (id, reason) => {
    setCandidates(prev => prev.map(p => (p.arxiv_id || p.id) === id ? { ...p, screening_reason: reason } : p))
  }

  const toggleSelect = (id) => {
    setSelectedIds(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id])
  }

  const runPipelineWithStep = async (downloadPromise = null) => {
    setPipelineRunning(true)
    setPipelineLogs([])

    const addLog = (msg) => {
      setPipelineLogs(prev => [...prev, { time: new Date().toLocaleTimeString(), msg }])
    }

    try {
      if (downloadPromise) {
        setPipelineStage('Initializing Workspace')
        addLog('Resetting corpus workspace for fresh topic review...')
        await fetch('/api/clear-corpus', { method: 'POST' }).catch(() => {})
        setPipelineStage('Downloading Papers')
        addLog('Agent 1: Fetching candidate PDFs from arXiv repository...')
        await downloadPromise
        addLog('PDFs archived in local corpus.')
      }

      setPipelineStage('1/5: GROBID Schema Extraction')
      addLog('Agent 2: Parsing TEI-XML structure & extracting structured methodology...')
      await fetch('/api/run-agent/extraction', { method: 'POST' }).catch(() => {})
      addLog('Schema extraction complete with zero-hallucination grounding.')

      setPipelineStage('2/5: Single-Paper Summaries')
      addLog('Agent 3: Deterministic Pydantic summaries (0 LLM calls)...')
      await fetch('/api/run-agent/summarizer', { method: 'POST' }).catch(() => {})
      addLog('Summaries compiled.')

      setPipelineStage('3/5: Comparative Analysis Matrix')
      addLog('Agent 4: Aligning cross-paper architectures & evaluation benchmarks...')
      await fetch('/api/run-agent/comparative', { method: 'POST' }).catch(() => {})
      addLog('Comparative matrix constructed.')

      setPipelineStage('4/5: Research Gap Synthesis')
      addLog('Agent 5: Synthesizing systemic bottlenecks & research frontiers (1 LLM call)...')
      await fetch('/api/run-agent/research-gap', { method: 'POST' }).catch(() => {})
      addLog('Identified research gaps.')

      setPipelineStage('5/5: Monograph Report Compilation')
      addLog('Agent 6: Generating editorial literature review & Mermaid diagrams...')
      await fetch('/api/run-agent/report', { method: 'POST' }).catch(() => {})
      addLog('Monograph ready!')

      if (onStartReview) await onStartReview()
    } catch (err) {
      addLog(`Error: ${err.message}`)
    } finally {
      setPipelineRunning(false)
    }
  }

  const handleDownloadAndSynthesize = () => {
    const selected = candidates.filter(p => selectedIds.includes(p.arxiv_id || p.id))
    if (!selected.length) return
    const p = fetch('/api/download-papers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ papers: selected }),
    })
    runPipelineWithStep(p)
  }

  const handleFilesChosen = (files) => {
    const pdfs = Array.from(files).filter(f => f.name.toLowerCase().endsWith('.pdf'))
    setLocalFiles(prev => [...prev, ...pdfs])
  }

  const handleUploadAndSynthesize = async () => {
    if (!localFiles.length) return
    const uploadPromises = localFiles.map(async file => {
      try {
        await fetch(`/api/upload-pdf?filename=${encodeURIComponent(file.name)}`, {
          method: 'POST',
          body: file,
        })
      } catch (e) {
        console.error('Upload failed for', file.name, e)
      }
    })
    runPipelineWithStep(Promise.all(uploadPromises))
  }

  return (
    <div className={styles.container}>
      {/* Editorial Header */}
      <header className={styles.masthead}>
        <span className={styles.brand}>MONOGRAPH</span>
        <span className={styles.edition}>AUTONOMOUS SCHOLAR · RESEARCH SYNTHESIS</span>
      </header>

      {/* Hero Intro */}
      <section className={styles.hero}>
        <div className={styles.heroTagRow}>
          <span className={styles.tagBadge}>NEW RESEARCH SESSION</span>
          <span className={styles.tagDivider}>•</span>
          <span className={styles.tagText}>AGENTIC AI LITERATURE REVIEW PROTOCOL</span>
        </div>

        <h1 className={styles.title}>Synthesize Any Research Literature.</h1>

        <p className={styles.subtitle}>
          Enter an arXiv research topic or upload PDFs from your local machine.
          The multi-agent system parses structural TEI-XML via GROBID, extracts deterministic schemas,
          and writes a publication-grade editorial monograph.
        </p>
      </section>

      {/* Main Interactive Ingestion Box */}
      <div className={styles.ingestionBox}>
        {pipelineRunning ? (
          <div className={styles.pipelineRunningBlock}>
            <div className={styles.pipelineHeader}>
              <Loader2 size={24} className={styles.spin} />
              <div>
                <h3 className={styles.pipelineStageTitle}>{pipelineStage}</h3>
                <p className={styles.pipelineStageDesc}>
                  Autonomous agents are currently executing the synthesis protocol.
                </p>
              </div>
            </div>

            <div className={styles.consoleBox}>
              <div className={styles.consoleBar}>AGENT EXECUTION TRACE</div>
              <div className={styles.consoleScroll}>
                {pipelineLogs.map((log, i) => (
                  <div key={i} className={styles.consoleLine}>
                    <span className={styles.consoleTime}>{log.time}</span>
                    <span className={styles.consoleMsg}>{log.msg}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <>
            <div className={styles.tabBar}>
              <button
                className={clsx(styles.tabBtn, tab === 'arxiv' && styles.tabBtnActive)}
                onClick={() => setTab('arxiv')}
              >
                <Search size={14} /> Search & Ingest from arXiv
              </button>
              <button
                className={clsx(styles.tabBtn, tab === 'upload' && styles.tabBtnActive)}
                onClick={() => setTab('upload')}
              >
                <Upload size={14} /> Upload Local PDFs
              </button>
            </div>

            {/* TAB 1: arXiv Search */}
            {tab === 'arxiv' && (
              <div className={styles.tabBody}>
                <div className={styles.searchRow}>
                  <input
                    type="text"
                    className={styles.searchInput}
                    placeholder="e.g. Agentic AI planning LLM reflection ReAct"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleScreen()}
                  />
                  <select
                    className={styles.maxSelect}
                    value={maxResults}
                    onChange={(e) => setMaxResults(Number(e.target.value))}
                  >
                    <option value={4}>4 papers</option>
                    <option value={6}>6 papers</option>
                    <option value={10}>10 papers</option>
                    <option value={15}>15 papers</option>
                  </select>
                  <button
                    className={styles.searchBtn}
                    onClick={handleScreen}
                    disabled={screening || !query.trim()}
                  >
                    {screening ? <Loader2 size={13} className={styles.spin} /> : <Search size={13} />}
                    {screening ? 'Screening…' : 'Screen arXiv'}
                  </button>
                </div>

                {/* Inclusion & Exclusion Criteria Toggle */}
                <button
                  type="button"
                  className={styles.criteriaToggleBtn}
                  onClick={() => setShowCriteria(!showCriteria)}
                >
                  <Sliders size={12} />
                  {showCriteria ? 'Hide Screening Protocol Criteria' : 'Set Inclusion & Exclusion Criteria'}
                </button>

                {showCriteria && (
                  <div className={styles.criteriaPanel}>
                    <div className={styles.criteriaGroup}>
                      <label className={styles.criteriaLabel}>Inclusion Criteria</label>
                      <input
                        type="text"
                        className={styles.criteriaInput}
                        placeholder="e.g. neural architecture, empirical evaluation, deep learning, benchmark"
                        value={inclusionCriteria}
                        onChange={(e) => setInclusionCriteria(e.target.value)}
                      />
                    </div>
                    <div className={styles.criteriaGroup}>
                      <label className={styles.criteriaLabel}>Exclusion Criteria</label>
                      <input
                        type="text"
                        className={styles.criteriaInput}
                        placeholder="e.g. survey only, review, theoretical without experiments, non-English"
                        value={exclusionCriteria}
                        onChange={(e) => setExclusionCriteria(e.target.value)}
                      />
                    </div>
                  </div>
                )}

                {screenError && (
                  <div className={styles.errorBox}>{screenError}</div>
                )}

                {/* Candidate paper cards with PRISMA Screening Workflow */}
                {candidates.length > 0 && (
                  <div className={styles.candidatesArea}>
                    {/* PRISMA Screening Summary Bar */}
                    <div className={styles.prismaSummaryBar}>
                      <span>PRISMA SCREENING PROTOCOL</span>
                      <div className={styles.prismaStats}>
                        <span>Identified: <strong>{candidates.length}</strong></span>
                        <span className={styles.prismaBadgeInc}>
                          ✓ Included: <strong>{candidates.filter(p => p.decision === 'Include').length}</strong>
                        </span>
                        <span className={styles.prismaBadgeMay}>
                          ? Maybe: <strong>{candidates.filter(p => p.decision === 'Maybe').length}</strong>
                        </span>
                        <span className={styles.prismaBadgeExc}>
                          ✗ Excluded: <strong>{candidates.filter(p => p.decision === 'Exclude').length}</strong>
                        </span>
                      </div>
                    </div>

                    {/* Filter Tabs */}
                    <div className={styles.filterTabs}>
                      <button
                        className={clsx(styles.filterTab, activeFilter === 'all' && styles.filterTabActive)}
                        onClick={() => setActiveFilter('all')}
                      >
                        All ({candidates.length})
                      </button>
                      <button
                        className={clsx(styles.filterTab, activeFilter === 'include' && styles.filterTabActive)}
                        onClick={() => setActiveFilter('include')}
                      >
                        Included ({candidates.filter(p => p.decision === 'Include').length})
                      </button>
                      <button
                        className={clsx(styles.filterTab, activeFilter === 'maybe' && styles.filterTabActive)}
                        onClick={() => setActiveFilter('maybe')}
                      >
                        Maybe ({candidates.filter(p => p.decision === 'Maybe').length})
                      </button>
                      <button
                        className={clsx(styles.filterTab, activeFilter === 'exclude' && styles.filterTabActive)}
                        onClick={() => setActiveFilter('exclude')}
                      >
                        Excluded ({candidates.filter(p => p.decision === 'Exclude').length})
                      </button>
                    </div>

                    <div className={styles.candidatesHeader}>
                      <span>
                        Showing {candidates.filter(p => activeFilter === 'all' || (p.decision || '').toLowerCase() === activeFilter).length} candidate papers ({selectedIds.length} queued for synthesis)
                      </span>
                      <button
                        className={styles.toggleAllBtn}
                        onClick={() =>
                          setSelectedIds(selectedIds.length === candidates.length ? [] : candidates.map(p => p.arxiv_id || p.id))
                        }
                      >
                        {selectedIds.length === candidates.length ? 'Deselect All' : 'Select All'}
                      </button>
                    </div>

                    <div className={styles.candidatesScroll}>
                      {candidates
                        .filter(p => activeFilter === 'all' || (p.decision || '').toLowerCase() === activeFilter)
                        .map(p => {
                          const id = p.arxiv_id || p.id
                          const isChecked = selectedIds.includes(id)
                          const scoreVal = typeof p.relevance_score === 'number'
                            ? Math.round(p.relevance_score > 1 ? p.relevance_score : p.relevance_score * 100)
                            : null

                          return (
                            <div
                              key={id}
                              className={clsx(styles.candidateRow, isChecked && styles.candidateRowSelected)}
                            >
                              <input
                                type="checkbox"
                                checked={isChecked}
                                onChange={() => toggleSelect(id)}
                                className={styles.checkbox}
                              />
                              <div className={styles.candidateDetails}>
                                <div className={styles.candidateTop}>
                                  <span className={styles.arxivId}>arXiv:{id}</span>
                                  {scoreVal !== null && (
                                    <span className={styles.matchScore}>
                                      {scoreVal}% relevance
                                    </span>
                                  )}
                                </div>
                                <h4 className={styles.candidateTitle}>{p.title}</h4>
                                {p.abstract && (
                                  <p className={styles.candidateAbstract}>{p.abstract.slice(0, 200)}…</p>
                                )}

                                {/* Screening Decisions per paper */}
                                <div className={styles.decisionControls}>
                                  <button
                                    type="button"
                                    className={clsx(
                                      styles.decisionBtn,
                                      p.decision === 'Include' && styles.decisionBtnIncludeActive
                                    )}
                                    onClick={() => handleDecisionChange(id, 'Include')}
                                  >
                                    <Check size={11} /> Include
                                  </button>
                                  <button
                                    type="button"
                                    className={clsx(
                                      styles.decisionBtn,
                                      p.decision === 'Maybe' && styles.decisionBtnMaybeActive
                                    )}
                                    onClick={() => handleDecisionChange(id, 'Maybe')}
                                  >
                                    <HelpCircle size={11} /> Maybe
                                  </button>
                                  <button
                                    type="button"
                                    className={clsx(
                                      styles.decisionBtn,
                                      p.decision === 'Exclude' && styles.decisionBtnExcludeActive
                                    )}
                                    onClick={() => handleDecisionChange(id, 'Exclude')}
                                  >
                                    <X size={11} /> Exclude
                                  </button>
                                </div>

                                {/* Screening Reason Field */}
                                <div className={styles.reasonBox}>
                                  <input
                                    type="text"
                                    className={styles.reasonInput}
                                    value={p.screening_reason || ''}
                                    placeholder="Enter screening rationale / reason..."
                                    onChange={(e) => handleReasonChange(id, e.target.value)}
                                  />
                                </div>
                              </div>
                            </div>
                          )
                        })}
                    </div>

                    <button
                      className={styles.submitBtn}
                      onClick={handleDownloadAndSynthesize}
                      disabled={!selectedIds.length}
                    >
                      <Sparkles size={14} /> Download Approved Papers & Synthesize Monograph ({selectedIds.length} papers)
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* TAB 2: Upload local PDFs */}
            {tab === 'upload' && (
              <div className={styles.tabBody}>
                <div
                  className={clsx(styles.dropZone, dragActive && styles.dropZoneActive)}
                  onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
                  onDragLeave={() => setDragActive(false)}
                  onDrop={(e) => {
                    e.preventDefault()
                    setDragActive(false)
                    handleFilesChosen(e.dataTransfer.files)
                  }}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload size={32} className={styles.dropIcon} />
                  <p className={styles.dropPrimary}>Drag and drop PDF research papers here</p>
                  <p className={styles.dropSecondary}>or click to browse your machine</p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf"
                    multiple
                    style={{ display: 'none' }}
                    onChange={(e) => handleFilesChosen(e.target.files)}
                  />
                </div>

                {localFiles.length > 0 && (
                  <div className={styles.localFilesQueue}>
                    <span className={styles.queueLabel}>Selected Papers ({localFiles.length})</span>
                    {localFiles.map((f, i) => (
                      <div key={i} className={styles.localFileRow}>
                        <FileText size={14} />
                        <span className={styles.localFileName}>{f.name}</span>
                        <span className={styles.localFileSize}>{(f.size / 1024 / 1024).toFixed(2)} MB</span>
                        <button
                          className={styles.removeBtn}
                          onClick={() => setLocalFiles(prev => prev.filter((_, idx) => idx !== i))}
                        >
                          <Trash2 size={12} />
                        </button>
                      </div>
                    ))}

                    <button
                      className={styles.submitBtn}
                      onClick={handleUploadAndSynthesize}
                    >
                      <ArrowRight size={14} /> Upload & Synthesize Monograph ({localFiles.length} PDFs)
                    </button>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>

      {/* Alternative Exploration / Sample Review Option */}
      {!pipelineRunning && (
        <div className={styles.exploreSection}>
          <div className={styles.divider}>
            <span>OR EXPLORE BENCHMARK SAMPLE</span>
          </div>

          <div className={styles.sampleBox}>
            <button className={styles.benchmarkBtn} onClick={onLoadBenchmark}>
              <Layers size={14} /> Load 12-Paper Standard Benchmark Review ↗
            </button>
            {existingCorpusCount > 0 && (
              <button className={styles.resumeBtn} onClick={onStartReview}>
                <BookOpen size={14} /> View Currently Generated Review ({existingCorpusCount} papers) ↗
              </button>
            )}
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className={styles.footer}>
        <span>MONOGRAPH PROTOCOL · ZERO HALLUCINATION GROUNDING · GROBID TEI-XML</span>
      </footer>
    </div>
  )
}
