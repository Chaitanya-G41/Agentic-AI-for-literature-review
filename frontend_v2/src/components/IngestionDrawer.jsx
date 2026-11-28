import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  X,
  Search,
  Upload,
  BookOpen,
  ArrowRight,
  CheckCircle2,
  Loader2,
  Sparkles,
  FileText,
  Trash2,
  Layers,
  Cpu,
  Sliders,
  Check,
  HelpCircle
} from 'lucide-react'
import clsx from 'clsx'
import styles from './IngestionDrawer.module.css'

export default function IngestionDrawer({ isOpen, onClose, onRefreshData }) {
  const [tab, setTab] = useState('arxiv') // 'arxiv' | 'upload'
  const [query, setQuery] = useState('')
  const [maxResults, setMaxResults] = useState(6)
  const [screening, setScreening] = useState(false)
  const [screenError, setScreenError] = useState(null)
  const [candidates, setCandidates] = useState([])
  const [selectedIds, setSelectedIds] = useState([])

  // Screening criteria state
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

  // Pipeline execution state
  const [pipelineRunning, setPipelineRunning] = useState(false)
  const [pipelineStage, setPipelineStage] = useState('')
  const [pipelineLogs, setPipelineLogs] = useState([])
  const [pipelineFinished, setPipelineFinished] = useState(false)

  // Load existing screening state when drawer opens
  useEffect(() => {
    if (isOpen) {
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
    }
  }, [isOpen])

  // Screen arXiv
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

      if (newDecision === 'Include') {
        setSelectedIds(s => s.includes(id) ? s : [...s, id])
      } else if (newDecision === 'Exclude') {
        setSelectedIds(s => s.filter(x => x !== id))
      }

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

  // Pipeline runner function
  const runFullPipeline = async (downloadPromise = null) => {
    setPipelineRunning(true)
    setPipelineFinished(false)
    setPipelineLogs([])

    const addLog = (msg) => {
      setPipelineLogs(prev => [...prev, { time: new Date().toLocaleTimeString(), msg }])
    }

    const runStage = async (label, url) => {
      const res = await fetch(url, { method: 'POST' })
      if (!res.ok) {
        let detail = ''
        try { const j = await res.json(); detail = j.detail || JSON.stringify(j) } catch (_) {
          try { detail = await res.text() } catch (_) { detail = `HTTP ${res.status}` }
        }
        throw new Error(`${label} failed (${res.status}): ${detail}`)
      }
      return res
    }

    try {
      if (downloadPromise) {
        setPipelineStage('Downloading Papers')
        addLog('Connecting to arXiv and saving candidate PDFs...')
        await downloadPromise
        addLog('PDFs safely archived in corpus.')
      }

      setPipelineStage('1/5: GROBID Parsing & Schema Extraction')
      addLog('Agent 2: Extracting TEI-XML schemas & methodology via GROBID...')
      await runStage('Extraction Agent', '/api/run-agent/extraction')
      addLog('Extracted structured schemas with zero hallucination grounding.')

      setPipelineStage('2/5: Single-Paper Deterministic Summaries')
      addLog('Agent 3: Generating deterministic Pydantic summaries (0 LLM calls)...')
      await runStage('Summarizer Agent', '/api/run-agent/summarizer')
      addLog('Summaries compiled.')

      setPipelineStage('3/5: Cross-Paper Comparative Analysis')
      addLog('Agent 4: Computing architectural alignment matrix...')
      await runStage('Comparative Agent', '/api/run-agent/comparative')
      addLog('Comparative matrix constructed.')

      setPipelineStage('4/5: Research Gap Synthesis')
      addLog('Agent 5: Synthesizing systemic bottlenecks & frontier research gaps...')
      await runStage('Research Gap Agent', '/api/run-agent/research-gap')
      addLog('Identified research gaps.')

      setPipelineStage('5/5: Compiling Monograph & Mermaid Diagrams')
      addLog('Agent 6: Generating high-density review with Mermaid.js architecture diagrams (~45-90s on NIM)...')
      await runStage('Report Agent', '/api/run-agent/report')
      addLog('Monograph synthesis ready!')

      setPipelineFinished(true)
      setPipelineStage('Complete')
      if (onRefreshData) await onRefreshData()
    } catch (err) {
      addLog(`❌ Error: ${err.message}`)
      setPipelineStage('Pipeline Error')
    } finally {
      setPipelineRunning(false)
    }
  }

  // Handle arXiv download & run
  const handleDownloadAndSynthesize = () => {
    const selected = candidates.filter(p => selectedIds.includes(p.arxiv_id || p.id))
    if (!selected.length) return
    const downloadPromise = (async () => {
      // Clear out old corpus workspace first so old topics are not mixed
      await fetch('/api/clear-corpus', { method: 'POST' }).catch(() => {})
      const res = await fetch('/api/download-papers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ papers: selected, query: query.trim() }),
      })
      if (!res.ok) {
        const j = await res.json().catch(() => ({}))
        throw new Error(`Download failed (${res.status}): ${j.detail || 'unknown'}`)
      }
      return res
    })()
    runFullPipeline(downloadPromise)
  }

  // Handle Local PDF file selection
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
    runFullPipeline(Promise.all(uploadPromises))
  }

  // Load 12 Benchmark Papers
  const handleLoadBenchmark = () => {
    const p = fetch('/api/load-benchmark', { method: 'POST' })
    runFullPipeline(p)
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            className={styles.backdrop}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={pipelineRunning ? null : onClose}
          />

          {/* Drawer container */}
          <motion.aside
            className={styles.drawer}
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 26, stiffness: 240 }}
          >
            {/* Drawer Header */}
            <div className={styles.header}>
              <div className={styles.headerTitleGroup}>
                <span className={styles.headerTag}>CORPUS DISCOVERY & INGESTION</span>
                <h3 className={styles.headerTitle}>Add Papers to Monograph</h3>
              </div>
              <button
                className={styles.closeBtn}
                onClick={onClose}
                disabled={pipelineRunning}
                title="Close drawer"
              >
                <X size={18} />
              </button>
            </div>

            {/* Pipeline Live Progress View */}
            {pipelineRunning || pipelineFinished ? (
              <div className={styles.pipelineView}>
                <div className={styles.pipelineStatusBox}>
                  {pipelineFinished ? (
                    <CheckCircle2 size={24} className={styles.iconSuccess} />
                  ) : (
                    <Loader2 size={24} className={clsx(styles.iconRunning, styles.spin)} />
                  )}
                  <div>
                    <h4 className={styles.stageTitle}>
                      {pipelineFinished ? 'Synthesis Complete' : pipelineStage}
                    </h4>
                    <p className={styles.stageSub}>
                      {pipelineFinished
                        ? 'Your Monograph has been updated with the synthesized findings.'
                        : 'Autonomous agents are parsing, aligning, and synthesizing the literature.'}
                    </p>
                  </div>
                </div>

                {/* Console Log ticker */}
                <div className={styles.logConsole}>
                  <div className={styles.logHeader}>
                    <span>AGENT EXECUTION LOG</span>
                    <span className={styles.logPulse} />
                  </div>
                  <div className={styles.logScroll}>
                    {pipelineLogs.map((item, idx) => (
                      <div key={idx} className={styles.logRow}>
                        <span className={styles.logTime}>{item.time}</span>
                        <span className={styles.logMsg}>{item.msg}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className={styles.pipelineActions}>
                  {pipelineFinished ? (
                    <button className={styles.primaryActionBtn} onClick={onClose}>
                      View Synthesized Monograph <ArrowRight size={14} />
                    </button>
                  ) : (
                    <p className={styles.waitNotice}>
                      ⚡ Strict 3-point LLM budget protocol active. Please keep this drawer open.
                    </p>
                  )}
                </div>
              </div>
            ) : (
              /* Regular Ingestion Tabs */
              <div className={styles.body}>
                <div className={styles.tabsRow}>
                  <button
                    className={clsx(styles.tabBtn, tab === 'arxiv' && styles.tabBtnActive)}
                    onClick={() => setTab('arxiv')}
                  >
                    <Search size={14} /> Fetch from arXiv
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
                  <div className={styles.tabContent}>
                    <p className={styles.tabDescription}>
                      Query arXiv API in real-time. The screening agent ranks relevant research papers automatically.
                    </p>

                    <div className={styles.searchBox}>
                      <input
                        type="text"
                        className={styles.searchInput}
                        placeholder="e.g. LLM agent reasoning planning ReAct"
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
                        {screening ? <Loader2 size={14} className={styles.spin} /> : <Search size={14} />}
                        {screening ? 'Fetching…' : 'Search'}
                      </button>
                    </div>

                    {/* Criteria Toggle */}
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
                      <div className={styles.errorNotice}>
                        {screenError}
                      </div>
                    )}

                    {candidates.length > 0 && (
                      <div className={styles.candidatesSection}>
                        {/* PRISMA summary bar */}
                        <div className={styles.prismaSummaryBar}>
                          <span>PRISMA SCREENING</span>
                          <div className={styles.prismaStats}>
                            <span>Total: <strong>{candidates.length}</strong></span>
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

                        <div className={styles.candidatesBar}>
                          <span className={styles.candidateCount}>
                            Showing {candidates.filter(p => activeFilter === 'all' || (p.decision || '').toLowerCase() === activeFilter).length} papers · {selectedIds.length} queued
                          </span>
                          <button
                            className={styles.selectToggleBtn}
                            onClick={() =>
                              setSelectedIds(
                                selectedIds.length === candidates.length
                                  ? []
                                  : candidates.map(p => p.arxiv_id || p.id)
                              )
                            }
                          >
                            {selectedIds.length === candidates.length ? 'Deselect All' : 'Select All'}
                          </button>
                        </div>

                        <div className={styles.candidatesList}>
                          {candidates
                            .filter(p => activeFilter === 'all' || (p.decision || '').toLowerCase() === activeFilter)
                            .map(p => {
                              const id = p.arxiv_id || p.id
                              const isSelected = selectedIds.includes(id)
                              const scoreVal = typeof p.relevance_score === 'number'
                                ? Math.round(p.relevance_score > 1 ? p.relevance_score : p.relevance_score * 100)
                                : null

                              return (
                                <div
                                  key={id}
                                  className={clsx(styles.candidateCard, isSelected && styles.candidateSelected)}
                                >
                                  <input
                                    type="checkbox"
                                    checked={isSelected}
                                    onChange={() => toggleSelect(id)}
                                    className={styles.candidateCheckbox}
                                  />
                                  <div className={styles.candidateInfo}>
                                    <div className={styles.candidateMetaRow}>
                                      <span className={styles.candidateId}>arXiv:{id}</span>
                                      {scoreVal !== null && (
                                        <span className={styles.scoreBadge}>
                                          {scoreVal}% relevance
                                        </span>
                                      )}
                                    </div>
                                    <h5 className={styles.candidateTitle}>{p.title}</h5>
                                    {p.abstract && (
                                      <p className={styles.candidateAbstract}>
                                        {p.abstract.slice(0, 160)}…
                                      </p>
                                    )}

                                    {/* Screening decisions */}
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

                                    {/* Screening reason */}
                                    <div className={styles.reasonBox}>
                                      <input
                                        type="text"
                                        className={styles.reasonInput}
                                        value={p.screening_reason || ''}
                                        placeholder="Screening rationale / reason..."
                                        onChange={(e) => handleReasonChange(id, e.target.value)}
                                      />
                                    </div>
                                  </div>
                                </div>
                              )
                            })}
                        </div>

                        <button
                          className={styles.primaryActionBtn}
                          onClick={handleDownloadAndSynthesize}
                          disabled={!selectedIds.length}
                        >
                          <Sparkles size={14} /> Download Approved & Synthesize ({selectedIds.length} papers)
                        </button>
                      </div>
                    )}
                  </div>
                )}

                {/* TAB 2: Local Upload */}
                {tab === 'upload' && (
                  <div className={styles.tabContent}>
                    <p className={styles.tabDescription}>
                      Upload existing research papers (.pdf) from your filesystem. They will be parsed with GROBID TEI-XML.
                    </p>

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
                      <Upload size={28} className={styles.dropIcon} />
                      <p className={styles.dropPrimary}>Drag & drop research papers here</p>
                      <p className={styles.dropSecondary}>or click to select PDFs from your machine</p>
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept=".pdf"
                        multiple
                        className={styles.hiddenInput}
                        onChange={(e) => handleFilesChosen(e.target.files)}
                      />
                    </div>

                    {localFiles.length > 0 && (
                      <div className={styles.localFilesList}>
                        <span className={styles.filesLabel}>Queued for Analysis ({localFiles.length})</span>
                        {localFiles.map((file, i) => (
                          <div key={i} className={styles.localFileItem}>
                            <FileText size={15} className={styles.fileIcon} />
                            <div className={styles.fileNameCol}>
                              <span className={styles.fileName}>{file.name}</span>
                              <span className={styles.fileSize}>
                                {(file.size / 1024 / 1024).toFixed(2)} MB
                              </span>
                            </div>
                            <button
                              className={styles.removeFileBtn}
                              onClick={(e) => {
                                e.stopPropagation()
                                setLocalFiles(prev => prev.filter((_, idx) => idx !== i))
                              }}
                            >
                              <Trash2 size={13} />
                            </button>
                          </div>
                        ))}

                        <button
                          className={styles.primaryActionBtn}
                          onClick={handleUploadAndSynthesize}
                        >
                          <Cpu size={14} /> Upload & Synthesize Monograph
                        </button>
                      </div>
                    )}
                  </div>
                )}

                {/* Quick Benchmark Preset Divider */}
                <div className={styles.presetSection}>
                  <div className={styles.presetDivider}>
                    <span>OR USE EXPERIMENTAL BENCHMARK</span>
                  </div>
                  <button className={styles.presetBtn} onClick={handleLoadBenchmark}>
                    <Layers size={14} /> Load 12 Standard Agentic AI Papers
                  </button>
                </div>
              </div>
            )}
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}
