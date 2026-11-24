import { useState, useEffect, useCallback } from 'react'
import CommandBar from './components/CommandBar'
import MonographReader from './components/MonographReader'
import IngestionDrawer from './components/IngestionDrawer'
import WelcomeScreen from './components/WelcomeScreen'
import LoadingCover from './components/LoadingCover'
import './App.css'

export default function App() {
  const [loading, setLoading]                     = useState(true)
  const [corpus, setCorpus]                       = useState([])
  const [reportData, setReportData]               = useState(null)
  const [reviews, setReviews]                     = useState([])
  const [selectedReviewId, setSelectedReviewId]   = useState(null)
  const [inspectorState, setInspectorState]       = useState(null)
  const [activePillars, setActivePillars]         = useState({
    methodology: true, datasets: true, limitations: true, gaps: true,
  })
  const [activePillarLens, setActivePillarLens]   = useState(null) // 'methodology' | 'datasets' | 'limitations' | 'gaps' | null
  const [readingMode, setReadingMode]             = useState('technical') // 'technical' | 'plain'
  const [activePdfPaper, setActivePdfPaper]       = useState(null) // paper object for split screen PDF
  const [isIngestOpen, setIsIngestOpen]           = useState(false) // slide out drawer
  const [error, setError]                         = useState(null)

  // Track if this browser has an active review session
  // Default to false for new users so they start on the fresh WelcomeScreen
  const [hasActiveSession, setHasActiveSession]   = useState(() => {
    return localStorage.getItem('monograph_session_active') === 'true'
  })

  const loadData = useCallback(async (targetReviewId = null) => {
    try {
      const reportUrl = targetReviewId
        ? `/api/report?review_id=${encodeURIComponent(targetReviewId)}`
        : '/api/report'
      const [cRes, rRes, revRes] = await Promise.all([
        fetch('/api/corpus'),
        fetch(reportUrl),
        fetch('/api/reviews').catch(() => ({ ok: false })),
      ])
      if (!cRes.ok || !rRes.ok) throw new Error('Backend API unavailable')
      const cData = await cRes.json()
      const rData = await rRes.json()
      const revData = revRes.ok ? await revRes.json() : { reviews: [] }
      const papers = cData.papers || cData || []
      setCorpus(papers)
      setReportData(rData)
      setReviews(revData.reviews || [])
      if (rData.review_id) {
        setSelectedReviewId(rData.review_id)
      } else if (targetReviewId) {
        setSelectedReviewId(targetReviewId)
      }

      // Set default inspector to first paper if none selected
      if (!inspectorState && papers.length > 0) {
        setInspectorState({
          paperId: papers[0].paper_id,
          quote: 'Interleaved reasoning traces with discrete tool executions.',
          loc: 'Section 3.1, Page 3',
          similarity: 0.94,
        })
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [inspectorState])

  useEffect(() => {
    loadData()
  }, [])

  const updateInspector = useCallback((paperId, quote, loc, similarity) => {
    setInspectorState({ paperId, quote, loc, similarity })
  }, [])

  // Start / activate a session (e.g. from WelcomeScreen)
  const handleStartReview = async () => {
    localStorage.setItem('monograph_session_active', 'true')
    setHasActiveSession(true)
    setSelectedReviewId(null)
    await loadData(null)
  }

  // Load sample benchmark papers
  const handleLoadBenchmark = async () => {
    try {
      setLoading(true)
      await fetch('/api/load-benchmark', { method: 'POST' })
      localStorage.setItem('monograph_session_active', 'true')
      setHasActiveSession(true)
      setSelectedReviewId(null)
      await loadData(null)
    } catch (e) {
      alert('Failed to load benchmark: ' + e.message)
    } finally {
      setLoading(false)
    }
  }

  // Clear session to return to fresh WelcomeScreen
  const handleStartFresh = async () => {
    const confirmReset = window.confirm(
      'Start a new research review? This will return to the welcome screen so you can search a new topic or upload fresh papers.'
    )
    if (confirmReset) {
      localStorage.removeItem('monograph_session_active')
      setHasActiveSession(false)
      setActivePdfPaper(null)
      setActivePillarLens(null)
      setInspectorState(null)
      try {
        await fetch('/api/clear-corpus', { method: 'POST' })
      } catch (e) {
        console.error('Failed to clear backend corpus:', e)
      }
      setCorpus([])
      setReportData(null)
      setSelectedReviewId(null)
    }
  }

  if (loading) return <LoadingCover />

  // If new user or no session active, show clean Welcome / Ingestion screen
  if (!hasActiveSession) {
    return (
      <WelcomeScreen
        onStartReview={handleStartReview}
        onLoadBenchmark={handleLoadBenchmark}
        existingCorpusCount={corpus.length}
      />
    )
  }

  if (error && !corpus.length) {
    return (
      <div className="error-screen">
        <p className="error-mono">MONOGRAPH PROTOCOL</p>
        <p className="error-msg">Backend API unavailable. Ensure <code>server.py</code> is running on port 8000.</p>
        <p className="error-sub">{error}</p>
        <button className="primary-reload-btn" onClick={() => { setLoading(true); setError(null); loadData(); }}>
          Retry Connection
        </button>
      </div>
    )
  }

  return (
    <div className="app-root">
      <CommandBar
        corpus={corpus}
        readingMode={readingMode}
        setReadingMode={setReadingMode}
        activePillars={activePillars}
        setActivePillars={setActivePillars}
        onOpenIngest={() => setIsIngestOpen(true)}
        activePillarLens={activePillarLens}
        setActivePillarLens={setActivePillarLens}
        onStartFresh={handleStartFresh}
        reviews={reviews}
        selectedReviewId={selectedReviewId}
        onSelectReview={(rId) => {
          setSelectedReviewId(rId)
          loadData(rId)
        }}
      />

      <MonographReader
        corpus={corpus}
        reportData={reportData}
        inspectorState={inspectorState}
        updateInspector={updateInspector}
        activePillars={activePillars}
        readingMode={readingMode}
        activePillarLens={activePillarLens}
        setActivePillarLens={setActivePillarLens}
        activePdfPaper={activePdfPaper}
        setActivePdfPaper={setActivePdfPaper}
      />

      {/* Slide-out Corpus Ingestion Drawer */}
      <IngestionDrawer
        isOpen={isIngestOpen}
        onClose={() => setIsIngestOpen(false)}
        onRefreshData={loadData}
      />
    </div>
  )
}
