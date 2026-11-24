import clsx from 'clsx'
import { useState } from 'react'
import { Download, Plus, RotateCcw, Cpu, Award } from 'lucide-react'
import ReviewInfoModal from './ReviewInfoModal'
import styles from './CommandBar.module.css'

const PILLARS = [
  { key: 'methodology', label: 'Methodology' },
  { key: 'datasets',    label: 'Datasets' },
  { key: 'limitations', label: 'Limitations' },
  { key: 'gaps',        label: 'Research Gaps' },
]

export default function CommandBar({
  corpus,
  reportData,
  readingMode,
  setReadingMode,
  activePillars,
  setActivePillars,
  onOpenIngest,
  activePillarLens,
  setActivePillarLens,
  onStartFresh,
  reviews = [],
  selectedReviewId = null,
  onSelectReview,
}) {
  const [infoModal, setInfoModal] = useState(null)
  const togglePillar = (key) => {
    setActivePillars((p) => ({ ...p, [key]: !p[key] }))
  }

  const handlePillarLensClick = (key) => {
    setActivePillarLens(current => current === key ? null : key)
  }

  const handleExport = async () => {
    try {
      const r = await fetch('/api/report')
      const data = await r.json()
      const content = data.content || data.report || ''
      const blob = new Blob([content], { type: 'text/markdown' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'literature_review_monograph.md'
      a.click()
      URL.revokeObjectURL(url)
    } catch {}
  }

  return (
    <header className={styles.bar}>
      {/* Brand & Corpus Ingestion trigger */}
      <div className={styles.brand}>
        <span className={styles.brandWord}>MONOGRAPH</span>
        <span className={styles.brandRule} />
        <button
          className={styles.corpusBtn}
          onClick={onOpenIngest}
          title="Open Corpus Discovery & Ingestion Drawer"
        >
          <span className={styles.corpusCount}>{corpus.length} PAPERS</span>
          <span className={styles.corpusAddBadge}>
            <Plus size={10} /> Add
          </span>
        </button>

        {reviews && reviews.length > 1 && (
          <div className={styles.reviewPickerWrap}>
            <span className={styles.reviewPickerLabel}>REVIEW:</span>
            <select
              className={styles.reviewSelect}
              value={selectedReviewId || ''}
              onChange={(e) => onSelectReview && onSelectReview(e.target.value)}
              title="Switch between saved literature reviews"
            >
              {reviews.map((r) => (
                <option key={r.review_id} value={r.review_id}>
                  {r.title ? (r.title.length > 25 ? r.title.slice(0, 25) + '…' : r.title) : r.review_id} ({r.paper_count} papers)
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Center: Pillar Lenses (Pure Black & Ivory Monochrome) */}
      <div className={styles.pillarsContainer}>
        <span className={styles.pillarsLabel}>LENSES:</span>
        <div className={styles.pillars}>
          {PILLARS.map(({ key, label }) => {
            const isLensActive = activePillarLens === key
            const isToggled = activePillars[key]
            return (
              <button
                key={key}
                className={clsx(
                  styles.pillar,
                  (isToggled || isLensActive) && styles.pillarActive,
                  isLensActive && styles.pillarLensActive
                )}
                onClick={() => {
                  togglePillar(key)
                  handlePillarLensClick(key)
                }}
                title={`Filter monograph to ${label}`}
              >
                <span className={styles.pillarDot} />
                {label}
                {isLensActive && <span className={styles.lensBadge}>LENS</span>}
              </button>
            )
          })}
        </div>
      </div>

      {/* Right controls */}
      <div className={styles.right}>
        <button className={styles.resetBtn} onClick={() => setInfoModal('architecture')} aria-haspopup="dialog">
          <Cpu size={12} /> System Architecture
        </button>
        <button className={styles.resetBtn} onClick={() => setInfoModal('evaluation')} aria-haspopup="dialog">
          <Award size={12} /> Evaluation Rubric
        </button>
        {/* Start Fresh / New Review button */}
        <button
          className={styles.resetBtn}
          onClick={onStartFresh}
          title="Clear current session and start a fresh review"
        >
          <RotateCcw size={11} />
          <span>New Review</span>
        </button>

        <button className={styles.ingestActionBtn} onClick={onOpenIngest}>
          <Plus size={12} />
          <span>Ingest Papers</span>
        </button>

        <div className={styles.modeSwitch}>
          <button
            className={clsx(styles.modeBtn, readingMode === 'technical' && styles.modeBtnOn)}
            onClick={() => setReadingMode('technical')}
          >
            Technical
          </button>
          <button
            className={clsx(styles.modeBtn, readingMode === 'plain' && styles.modeBtnOn)}
            onClick={() => setReadingMode('plain')}
          >
            Plain
          </button>
        </div>

        <button className={styles.exportBtn} onClick={handleExport} title="Download Monograph Markdown">
          <Download size={11} />
          <span>Export</span>
        </button>
      </div>
      {infoModal && <ReviewInfoModal kind={infoModal} onClose={() => setInfoModal(null)} corpus={corpus} content={reportData?.content || reportData?.report || ''} />}
    </header>
  )
}
