import { useState } from 'react'
import { X, ExternalLink, FileText, CheckCircle2, ChevronDown, ChevronUp } from 'lucide-react'
import styles from './SplitScreenPdf.module.css'

export default function SplitScreenPdf({ paper, quote, loc, similarity, onClose }) {
  const [showCitationDetails, setShowCitationDetails] = useState(true)

  if (!paper) return null

  const id = paper.paper_id || paper.arxiv_id || ''
  const filename = paper.filename || id
  const pdfUrl = `/api/pdf/${encodeURIComponent(filename)}`
  const externalArxivUrl = id.includes('.') ? `https://arxiv.org/abs/${id}` : null
  const title = paper.title?.replace(/^\d+\.\d+\s+/, '').replace(/_/g, ' ') || id

  return (
    <div className={styles.container}>
      {/* Split screen top command bar */}
      <div className={styles.topBar}>
        <div className={styles.metaLeft}>
          <span className={styles.badge}>
            <FileText size={11} /> PDF GROUNDING
          </span>
          <span className={styles.paperId}>arXiv:{id}</span>
          <span className={styles.paperTitle} title={title}>
            {title}
          </span>
        </div>

        <div className={styles.controlsRight}>
          {externalArxivUrl && (
            <a
              href={externalArxivUrl}
              target="_blank"
              rel="noopener noreferrer"
              className={styles.toolBtn}
              title="Open arXiv abstract page"
            >
              <ExternalLink size={12} />
              <span className={styles.toolBtnLabel}>arXiv</span>
            </a>
          )}
          <a
            href={pdfUrl}
            target="_blank"
            rel="noopener noreferrer"
            className={styles.toolBtn}
            title="Open raw PDF in new browser tab"
          >
            <ExternalLink size={12} />
            <span className={styles.toolBtnLabel}>New Tab</span>
          </a>
          <button
            className={styles.closeBtn}
            onClick={onClose}
            title="Exit split screen view"
          >
            <X size={15} />
            <span className={styles.closeLabel}>Close PDF</span>
          </button>
        </div>
      </div>

      {/* PDF Viewer frame */}
      <div className={styles.frameWrapper}>
        <iframe
          src={`${pdfUrl}#navpanes=0&toolbar=1`}
          className={styles.pdfFrame}
          title={`PDF for ${title}`}
        />
      </div>

      {/* Collapsible Provenance Evidence Dock */}
      {(quote || paper.core_problem) && (
        <div className={styles.provenanceDock}>
          <div
            className={styles.dockHeader}
            onClick={() => setShowCitationDetails(v => !v)}
          >
            <div className={styles.dockTitleGroup}>
              <CheckCircle2 size={13} className={styles.dockCheck} />
              <span className={styles.dockTitle}>VERBATIM GROUNDED EVIDENCE</span>
              {similarity && (
                <span className={styles.dockSim}>
                  Cosine: {(similarity * 100).toFixed(1)}%
                </span>
              )}
            </div>
            <button className={styles.dockToggleBtn}>
              {showCitationDetails ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
            </button>
          </div>

          {showCitationDetails && (
            <div className={styles.dockBody}>
              {quote && (
                <div className={styles.dockField}>
                  <span className={styles.dockLabel}>CITED STATEMENT</span>
                  <p className={styles.dockQuote}>"{quote}"</p>
                </div>
              )}
              {paper.core_problem && (
                <div className={styles.dockField}>
                  <span className={styles.dockLabel}>EXTRACTED PROBLEM & SCOPE</span>
                  <p className={styles.dockText}>{paper.core_problem}</p>
                </div>
              )}
              {paper.methodology && (
                <div className={styles.dockField}>
                  <span className={styles.dockLabel}>METHODOLOGY EXTRACT</span>
                  <p className={styles.dockText}>{paper.methodology.slice(0, 300)}…</p>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
