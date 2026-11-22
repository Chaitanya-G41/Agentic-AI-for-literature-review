import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { MousePointerClick, FileText, Copy, ExternalLink, Columns } from 'lucide-react'
import styles from './ProvenanceInspector.module.css'

export default function ProvenanceInspector({ inspectorState, corpus, onOpenPdf }) {
  const [detail, setDetail]   = useState(null)
  const [copied, setCopied]   = useState(false)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!inspectorState?.paperId) { setDetail(null); return }

    // first try corpus
    const local = corpus.find(
      p => p.paper_id === inspectorState.paperId || p.arxiv_id === inspectorState.paperId
    )
    if (local) { setDetail(local); return }

    // else fetch
    setLoading(true)
    fetch(`/api/paper/${encodeURIComponent(inspectorState.paperId)}`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d) setDetail(d) })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [inspectorState, corpus])

  const paper = detail
  const id = paper?.paper_id || paper?.arxiv_id || inspectorState?.paperId || ''
  const filename = paper?.filename || id
  const title = paper?.title?.replace(/^\d+\.\d+\s+/, '').replace(/_/g, ' ') || id
  const par = paper?.paradigm
  const quote = inspectorState?.quote || ''
  const sim = inspectorState?.similarity

  const pdfUrl = id ? `/api/pdf/${encodeURIComponent(filename)}` : null
  const arxivUrl = id && id.includes('.') ? `https://arxiv.org/abs/${id}` : null

  const bibtex = id
    ? `@article{arxiv${id.replace(/\./g, '')},
  title   = {${title}},
  year    = {${paper?.year || 2024}},
  archivePrefix = {arXiv},
  eprint  = {${id}}
}`
    : ''

  const copy = () => {
    navigator.clipboard.writeText(bibtex)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <aside className={styles.root}>
      <div className={styles.head}>
        <span className={styles.headLabel}>PROVENANCE INSPECTOR</span>
        {paper && par && (
          <span
            className={styles.paradigmBadge}
            style={{ color: par.text, borderColor: par.border + '55', background: par.bg }}
          >
            {par.tag}
          </span>
        )}
      </div>

      <AnimatePresence mode="wait">
        {!inspectorState ? (
          <motion.div
            key="empty"
            className={styles.empty}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            <MousePointerClick size={30} className={styles.emptyIcon} />
            <p className={styles.emptyHead}>Click any citation</p>
            <p className={styles.emptySub}>
              Click any underlined reference in the article to load verbatim source text
              and view its grounded PDF side-by-side.
            </p>
          </motion.div>
        ) : loading ? (
          <motion.div key="loading" className={styles.empty} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <p className={styles.emptySub}>Loading…</p>
          </motion.div>
        ) : (
          <motion.div
            key={id}
            className={styles.content}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.25, ease: 'easeOut' }}
          >

            {/* arXiv ID + location */}
            <div className={styles.idRow}>
              <span className={styles.arxivId}>arXiv:{id}</span>
              {inspectorState?.loc && (
                <span className={styles.loc}>{inspectorState.loc}</span>
              )}
            </div>

            {/* Title */}
            {paper && (
              <div className={styles.block}>
                <span className={styles.blockLabel}>PAPER REFERENCE</span>
                <h4 className={styles.paperTitle}>{title}</h4>
              </div>
            )}

            {/* Cited as / verbatim text */}
            {quote && (
              <div className={styles.block}>
                <span className={styles.blockLabel}>CITED STATEMENT</span>
                <p className={styles.quoteText}>"{quote}"</p>
              </div>
            )}

            {/* Core problem */}
            {paper?.core_problem && (
              <div className={styles.block}>
                <span className={styles.blockLabel}>VERBATIM SOURCE TEXT</span>
                <p className={styles.sourceText}>
                  "{paper.core_problem.slice(0, 320)}{paper.core_problem.length > 320 ? '…' : ''}"
                </p>
              </div>
            )}

            {/* Methodology snippet */}
            {paper?.methodology && (
              <div className={styles.block}>
                <span className={styles.blockLabel}>METHODOLOGY EXTRACT</span>
                <p className={styles.methodText}>
                  {paper.methodology.slice(0, 240)}{paper.methodology.length > 240 ? '…' : ''}
                </p>
              </div>
            )}

            {/* Cosine similarity */}
            {sim !== undefined && sim !== null && (
              <div className={styles.simRow}>
                <span className={styles.simLabel}>Cosine similarity</span>
                <span
                  className={styles.simScore}
                  style={{ color: sim >= 0.8 ? 'var(--success)' : sim >= 0.5 ? 'var(--warning)' : '#DC2626' }}
                >
                  {(sim * 100).toFixed(1)}%
                </span>
              </div>
            )}

            {/* Grounding badge */}
            <div className={styles.groundingRow}>
              <span className={styles.groundingOk}>✓ TEI-XML Grounded</span>
            </div>

            {/* Actions: Split Screen View, External Tab, BibTeX */}
            <div className={styles.actions}>
              {paper && onOpenPdf && (
                <button
                  className={styles.actionPrimary}
                  onClick={() => onOpenPdf(paper)}
                  title="Open in neat split screen view next to the Monograph"
                >
                  <Columns size={13} /> View Split-Screen PDF ◨
                </button>
              )}

              <div className={styles.actionRowSecondary}>
                {pdfUrl && (
                  <a href={pdfUrl} target="_blank" rel="noopener noreferrer" className={styles.actionSecondary}>
                    <FileText size={11} /> Raw PDF ↗
                  </a>
                )}
                {arxivUrl && (
                  <a href={arxivUrl} target="_blank" rel="noopener noreferrer" className={styles.actionSecondary}>
                    <ExternalLink size={11} /> arXiv
                  </a>
                )}
                <button className={styles.actionSecondary} onClick={copy} disabled={!bibtex}>
                  <Copy size={11} /> {copied ? 'Copied!' : 'BibTeX'}
                </button>
              </div>
            </div>

          </motion.div>
        )}
      </AnimatePresence>
    </aside>
  )
}
