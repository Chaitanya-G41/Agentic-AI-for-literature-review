
import { useEffect, useRef, useState } from 'react'
import mermaid from 'mermaid'
import styles from './MermaidDiagrams.module.css'

mermaid.initialize({
  startOnLoad: false,
  theme: 'base',
  themeVariables: {
    primaryColor: '#FFFFFF',
    primaryTextColor: '#121316',
    primaryBorderColor: '#121316',
    lineColor: '#3D3D3D',
    background: '#FAF8F5',
    fontFamily: 'Plus Jakarta Sans, system-ui, sans-serif',
    fontSize: '12px',
  },
  securityLevel: 'loose',
})

const LABELS = {
  methodology_flowchart: 'Methodology Flowchart',
  themes_mindmap: 'Thematic Clusters',
  research_gaps_flowchart: 'Research Gaps',
}

let renderSequence = 0

function applySvgDimensions(svgEl, zoom) {
  if (!svgEl) return
  svgEl.style.height = 'auto'
  svgEl.style.transition = 'width 0.2s ease, max-width 0.2s ease, transform 0.2s ease'

  let vbWidth = 0
  if (svgEl.viewBox && svgEl.viewBox.baseVal && svgEl.viewBox.baseVal.width > 0) {
    vbWidth = svgEl.viewBox.baseVal.width
  } else {
    const vbAttr = svgEl.getAttribute('viewBox')
    if (vbAttr) {
      const parts = vbAttr.split(/[\s,]+/)
      if (parts.length >= 4) {
        vbWidth = parseFloat(parts[2]) || 0
      }
    }
  }

  // Diagnostic assessment diagrams (single card) or narrow compact charts
  // should render as readable, well-proportioned horizontal cards rather than collapsing into narrow pillars
  const isDiagnostic = !!(
    svgEl.querySelector('.diagStyle') ||
    svgEl.querySelector('[class*="diagStyle"]') ||
    svgEl.textContent?.includes('Diagnostic Assessment')
  )

  if (isDiagnostic) {
    const scale = zoom === 'fit' ? 1.0 : zoom
    const targetWidth = Math.min(Math.max(Math.round(vbWidth * scale), 580), 760)
    svgEl.style.width = `${targetWidth}px`
    svgEl.style.maxWidth = '100%'
    svgEl.style.margin = '16px auto'
    svgEl.style.display = 'block'
  } else if (vbWidth > 0 && vbWidth < 650) {
    const scale = zoom === 'fit' ? 1.0 : zoom
    const targetWidth = Math.min(Math.round(vbWidth * scale), 700)
    svgEl.style.width = `${targetWidth}px`
    svgEl.style.maxWidth = '100%'
    svgEl.style.margin = '0 auto'
    svgEl.style.display = 'block'
  } else {
    if (zoom === 'fit') {
      svgEl.style.maxWidth = '100%'
      svgEl.style.width = '100%'
      svgEl.style.margin = '0 auto'
      svgEl.style.display = 'block'
    } else {
      svgEl.style.maxWidth = 'none'
      svgEl.style.width = `${Math.round(zoom * 100)}%`
      svgEl.style.margin = '0 auto'
      svgEl.style.display = 'block'
    }
  }
}

function Chart({ code, id }) {
  const ref = useRef(null)
  const [err, setErr] = useState(null)
  const [zoom, setZoom] = useState(1.0)

  useEffect(() => {
    let isCancelled = false

    if (!ref.current || !code) return

    setErr(null)
    ref.current.innerHTML = ''

    const clean = code
      .replace(/^```mermaid\s*/i, '')
      .replace(/```\s*$/, '')
      .replace(/^%%(?!\{init)[^\n]*\n?/gm, '')
      .trim()

    const uniqueId = `mermaid-${id}-${++renderSequence}`

    try {
      mermaid.render(uniqueId, clean)
        .then(({ svg }) => {
          if (!isCancelled && ref.current) {
            ref.current.innerHTML = svg

            const svgEl = ref.current.querySelector('svg')
            if (svgEl) {
              applySvgDimensions(svgEl, zoom)
            }
          }
        })
        .catch(e => {
          if (!isCancelled) {
            setErr(e.message || String(e))
          }
        })
    } catch (e) {
      if (!isCancelled) {
        setErr(e.message || String(e))
      }
    }

    return () => {
      isCancelled = true
    }
  }, [code, id])

  useEffect(() => {
    if (!ref.current) return
    const svgEl = ref.current.querySelector('svg')
    if (svgEl) {
      applySvgDimensions(svgEl, zoom)
    }
  }, [zoom])

  const handleZoom = (delta) => {
    setZoom(prev => {
      const curr = prev === 'fit' ? 1.0 : prev
      const next = Math.max(0.6, Math.min(2.5, Math.round((curr + delta) * 100) / 100))
      return next
    })
  }

  if (err) {
    return (
      <div className={styles.error}>
        <p className={styles.errorLabel}>Diagram Render Notice</p>
        <pre className={styles.errorPre}>{err}</pre>
      </div>
    )
  }

  return (
    <div className={styles.chartContainer}>
      <div className={styles.chartControlsRow}>
        <span className={styles.zoomIndicator}>
          Scale: {zoom === 'fit' ? 'Fit to Screen' : `${Math.round(zoom * 100)}%`}
        </span>
        <div className={styles.zoomButtonGroup}>
          <button
            type="button"
            className={styles.zoomBtn}
            onClick={() => handleZoom(-0.15)}
            title="Zoom Out"
          >
            −
          </button>
          <button
            type="button"
            className={zoom === 1.0 ? styles.zoomBtnActive : styles.zoomBtn}
            onClick={() => setZoom(1.0)}
            title="100% Readable Actual Size"
          >
            100%
          </button>
          <button
            type="button"
            className={styles.zoomBtn}
            onClick={() => handleZoom(0.15)}
            title="Zoom In"
          >
            +
          </button>
          <button
            type="button"
            className={zoom === 'fit' ? styles.zoomBtnActive : styles.zoomBtn}
            onClick={() => setZoom('fit')}
            title="Fit to Screen Width"
          >
            Fit
          </button>
        </div>
      </div>
      <div className={styles.chartScrollArea}>
        <div ref={ref} className={styles.chart} />
      </div>
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────
   Tab 1: Methodology Flowchart
   Research Corpus → Main Methodological Approaches → Representative Papers
───────────────────────────────────────────────────────────── */

function MethodologyExplorer({ data }) {
  const [expandedCats, setExpandedCats] = useState({})
  const categories = data?.categories || []
  const corpusName = data?.corpus_name || 'Evaluated Research Corpus'
  const isInsufficient = Boolean(data?.is_insufficient)

  const toggleCat = (catId) => {
    setExpandedCats(prev => ({
      ...prev,
      [catId]: !prev[catId],
    }))
  }

  if (!categories.length) {
    return (
      <div className={styles.noData} style={{ padding: '24px', textAlign: 'center', background: '#FEF2F2', border: '1px solid #FECACA', borderRadius: '8px' }}>
        <h4 style={{ color: '#991B1B', margin: '0 0 8px 0', fontSize: '15px' }}>
          Diagnostic Assessment: Scope Relevance Exclusions
        </h4>
        <p style={{ color: '#7F1D1D', margin: 0, fontSize: '13px', lineHeight: '1.5' }}>
          No directly or partially relevant papers identified for <strong>"{corpusName}"</strong>.
          Unrelated or off-topic publications have been strictly excluded to prevent corpus contamination.
          Please ingest papers directly aligned with this research topic.
        </p>
      </div>
    )
  }

  return (
    <div className={styles.explorerRoot}>
      <div className={styles.hierarchyNav}>
        <span className={styles.hierarchyStep}>
          Corpus: <strong>{corpusName}</strong>
        </span>

        <span className={styles.hierarchyArrow}>→</span>

        <span className={styles.hierarchyStep}>
          {isInsufficient
            ? 'Dimensional Exclusion Groups'
            : 'Methodological Approaches'} (
          <strong>{categories.length}</strong>)
        </span>

        <span className={styles.hierarchyArrow}>→</span>

        <span className={styles.hierarchyStep}>
          {isInsufficient
            ? 'Excluded Papers & Exclusion Reasons'
            : 'Representative Papers'}
        </span>
      </div>

      <div className={styles.categoriesGrid}>
        {categories.map((cat, idx) => {
          const isExpanded = !!expandedCats[cat.id || idx]
          const repPapers = cat.representative_papers || []
          const allPapers = cat.all_papers || repPapers
          const shownPapers = isExpanded
            ? allPapers
            : repPapers.slice(0, 2)

          const hasMore = allPapers.length > 2

          return (
            <div
              key={cat.id || idx}
              className={styles.categoryCard}
            >
              <div className={styles.categoryHeader}>
                <div className={styles.categoryTitleWrap}>
                  <span className={styles.categoryIndex}>
                    0{idx + 1}
                  </span>

                  <h4 className={styles.categoryName}>
                    {cat.name}
                  </h4>
                </div>

                <span className={styles.paperCountBadge}>
                  {cat.paper_count}{' '}
                  {cat.paper_count === 1
                    ? (isInsufficient ? 'Excluded Paper' : 'Paper')
                    : (isInsufficient ? 'Excluded Papers' : 'Papers')}
                </span>
              </div>

              <p className={styles.categoryDesc}>
                {cat.description}
              </p>

              <div className={styles.papersSection}>
                <div className={styles.papersSectionHeader}>
                  <span className={styles.papersSectionLabel}>
                    {isInsufficient
                      ? 'Excluded Papers & Reasons'
                      : isExpanded
                        ? 'All Grouped Papers'
                        : 'Representative Papers'}
                  </span>
                </div>

                <div className={styles.papersList}>
                  {shownPapers.map((p, pIdx) => (
                    <div
                      key={p.paper_id || pIdx}
                      className={styles.paperCard}
                    >
                      <div className={styles.paperCardTop}>
                        <span className={styles.paperIdTag}>
                          {p.paper_id}
                        </span>

                        <span className={styles.paperTitle}>
                          {p.title}
                        </span>
                      </div>

                      {p.detail && (
                        <p className={styles.paperDetail}>
                          {p.detail}
                        </p>
                      )}

                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: '6px' }}>
                        {p.model_architecture && !p.model_architecture.toLowerCase().includes('not specified') && (
                          <span className={styles.paperChip} title="Model Architecture">
                            Arch: {p.model_architecture}
                          </span>
                        )}
                        {p.learning_paradigm && !p.learning_paradigm.toLowerCase().includes('not specified') && (
                          <span className={styles.paperChip} title="Learning Paradigm">
                            {p.learning_paradigm}
                          </span>
                        )}
                        {p.data_modality && !p.data_modality.toLowerCase().includes('not specified') && (
                          <span className={styles.paperChip} title="Data Modality">
                            {p.data_modality}
                          </span>
                        )}
                        {p.research_design && !p.research_design.toLowerCase().includes('not specified') && (
                          <span className={styles.paperChip} title="Research Design">
                            {p.research_design}
                          </span>
                        )}
                        {p.relevance_level && (
                          <span
                            className={styles.paperChip}
                            style={{
                              borderColor: p.relevance_level === 'Directly Relevant' ? '#86EFAC' : '#FDE047',
                              color: p.relevance_level === 'Directly Relevant' ? '#166534' : '#854D0E',
                            }}
                          >
                            {p.relevance_level}
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                {hasMore && (
                  <button
                    className={styles.expandPapersBtn}
                    onClick={() => toggleCat(cat.id || idx)}
                  >
                    {isExpanded
                      ? 'Show fewer papers ↑'
                      : `+${allPapers.length - 2} more papers (Click to view all) ↓`}
                  </button>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────
   Tab 2: Thematic Clusters
   3-6 Major Research Themes, Explanation, Paper Count, Rep Papers
───────────────────────────────────────────────────────────── */

function ThematicClustersExplorer({ data }) {
  const [expandedThemes, setExpandedThemes] = useState({})
  const themes = data?.themes || []
  const topic = data?.corpus_topic || 'Thematic Scope'
  const isInsufficient = Boolean(data?.is_insufficient)

  const toggleTheme = (themeId) => {
    setExpandedThemes(prev => ({
      ...prev,
      [themeId]: !prev[themeId],
    }))
  }

  if (!themes.length) {
    return (
      <div className={styles.noData} style={{ padding: '24px', textAlign: 'center', background: '#FFFBEB', border: '1px solid #FDE68A', borderRadius: '8px' }}>
        <h4 style={{ color: '#92400E', margin: '0 0 8px 0', fontSize: '15px' }}>
          Diagnostic Assessment: Thematic Clusters
        </h4>
        <p style={{ color: '#78350F', margin: 0, fontSize: '13px', lineHeight: '1.5' }}>
          No coherent multi-paper thematic clusters (minimum 2 publications sharing problem characteristics)
          could be established for <strong>"{topic}"</strong>.
          Individual publications are not forced into isolated singleton clusters.
        </p>
      </div>
    )
  }

  return (
    <div className={styles.explorerRoot}>
      <div className={styles.hierarchyNav}>
        <span className={styles.hierarchyStep}>
          Corpus Scope: <strong>{topic}</strong>
        </span>

        <span className={styles.hierarchyArrow}>·</span>

        <span className={styles.hierarchyStep}>
          {isInsufficient
            ? 'Topical Divergence & Exclusion Groups'
            : 'Major Research Themes'} (
          <strong>{themes.length}</strong>)
        </span>
      </div>

      <div className={styles.themesGrid}>
        {themes.map((th, idx) => {
          const isExpanded = !!expandedThemes[th.id || idx]
          const papers = th.papers || []
          const repPapers = th.representative_papers || []

          return (
            <div
              key={th.id || idx}
              className={styles.themeCard}
            >
              <div className={styles.themeTopRow}>
                <div className={styles.themeTitleWrap}>
                  <span className={styles.themeBullet}>●</span>

                  <h4 className={styles.themeTitle}>
                    {th.title || th.theme_title}
                  </h4>
                </div>

                <span className={styles.themeBadge}>
                  {th.paper_count}{' '}
                  {th.paper_count === 1 ? 'Paper' : 'Papers'}
                </span>
              </div>

              <p className={styles.themeDesc}>
                {th.description}
              </p>

              <div className={styles.themeChipsRow}>
                <span className={styles.chipsLabel}>
                  Papers:
                </span>

                {repPapers.map((rp, rIdx) => {
                  const pid = typeof rp === 'object'
                    ? rp.paper_id
                    : rp

                  return (
                    <span
                      key={pid || rIdx}
                      className={styles.paperChip}
                    >
                      [{pid}]
                    </span>
                  )
                })}
              </div>

              <button
                className={styles.themeActionBtn}
                onClick={() => toggleTheme(th.id || idx)}
              >
                {isExpanded
                  ? 'Hide Associated Papers & Findings ↑'
                  : 'View Associated Papers & Research Findings ↓'}
              </button>

              {isExpanded && papers.length > 0 && (
                <div className={styles.themeDetailDrawer}>
                  {papers.map((p, pIdx) => (
                    <div
                      key={p.paper_id || pIdx}
                      className={styles.themeDetailItem}
                    >
                      <div className={styles.themeDetailItemHeader}>
                        <span className={styles.citationTag}>
                          [{p.paper_id}]
                        </span>

                        <span className={styles.themeDetailItemTitle}>
                          {p.title}
                        </span>
                      </div>

                      {p.finding && (
                        <p className={styles.themeFindingText}>
                          <strong>Finding / Focus:</strong>{' '}
                          {p.finding}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────
   Tab 3: Research Gaps
   Observed Limitations → Research Gaps → Future Opportunities
───────────────────────────────────────────────────────────── */

function ResearchGapsExplorer({ data }) {
  const [expandedGaps, setExpandedGaps] = useState({})
  const gaps = data?.gaps || []

  const toggleGap = (gapId) => {
    setExpandedGaps(prev => ({
      ...prev,
      [gapId]: !prev[gapId],
    }))
  }

  if (!gaps.length) {
    return (
      <div className={styles.noData} style={{ padding: '24px', textAlign: 'center', background: '#FFF1F2', border: '1px solid #FECDD3', borderRadius: '8px' }}>
        <h4 style={{ color: '#881337', margin: '0 0 8px 0', fontSize: '15px' }}>
          Diagnostic Assessment: Research Gaps & Frontiers
        </h4>
        <p style={{ color: '#9F1239', margin: 0, fontSize: '13px', lineHeight: '1.5' }}>
          No verifiable author-stated limitations or cross-paper synthesized gaps identified.
          Unsubstantiated or speculative claims are strictly excluded to preserve scientific accuracy.
        </p>
      </div>
    )
  }

  return (
    <div className={styles.explorerRoot}>
      <div className={styles.pipelineHeader}>
        <div className={styles.pipelineStage}>
          <span className={styles.stageNumber}>Stage 1</span>
          <span className={styles.stageTitle}>
            Observed Limitations
          </span>
        </div>

        <span className={styles.pipelineArrow}>→</span>

        <div className={styles.pipelineStage}>
          <span className={styles.stageNumber}>Stage 2</span>
          <span className={styles.stageTitle}>
            Research Gaps
          </span>
        </div>

        <span className={styles.pipelineArrow}>→</span>

        <div className={styles.pipelineStage}>
          <span className={styles.stageNumber}>Stage 3</span>
          <span className={styles.stageTitle}>
            Future Opportunities
          </span>
        </div>
      </div>

      <div className={styles.gapsList}>
        {gaps.map((gap, idx) => {
          const gType = String(gap.type || '').toLowerCase()
          const isCorpus = gType.includes('corpus') || gType.includes('retrieval')
          const isSynth = gType.includes('synth')
          const isFutureWork = gType.includes('future')
          const isInferred = gType.includes('infer')
          const isUnavailable =
            gType.includes('unavail') ||
            gType.includes('insufficient') ||
            gap.has_limitation === false

          const isExpanded = !!expandedGaps[gap.gap_id || idx]
          const supporting = gap.supporting_papers || []

          return (
            <div
              key={gap.gap_id || idx}
              className={styles.gapCard}
            >
              <div className={styles.gapHeader}>
                <div className={styles.gapHeaderLeft}>
                  <span className={styles.gapIdTag}>
                    {gap.gap_id}
                  </span>

                  <span
                    className={
                      isCorpus
                        ? styles.neutralBadge
                        : isSynth
                          ? styles.synthBadge
                          : isFutureWork
                            ? styles.futureBadge
                            : isInferred
                              ? styles.inferredBadge
                              : isUnavailable
                                ? styles.neutralBadge
                                : styles.authorBadge
                    }
                  >
                    {gap.category_label ||
                      (isCorpus
                        ? 'Corpus Retrieval Gap'
                        : isSynth
                          ? 'Evidence-Supported Synthesized Gap'
                          : isFutureWork
                            ? 'Author-Stated Future Work'
                            : isInferred
                              ? 'Inferred Cross-Domain Opportunity'
                              : isUnavailable
                                ? 'Unavailable Evidence'
                                : 'Author-Stated Limitation')}
                  </span>

                  <h4 className={styles.gapTitle}>
                    {gap.title}
                  </h4>
                </div>
              </div>

              <div className={styles.gapStagesGrid}>
                <div className={styles.gapStageBox}>
                  <div className={styles.gapStageLabelRow}>
                    <span className={styles.limitationDot}>●</span>

                    <span className={styles.gapStageBoxLabel}>
                      {gap.category_label ||
                        (isCorpus
                          ? 'Observed Dimensional Deficit'
                          : isFutureWork
                            ? 'Author-Stated Future Work'
                            : isSynth
                              ? 'Cross-Paper Evidence'
                              : isUnavailable
                                ? 'Inspected Sections Status'
                                : 'Observed Limitation')}
                    </span>
                  </div>

                  <p className={styles.gapStageBoxText}>
                    {gap.observed_limitation ||
                      gap.limitation_summary ||
                      gap.explanation}
                  </p>

                  {!isCorpus && supporting.length > 0 && (
                    <div className={styles.supportingPapersRow}>
                      <span className={styles.supportingLabel}>
                        Cited In:
                      </span>

                      {supporting.map((pid, sIdx) => (
                        <span
                          key={pid || sIdx}
                          className={styles.supportingChip}
                        >
                          [{pid}]
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                <div className={styles.gapStageBox}>
                  <div className={styles.gapStageLabelRow}>
                    <span className={styles.gapDot}>●</span>

                    <span className={styles.gapStageBoxLabel}>
                      Research Gap Formulation
                    </span>
                  </div>

                  <p className={styles.gapStageBoxText}>
                    {gap.explanation}
                  </p>
                </div>

                <div className={styles.gapStageBox}>
                  <div className={styles.gapStageLabelRow}>
                    <span className={styles.oppDot}>●</span>

                    <span className={styles.gapStageBoxLabel}>
                      Future Opportunity
                    </span>
                  </div>

                  <p className={styles.gapStageBoxText}>
                    {gap.future_opportunity ||
                      'Develop unified frameworks mitigating reported bottlenecks.'}
                  </p>
                </div>
              </div>

              {!isCorpus && gap.evidence && gap.evidence.trim() && (
                <div className={styles.evidenceWrap}>
                  <button
                    className={styles.evidenceToggleBtn}
                    onClick={() => toggleGap(gap.gap_id || idx)}
                  >
                    {isExpanded
                      ? 'Hide Supporting Evidence Quotes ↑'
                      : 'Inspect Grounded Evidence & Quotes ↓'}
                  </button>

                  {isExpanded && (
                    <div className={styles.evidenceBox}>
                      <p className={styles.evidenceQuote}>
                        "{gap.evidence}"
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────
   Main MermaidDiagrams Component
───────────────────────────────────────────────────────────── */

export default function MermaidDiagrams({
  diagrams = {},
  diagramData = null,
}) {
  const defaultKeys = [
    'methodology_flowchart',
    'themes_mindmap',
    'research_gaps_flowchart',
  ]

  const passedKeys = Object.keys(diagrams || {}).filter(
    k => Boolean(diagrams[k] && typeof diagrams[k] === 'string' && diagrams[k].trim())
  )

  const keys = passedKeys.length > 0 && !passedKeys.some(k => defaultKeys.includes(k))
    ? passedKeys
    : defaultKeys

  const [tab, setTab] = useState(() => keys[0] || 'methodology_flowchart')
  const [viewMode, setViewMode] = useState('mermaid')

  useEffect(() => {
    if (!keys.includes(tab)) {
      setTab(keys[0] || 'methodology_flowchart')
    }
  }, [keys, tab])

  const hasDiagramData = Boolean(
    diagramData && (
      (diagramData.methodology?.categories && diagramData.methodology.categories.length > 0) ||
      (diagramData.thematic_clusters?.themes && diagramData.thematic_clusters.themes.length > 0) ||
      (diagramData.research_gaps?.gaps && diagramData.research_gaps.gaps.length > 0)
    )
  )

  const hasMermaidCode = Object.keys(diagrams || {}).some(
    k => Boolean(
      diagrams[k] &&
      typeof diagrams[k] === 'string' &&
      diagrams[k].trim()
    )
  )

  if (!hasDiagramData && !hasMermaidCode && !diagramData) {
    return (
      <div className={styles.root}>
        <div className={styles.emptyWrap}>
          <p className={styles.emptyTitle}>
            No Architecture Diagrams Available
          </p>

          <p className={styles.emptyText}>
            No diagram data is associated with the selected review or corpus.
            Ingest papers and run the synthesis pipeline to view dynamic
            architecture diagrams.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.root}>
      <div className={styles.topBar}>
        <div className={styles.tabs}>
          {keys.map(k => (
            <button
              key={k}
              type="button"
              className={tab === k ? styles.tabOn : styles.tab}
              onClick={() => setTab(k)}
            >
              {LABELS[k] || k.replace(/_/g, ' ')}
            </button>
          ))}
        </div>

        <div className={styles.viewControls}>
          <button
            type="button"
            className={
              viewMode === 'interactive'
                ? styles.viewBtnActive
                : styles.viewBtn
            }
            onClick={() => setViewMode('interactive')}
            title="Interactive hierarchical cards with expandable papers & details"
          >
            Interactive Cards
          </button>

          <button
            type="button"
            className={
              viewMode === 'mermaid'
                ? styles.viewBtnActive
                : styles.viewBtn
            }
            onClick={() => setViewMode('mermaid')}
            title="Rendered SVG flowchart diagram"
          >
            Mermaid Diagram
          </button>
        </div>
      </div>

      <div className={styles.contentArea}>
        {viewMode === 'interactive' ? (
          <div>
            {tab === 'methodology_flowchart' && (
              <MethodologyExplorer
                data={diagramData?.methodology}
              />
            )}

            {tab === 'themes_mindmap' && (
              <ThematicClustersExplorer
                data={diagramData?.thematic_clusters}
              />
            )}

            {tab === 'research_gaps_flowchart' && (
              <ResearchGapsExplorer
                data={diagramData?.research_gaps}
              />
            )}

            {!defaultKeys.includes(tab) && (
              <p className={styles.noData}>
                Interactive cards are available for Methodology Flowchart, Thematic Clusters, and Research Gaps. Switch to Mermaid Diagram view to inspect custom architecture graphs.
              </p>
            )}
          </div>
        ) : (
          <div className={styles.chartWrap}>
            {tab === 'methodology_flowchart' && (
              <div className={styles.diagramViewHeader}>
                <div className={styles.diagramViewTitleRow}>
                  <span className={styles.diagramViewTitle}>
                    Methodological Taxonomy Flowchart
                  </span>

                  <span className={styles.diagramViewSubtitle}>
                    Research Corpus → Methodological Approaches → Research Papers
                  </span>
                </div>
              </div>
            )}

            {tab === 'themes_mindmap' && (
              <div className={styles.diagramViewHeader}>
                <div className={styles.diagramViewTitleRow}>
                  <span className={styles.diagramViewTitle}>
                    Thematic Clusters Architecture
                  </span>

                  <span className={styles.diagramViewSubtitle}>
                    Research Corpus → Shared Research Themes → Supporting Papers
                  </span>
                </div>
              </div>
            )}

            {tab === 'research_gaps_flowchart' && (
              <div className={styles.diagramViewHeader}>
                <div className={styles.diagramViewTitleRow}>
                  <span className={styles.diagramViewTitle}>
                    Research Gaps & Future Frontiers
                  </span>

                  <span className={styles.diagramViewSubtitle}>
                    Observed Limitations → Research Gaps → Future Opportunities
                  </span>
                </div>
              </div>
            )}

            {diagrams && diagrams[tab] ? (
              <Chart
                key={tab}
                code={diagrams[tab]}
                id={tab}
              />
            ) : (
              <div className={styles.noDataWrap}>
                <p className={styles.noData}>
                  No diagram code generated for this view.
                </p>
                <p className={styles.noDataSub} style={{ fontSize: '11.5px', color: '#64748B', textAlign: 'center' }}>
                  Switch to <strong>Interactive Cards</strong> to explore structured taxonomy data.
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
