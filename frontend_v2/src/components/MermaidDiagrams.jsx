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
  methodology_flowchart:   'Methodology Flowchart',
  themes_mindmap:          'Thematic Clusters',
  research_gaps_flowchart: 'Research Gaps',
}

let renderSequence = 0

function Chart({ code, id }) {
  const ref = useRef()
  const [err, setErr] = useState(null)

  useEffect(() => {
    let isCancelled = false
    if (!ref.current || !code) return
    let cancelled = false
    setErr(null)
    const clean = code
      .replace(/^```mermaid\s*/i, '')
      .replace(/```\s*$/, '')
      .replace(/^%%(?!\{init)[^\n]*\n?/gm, '')
      .trim()

    const uniqueId = mmd--
    try {
      mermaid.render(uniqueId, clean)
        .then(({ svg }) => {
          if (!isCancelled && ref.current) {
            ref.current.innerHTML = svg
            const svgEl = ref.current.querySelector('svg')
            if (svgEl) {
              svgEl.style.maxWidth = '100%'
              svgEl.style.height = 'auto'
            }
          }
        })
        .catch(e => {
          if (!isCancelled) setErr(e.message || String(e))
        })
    } catch (e) {
      if (!isCancelled) setErr(e.message || String(e))
    }

    return () => { isCancelled = true }
  }, [code, id])

  if (err) {
    return (
      <div className={styles.error}>
        <p className={styles.errorLabel}>Diagram Render Notice</p>
        <pre className={styles.errorPre}>{err}</pre>
      </div>
    )
  }
  return <div ref={ref} className={styles.chart} />
}

/* ─────────────────────────────────────────────────────────────
   Tab 1: Methodology Flowchart (Compact Hierarchy)
   Research Corpus → Main Methodological Approaches → Representative Papers
───────────────────────────────────────────────────────────── */
function MethodologyExplorer({ data }) {
  const [expandedCats, setExpandedCats] = useState({})
  const categories = data?.categories || []
  const corpusName = data?.corpus_name || 'Evaluated Research Corpus'
  const isInsufficient = Boolean(data?.is_insufficient)

  const toggleCat = (catId) => {
    setExpandedCats(prev => ({ ...prev, [catId]: !prev[catId] }))
  }

  if (!categories.length) {
    return <p className={styles.noData}>No methodology taxonomy available for this corpus.</p>
  }

  return (
    <div className={styles.explorerRoot}>
      {/* Hierarchy Header */}
      <div className={styles.hierarchyNav}>
        <span className={styles.hierarchyStep}>Corpus: <strong>{corpusName}</strong></span>
        <span className={styles.hierarchyArrow}>→</span>
        <span className={styles.hierarchyStep}>
          {isInsufficient ? 'Dimensional Exclusion Groups' : 'Methodological Approaches'} (<strong>{categories.length}</strong>)
        </span>
        <span className={styles.hierarchyArrow}>→</span>
        <span className={styles.hierarchyStep}>
          {isInsufficient ? 'Excluded Papers & Exclusion Reasons' : 'Representative Papers'}
        </span>
      </div>

      <div className={styles.categoriesGrid}>
        {categories.map((cat, idx) => {
          const isExpanded = !!expandedCats[cat.id || idx]
          const repPapers = cat.representative_papers || []
          const allPapers = cat.all_papers || repPapers
          const shownPapers = isExpanded ? allPapers : repPapers.slice(0, 2)
          const hasMore = allPapers.length > 2

          return (
            <div key={cat.id || idx} className={styles.categoryCard}>
              <div className={styles.categoryHeader}>
                <div className={styles.categoryTitleWrap}>
                  <span className={styles.categoryIndex}>0{idx + 1}</span>
                  <h4 className={styles.categoryName}>{cat.name}</h4>
                </div>
                <span className={styles.paperCountBadge}>
                  {cat.paper_count} {cat.paper_count === 1 ? (isInsufficient ? 'Excluded Paper' : 'Paper') : (isInsufficient ? 'Excluded Papers' : 'Papers')}
                </span>
              </div>

              <p className={styles.categoryDesc}>{cat.description}</p>

              <div className={styles.papersSection}>
                <div className={styles.papersSectionHeader}>
                  <span className={styles.papersSectionLabel}>
                    {isInsufficient ? 'Excluded Papers & Reasons' : isExpanded ? 'All Grouped Papers' : 'Representative Papers'}
                  </span>
                </div>

                <div className={styles.papersList}>
                  {shownPapers.map((p, pIdx) => (
                    <div key={p.paper_id || pIdx} className={styles.paperCard}>
                      <div className={styles.paperCardTop}>
                        <span className={styles.paperIdTag}>{p.paper_id}</span>
                        <span className={styles.paperTitle}>{p.title}</span>
                      </div>
                      {p.detail && (
                        <p className={styles.paperDetail}>{p.detail}</p>
                      )}
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
    setExpandedThemes(prev => ({ ...prev, [themeId]: !prev[themeId] }))
  }

  if (!themes.length) {
    return <p className={styles.noData}>No thematic clusters generated for this corpus.</p>
  }

  return (
    <div className={styles.explorerRoot}>
      <div className={styles.hierarchyNav}>
        <span className={styles.hierarchyStep}>Corpus Scope: <strong>{topic}</strong></span>
        <span className={styles.hierarchyArrow}>·</span>
        <span className={styles.hierarchyStep}>
          {isInsufficient ? 'Topical Divergence & Exclusion Groups' : 'Major Research Themes'} (<strong>{themes.length}</strong>)
        </span>
      </div>

      <div className={styles.themesGrid}>
        {themes.map((th, idx) => {
          const isExpanded = !!expandedThemes[th.id || idx]
          const papers = th.papers || []
          const repPapers = th.representative_papers || []

          return (
            <div key={th.id || idx} className={styles.themeCard}>
              <div className={styles.themeTopRow}>
                <div className={styles.themeTitleWrap}>
                  <span className={styles.themeBullet}>●</span>
                  <h4 className={styles.themeTitle}>{th.title || th.theme_title}</h4>
                </div>
                <span className={styles.themeBadge}>
                  {th.paper_count} {th.paper_count === 1 ? 'Paper' : 'Papers'}
                </span>
              </div>

              <p className={styles.themeDesc}>{th.description}</p>

              <div className={styles.themeChipsRow}>
                <span className={styles.chipsLabel}>Papers:</span>
                {repPapers.map((rp, rIdx) => {
                  const pid = typeof rp === 'object' ? rp.paper_id : rp
                  return (
                    <span key={pid || rIdx} className={styles.paperChip}>
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
                    <div key={p.paper_id || pIdx} className={styles.themeDetailItem}>
                      <div className={styles.themeDetailItemHeader}>
                        <span className={styles.citationTag}>[{p.paper_id}]</span>
                        <span className={styles.themeDetailItemTitle}>{p.title}</span>
                      </div>
                      {p.finding && (
                        <p className={styles.themeFindingText}>
                          <strong>Finding / Focus:</strong> {p.finding}
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
  const topic = data?.corpus_topic || 'Research Frontier'

  const toggleGap = (gapId) => {
    setExpandedGaps(prev => ({ ...prev, [gapId]: !prev[gapId] }))
  }

  if (!gaps.length) {
    return <p className={styles.noData}>No research gaps synthesized for this corpus.</p>
  }

  return (
    <div className={styles.explorerRoot}>
      {/* 3-Stage Pipeline Header */}
      <div className={styles.pipelineHeader}>
        <div className={styles.pipelineStage}>
          <span className={styles.stageNumber}>Stage 1</span>
          <span className={styles.stageTitle}>Observed Limitations</span>
        </div>
        <span className={styles.pipelineArrow}>→</span>
        <div className={styles.pipelineStage}>
          <span className={styles.stageNumber}>Stage 2</span>
          <span className={styles.stageTitle}>Research Gaps</span>
        </div>
        <span className={styles.pipelineArrow}>→</span>
        <div className={styles.pipelineStage}>
          <span className={styles.stageNumber}>Stage 3</span>
          <span className={styles.stageTitle}>Future Opportunities</span>
        </div>
      </div>

      <div className={styles.gapsList}>
        {gaps.map((gap, idx) => {
          const gType = String(gap.type || '').toLowerCase()
          const isCorpus = gType.includes('corpus') || gType.includes('retrieval')
          const isSynth = gType.includes('synth')
          const isFutureWork = gType.includes('future')
          const isInferred = gType.includes('infer')
          const isUnavailable = gType.includes('unavail') || gType.includes('insufficient') || gap.has_limitation === false
          const isExpanded = !!expandedGaps[gap.gap_id || idx]
          const supporting = gap.supporting_papers || []

          return (
            <div key={gap.gap_id || idx} className={styles.gapCard}>
              <div className={styles.gapHeader}>
                <div className={styles.gapHeaderLeft}>
                  <span className={styles.gapIdTag}>{gap.gap_id}</span>
                  <span className={isCorpus ? styles.neutralBadge : isSynth ? styles.synthBadge : isFutureWork ? styles.futureBadge : isInferred ? styles.inferredBadge : isUnavailable ? styles.neutralBadge : styles.authorBadge}>
                    {isCorpus ? 'Corpus Retrieval Gap' : isSynth ? 'Evidence-Supported Synthesized Gap' : isFutureWork ? 'Author-Stated Future Work' : isInferred ? 'Inferred Cross-Domain Opportunity' : isUnavailable ? 'Unavailable Evidence' : 'Author-Stated Limitation'}
                  </span>
                  <h4 className={styles.gapTitle}>{gap.title}</h4>
                </div>
              </div>

              {/* 3-Stage Inline Progression */}
              <div className={styles.gapStagesGrid}>
                {/* Stage 1: Limitation */}
                <div className={styles.gapStageBox}>
                  <div className={styles.gapStageLabelRow}>
                    <span className={styles.limitationDot}>●</span>
                    <span className={styles.gapStageBoxLabel}>
                      {isCorpus ? 'Observed Dimensional Deficit' : isFutureWork ? 'Author-Stated Future Work' : isSynth ? 'Cross-Paper Evidence' : isUnavailable ? 'Inspected Sections Status' : 'Observed Limitation'}
                    </span>
                  </div>
                  <p className={styles.gapStageBoxText}>
                    {gap.observed_limitation || gap.limitation_summary || gap.explanation}
                  </p>
                  {!isCorpus && supporting.length > 0 && (
                    <div className={styles.supportingPapersRow}>
                      <span className={styles.supportingLabel}>Cited In:</span>
                      {supporting.map((pid, sIdx) => (
                        <span key={pid || sIdx} className={styles.supportingChip}>
                          [{pid}]
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* Stage 2: Gap Explanation */}
                <div className={styles.gapStageBox}>
                  <div className={styles.gapStageLabelRow}>
                    <span className={styles.gapDot}>●</span>
                    <span className={styles.gapStageBoxLabel}>Research Gap Formulation</span>
                  </div>
                  <p className={styles.gapStageBoxText}>{gap.explanation}</p>
                </div>

                {/* Stage 3: Future Opportunity */}
                <div className={styles.gapStageBox}>
                  <div className={styles.gapStageLabelRow}>
                    <span className={styles.oppDot}>●</span>
                    <span className={styles.gapStageBoxLabel}>Future Opportunity</span>
                  </div>
                  <p className={styles.gapStageBoxText}>
                    {gap.future_opportunity || 'Develop unified frameworks mitigating reported bottlenecks.'}
                  </p>
                </div>
              </div>

              {/* Evidence Expansion Drawer (Only for real author quotes, never for corpus-level gaps) */}
              {!isCorpus && gap.evidence && gap.evidence.trim() && (
                <div className={styles.evidenceWrap}>
                  <button
                    className={styles.evidenceToggleBtn}
                    onClick={() => toggleGap(gap.gap_id || idx)}
                  >
                    {isExpanded ? 'Hide Supporting Evidence Quotes ↑' : 'Inspect Grounded Evidence & Quotes ↓'}
                  </button>
                  {isExpanded && (
                    <div className={styles.evidenceBox}>
                      <p className={styles.evidenceQuote}>"{gap.evidence}"</p>
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
export default function MermaidDiagrams({ diagrams = {}, diagramData = null }) {
  const keys = ['methodology_flowchart', 'themes_mindmap', 'research_gaps_flowchart']
  const [tab, setTab] = useState('methodology_flowchart')
  const [viewMode, setViewMode] = useState('interactive') // 'interactive' | 'mermaid'

  const hasDiagramData = diagramData && (
    (diagramData.methodology && diagramData.methodology.categories?.length > 0) ||
    (diagramData.thematic_clusters && diagramData.thematic_clusters.themes?.length > 0) ||
    (diagramData.research_gaps && diagramData.research_gaps.gaps?.length > 0)
  )

  const hasMermaidCode = Object.keys(diagrams || {}).some(k => Boolean(diagrams[k] && diagrams[k].trim()))

  if (!hasDiagramData && !hasMermaidCode) {
    return (
      <div className={styles.root}>
        <div className={styles.emptyWrap}>
          <p className={styles.emptyTitle}>No Architecture Diagrams Available</p>
          <p className={styles.emptyText}>
            No diagram data is associated with the selected review or corpus. Ingest papers and run the synthesis pipeline to view dynamic architecture diagrams.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.root}>
      {/* Top Header & View Controls */}
      <div className={styles.topBar}>
        <div className={styles.tabs}>
          {keys.map(k => (
            <button
              key={k}
              className={tab === k ? styles.tabOn : styles.tab}
              onClick={() => setTab(k)}
            >
              {LABELS[k] || k}
            </button>
          ))}
        </div>

        <div className={styles.viewControls}>
          <button
            className={viewMode === 'interactive' ? styles.viewBtnActive : styles.viewBtn}
            onClick={() => setViewMode('interactive')}
            title="Interactive hierarchical cards with expandable papers & details"
          >
            Interactive Cards
          </button>
          <button
            className={viewMode === 'mermaid' ? styles.viewBtnActive : styles.viewBtn}
            onClick={() => setViewMode('mermaid')}
            title="Rendered SVG flowchart diagram"
          >
            Mermaid Diagram
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className={styles.contentArea}>
        {viewMode === 'interactive' && hasDiagramData ? (
          <div>
            {tab === 'methodology_flowchart' && (
              <MethodologyExplorer data={diagramData?.methodology} />
            )}
            {tab === 'themes_mindmap' && (
              <ThematicClustersExplorer data={diagramData?.thematic_clusters} />
            )}
            {tab === 'research_gaps_flowchart' && (
              <ResearchGapsExplorer data={diagramData?.research_gaps} />
            )}
          </div>
        ) : (
          <div className={styles.chartWrap}>
            {tab === 'methodology_flowchart' && (
              <div className={styles.diagramViewHeader}>
                <div className={styles.diagramViewTitleRow}>
                  <span className={styles.diagramViewTitle}>Methodological Taxonomy Flowchart</span>
                  <span className={styles.diagramViewSubtitle}>
                    Research Corpus → Methodological Approaches → Research Papers
                  </span>
                </div>
              </div>
            )}
            {tab === 'themes_mindmap' && (
              <div className={styles.diagramViewHeader}>
                <div className={styles.diagramViewTitleRow}>
                  <span className={styles.diagramViewTitle}>Thematic Clusters Architecture</span>
                  <span className={styles.diagramViewSubtitle}>
                    Research Corpus → Shared Research Themes → Supporting Papers
                  </span>
                </div>
              </div>
            )}
            {tab === 'research_gaps_flowchart' && (
              <div className={styles.diagramViewHeader}>
                <div className={styles.diagramViewTitleRow}>
                  <span className={styles.diagramViewTitle}>Research Gaps & Future Frontiers</span>
                  <span className={styles.diagramViewSubtitle}>
                    Observed Limitations → Research Gaps → Future Opportunities
                  </span>
                </div>
              </div>
            )}
            {diagrams[tab] ? (
              <Chart key={tab} code={diagrams[tab]} id={tab} />
            ) : (
              <p className={styles.noData}>No diagram code generated for this view.</p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
