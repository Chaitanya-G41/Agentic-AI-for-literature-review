import { useEffect, useRef, useState } from 'react'
import mermaid from 'mermaid'
import styles from './MermaidDiagrams.module.css'

mermaid.initialize({
  startOnLoad: false,
  theme: 'base',
  themeVariables: {
    primaryColor: '#F8FAFC',
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
    if (!ref.current || !code) return
    let cancelled = false
    setErr(null)
    const clean = code
      .replace(/^```mermaid\s*/i, '')
      .replace(/```\s*$/, '')
      .replace(/%%[^\n]*/g, '')   // strip %% directives that might error
      .trim()

    mermaid.render(`mmd-${id.replace(/[^a-zA-Z0-9_-]/g, '-')}-${++renderSequence}`, clean)
      .then(({ svg }) => { if (!cancelled && ref.current) ref.current.innerHTML = svg })
      .catch(e => { if (!cancelled) setErr(e.message) })
    return () => { cancelled = true }
  }, [code, id])

  if (err) return (
    <div className={styles.error}>
      <p className={styles.errorLabel}>Diagram parse error</p>
      <pre className={styles.errorPre}>{err}</pre>
    </div>
  )
  return <div ref={ref} className={styles.chart} />
}

export default function MermaidDiagrams({ diagrams }) {
  const keys   = Object.keys(diagrams).filter(k => diagrams[k])
  const [tab, setTab] = useState(keys[0] || '')
  if (!keys.length) return null

  return (
    <div className={styles.root}>
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
      <div className={styles.chartWrap}>
        <Chart key={tab} code={diagrams[tab]} id={tab} />
      </div>
    </div>
  )
}
