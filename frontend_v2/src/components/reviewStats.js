export function extractionStats(corpus = []) {
  const active = corpus.filter(p => p.active !== false)
  const llm = active.filter(p => (p.generation_mode || p.schema?.generation_mode) === 'GENUINE_LLM').length
  const heuristic = active.filter(p => (p.generation_mode || p.schema?.generation_mode) === 'HEURISTIC_FALLBACK').length
  const total = active.length
  return { total, llm, heuristic, unknown: total - llm - heuristic,
    percent: total ? Math.round(llm / total * 100) : null }
}

export function citationStats(content = '', corpus = []) {
  const ids = new Set(corpus.filter(p => p.active !== false).map(p => p.paper_id || p.arxiv_id))
  const text = content.replace(/```[\s\S]*?```/g, '')
  const tags = [...text.matchAll(/\[([^\]\n]+)\]/g)].flatMap(m =>
    m[1].match(/\b\d{4}\.\d{4,5}(?:v\d+)?\b/g) || (ids.has(m[1]) ? [m[1]] : []))
  const valid = tags.filter(id => ids.has(id) || ids.has(id.replace(/v\d+$/, ''))).length
  return { total: tags.length, valid, percent: tags.length && ids.size ? Math.round(valid / tags.length * 100) : null }
}
