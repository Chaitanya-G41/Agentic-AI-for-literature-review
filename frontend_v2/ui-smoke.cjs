// Requires a temporary @playwright/test install and Vite on port 5178.
const { chromium, expect } = require('@playwright/test')
const fs = require('node:fs')
const path = require('node:path')
const os = require('node:os')

async function main() {
  const browser = await chromium.launch({ channel: 'chrome', headless: true })
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
    const ids = JSON.parse(fs.readFileSync('../data/comparative_analysis.json', 'utf8')).paper_ids
    const papers = ids.map((id, i) => {
      const schema = JSON.parse(fs.readFileSync(`../data/extracted_schemas/${id}_schema.json`, 'utf8'))
      return { paper_id: id, title: schema.paper_title, methodology: schema.methodology_summary,
        generation_mode: i === 0 ? 'HEURISTIC_FALLBACK' : 'GENUINE_LLM', figures: [] }
    })
    const content = fs.readFileSync('../data/reports/literature_review_report.md', 'utf8')
    await page.route('**/api/**', route => {
      const url = new URL(route.request().url())
      route.fulfill({ json: url.pathname === '/api/corpus' ? { papers } :
        url.pathname === '/api/report' ? { content, title: 'Brain Tumor Segmentation', diagrams: {} } : {} })
    })
    await page.addInitScript(() => localStorage.setItem('monograph_session_active', 'true'))
    await page.goto('http://127.0.0.1:5178')
    const architecture = page.getByRole('button', { name: 'System Architecture' })
    await architecture.click()
    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible()
    try { await expect(dialog.locator('svg.flowchart')).toBeVisible() }
    catch { throw new Error(await dialog.innerText()) }
    await expect(dialog.getByRole('cell', { name: 'PaperSchema', exact: true })).toBeVisible()
    await page.keyboard.press('Shift+Tab')
    expect(await page.evaluate(() => document.querySelector('dialog').contains(document.activeElement))).toBe(true)
    const screenshots = path.join(os.tmpdir(), 'nlp05-ui-validation')
    fs.mkdirSync(screenshots, { recursive: true })
    await page.screenshot({ path: path.join(screenshots, 'architecture.png') })
    await page.keyboard.press('Escape')
    await expect(dialog).toHaveCount(0)
    await expect(architecture).toBeFocused()
    await page.getByRole('button', { name: 'Evaluation Rubric' }).click()
    await expect(page.getByRole('dialog').getByText('83%', { exact: true })).toBeVisible()
    await expect(page.getByRole('dialog').getByRole('heading', { name: 'Hallucination Rate' })).toBeVisible()
    await page.screenshot({ path: path.join(screenshots, 'evaluation.png') })
    await page.getByRole('button', { name: 'Close dialog' }).click()
    await expect(page.getByText('5/6 LLM · 1/6 Heuristic (83% LLM)')).toBeVisible()
    await page.setViewportSize({ width: 390, height: 844 })
    await page.getByRole('button', { name: 'Evaluation Rubric' }).click()
    const bounds = await page.getByRole('dialog').boundingBox()
    expect(bounds.width).toBeLessThanOrEqual(390)
    await page.keyboard.press('Escape')
    expect(errors).toEqual([])
    console.log(`UI smoke passed: diagram, statistics, modal focus/Escape, mobile dialog. Screenshots: ${screenshots}`)
  } finally {
    await browser.close()
  }
}
main().catch(error => { console.error(error); process.exitCode = 1 })
