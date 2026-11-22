import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

function uploadPlugin() {
  return {
    name: 'pdf-uploader',
    configureServer(server) {
      server.middlewares.use('/api/upload-pdf', (req, res) => {
        if (req.method === 'POST') {
          const targetDir = path.resolve(__dirname, '../data/pdf_corpus')
          if (!fs.existsSync(targetDir)) {
            fs.mkdirSync(targetDir, { recursive: true })
          }
          const chunks = []
          req.on('data', chunk => chunks.push(chunk))
          req.on('end', () => {
            try {
              const buffer = Buffer.concat(chunks)
              const url = new URL(req.url, 'http://localhost')
              let filename = url.searchParams.get('filename') || req.headers['x-filename'] || `uploaded_${Date.now()}.pdf`
              filename = path.basename(filename)
              const dest = path.join(targetDir, filename)
              fs.writeFileSync(dest, buffer)
              res.setHeader('Content-Type', 'application/json')
              res.end(JSON.stringify({ status: 'ok', filename, size: buffer.length }))
            } catch (err) {
              res.statusCode = 500
              res.end(JSON.stringify({ error: err.message }))
            }
          })
        } else {
          res.statusCode = 405
          res.end()
        }
      })
    }
  }
}

export default defineConfig({
  plugins: [react(), uploadPlugin()],
  server: {
    host: true,
    port: 5174,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/static': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
