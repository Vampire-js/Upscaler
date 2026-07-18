import './style.css'

const API = import.meta.env.VITE_API_URL || 'https://upscaler-krzf.onrender.com'

document.querySelector('#app').innerHTML = `
  <h1>Upscaler</h1>
  <p class="sub">4× super-resolution. Upload an image, get it back sharper.</p>

  <div class="controls">
    <label class="filebtn" for="file">Choose image</label>
    <input type="file" id="file" accept="image/*" />
    <button id="run" disabled>Upscale</button>
    <button id="download" class="ghost" disabled>Download result</button>
  </div>

  <div class="status" id="status"></div>

  <div class="row">
    <div class="card">
      <h2>Input</h2>
      <div class="imgwrap empty" id="inWrap">no image selected</div>
      <div class="meta" id="inMeta"></div>
    </div>
    <div class="card">
      <h2>Output</h2>
      <div class="imgwrap empty" id="outWrap">no result yet</div>
      <div class="meta" id="outMeta"></div>
    </div>
  </div>

  <footer>
    API: <code>${API}</code> — max input side 1024 px, max file 5 MB.
  </footer>
`

const fileInput = document.getElementById('file')
const runBtn = document.getElementById('run')
const dlBtn = document.getElementById('download')
const statusEl = document.getElementById('status')
const inWrap = document.getElementById('inWrap')
const outWrap = document.getElementById('outWrap')
const inMeta = document.getElementById('inMeta')
const outMeta = document.getElementById('outMeta')

let currentFile = null
let resultBlobUrl = null
let resultFilename = 'upscaled.png'

function setStatus(msg, { err = false, loading = false } = {}) {
  statusEl.classList.toggle('err', err)
  statusEl.innerHTML = loading ? `<span class="spin"></span>${msg}` : msg
}

function showImage(el, url) {
  el.classList.remove('empty')
  el.innerHTML = ''
  const img = document.createElement('img')
  img.src = url
  el.appendChild(img)
  return img
}

fileInput.addEventListener('change', () => {
  const f = fileInput.files[0]
  if (!f) return
  if (f.size > 5 * 1024 * 1024) {
    setStatus(`File too large: ${(f.size / 1024 / 1024).toFixed(1)} MB (max 5 MB)`, { err: true })
    return
  }
  currentFile = f
  const url = URL.createObjectURL(f)
  const img = showImage(inWrap, url)
  img.onload = () => {
    inMeta.textContent = `${f.name} — ${img.naturalWidth}×${img.naturalHeight} — ${(f.size / 1024).toFixed(1)} KB`
  }
  // reset output
  outWrap.classList.add('empty')
  outWrap.innerHTML = 'no result yet'
  outMeta.textContent = ''
  if (resultBlobUrl) { URL.revokeObjectURL(resultBlobUrl); resultBlobUrl = null }
  dlBtn.disabled = true
  runBtn.disabled = false
  setStatus('')
})

runBtn.addEventListener('click', async () => {
  if (!currentFile) return
  runBtn.disabled = true
  dlBtn.disabled = true
  setStatus('Upscaling on CPU (this can take 10–30s the first time)…', { loading: true })
  const t0 = performance.now()

  try {
    const fd = new FormData()
    fd.append('file', currentFile)
    const res = await fetch(`${API}/upscale`, { method: 'POST', body: fd })

    if (!res.ok) {
      let detail = res.statusText
      try { const j = await res.json(); detail = j.detail || detail } catch {}
      throw new Error(`${res.status}: ${detail}`)
    }

    const blob = await res.blob()
    if (resultBlobUrl) URL.revokeObjectURL(resultBlobUrl)
    resultBlobUrl = URL.createObjectURL(blob)
    resultFilename = currentFile.name.replace(/\.[^.]+$/, '') + '_x4.png'

    const img = showImage(outWrap, resultBlobUrl)
    img.onload = () => {
      outMeta.textContent = `${img.naturalWidth}×${img.naturalHeight} — ${(blob.size / 1024).toFixed(1)} KB`
    }

    const dt = ((performance.now() - t0) / 1000).toFixed(1)
    setStatus(`Done in ${dt}s.`)
    dlBtn.disabled = false
  } catch (e) {
    setStatus(e.message || String(e), { err: true })
  } finally {
    runBtn.disabled = false
  }
})

dlBtn.addEventListener('click', () => {
  if (!resultBlobUrl) return
  const a = document.createElement('a')
  a.href = resultBlobUrl
  a.download = resultFilename
  a.click()
})
