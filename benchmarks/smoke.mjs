const API_BASE = process.env.API_BASE_URL || 'http://127.0.0.1:6346'
const OPENVINO_BASE = process.env.OPENVINO_BASE_URL || 'http://127.0.0.1:3012'

async function getJson(url, init) {
  const res = await fetch(url, init)
  const body = await res.text()
  if (!res.ok) {
    throw new Error(`${init?.method || 'GET'} ${url} failed with ${res.status}: ${body}`)
  }
  return body ? JSON.parse(body) : {}
}

async function waitForHealth(url, label, timeoutMs = 30000) {
  const started = Date.now()
  let lastError = null
  while (Date.now() - started < timeoutMs) {
    try {
      const payload = await getJson(url)
      console.log(`${label}_HEALTH`, JSON.stringify(payload))
      return payload
    } catch (error) {
      lastError = error
      await new Promise((resolve) => setTimeout(resolve, 1000))
    }
  }
  throw new Error(`Timed out waiting for ${label} at ${url}: ${lastError?.message || 'unknown error'}`)
}

await waitForHealth(`${OPENVINO_BASE}/health`, 'OPENVINO')
await waitForHealth(`${API_BASE}/health`, 'API')

const text = 'edumem smoke test memory: agent-memory keeps API Client on 8888 and edumem Qdrant on 6333.'
const retain = await getJson(`${API_BASE}/v1/default/banks/testbank/memories/retain`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    content: text,
    tags: ['user:smoke']
  })
})
console.log('STORE', JSON.stringify(retain))

let recalled = { results: [] }
for (let attempt = 0; attempt < 5; attempt += 1) {
  recalled = await getJson(`${API_BASE}/v1/default/banks/testbank/memories/recall`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: 'Which ports are used by API Client and edumem?',
      tags: ['user:smoke']
    })
  })

  const candidateResults = Array.isArray(recalled.results) ? recalled.results : []
  if (candidateResults.some((result) => result?.text?.includes(text))) {
    break
  }

  if (attempt < 4) {
    await new Promise((resolve) => setTimeout(resolve, 1500))
  }
}

const results = Array.isArray(recalled.results) ? recalled.results : []
console.log('RECALL_COUNT', results.length)
console.log('TOP_TEXT', results[0]?.text || JSON.stringify(results[0] || null))

if (!results.some((result) => result?.text?.includes(text))) {
  throw new Error('Stored smoke fact was not returned by recall')
}
