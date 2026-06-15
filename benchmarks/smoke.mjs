import { createMnemosyne as createEdumem } from 'mnemosy-ai'
const m = await createEdumem({
  vectorDbUrl: 'http://127.0.0.1:6333',
  embeddingUrl: 'http://127.0.0.1:3002/embed',
  embeddingModel: 'gte-modernbert-base',
  agentId: 'agent-memory-edumem-smoke',
  collections: {
    shared: 'edumem_shared',
    private: 'edumem_private',
    profiles: 'edumem_profiles',
    skills: 'edumem_skills'
  },
  enableGraph: false,
  enableBroadcast: false,
  enableExtraction: false
})
const text = 'edumem smoke test memory: agent-memory keeps Hindsight on 8888 and edumem Qdrant on 6333.'
const stored = await m.store({ text, memoryType: 'fact', importance: 0.7 })
console.log('STORE', JSON.stringify(stored))
const recalled = await m.recall({ query: 'Which ports are used by Hindsight and edumem?', limit: 5 })
console.log('RECALL_COUNT', recalled.length)
console.log('TOP_TEXT', recalled[0]?.text || recalled[0]?.entry?.text || JSON.stringify(recalled[0] || null))
