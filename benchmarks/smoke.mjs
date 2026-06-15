import { createMnemosyne } from 'mnemosy-ai'
const m = await createMnemosyne({
  vectorDbUrl: 'http://127.0.0.1:6333',
  embeddingUrl: 'http://127.0.0.1:6335/api/embed',
  embeddingModel: 'gte-modernbert-base',
  agentId: 'agent-memory-mnemosyne-smoke',
  collections: {
    shared: 'mnemosyne_shared',
    private: 'mnemosyne_private',
    profiles: 'mnemosyne_profiles',
    skills: 'mnemosyne_skills'
  },
  enableGraph: false,
  enableBroadcast: false,
  enableExtraction: false
})
const text = 'Mnemosyne smoke test memory: agent-memory keeps Hindsight on 8888 and Mnemosyne Qdrant on 6333.'
const stored = await m.store({ text, memoryType: 'fact', importance: 0.7 })
console.log('STORE', JSON.stringify(stored))
const recalled = await m.recall({ query: 'Which ports are used by Hindsight and Mnemosyne?', limit: 5 })
console.log('RECALL_COUNT', recalled.length)
console.log('TOP_TEXT', recalled[0]?.text || recalled[0]?.entry?.text || JSON.stringify(recalled[0] || null))
