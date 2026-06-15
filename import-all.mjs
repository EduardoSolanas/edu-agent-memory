import { createMnemosyne } from 'mnemosy-ai'
import fs from 'fs'

async function run() {
  const raw = fs.readFileSync('memories_dump.json', 'utf8')
  const data = JSON.parse(raw)
  console.log(`Loaded ${data.length} memories from dump file.`)

  const m = await createMnemosyne({
    vectorDbUrl: 'http://127.0.0.1:6333',
    embeddingUrl: 'http://127.0.0.1:6335/api/embed',
    embeddingModel: 'gte-modernbert-base',
    agentId: 'agent-memory-hindsight-import',
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

  let count = 0
  for (const item of data) {
    const text = item.text
    if (!text) continue

    // Determine category based on tags & text
    let category = 'fact'
    const tags = item.tags || []
    const isPreference = tags.some(t => t.includes('preference') || t.includes('personal')) ||
                         text.toLowerCase().includes('prefer') ||
                         text.toLowerCase().includes('like')
    if (isPreference) {
      category = 'preference'
    }

    const payload = {
      text,
      category,
      memoryType: item.fact_type || 'world',
      importance: item.fact_type === 'world' ? 0.8 : 0.6,
      metadata: {
        hindsight_id: item.id,
        tags: tags,
        document_id: item.document_id,
        created_at: item.created_at
      }
    }

    try {
      const storedId = await m.store(payload)
      count++
      if (count % 20 === 0 || count === data.length) {
        console.log(`Processed ${count}/${data.length} memories...`)
      }
    } catch (e) {
      console.error(`Error storing item ${item.id}:`, e.message)
    }
  }

  console.log(`IMPORT COMPLETE: Successfully imported ${count} memories into Mnemosyne.`)
  const stats = await m.stats()
  console.log('MNEMOSYNE STATS:', JSON.stringify(stats))
}

run().catch(console.error)
