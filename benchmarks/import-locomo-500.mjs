import { performance } from 'perf_hooks';

const HINDSIGHT_URL = 'http://127.0.0.1:8888';
const OPENVINO_URL = 'http://127.0.0.1:3002';
const QDRANT_URL = 'http://127.0.0.1:6333';
const COLLECTION_NAME = 'locomo_shared';
const LIMIT = 500;
const EMBED_BATCH_SIZE = 250;

async function run() {
  const tStart = performance.now();
  console.log('Starting limited (500) Locomo data import...');

  console.log(`Fetching up to ${LIMIT} memories...`);
  const pageRes = await fetch(`${HINDSIGHT_URL}/v1/default/banks/locomo/memories/list?limit=${LIMIT}`);
  if (!pageRes.ok) {
    throw new Error(`Failed to contact Hindsight: ${pageRes.statusText}`);
  }
  const pageData = await pageRes.json();
  const items = pageData.items || [];
  console.log(`Fetched ${items.length} memories from Hindsight.`);

  if (items.length === 0) {
    console.log('No items found!');
    return;
  }

  // Process items in embedding batches
  for (let i = 0; i < items.length; i += EMBED_BATCH_SIZE) {
    const batchItems = items.slice(i, i + EMBED_BATCH_SIZE);
    const texts = batchItems.map(item => item.text);
    console.log(`Embedding batch ${i} with size ${batchItems.length}...`);

    // Generate embeddings directly via local OpenVINO
    const embedRes = await fetch(`${OPENVINO_URL}/embed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ inputs: texts })
    });
    if (!embedRes.ok) {
      throw new Error(`Embedding failed at batch ${i}: ${embedRes.statusText}`);
    }
    const embeddings = await embedRes.json();

    // Create Qdrant points
    const points = batchItems.map((item, idx) => {
      const userTag = item.tags?.find(t => t.startsWith('user:'));
      const userId = userTag ? userTag.split('user:')[1] : 'locomo_user';

      return {
        id: item.id,
        vector: embeddings[idx],
        payload: {
          text: item.text,
          memory_type: item.fact_type || 'semantic',
          classification: 'public',
          agent_id: 'locomo',
          user_id: userId,
          scope: 'public',
          urgency: 'reference',
          domain: 'general',
          confidence: 0.8,
          confidence_tag: 'grounded',
          priority_score: 0.5,
          importance: 0.5,
          deleted: false,
          created_at: item.date || new Date().toISOString(),
          updated_at: item.date || new Date().toISOString(),
          ingested_at: new Date().toISOString(),
          metadata: {
            original_id: item.id,
            tags: item.tags || [],
            date: item.date || '',
            entities: item.entities || ''
          }
        }
      };
    });

    console.log(`Uploading ${points.length} points to Qdrant...`);
    // Upload to Qdrant
    const uploadRes = await fetch(`${QDRANT_URL}/collections/${COLLECTION_NAME}/points?wait=true`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ points })
    });
    if (!uploadRes.ok) {
      const errText = await uploadRes.text();
      throw new Error(`Upload to Qdrant failed: ${errText}`);
    }
  }

  const tEnd = performance.now();
  console.log(`Locomo 500 import completed successfully in ${((tEnd - tStart) / 1000).toFixed(1)} seconds.`);
}

run().catch(err => {
  console.error('FATAL ERROR:', err);
  process.exit(1);
});
