import { performance } from "perf_hooks";

const HINDSIGHT_URL = "http://127.0.0.1:8888";
const OPENVINO_URL = "http://127.0.0.1:3002";
const QDRANT_URL = "http://127.0.0.1:6333";
const COLLECTION_NAME = "locomo_shared";
const PAGE_SIZE = 1000;
const EMBED_BATCH_SIZE = 512;

async function run() {
  const tStart = performance.now();
  console.log("Starting high-fidelity Locomo data import...");

  // 1. Get total count
  const firstRes = await fetch(`${HINDSIGHT_URL}/v1/default/banks/locomo/memories/list?limit=1`);
  if (!firstRes.ok) {
    throw new Error(`Failed to contact API Client: ${firstRes.statusText}`);
  }
  const firstData = await firstRes.json();
  const total = firstData.total;
  console.log(`Total memories to import: ${total}`);

  let offset = 0;
  while (offset < total) {
    console.log(`Fetching offset ${offset}...`);
    const pageRes = await fetch(`${HINDSIGHT_URL}/v1/default/banks/locomo/memories/list?limit=${PAGE_SIZE}&offset=${offset}`);
    const pageData = await pageRes.json();
    const items = pageData.items || [];
    
    if (items.length === 0) break;

    // Process items in embedding batches
    for (let i = 0; i < items.length; i += EMBED_BATCH_SIZE) {
      const batchItems = items.slice(i, i + EMBED_BATCH_SIZE);
      const texts = batchItems.map(item => item.text);

      // Generate embeddings
      const embedRes = await fetch(`${OPENVINO_URL}/embed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ inputs: texts })
      });
      if (!embedRes.ok) {
        throw new Error(`Embedding failed at offset ${offset} batch ${i}: ${embedRes.statusText}`);
      }
      const embeddings = await embedRes.json();

      // Create Qdrant points with full payload fields mapped to payloadToMemCell
      const points = batchItems.map((item, idx) => {
        // High-fidelity extraction of userId from tags (e.g. "user:locomo_exp_user_4_speaker_b_default")
        const userTag = item.tags?.find(t => t.startsWith("user:"));
        const userId = userTag ? userTag.split("user:")[1] : "locomo_user";

        return {
          id: item.id,
          vector: embeddings[idx],
          payload: {
            text: item.text,
            memory_type: item.fact_type || "semantic",
            classification: "public",
            agent_id: "locomo",
            user_id: userId,
            scope: "public",
            urgency: "reference",
            domain: "general",
            confidence: 0.8,
            confidence_tag: "grounded",
            priority_score: 0.5,
            importance: 0.5,
            deleted: false,
            created_at: item.date || new Date().toISOString(),
            updated_at: item.date || new Date().toISOString(),
            ingested_at: new Date().toISOString(),
            metadata: {
              original_id: item.id,
              tags: item.tags || [],
              date: item.date || "",
              entities: item.entities || ""
            }
          }
        };
      });

      // Upload to Qdrant
      const uploadRes = await fetch(`${QDRANT_URL}/collections/${COLLECTION_NAME}/points?wait=true`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ points })
      });
      if (!uploadRes.ok) {
        const errText = await uploadRes.text();
        throw new Error(`Upload to Qdrant failed: ${errText}`);
      }
    }

    offset += items.length;
    const progress = ((offset / total) * 100).toFixed(1);
    console.log(`Progress: ${offset}/${total} (${progress}%)`);
  }

  const tEnd = performance.now();
  console.log(`High-fidelity Locomo import completed successfully in ${((tEnd - tStart) / 1000).toFixed(1)} seconds.`);
}

run().catch(err => {
  console.error("FATAL ERROR:", err);
  process.exit(1);
});
