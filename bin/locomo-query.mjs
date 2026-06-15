import { createMnemosyne } from "mnemosy-ai";

const m = await createMnemosyne({
  vectorDbUrl: "http://127.0.0.1:6333",
  embeddingUrl: "http://127.0.0.1:6335/api/embed",
  embeddingModel: "gte-modernbert-base",
  agentId: "locomo-query",
  collections: {
    shared: "locomo_shared",
    private: "locomo_private",
    profiles: "locomo_profiles",
    skills: "locomo_skills"
  },
  enableGraph: false,
  enableBroadcast: false,
  enableExtraction: false,
  enableBM25: true // Re-enabled BM25!
});

// Sleep 1.5 seconds to allow the BM25 index to bootstrap in the background
console.log("Initializing BM25 full-text index...");
await new Promise(resolve => setTimeout(resolve, 1500));

const query = process.argv[2] || "Where does Tim want to travel?";
console.log(`Querying: "${query}"`);

const res = await m.recall({ query, limit: 3 });
console.log("\nRECALLED (HYBRID SEARCH - VECTOR + BM25):");
res.forEach((r, idx) => {
  console.log(`  ${idx+1}: ${r.entry.text} (score: ${r.score.toFixed(4)})`);
});
