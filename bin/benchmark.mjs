import { createMnemosyne } from "mnemosy-ai";
import { performance } from "perf_hooks";

const m = await createMnemosyne({
  vectorDbUrl: "http://127.0.0.1:6333",
  embeddingUrl: "http://127.0.0.1:6335/api/embed",
  embeddingModel: "gte-modernbert-base",
  agentId: "benchmark-agent",
  collections: {
    shared: "mnemosyne_shared",
    private: "mnemosyne_private",
    profiles: "mnemosyne_profiles",
    skills: "mnemosyne_skills"
  },
  enableGraph: false,
  enableBroadcast: false,
  enableExtraction: false
});

const queries = [
  "Aircall interview prep",
  "iGPU passthrough configuration",
  "Gemini Flash Lite model preference",
  "GL-MT6000 router configuration",
  "Neovim tmux tool preferences"
];

// Warmup
for (const q of queries) {
  try {
    await fetch("http://127.0.0.1:8888/v1/default/banks/hermes/memories/recall", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q })
    });
    await m.recall({ query: q, limit: 3 });
  } catch (e) {
    // Ignore warmup errors
  }
}

const results = [];

for (const q of queries) {
  // Hindsight Recall
  let hsLatency = 0;
  let hsCount = 0;
  let hsTopText = "No match";
  try {
    const t0 = performance.now();
    const hsRes = await fetch("http://127.0.0.1:8888/v1/default/banks/hermes/memories/recall", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q })
    });
    const t1 = performance.now();
    const hsData = await hsRes.json();
    hsCount = hsData.results?.length || 0;
    hsTopText = hsData.results?.[0]?.text || "No match";
    hsLatency = t1 - t0;
  } catch (e) {
    hsTopText = "Error: " + e.message;
  }

  // Mnemosyne Recall
  let mnLatency = 0;
  let mnCount = 0;
  let mnTopText = "No match";
  try {
    const t2 = performance.now();
    const mnRes = await m.recall({ query: q, limit: 3 });
    const t3 = performance.now();
    mnCount = mnRes.length;
    mnTopText = mnRes[0]?.text || mnRes[0]?.entry?.text || "No match";
    mnLatency = t3 - t2;
  } catch (e) {
    mnTopText = "Error: " + e.message;
  }

  results.push({
    query: q,
    hindsight: { latency: hsLatency, count: hsCount, top: hsTopText },
    mnemosyne: { latency: mnLatency, count: mnCount, top: mnTopText }
  });
}

console.log(JSON.stringify(results, null, 2));
