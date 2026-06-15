import { createMnemosyne as createEdumem } from 'mnemosy-ai';
import { createServer } from 'http';

const PORT = 6336;
const INTERVAL_MS = 12 * 60 * 60 * 1000; // 12 hours

async function main() {
  console.log('Starting api-daemon with RAG/Recall API...');
  
  // Single master edumem instance scoped to personal collections
  const m = await createEdumem({
    vectorDbUrl: 'http://127.0.0.1:6333',
    embeddingUrl: 'http://127.0.0.1:3002/embed',
    embeddingModel: 'gte-modernbert-base',
    agentId: 'agent-memory-daemon',
    collections: {
      shared: 'edumem_shared',
      private: 'edumem_private',
      profiles: 'edumem_profiles',
      skills: 'edumem_skills'
    },
    enableGraph: false,
    enableBroadcast: false,
    enableExtraction: true,
    extractionUrl: 'http://127.0.0.1:6335',
    enableBM25: true,
    bm25MaxDocs: 10000,
    bm25PageSize: 100
  });

  let lastDream = null;
  let lastConsolidate = null;
  let isRunning = false;

  async function runCognitiveJobs() {
    if (isRunning) return;
    isRunning = true;
    console.log('[' + new Date().toISOString() + '] Starting scheduled dreaming & consolidation on personal data...');
    try {
      console.log('Running edumem Dream (deduplication & decay)...');
      const dreamRes = await m.dream();
      lastDream = { timestamp: new Date().toISOString(), result: dreamRes };
      console.log('Dream complete:', JSON.stringify(dreamRes));

      console.log('Running edumem Consolidate...');
      const consRes = await m.consolidate();
      lastConsolidate = { timestamp: new Date().toISOString(), result: consRes };
      console.log('Consolidate complete:', JSON.stringify(consRes));
    } catch (err) {
      console.error('Error running cognitive jobs:', err);
    } finally {
      isRunning = false;
    }
  }

  // Run cognitive dreaming/decay immediately on startup
  await runCognitiveJobs().catch(console.error);

  // Set interval
  setInterval(runCognitiveJobs, INTERVAL_MS);

  // Helper to parse JSON body
  function getJsonBody(req) {
    return new Promise((resolve, reject) => {
      let body = '';
      req.on('data', chunk => { body += chunk; });
      req.on('end', () => {
        try {
          resolve(body ? JSON.parse(body) : {});
        } catch (e) {
          reject(e);
        }
      });
      req.on('error', reject);
    });
  }

  // Start HTTP server supporting API Client RAG API protocol
  const server = createServer(async (req, res) => {
    const url = req.url;
    const method = req.method;

    if (method === 'GET' && (url === '/health' || url === '/')) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        status: 'healthy',
        daemon: 'edumem-cognitive-daemon',
        last_dream: lastDream,
        last_consolidate: lastConsolidate,
        is_running: isRunning,
        uptime_seconds: Math.floor(process.uptime())
      }));
      return;
    }

    // 1. RECALL API (compatible with API Client recall spec)
    if (method === 'POST' && url.match(/^\/v1\/default\/banks\/([^\/]+)\/memories\/recall$/)) {
      try {
        const match = url.match(/^\/v1\/default\/banks\/([^\/]+)\/memories\/recall$/);
        const bankId = match[1];
        const body = await getJsonBody(req);
        const query = body.query;
        const tags = body.tags || [];
        
        // Strip user: prefix from tags
        const userTag = tags.find(t => t.startsWith('user:'));
        const userId = userTag ? userTag.replace('user:', '') : null;
        
        console.log('[RECALL] bank=' + bankId + ' user=' + userId + ' query= + query + ');
        
        // Dynamic collection routing: locomo -> locomo_shared, anything else -> personal shared
        const targetCollection = bankId === 'locomo' ? 'locomo_shared' : (bankId === 'pm' ? 'pm_shared' : m.config.sharedCollection);
        const filters = userId ? { user_id: userId } : undefined;
        
        // Generate embeddings and query Qdrant directly via our master client\'s db handler
        const vector = await m.embeddings.embed(query);
        const searchResults = await m.db.search(targetCollection, vector, 15, 0.3, filters);
        
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          results: searchResults.map(r => ({
            text: r.entry.text,
            score: r.score
          }))
        }));
      } catch (err) {
        console.error('Recall API error:', err);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
      return;
    }

    // 2. REFLECT API (compatible with API Client reflect spec)
    if (method === 'POST' && url.match(/^\/v1\/default\/banks\/([^\/]+)\/reflect$/)) {
      try {
        const match = url.match(/^\/v1\/default\/banks\/([^\/]+)\/reflect$/);
        const bankId = match[1];
        const body = await getJsonBody(req);
        const query = body.query;
        const tags = body.tags || [];
        
        const userTags = tags.filter(t => t.startsWith('user:')).map(t => t.replace('user:', ''));
        console.log('[REFLECT] bank=' + bankId + ' users=' + JSON.stringify(userTags) + ' query= + query + ');

        const targetCollection = bankId === 'locomo' ? 'locomo_shared' : (bankId === 'pm' ? 'pm_shared' : m.config.sharedCollection);
        
        // Retrieve user-scoped memories in parallel
        const results = await Promise.all(userTags.map(async (uid) => {
          const vector = await m.embeddings.embed(query);
          const userMemories = await m.db.search(targetCollection, vector, 15, 0.3, { user_id: uid });
          return {
            uid,
            memories: userMemories.map(r => r.entry.text)
          };
        }));

        // Format into combined context string matching API Client\'s structure
        let formattedContext = '';
        results.forEach(({ uid, memories }) => {
          const speakerName = uid.includes('speaker_a') ? 'Speaker A' : 'Speaker B';
          formattedContext += '### Historical memories for ' + speakerName + ' (' + uid + '):\n';
          if (memories.length === 0) {
            formattedContext += '- No relevant memories found.\n';
          } else {
            memories.forEach(m => {
              formattedContext += '- ' + m + '\n';
            });
          }
          formattedContext += '\n';
        });

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          text: formattedContext.trim()
        }));
      } catch (err) {
        console.error('Reflect API error:', err);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
      return;
    }

    // Fallback 404
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not Found' }));
  });

  server.listen(PORT, '0.0.0.0', () => {
    console.log('api-daemon Server running on http://0.0.0.0:' + PORT);
  });
}

main().catch(console.error);
