import { createServer } from 'http';
import { randomUUID } from 'crypto';

// Production configuration via environment variables
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 6336;
const VECTOR_DB_URL = process.env.VECTOR_DB_URL || 'http://127.0.0.1:6333';
const EMBEDDING_URL = process.env.EMBEDDING_URL || 'http://127.0.0.1:3002/v1/embeddings';
const SHARED_COLLECTION = process.env.SHARED_COLLECTION || 'edumem_shared';

// Ensure Qdrant collection is initialized on startup
async function ensureCollection(collection) {
  try {
    const checkRes = await fetch(`${VECTOR_DB_URL}/collections/${collection}`);
    if (checkRes.status === 200) {
      console.log(`[Collection] "${collection}" is verified and active.`);
      return;
    }
    
    console.log(`[Collection] Creating "${collection}" (768-dim, Cosine distance)...`);
    const createRes = await fetch(`${VECTOR_DB_URL}/collections/${collection}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        vectors: {
          size: 768,
          distance: 'Cosine'
        }
      })
    });
    
    if (!createRes.ok) {
      throw new Error(`PUT /collections/${collection} returned status ${createRes.status}`);
    }
    console.log(`[Collection] Successfully initialized "${collection}"!`);
  } catch (err) {
    console.error(`[Collection Error] Failed to ensure collection "${collection}":`, err.message);
  }
}

// Compute embeddings using the local OpenVINO GenAI server
async function getEmbedding(text) {
  const res = await fetch(EMBEDDING_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input: text })
  });
  if (!res.ok) {
    throw new Error(`Embedding request failed: ${res.statusText}`);
  }
  const data = await res.json();
  return data.data[0].embedding; // Extract the first float array
}

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

async function main() {
  console.log(`Starting native edumem api-daemon on port ${PORT}...`);
  console.log(`Configured backend URLs:`);
  console.log(`- Vector DB (Qdrant): ${VECTOR_DB_URL}`);
  console.log(`- Embeddings (OpenVINO): ${EMBEDDING_URL}`);
  console.log(`- Active Collection: ${SHARED_COLLECTION}`);

  // Bootstrap active collection
  await ensureCollection(SHARED_COLLECTION);

  const server = createServer(async (req, res) => {
    const url = req.url;
    const method = req.method;

    // Health Endpoint
    if (method === 'GET' && (url === '/health' || url === '/')) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        status: 'healthy',
        daemon: 'edumem-cognitive-daemon',
        uptime_seconds: Math.floor(process.uptime()),
        dependencies: {
          vector_db: VECTOR_DB_URL,
          embeddings: EMBEDDING_URL
        }
      }));
      return;
    }

    // 1. RECALL API
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
        
        console.log(`[RECALL] bank=${bankId} user=${userId} query="${query}"`);
        
        const targetCollection = bankId === 'locomo' ? 'locomo_shared' : (bankId === 'pm' ? 'pm_shared' : SHARED_COLLECTION);
        
        // Compute embedding using OpenVINO
        const vector = await getEmbedding(query);
        
        // Formulate Qdrant search request
        const must = [{ key: "deleted", match: { value: false } }];
        if (userId) {
          must.push({ key: "metadata.user_id", match: { value: userId } });
        }

        const qdrantRes = await fetch(`${VECTOR_DB_URL}/collections/${targetCollection}/points/search`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            vector,
            limit: 15,
            filter: { must },
            with_payload: true
          })
        });

        if (!qdrantRes.ok) {
          throw new Error(`Qdrant search failed with status ${qdrantRes.status}`);
        }

        const data = await qdrantRes.json();
        const searchResults = (data.result || []).map(r => ({
          text: r.payload?.text || '',
          score: r.score
        }));

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ results: searchResults }));
      } catch (err) {
        console.error('Recall API error:', err.message);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
      return;
    }

    // 2. RETAIN API
    if (method === 'POST' && url.match(/^\/v1\/default\/banks\/([^\/]+)\/memories\/retain$/)) {
      try {
        const match = url.match(/^\/v1\/default\/banks\/([^\/]+)\/memories\/retain$/);
        const bankId = match[1];
        const body = await getJsonBody(req);
        const content = body.content || body.text || '';
        const tags = body.tags || [];
        
        // Strip user: prefix from tags
        const userTag = tags.find(t => t.startsWith('user:'));
        const userId = userTag ? userTag.replace('user:', '') : null;

        console.log(`[RETAIN] bank=${bankId} user=${userId} content="${content.slice(0, 60)}..."`);
        
        const targetCollection = bankId === 'locomo' ? 'locomo_shared' : (bankId === 'pm' ? 'pm_shared' : SHARED_COLLECTION);
        
        // Compute embedding using OpenVINO
        const vector = await getEmbedding(content);
        
        // Assemble Qdrant point paylod
        const id = randomUUID();
        const now = new Date().toISOString();
        const payload = {
          text: content,
          agent_id: "agent-memory-daemon",
          user_id: null,
          memory_type: "fact",
          deleted: false,
          metadata: userId ? { user_id: userId } : {},
          created_at: now,
          updated_at: now
        };

        const qdrantRes = await fetch(`${VECTOR_DB_URL}/collections/${targetCollection}/points`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            wait: true,
            points: [{ id, vector, payload }]
          })
        });

        if (!qdrantRes.ok) {
          throw new Error(`Qdrant PUT points failed with status ${qdrantRes.status}`);
        }

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          result: "Memory stored successfully.",
          id: id
        }));
      } catch (err) {
        console.error('Retain API error:', err.message);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
      return;
    }

    // 3. REFLECT API
    if (method === 'POST' && url.match(/^\/v1\/default\/banks\/([^\/]+)\/reflect$/)) {
      try {
        const match = url.match(/^\/v1\/default\/banks\/([^\/]+)\/reflect$/);
        const bankId = match[1];
        const body = await getJsonBody(req);
        const query = body.query;
        const tags = body.tags || [];
        
        const userTags = tags.filter(t => t.startsWith('user:')).map(t => t.replace('user:', ''));
        console.log(`[REFLECT] bank=${bankId} users=${JSON.stringify(userTags)} query="${query}"`);

        const targetCollection = bankId === 'locomo' ? 'locomo_shared' : (bankId === 'pm' ? 'pm_shared' : SHARED_COLLECTION);
        const vector = await getEmbedding(query);
        
        // Retrieve user-scoped memories in parallel
        const results = await Promise.all(userTags.map(async (uid) => {
          const must = [
            { key: "deleted", match: { value: false } },
            { key: "metadata.user_id", match: { value: uid } }
          ];
          
          const qdrantRes = await fetch(`${VECTOR_DB_URL}/collections/${targetCollection}/points/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              vector,
              limit: 15,
              filter: { must },
              with_payload: true
            })
          });

          if (!qdrantRes.ok) return { uid, memories: [] };

          const data = await qdrantRes.json();
          const memories = (data.result || []).map(r => r.payload?.text || '');
          return { uid, memories };
        }));

        // Format into combined context string matching API specification
        let formattedContext = '';
        results.forEach(({ uid, memories }) => {
          const speakerName = uid.includes('speaker_a') ? 'Speaker A' : 'Speaker B';
          formattedContext += `### Historical memories for ${speakerName} (${uid}):\n`;
          if (memories.length === 0) {
            formattedContext += '- No relevant memories found.\n';
          } else {
            memories.forEach(m => {
              formattedContext += `- ${m}\n`;
            });
          }
          formattedContext += '\n';
        });

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ text: formattedContext.trim() }));
      } catch (err) {
        console.error('Reflect API error:', err.message);
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
    console.log(`api-daemon Server running on http://0.0.0.0:${PORT}`);
  });
}

main().catch(console.error);