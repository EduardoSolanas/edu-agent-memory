import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { test, describe, before, after } from 'node:test';
import assert from 'node:assert';

const PORT = 6339;
const BASE_URL = `http://127.0.0.1:${PORT}`;
const EMBEDDING_DIMENSION = 768;

let daemonProcess;
let qdrantServer;
let embeddingServer;
let qdrantUrl;
let embeddingUrl;
let uniqueFactText;
let uniqueId;

const collections = new Map();

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
    });
    req.on('end', () => {
      if (!body) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(body));
      } catch (err) {
        reject(err);
      }
    });
    req.on('error', reject);
  });
}

function getPathValue(object, path) {
  return path.split('.').reduce((value, segment) => (value == null ? value : value[segment]), object);
}

function matchesFilter(point, must = []) {
  return must.every((rule) => {
    if (!rule || typeof rule !== 'object') {
      return true;
    }
    const value = getPathValue(point.payload ?? {}, rule.key ?? '');
    if (!rule.match || !Object.prototype.hasOwnProperty.call(rule.match, 'value')) {
      return true;
    }
    return value === rule.match.value;
  });
}

function getCollection(name) {
  if (!collections.has(name)) {
    collections.set(name, []);
  }
  return collections.get(name);
}

function buildEmbedding() {
  const vector = new Array(EMBEDDING_DIMENSION).fill(0);
  vector[0] = 1;
  return vector;
}

async function startServer(handler) {
  const server = createServer(handler);
  await new Promise((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  const address = server.address();
  if (!address || typeof address === 'string') {
    throw new Error('Failed to bind test server.');
  }
  return { server, port: address.port };
}

function closeServer(server) {
  if (!server) {
    return Promise.resolve();
  }
  return new Promise((resolve) => server.close(resolve));
}

async function startQdrantStub() {
  return startServer(async (req, res) => {
    const pathname = new URL(req.url, 'http://127.0.0.1').pathname;
    const collectionMatch = pathname.match(/^\/collections\/([^/]+)$/);
    const pointsMatch = pathname.match(/^\/collections\/([^/]+)\/points$/);
    const searchMatch = pathname.match(/^\/collections\/([^/]+)\/points\/search$/);

    if (req.method === 'GET' && collectionMatch) {
      const collectionName = collectionMatch[1];
      getCollection(collectionName);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ result: { status: 'ok', name: collectionName } }));
      return;
    }

    if (req.method === 'PUT' && collectionMatch) {
      const collectionName = collectionMatch[1];
      getCollection(collectionName);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ result: { status: 'ok', name: collectionName } }));
      return;
    }

    if (req.method === 'PUT' && pointsMatch) {
      const collectionName = pointsMatch[1];
      const body = await readJsonBody(req);
      const points = Array.isArray(body.points) ? body.points : [];
      const collection = getCollection(collectionName);
      for (const point of points) {
        collection.push({
          id: point.id ?? randomUUID(),
          vector: Array.isArray(point.vector) ? point.vector : buildEmbedding(),
          payload: point.payload ?? {}
        });
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ result: { status: 'ok', points: points.length } }));
      return;
    }

    if (req.method === 'POST' && searchMatch) {
      const collectionName = searchMatch[1];
      const body = await readJsonBody(req);
      const must = Array.isArray(body.filter?.must) ? body.filter.must : [];
      const collection = getCollection(collectionName);
      const matches = collection
        .filter((point) => matchesFilter(point, must))
        .map((point, index) => ({
          id: point.id,
          payload: point.payload,
          score: 1 - index * 0.001
        }));

      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ result: matches.slice(0, body.limit ?? matches.length) }));
      return;
    }

    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not found' }));
  });
}

async function startEmbeddingStub() {
  return startServer(async (req, res) => {
    const pathname = new URL(req.url, 'http://127.0.0.1').pathname;
    if (req.method === 'POST' && pathname === '/v1/embeddings') {
      await readJsonBody(req);
      const embedding = buildEmbedding();
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        object: 'list',
        data: [
          {
            object: 'embedding',
            embedding,
            index: 0
          }
        ],
        model: 'stub-openvino-embedding',
        usage: {
          prompt_tokens: 0,
          total_tokens: 0
        }
      }));
      return;
    }

    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not found' }));
  });
}

describe('api-daemon Integration Tests', () => {
  before(async () => {
    uniqueId = Date.now();
    uniqueFactText = `TDD story ${uniqueId}: a brilliant blue giraffe flew over the orange fence on port ${PORT}`;

    const qdrant = await startQdrantStub();
    const embedding = await startEmbeddingStub();
    qdrantServer = qdrant.server;
    embeddingServer = embedding.server;
    qdrantUrl = `http://127.0.0.1:${qdrant.port}`;
    embeddingUrl = `http://127.0.0.1:${embedding.port}/v1/embeddings`;

    console.log('[Test Setup] Launching api-daemon on test port', PORT);

    daemonProcess = spawn(process.execPath, ['bin/api-daemon.mjs'], {
      cwd: process.cwd(),
      env: {
        ...process.env,
        PORT: PORT.toString(),
        VECTOR_DB_URL: qdrantUrl,
        EMBEDDING_URL: embeddingUrl
      },
      stdio: 'pipe'
    });

    daemonProcess.stdout.on('data', (data) => {
      console.log(`[Daemon Stdout] ${data.toString().trim()}`);
    });
    daemonProcess.stderr.on('data', (data) => {
      console.error(`[Daemon Stderr] ${data.toString().trim()}`);
    });

    let healthy = false;
    for (let i = 0; i < 20; i++) {
      try {
        const res = await fetch(`${BASE_URL}/health`);
        if (res.status === 200) {
          const body = await res.json();
          if (body.status === 'healthy') {
            healthy = true;
            break;
          }
        }
      } catch (err) {
        // Ignored during startup poll
      }
      await new Promise((resolve) => setTimeout(resolve, 500));
    }

    if (!healthy) {
      throw new Error('api-daemon failed to start or pass health check on port ' + PORT);
    }
    console.log('[Test Setup] api-daemon is ready and healthy!');
  });

  after(async () => {
    if (daemonProcess) {
      console.log('[Test Teardown] Killing api-daemon process...');
      daemonProcess.kill('SIGTERM');
    }
    await closeServer(qdrantServer);
    await closeServer(embeddingServer);
  });

  test('GET /health returns 200 and healthy status payload', async () => {
    const res = await fetch(`${BASE_URL}/health`);
    assert.strictEqual(res.status, 200);
    const body = await res.json();
    assert.strictEqual(body.status, 'healthy');
    assert.strictEqual(body.daemon, 'edumem-cognitive-daemon');
    assert.ok(typeof body.uptime_seconds === 'number');
  });

  test('POST /v1/default/banks/testbank/memories/retain stores fact', async () => {
    const res = await fetch(`${BASE_URL}/v1/default/banks/testbank/memories/retain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content: uniqueFactText,
        tags: ['user:tdd-tester']
      })
    });
    assert.strictEqual(res.status, 200);
    const body = await res.json();
    assert.strictEqual(body.result, 'Memory stored successfully.');
    assert.ok(body.id);
  });

  test('POST /v1/default/banks/testbank/memories/recall retrieves stored fact', async () => {
    await new Promise((resolve) => setTimeout(resolve, 1500));

    const res = await fetch(`${BASE_URL}/v1/default/banks/testbank/memories/recall`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: `What blue animal flew over the fence in test ${uniqueId}?`,
        tags: ['user:tdd-tester']
      })
    });
    assert.strictEqual(res.status, 200);
    const body = await res.json();
    assert.ok(Array.isArray(body.results));
    assert.ok(body.results.length > 0, 'Should return search results');

    const found = body.results.some((r) => r.text.includes(uniqueId.toString()));
    assert.ok(found, `Should find the stored TDD test fact containing ${uniqueId} in search results`);
    assert.ok(body.results[0].score > 0.1, 'Matches should have valid confidence scores');
  });

  test('POST /v1/default/banks/testbank/reflect synthesizes user memories', async () => {
    const res = await fetch(`${BASE_URL}/v1/default/banks/testbank/reflect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: `active test story about blue giraffe ${uniqueId}`,
        tags: ['user:tdd-tester']
      })
    });
    assert.strictEqual(res.status, 200);
    const body = await res.json();
    assert.ok(body.text);
    assert.ok(body.text.includes('Historical memories for'), 'Should include the formatted speaker sections');
    assert.ok(body.text.includes(uniqueId.toString()), `Should list retrieved facts containing ${uniqueId} in the reflection output`);
  });
});
