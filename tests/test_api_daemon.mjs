import { test, describe, before, after } from 'node:test';
import assert from 'node:assert';
import { spawn } from 'child_process';

const PORT = 6339;
const BASE_URL = `http://127.0.0.1:${PORT}`;

describe('api-daemon Integration Tests', () => {
  let daemonProcess;
  let uniqueFactText;
  let uniqueId;

  before(async () => {
    uniqueId = Date.now();
    // Use a completely novel, non-overlapping semantic topic to prevent deduplication/merging
    uniqueFactText = `TDD story ${uniqueId}: a brilliant blue giraffe flew over the orange fence on port ${PORT}`;
    
    console.log('[Test Setup] Launching api-daemon on test port', PORT);
    
    // Launch api-daemon as a background subprocess, overriding PORT via env
    daemonProcess = spawn('node', ['bin/api-daemon.mjs'], {
      cwd: '/opt/edumem',
      env: { ...process.env, PORT: PORT.toString() },
      stdio: 'pipe'
    });

    // Capture logs for debugging
    daemonProcess.stdout.on('data', (data) => {
      console.log(`[Daemon Stdout] ${data.toString().trim()}`);
    });
    daemonProcess.stderr.on('data', (data) => {
      console.error(`[Daemon Stderr] ${data.toString().trim()}`);
    });

    // Wait for the server to spin up and become healthy (poll /health)
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
      await new Promise(resolve => setTimeout(resolve, 500));
    }

    if (!healthy) {
      throw new Error('api-daemon failed to start or pass health check on port ' + PORT);
    }
    console.log('[Test Setup] api-daemon is ready and healthy!');
  });

  after(() => {
    if (daemonProcess) {
      console.log('[Test Teardown] Killing api-daemon process...');
      daemonProcess.kill('SIGTERM');
    }
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
    // Wait a brief moment to ensure write-index synchronization completes in Qdrant
    await new Promise(resolve => setTimeout(resolve, 1500));

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
    
    // Find the stored fact in the returned results
    const found = body.results.some(r => r.text.includes(uniqueId.toString()));
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
