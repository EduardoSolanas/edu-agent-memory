# edumem on agent-memory

Installed side-by-side with Hindsight.

- edumem SDK path: `/opt/edumem`
- NPM package: `mnemosy-ai`
- Qdrant container: `edumem-qdrant`
- Qdrant HTTP: `http://127.0.0.1:6333` / LAN `http://192.168.1.160:6333`
- Qdrant gRPC: `127.0.0.1:6334` only
- Embedding proxy: `http://127.0.0.1:6335/api/embed`
- Embedding upstream: existing OpenVINO server `http://127.0.0.1:3002/embed`
- Smoke test: `cd /opt/edumem && node smoke.mjs`

Hindsight remains on:
- API: `http://192.168.1.160:8888`
- UI: `http://192.168.1.160:9999/dashboard`
- OpenVINO: `http://192.168.1.160:3002`

Recommended edumem config:

```js
{
  vectorDbUrl: 'http://127.0.0.1:6333',
  embeddingUrl: 'http://127.0.0.1:6335/api/embed',
  embeddingModel: 'gte-modernbert-base',
  agentId: '<your-agent-id>',
  collections: {
    shared: 'edumem_shared',
    private: 'edumem_private',
    profiles: 'edumem_profiles',
    skills: 'edumem_skills'
  },
  enableGraph: false,
  enableBroadcast: false,
  enableExtraction: false
}
```
