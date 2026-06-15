# Mnemosyne on agent-memory

Installed side-by-side with Hindsight.

- Mnemosyne SDK path: `/opt/mnemosyne`
- NPM package: `mnemosy-ai`
- Qdrant container: `mnemosyne-qdrant`
- Qdrant HTTP: `http://127.0.0.1:6333` / LAN `http://192.168.1.160:6333`
- Qdrant gRPC: `127.0.0.1:6334` only
- Embedding proxy: `http://127.0.0.1:6335/api/embed`
- Embedding upstream: existing OpenVINO server `http://127.0.0.1:3002/embed`
- Smoke test: `cd /opt/mnemosyne && node smoke.mjs`

Hindsight remains on:
- API: `http://192.168.1.160:8888`
- UI: `http://192.168.1.160:9999/dashboard`
- OpenVINO: `http://192.168.1.160:3002`

Recommended Mnemosyne config:

```js
{
  vectorDbUrl: 'http://127.0.0.1:6333',
  embeddingUrl: 'http://127.0.0.1:6335/api/embed',
  embeddingModel: 'gte-modernbert-base',
  agentId: '<your-agent-id>',
  collections: {
    shared: 'mnemosyne_shared',
    private: 'mnemosyne_private',
    profiles: 'mnemosyne_profiles',
    skills: 'mnemosyne_skills'
  },
  enableGraph: false,
  enableBroadcast: false,
  enableExtraction: false
}
```
