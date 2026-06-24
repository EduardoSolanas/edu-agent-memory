# TEMPORARY -- judged graph A/B (graph fusion ON vs OFF), not committed.
# Both arms ingest identically (EDUMEM_LLM_EXTRACTION=1 -> graph stored);
# they differ ONLY in EDUMEM_KG_FUSION at read time. Per-case isolation
# (separate process per conversation) to survive Windows temp-dir cleanup.

$ErrorActionPreference = "Continue"

# --- NAN LLM endpoint (qwen3.6 answer, deepseek-v4-flash judge) ---
$nan = (Select-String -Path ".env" -Pattern '^NAN_APY_KEY=').Line -replace '^NAN_APY_KEY=', '' -replace '[\r"]', ''
$env:OPENROUTER_API_KEY = $nan
$env:OPENROUTER_BASE_URL = "https://api.nan.builders/v1"

# --- Dense recall path (gte-modernbert via local container) ---
$env:EDUMEM_EMBEDDING_API_URL = "http://localhost:3002"
$env:EDUMEM_EMBEDDING_MODEL   = "Alibaba-NLP/gte-modernbert-base"
$env:EDUMEM_EMBEDDINGS_VIA_API = "1"

# --- Shared eval config (better_data.md S6) ---
$env:EDUMEM_LLM_EXTRACTION   = "1"      # graph + canonical facts written at ingest
$env:EDUMEM_MAX_CONTEXT_CHARS = "16000"
$env:PYTHONUNBUFFERED = "1"
# 4-way question workers: safe now that init_beam guards its trigger rewrite
# (re-init on an existing DB issues no write -> no "database is locked").
$env:BEAM_QUESTION_WORKERS = "4"

$arms  = @{ "graph" = "1"; "nograph" = "0" }
$cases = @(4)

foreach ($arm in $arms.Keys) {
  $env:EDUMEM_KG_FUSION = $arms[$arm]
  foreach ($c in $cases) {
    $tag = "graphab-$arm-c$c"
    $out = "results_graphab/$arm/c$c"
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    Write-Host "=== RUN $tag (EDUMEM_KG_FUSION=$($arms[$arm]), case $c) ==="
    python -u tools/evaluate_beam_end_to_end.py `
      --scales 100K --case-index $c `
      --model qwen3.6 --judge-model deepseek-v4-flash `
      --pure-recall --config-id $tag --output-dir $out 2>&1 |
      Tee-Object -FilePath "graphab_$arm`_c$c.log"
    Write-Host "=== DONE $tag exit=$LASTEXITCODE ==="
  }
}
Write-Host "ALL GRAPH A/B RUNS COMPLETE"
