from __future__ import annotations

from collections import deque
import contextlib
import gc
import hashlib
import json
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from edumem.core.beam import BeamMemory, _spo_fact_id
from edumem.extraction import ExtractionClient
from tools.evaluate_beam_end_to_end import LLMClient, ingest_conversation


MANIFEST_NAME = "manifest.json"
CONVERSATION_NAME = "conversation.json"
OPS_NAME = "generated_ops.jsonl"
CONTRACT_NAME = "write_contract.json"
REPLAY_CONTRACT_NAME = "replay_write_contract.json"
FINAL_DB_NAME = "final.db"


def _cache_path(cache_dir: str | Path, name: str) -> Path:
    return Path(cache_dir) / name


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text((text + "\n") if text else "", encoding="utf-8")


def load_generated_write_cache(cache_dir: str | Path) -> dict:
    cache_path = Path(cache_dir)
    contract_path = _cache_path(cache_path, CONTRACT_NAME)
    return {
        "manifest": _read_json(_cache_path(cache_path, MANIFEST_NAME)),
        "conversation": _read_json(_cache_path(cache_path, CONVERSATION_NAME)),
        "operations": _read_jsonl(_cache_path(cache_path, OPS_NAME)),
        "contract": _read_json(contract_path) if contract_path.exists() else None,
    }


def _round_float(value: Any, digits: int = 4) -> float:
    return round(float(value or 0.0), digits)


def _sleep_summary_sha1(memories: list[str], source: str) -> str:
    payload = {
        "source": str(source or ""),
        "memories": [str(memory or "") for memory in (memories or [])],
    }
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _normalize_source_memory_id(value: Any, mapping: dict[str, str]) -> str:
    text = str(value or "")
    if len(text) == 16 and all(ch in "0123456789abcdef" for ch in text):
        existing = mapping.get(text)
        if existing is not None:
            return existing
        normalized = f"auto_source_{len(mapping) + 1}"
        mapping[text] = normalized
        return normalized
    return text


def _normalize_state_json(raw: str) -> dict:
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def replay_generated_write_ops(
    beam: BeamMemory,
    operations: list[dict],
    *,
    session_id: str | None = None,
) -> dict:
    session = session_id or beam.session_id
    counts = {
        "store_llm_extraction": 0,
        "store_spo_facts": 0,
        "store_conclusions": 0,
        "apply_card_patch": 0,
    }

    for op in operations:
        kind = op.get("op")
        if kind == "store_llm_extraction":
            beam._store_llm_extraction(
                session,
                int(op.get("msg_idx") or 0),
                op.get("parsed") or {},
                str(op.get("ctx") or ""),
                source_memory_id=op.get("source_memory_id"),
            )
            counts[kind] += 1
            continue

        if kind == "store_spo_facts":
            for fact in op.get("facts") or []:
                fid = _spo_fact_id(
                    session,
                    fact.get("subject", ""),
                    fact.get("predicate", "stated"),
                    fact.get("object", ""),
                )
                beam.conn.execute(
                    "INSERT OR IGNORE INTO facts "
                    "(fact_id, session_id, subject, predicate, object, timestamp, source_msg_id, confidence) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        fid,
                        session,
                        fact.get("subject", ""),
                        fact.get("predicate", "stated"),
                        fact.get("object", ""),
                        fact.get("timestamp", ""),
                        fact.get("source_msg_id", ""),
                        fact.get("confidence", 0.7),
                    ),
                )
            counts[kind] += 1
            continue

        if kind == "store_conclusions":
            for concl in op.get("conclusions") or []:
                text = (concl or {}).get("text", "")
                theme = (concl or {}).get("theme", "general")
                source = (concl or {}).get("source")
                if not text:
                    continue
                key = beam._conclusion_fact_key(theme, text, source)
                source_id = (
                    f"concl:{beam._normalize_fact_key_component(theme)}:"
                    f"{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"
                )
                changed = beam._store_memoria_fact(
                    session,
                    0,
                    "conclusion",
                    key,
                    text,
                    text,
                    float((concl or {}).get("confidence", 0.7)),
                    source_memory_id=source_id,
                )
                if changed:
                    beam._refresh_conclusion_aggregate(
                        session,
                        theme,
                        0,
                        source_memory_id=f"agg:{beam._normalize_fact_key_component(theme)}",
                    )
            counts[kind] += 1
            continue

        if kind == "apply_card_patch":
            beam._apply_card_patch(
                session,
                op.get("patch") or {},
                int(op.get("msg_idx") or 0),
            )
            counts[kind] += 1
            continue

        if kind in {"chat_response", "extraction_chat_response", "llm_chat_response", "sleep_summary_response"}:
            continue

        raise ValueError(f"unsupported generated write op: {kind!r}")

    beam.conn.commit()
    return counts


def _prime_replay_base_ingest(beam: BeamMemory, conversation: list[dict]) -> None:
    original_use_cloud = getattr(beam, "use_cloud", False)
    original_llm_client = getattr(beam, "_llm_client", None)
    original_extraction_client = getattr(beam, "_extraction_client", None)
    try:
        beam.use_cloud = False
        beam._llm_client = None
        beam._extraction_client = None
        ingest_conversation(beam, conversation, llm=None)
    finally:
        beam.use_cloud = original_use_cloud
        beam._llm_client = original_llm_client
        beam._extraction_client = original_extraction_client


class ReplayExtractionClient:
    """Replay recorded model-side write outputs through the normal ingest path."""

    def __init__(self, events: list[dict]):
        self._events = [
            event for event in events
            if event.get("op") in {
                "extraction_chat_response",
                "store_spo_facts",
                "store_conclusions",
                "apply_card_patch",
            }
        ]
        self._idx = 0

    def _next(self, kind: str) -> dict:
        if self._idx >= len(self._events):
            raise AssertionError(f"no recorded event left for {kind}")
        event = self._events[self._idx]
        self._idx += 1
        actual = event.get("op")
        if actual != kind:
            raise AssertionError(f"expected recorded event {kind!r}, got {actual!r}")
        return event

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        event = self._next("extraction_chat_response")
        user_content = ""
        if messages:
            user_content = str((messages[-1] or {}).get("content") or "")
        prompt_sha1 = hashlib.sha1(user_content.encode("utf-8")).hexdigest()
        expected_sha1 = event.get("prompt_sha1")
        if expected_sha1 and expected_sha1 != prompt_sha1:
            raise AssertionError(
                f"recorded chat prompt mismatch: expected {expected_sha1}, got {prompt_sha1}"
            )
        return str(event.get("response") or "")

    def extract_facts(self, messages: list) -> list:
        event = self._next("store_spo_facts")
        return list(event.get("facts") or [])

    def extract_conclusions(self, messages: list) -> list:
        event = self._next("store_conclusions")
        return list(event.get("conclusions") or [])

    def update_card(
        self,
        current_card: dict | None,
        agenda: dict,
        evidence_rows: list[dict],
        session_overview: dict | None = None,
    ) -> dict:
        event = self._next("apply_card_patch")
        recorded_agenda = event.get("agenda") or {}
        if recorded_agenda.get("agenda_key") != agenda.get("agenda_key"):
            raise AssertionError(
                "recorded card agenda mismatch: "
                f"expected {recorded_agenda.get('agenda_key')!r}, got {agenda.get('agenda_key')!r}"
            )
        return dict(event.get("patch") or {})

    def refresh_session_overview(self, live_cards: list[dict]) -> dict:
        event = self._next("apply_card_patch")
        recorded_agenda = event.get("agenda") or {}
        if recorded_agenda.get("agenda_key") != "session:overview":
            raise AssertionError(
                "expected recorded session overview patch, "
                f"got {recorded_agenda.get('agenda_key')!r}"
            )
        return dict(event.get("patch") or {})

    def assert_drained(self) -> None:
        if self._idx != len(self._events):
            remaining = [event.get("op") for event in self._events[self._idx:self._idx + 10]]
            raise AssertionError(f"unused recorded events remain: {remaining}")


class RecordingLLMClient:
    """Wrap the main LLM client and record chat completions used during ingest."""

    def __init__(self, inner: LLMClient, events: list[dict]):
        self._inner = inner
        self._events = events

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        response = self._inner.chat(messages, temperature=temperature, max_tokens=max_tokens)
        user_content = ""
        if messages:
            user_content = str((messages[-1] or {}).get("content") or "")
        self._events.append({
            "op": "llm_chat_response",
            "prompt_sha1": hashlib.sha1(user_content.encode("utf-8")).hexdigest(),
            "response": response,
        })
        return response


class ReplayLLMClient:
    """Replay cached main-LLM chat completions through the normal ingest path."""

    def __init__(self, events: list[dict], *, model: str = "cached-llm"):
        self._events = [event for event in events if event.get("op") == "llm_chat_response"]
        self._events_by_sha: dict[str, deque[dict]] = {}
        self._events_without_sha: deque[dict] = deque()
        self._remaining = 0
        self._lock = threading.Lock()
        for event in self._events:
            prompt_sha1 = str(event.get("prompt_sha1") or "").strip()
            if prompt_sha1:
                self._events_by_sha.setdefault(prompt_sha1, deque()).append(event)
            else:
                self._events_without_sha.append(event)
            self._remaining += 1
        self.model = model
        self.api_key = ""
        self.base_url = "cached://local"
        self.last_error_message = ""

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        user_content = ""
        if messages:
            user_content = str((messages[-1] or {}).get("content") or "")
        prompt_sha1 = hashlib.sha1(user_content.encode("utf-8")).hexdigest()
        with self._lock:
            event = None
            bucket = self._events_by_sha.get(prompt_sha1)
            if bucket:
                event = bucket.popleft()
            elif self._events_without_sha:
                event = self._events_without_sha.popleft()
            if event is None:
                available = [
                    sha for sha, remaining in self._events_by_sha.items()
                    if remaining
                ][:5]
                raise AssertionError(
                    "no recorded llm chat event left for prompt "
                    f"{prompt_sha1}; remaining hashes={available}"
                )
            self._remaining -= 1
        expected_sha1 = event.get("prompt_sha1")
        if expected_sha1 and expected_sha1 != prompt_sha1:
            raise AssertionError(
                f"recorded llm prompt mismatch: expected {expected_sha1}, got {prompt_sha1}"
            )
        return str(event.get("response") or "")

    def assert_drained(self) -> None:
        if self._remaining != 0:
            remaining = [
                sha for sha, bucket in self._events_by_sha.items()
                if bucket
            ][:10]
            raise AssertionError(
                "unused recorded llm events remain: "
                f"{self._remaining} hashes={remaining}"
            )


class RecordingSleepSummarizer:
    """Wrap local_llm.summarize_memories and record summary outputs."""

    def __init__(self, inner: Any, events: list[dict]):
        self._inner = inner
        self._events = events

    def __call__(self, memories: list[str], source: str = "") -> str | None:
        response = self._inner(memories, source=source)
        self._events.append({
            "op": "sleep_summary_response",
            "source": str(source or ""),
            "memories_sha1": _sleep_summary_sha1(memories, source),
            "response": response,
        })
        return response


class ReplaySleepSummarizer:
    """Replay cached local_llm.summarize_memories outputs."""

    def __init__(self, events: list[dict]):
        self._events = [event for event in events if event.get("op") == "sleep_summary_response"]
        self._idx = 0

    def __call__(self, memories: list[str], source: str = "") -> str | None:
        if self._idx >= len(self._events):
            raise AssertionError("no recorded sleep summary event left")
        event = self._events[self._idx]
        self._idx += 1
        actual_source = str(source or "")
        expected_source = str(event.get("source") or "")
        if expected_source != actual_source:
            raise AssertionError(
                f"recorded sleep summary source mismatch: expected {expected_source!r}, got {actual_source!r}"
            )
        expected_sha1 = event.get("memories_sha1")
        actual_sha1 = _sleep_summary_sha1(memories, source)
        if expected_sha1 and expected_sha1 != actual_sha1:
            raise AssertionError(
                f"recorded sleep summary payload mismatch: expected {expected_sha1}, got {actual_sha1}"
            )
        response = event.get("response")
        return None if response is None else str(response)

    def assert_drained(self) -> None:
        if self._idx != len(self._events):
            remaining = self._events[self._idx:self._idx + 10]
            raise AssertionError(f"unused recorded sleep summary events remain: {len(remaining)}")


@contextlib.contextmanager
def _patched_sleep_summarizer(replacement: Any):
    from edumem.core import local_llm

    original = local_llm.summarize_memories
    local_llm.summarize_memories = replacement
    try:
        yield
    finally:
        local_llm.summarize_memories = original


def replay_generated_write_cache(
    cache_dir: str | Path,
    *,
    db_path: str | Path | None = None,
    beam: BeamMemory | None = None,
    session_id: str | None = None,
    prime_conversation: bool | None = None,
) -> BeamMemory:
    payload = load_generated_write_cache(cache_dir)
    manifest = payload["manifest"]
    conversation = payload["conversation"]
    target_session = session_id or manifest.get("session_id") or "write-cache-session"
    replay_mode = str(manifest.get("replay_mode") or "").strip().lower()
    should_prime = (
        bool(prime_conversation)
        if prime_conversation is not None
        else bool(manifest.get("replay_base_ingest"))
    )

    if beam is None:
        if db_path is None:
            raise ValueError("db_path is required when beam is not provided")
        beam = BeamMemory(db_path=db_path, session_id=target_session)

    if replay_mode == "cached_live_ingest":
        if beam is None:
            raise ValueError("beam is required for cached_live_ingest replay")
        beam.use_cloud = True
        replay_client = ReplayExtractionClient(payload["operations"])
        replay_llm = ReplayLLMClient(payload["operations"])
        replay_sleep = ReplaySleepSummarizer(payload["operations"])
        beam._llm_client = replay_llm
        beam._extraction_client = replay_client
        with _patched_sleep_summarizer(replay_sleep):
            ingest_conversation(beam, conversation, llm=replay_llm)
            while True:
                result = beam.sleep(force=True)
                if result.get("status") in ("no_op", "error"):
                    break
        replay_client.assert_drained()
        replay_llm.assert_drained()
        replay_sleep.assert_drained()
        return beam

    if should_prime:
        _prime_replay_base_ingest(beam, conversation)
    replay_generated_write_ops(beam, payload["operations"], session_id=target_session)
    return beam


def materialize_generated_write_cache_db(
    cache_dir: str | Path,
    *,
    db_path: str | Path | None = None,
    session_id: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Materialize a canonical SQLite DB snapshot from a write-cache artifact."""
    cache_path = Path(cache_dir)
    target_db = Path(db_path) if db_path is not None else _cache_path(cache_path, FINAL_DB_NAME)
    if target_db.exists():
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite existing DB snapshot: {target_db}")
        for sidecar in (target_db, Path(f"{target_db}-wal"), Path(f"{target_db}-shm")):
            try:
                if sidecar.exists():
                    sidecar.unlink()
            except FileNotFoundError:
                pass
    beam = replay_generated_write_cache(
        cache_path,
        db_path=target_db,
        session_id=session_id,
    )
    try:
        expected_contract = load_generated_write_cache(cache_path).get("contract")
        if expected_contract is not None:
            actual_contract = capture_write_contract(beam, session_id=session_id)
            if actual_contract != expected_contract:
                raise AssertionError(
                    "materialized final.db diverged from cached contract: "
                    f"expected={_contract_shape(expected_contract)} actual={_contract_shape(actual_contract)}"
                )
    finally:
        beam.conn.close()
    return target_db


def capture_write_contract(beam: BeamMemory, *, session_id: str | None = None) -> dict:
    session = session_id or beam.session_id
    source_id_map: dict[str, str] = {}

    facts_rows = beam.conn.execute(
        "SELECT subject, predicate, object, timestamp, source_msg_id, confidence "
        "FROM facts WHERE session_id = ? "
        "ORDER BY subject, predicate, object, timestamp, source_msg_id",
        (session,),
    ).fetchall()

    memoria_rows = beam.conn.execute(
        "SELECT fact_type, key, value, COALESCE(previous_value, ''), COALESCE(version_id, 0), "
        "COALESCE(source_memory_id, '') "
        "FROM memoria_facts WHERE session_id = ? "
        "ORDER BY fact_type, key, COALESCE(version_id, 0), value",
        (session,),
    ).fetchall()

    kg_rows = beam.conn.execute(
        "SELECT subject, predicate, object, COALESCE(message_idx, 0) "
        "FROM memoria_kg WHERE session_id = ? "
        "ORDER BY subject, predicate, object, COALESCE(message_idx, 0)",
        (session,),
    ).fetchall()

    timeline_rows = beam.conn.execute(
        "SELECT date, COALESCE(message_idx, 0), COALESCE(description, ''), COALESCE(source, '') "
        "FROM memoria_timelines WHERE session_id = ? "
        "ORDER BY date, COALESCE(message_idx, 0), COALESCE(description, '')",
        (session,),
    ).fetchall()

    card_rows = beam.conn.execute(
        "SELECT card_type, card_key, title, summary, state_json, confidence, "
        "COALESCE(version_id, 0), COALESCE(valid_to_msg_idx, -1) "
        "FROM memory_cards WHERE session_id = ? "
        "ORDER BY card_type, card_key, COALESCE(version_id, 0)",
        (session,),
    ).fetchall()

    evidence_rows = beam.conn.execute(
        "SELECT c.card_key, e.evidence_table, COALESCE(e.message_idx, 0), e.snippet, e.weight "
        "FROM memory_card_evidence e "
        "JOIN memory_cards c ON c.id = e.card_id "
        "WHERE c.session_id = ? "
        "ORDER BY c.card_key, e.evidence_table, COALESCE(e.message_idx, 0), e.snippet",
        (session,),
    ).fetchall()

    episodic_rows = beam.conn.execute(
        "SELECT content, COALESCE(source, ''), importance, COALESCE(metadata_json, '{}'), "
        "COALESCE(veracity, 'unknown') "
        "FROM episodic_memory WHERE session_id = ? "
        "ORDER BY COALESCE(source, ''), content",
        (session,),
    ).fetchall()

    consolidation_rows = beam.conn.execute(
        "SELECT items_consolidated, summary_preview "
        "FROM consolidation_log WHERE session_id = ? "
        "ORDER BY id",
        (session,),
    ).fetchall()

    return {
        "session_id": session,
        "facts": [
            {
                "subject": row[0],
                "predicate": row[1],
                "object": row[2],
                "timestamp": row[3],
                "source_msg_id": row[4],
                "confidence": _round_float(row[5]),
            }
            for row in facts_rows
        ],
        "memoria_facts": [
            {
                "fact_type": row[0],
                "key": row[1],
                "value": row[2],
                "previous_value": row[3],
                "version_id": int(row[4] or 0),
                "source_memory_id": _normalize_source_memory_id(row[5], source_id_map),
            }
            for row in memoria_rows
        ],
        "memoria_kg": [
            {
                "subject": row[0],
                "predicate": row[1],
                "object": row[2],
                "message_idx": int(row[3] or 0),
            }
            for row in kg_rows
        ],
        "memoria_timelines": [
            {
                "date": row[0],
                "message_idx": int(row[1] or 0),
                "description": row[2],
                "source": row[3],
            }
            for row in timeline_rows
        ],
        "memory_cards": [
            {
                "card_type": row[0],
                "card_key": row[1],
                "title": row[2],
                "summary": row[3],
                "state": _normalize_state_json(row[4]),
                "confidence": _round_float(row[5]),
                "version_id": int(row[6] or 0),
                "valid_to_msg_idx": int(row[7] or -1),
            }
            for row in card_rows
        ],
        "memory_card_evidence": [
            {
                "card_key": row[0],
                "evidence_table": row[1],
                "message_idx": int(row[2] or 0),
                "snippet": row[3],
                "weight": _round_float(row[4]),
            }
            for row in evidence_rows
        ],
        "episodic_memory": [
            {
                "content": row[0],
                "source": row[1],
                "importance": _round_float(row[2]),
                "metadata": _normalize_state_json(row[3]),
                "veracity": row[4],
            }
            for row in episodic_rows
        ],
        "consolidation_log": [
            {
                "items_consolidated": int(row[0] or 0),
                "summary_preview": row[1],
            }
            for row in consolidation_rows
        ],
    }


class RecordingExtractionClient:
    """Wrap a real ExtractionClient and record parsed write operations."""

    def __init__(self, inner: ExtractionClient, events: list[dict] | None = None):
        self._inner = inner
        self.events: list[dict] = events if events is not None else []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        response = self._inner.chat(messages, temperature=temperature, max_tokens=max_tokens)
        user_content = ""
        if messages:
            user_content = str((messages[-1] or {}).get("content") or "")
        self.events.append({
            "op": "extraction_chat_response",
            "prompt_sha1": hashlib.sha1(user_content.encode("utf-8")).hexdigest(),
            "response": response,
        })
        return response

    def extract_facts(self, messages: list) -> list:
        facts = self._inner.extract_facts(messages)
        self.events.append({
            "op": "store_spo_facts",
            "facts": facts,
        })
        return facts

    def extract_conclusions(self, messages: list) -> list:
        conclusions = self._inner.extract_conclusions(messages)
        self.events.append({
            "op": "store_conclusions",
            "conclusions": conclusions,
        })
        return conclusions

    def update_card(
        self,
        current_card: dict | None,
        agenda: dict,
        evidence_rows: list[dict],
        session_overview: dict | None = None,
    ) -> dict:
        patch = self._inner.update_card(current_card, agenda, evidence_rows, session_overview)
        msg_idx = max((row.get("message_idx") or 0 for row in evidence_rows), default=0)
        self.events.append({
            "op": "apply_card_patch",
            "msg_idx": int(msg_idx),
            "patch": patch,
            "agenda": agenda,
        })
        return patch

    def refresh_session_overview(self, live_cards: list[dict]) -> dict:
        patch = self._inner.refresh_session_overview(live_cards)
        msg_idx = max(
            (
                card.get("source_end_msg_idx")
                or card.get("source_start_msg_idx")
                or 0
                for card in live_cards
            ),
            default=0,
        )
        self.events.append({
            "op": "apply_card_patch",
            "msg_idx": int(msg_idx),
            "patch": patch,
            "agenda": {"agenda_type": "session", "agenda_key": "session:overview"},
        })
        return patch


def _contract_shape(contract: dict) -> dict[str, int]:
    return {
        "facts": len(contract.get("facts", [])),
        "memoria_facts": len(contract.get("memoria_facts", [])),
        "memoria_kg": len(contract.get("memoria_kg", [])),
        "memoria_timelines": len(contract.get("memoria_timelines", [])),
        "memory_cards": len(contract.get("memory_cards", [])),
        "memory_card_evidence": len(contract.get("memory_card_evidence", [])),
        "episodic_memory": len(contract.get("episodic_memory", [])),
        "consolidation_log": len(contract.get("consolidation_log", [])),
    }


def _cleanup_temp_tree(path: str | Path) -> None:
    try:
        gc.collect()
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def build_generated_write_cache_for_conversation(
    conversation: list[dict],
    cache_dir: str | Path,
    *,
    session_id: str,
    conversation_id: str,
    scale: str = "100K",
    llm_model: str = "qwen3.6",
) -> dict:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    _write_json(_cache_path(cache_path, CONVERSATION_NAME), conversation)
    _write_json(
        _cache_path(cache_path, MANIFEST_NAME),
        {
            "session_id": session_id,
            "conversation_id": conversation_id,
            "scale": scale,
            "llm_model": llm_model,
            "replay_mode": "cached_live_ingest",
        },
    )

    recorded_events: list[dict] = []
    llm = RecordingLLMClient(LLMClient(model=llm_model), recorded_events)
    live_contract: dict

    live_td = tempfile.mkdtemp()
    try:
        db_path = Path(live_td) / "live-build.db"
        beam = BeamMemory(db_path=db_path, session_id=session_id, use_cloud=True, llm_client=llm)
        recorder = RecordingExtractionClient(ExtractionClient(), recorded_events)
        from edumem.core import local_llm
        summary_recorder = RecordingSleepSummarizer(local_llm.summarize_memories, recorded_events)
        beam._extraction_client = recorder

        with _patched_sleep_summarizer(summary_recorder):
            ingest_conversation(beam, conversation, llm=llm)
            while True:
                result = beam.sleep(force=True)
                if result.get("status") in ("no_op", "error"):
                    break
        live_contract = capture_write_contract(beam, session_id=session_id)
        beam.conn.close()
        del beam

        _write_jsonl(_cache_path(cache_path, OPS_NAME), recorded_events)
        _write_json(_cache_path(cache_path, CONTRACT_NAME), live_contract)
    finally:
        _cleanup_temp_tree(live_td)

    replay_contract: dict | None = None
    replay_td = tempfile.mkdtemp()
    try:
        replay_db = Path(replay_td) / "replay.db"
        replay_beam = replay_generated_write_cache(
            cache_path,
            beam=BeamMemory(db_path=replay_db, session_id=session_id, use_cloud=True),
            session_id=session_id,
        )
        try:
            replay_contract = capture_write_contract(replay_beam, session_id=session_id)
        finally:
            replay_beam.conn.close()
            del replay_beam
            gc.collect()
    finally:
        if replay_contract is not None:
            _write_json(_cache_path(cache_path, REPLAY_CONTRACT_NAME), replay_contract)
        _cleanup_temp_tree(replay_td)
    if replay_contract != live_contract:
        raise AssertionError(
            "replay contract diverged from live contract: "
            f"live={_contract_shape(live_contract)} replay={_contract_shape(replay_contract)}"
        )
    final_db_path = materialize_generated_write_cache_db(cache_path, session_id=session_id)
    return {
        "cache_dir": str(cache_path),
        "session_id": session_id,
        "conversation_id": conversation_id,
        "operation_count": len(recorded_events),
        "final_db": str(final_db_path),
    }
