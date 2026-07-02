from __future__ import annotations

import os


def _assemble_memory_context(memories: list, max_chars: int) -> tuple[str, list]:
    """Sort memories by relevance descending (stable), deduplicate, skip score<0.20,
    build [Memory] context string up to max_chars. Supports per-type sub-budgets
    (EDUMEM_SUB_BUDGET_FACT, EDUMEM_SUB_BUDGET_TIMELINE, EDUMEM_SUB_BUDGET_MEMORIA)
    so one type doesn't crowd out another. Mutates each mem's 'final_context_included' key.
    Returns (context_str, memories)."""
    fact_budget = int(os.environ.get("EDUMEM_SUB_BUDGET_FACT", "12000"))
    timeline_budget = int(os.environ.get("EDUMEM_SUB_BUDGET_TIMELINE", "6000"))
    memoria_budget = int(os.environ.get("EDUMEM_SUB_BUDGET_MEMORIA", "6000"))

    def _score(m):
        s = m.get("score", m.get("relevance", 0))
        return s if isinstance(s, (int, float)) else 0.0

    sorted_mems = sorted(memories, key=_score, reverse=True)

    for mem in memories:
        mem.setdefault("final_context_included", False)

    seen_content: set = set()
    lanes = {"fact": [], "timeline": [], "summary": [], "memoria": [], "other": []}

    for mem in sorted_mems:
        content = mem.get("content", "")
        content_key = content[:100]
        if content_key in seen_content:
            continue
        seen_content.add(content_key)

        score = _score(mem)
        if score < 0.20:
            continue

        t = mem.get("type", "other")
        if t not in lanes:
            t = "other"
        lanes[t].append((score, mem))

    sub_budgets = {"fact": fact_budget, "timeline": timeline_budget,
                   "summary": None, "memoria": memoria_budget, "other": None}
    selected = []

    for t in ("fact", "timeline", "summary", "memoria", "other"):
        items = sorted(lanes[t], key=lambda x: -x[0])
        budget = sub_budgets[t]
        used = 0
        for score, mem in items:
            content = mem.get("content", "")
            if budget is not None and used + len(content) > budget:
                continue
            selected.append((score, mem))
            used += len(content)

    selected.sort(key=lambda x: -x[0])

    parts: list = []
    total_chars = 0
    budget_hit = False
    omitted_by_budget = 0

    for score, mem in selected:
        content = mem.get("content", "")
        if not budget_hit and total_chars + len(content) > max_chars:
            budget_hit = True
            remaining = max_chars - total_chars
            if remaining > 100:
                mem["final_context_included"] = True
                parts.append(f"[Memory] {content[:remaining]}...")
            else:
                omitted_by_budget += 1
            continue

        if budget_hit:
            omitted_by_budget += 1
            continue

        mem["final_context_included"] = True
        parts.append(f"[Memory] {content}")
        total_chars += len(content)

    if omitted_by_budget > 0:
        suffix = "" if omitted_by_budget == 1 else "s"
        parts.append(
            f"... {omitted_by_budget} additional memory item{suffix} "
            f"omitted due to context budget ..."
        )

    return "\n\n".join(parts), sorted_mems


def assemble_card_context(cards: list, evidence: list, max_chars: int = 16000) -> str:
    """Render §6.6 card-context layout: cards first, evidence second.

    Layout:
        [Card TOPIC] Security hardening
        Security work progressed from password hashing to RBAC...

        [Card CHANGE] Deployment window
        Current: February 5 through February 12...

        [Evidence]
        - MSGIDX:40 password hashing was added...
        - MSGIDX:72 RBAC was introduced...

    Cards are uppercase card_type. Evidence lines include MSGIDX:N when
    message_idx is available. Truncates to max_chars (cards section first,
    then evidence) so the budget is always respected.
    """
    parts: list[str] = []
    total = 0

    # --- Cards section (first) ---
    for card in cards:
        card_type = (card.get('card_type') or 'card').upper()
        title = card.get('title', '').strip()
        summary = card.get('summary', '').strip()
        block = f"[Card {card_type}] {title}\n{summary}"
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 40:
                parts.append(block[:remaining] + '...')
                total += remaining
            break
        parts.append(block)
        total += len(block)

    # --- Evidence section (second) ---
    ev_lines: list[str] = []
    for ev in evidence:
        msg_idx = ev.get('message_idx')
        snippet = (ev.get('snippet') or '').strip()
        if not snippet:
            continue
        if msg_idx is not None:
            line = f"- MSGIDX:{msg_idx} {snippet}"
        else:
            line = f"- {snippet}"
        ev_lines.append(line)

    if ev_lines:
        ev_block = "[Evidence]\n" + "\n".join(ev_lines)
        if total + len(ev_block) <= max_chars:
            parts.append(ev_block)

    return "\n\n".join(parts)
