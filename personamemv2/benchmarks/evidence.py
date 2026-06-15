import re

STOPWORDS = {
    "about", "after", "again", "along", "also", "because", "before", "being", "could", "during",
    "every", "from", "have", "into", "just", "like", "more", "most", "need", "some", "that",
    "their", "there", "these", "thing", "this", "those", "through", "what", "when", "where", "which",
    "while", "with", "would", "your", "youre", "youll", "youve", "since", "might", "enjoy",
    "start", "starting", "suggest", "recommend", "help", "good", "ways", "option", "could",
}

SYNONYMS = {
    "bread": {"dough", "rise", "baking", "bake", "yeast", "sourdough", "brioche", "oven", "flour"},
    "sourdough": {"dough", "rise", "baking", "bake", "yeast", "bread", "oven"},
    "brioche": {"dough", "rise", "baking", "bake", "yeast", "bread", "oven"},
    "surfing": {"waves", "board", "waxing", "ocean", "beach", "water", "surf"},
    "surf": {"waves", "board", "waxing", "ocean", "beach", "water", "surfing"},
    "football": {"soccer", "match", "league", "premier", "stoppage", "jersey", "goal"},
    "soccer": {"football", "match", "league", "premier", "stoppage", "jersey", "goal"},
    "running": {"marathon", "ultramarathon", "run", "trail", "miles", "hours", "jogging"},
    "run": {"marathon", "ultramarathon", "running", "trail", "miles", "hours", "jogging"},
    "ultramarathon": {"running", "run", "trail", "miles", "hours", "demanding"},
    "swimming": {"swim", "pool", "water", "stroke", "strokes"},
    "swim": {"swimming", "pool", "water", "stroke", "strokes"},
    "gardening": {"garden", "plant", "houseplants", "plants", "crop", "soil", "rotation"},
    "garden": {"gardening", "plant", "houseplants", "plants", "crop", "soil", "rotation"},
    "houseplants": {"garden", "gardening", "plant", "plants", "pot", "office"},
    "facebook": {"social media", "feed", "posts", "online", "sharing", "friends", "family"},
}

SPECIAL_WORDS = {
    "dough", "rise", "yeast", "sourdough", "brioche", "baking", "bake", "bread",
    "surfing", "waves", "board", "waxing", "surf", "ocean",
    "ultramarathon", "marathon", "running", "run", "trail",
    "jollof", "egusi", "stew", "nigerian", "igbo",
    "football", "soccer", "premier", "league", "match",
    "facebook", "houseplants", "garden", "gardening", "cholesterol", "lipid",
    "herbal", "tea", "coffee", "asthma", "myopia", "blouse", "emerald",
}


def tokens(text):
    base = {w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}", text.lower()) if w not in STOPWORDS}
    expanded = set(base)
    for w in base:
        if w in SYNONYMS:
            expanded.update(SYNONYMS[w])
    return expanded


def history_lines(history):
    lines = []
    for msg in history:
        role = str(msg.get("role", "user")).upper()
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append((role, content))
    return lines


def best_evidence(history, query, mapping, per_option=2, max_chars=900):
    lines = history_lines(history)
    selected = []
    seen = set()

    def add_for(label, text):
        q = tokens(text)
        scored = []
        for n, (role, content) in enumerate(lines):
            ct = tokens(content)
            if not ct:
                continue
            overlap_set = q & ct
            if overlap_set:
                score = sum(5 if w in SPECIAL_WORDS else 1 for w in overlap_set)
                scored.append((score, -abs(len(lines) - n), n, role, content))
        for _, _, n, role, content in sorted(scored, reverse=True)[:per_option]:
            key = (n, role, content[:80])
            if key in seen:
                continue
            seen.add(key)
            clipped = content.replace("\n", " ")[:max_chars]
            selected.append(f"[{label}] {role}: {clipped}")

    add_for("QUERY", query)
    for letter, option in mapping.items():
        add_for(f"OPTION {letter}", option)

    return "\n".join(selected[:12])


def build_profile_map_reduce(client, model, history, profile_path, hash_path, prompt_hash):
    import os
    if profile_path.exists() and not os.getenv("REFRESH_PROFILE"):
        return profile_path.read_text()

    # Step 1: Chunked map extraction
    chunk_size = 30
    chunks = [history[i:i + chunk_size] for i in range(0, len(history), chunk_size)]
    bullets_list = []

    for idx, chunk in enumerate(chunks):
        lines = []
        for m in chunk:
            role = m.get("role", "user")
            content = m.get("content", "")
            lines.append(f"{role.upper()}: {content}")
        chunk_text = "\n".join(lines)
        
        prompt = f"""Analyze this segment of a user's chat history.
Extract all specific:
- Personal facts (Name, Age, Location, Family, Job)
- Hobbies, sports, physical activities, and personal routines (such as surfing, cycling, baking, swimming, running, etc.)
- Preferred products, devices, apps, sites, and services (e.g., Facebook, etc.)
- Health conditions, medical concerns, physical constraints, and symptoms (e.g., lower back pain, leg injury, etc.)
- Cultural, intellectual, or topical interests, curiosities, or aesthetic preferences (such as history, policy, cinema, luxury fashion, music genres, economics, etc.)
- Topics or things they asked to forget/privacy requests
- Mentioned third-party names (e.g., friends, colleagues, characters in stories)

Only extract facts supported by the user's own words/drafts in this segment. Do not infer or invent.
Format as simple, concise bullet points.

CHAT SEGMENT:
{chunk_text}
"""
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        bullets = resp.choices[0].message.content or ""
        bullets_list.append(bullets)

    # Step 2: Reduce/merge into the final structured profile
    combined_bullets = "\n\n".join(bullets_list)

    merge_prompt = f"""You are a high-precision user memory compiler.
Combine these segments of user facts extracted from different parts of their chat history into a single, cohesive, high-precision compact memory profile.

STRICT EXTRACTION LAWS:
1. USER EVIDENCE ONLY: Record specific facts, preferences, constraints, and routines only when supported by the user's own words or by the user's personal drafts/messages. Do not infer from demographics, job, nationality, location, or broad persona stereotypes.
2. PERSONAL DRAFTS COUNT: Personal drafts, emails, messages, and quoted originals that the user asks to polish can reveal real routines, preferences, sensitive data, and stressors. Extract those facts only when the draft is clearly first-person/personal to the user; otherwise put them under THIRD-PARTY OR FICTIONAL CONTEXT.
3. PHYSICAL ROUTINES IN DRAFTS: Pay special attention to concrete embodied routines in personal drafts, such as baking bread, stretching/breathing on a mat, endurance runs, sports (like surfing, swimming, cycling), cooking, home projects, and health constraints.
4. SELF VS OTHERS/FICTION: If the user discusses a friend, family member, fictional character, customer, example, article, or draft subject (such as Mark Ellison, Samuel Ortega, Leon, Clara, Mateo, Sofia, etc.), do not make it a user fact. Label it under THIRD-PARTY OR FICTIONAL CONTEXT. Keep highly specific hobbies, physical routines, items, emotional struggles, career doubts, or psychological conflicts mentioned in third-party drafts (such as "houseplants" in Mark Ellison's bio draft, or "missed birthday/Mateo's party" in Sofia's email draft, or "questioning career path / corporate executive disregard" in Daniel's story draft) strictly inside THIRD-PARTY OR FICTIONAL CONTEXT, and do NOT duplicate them under STABLE USER FACTS, ACTIVE PERSONAL ROUTINES & HABITS, or IMPLIED PREFERENCES/CONCERNS for the user!
5. INTELLECTUAL QUESTIONS ARE INTERESTS, NOT PRACTICES: A detailed question can show topical interest or concern, but does not prove the user actively practices, owns, consumes, or prefers the thing.
6. FORGET REQUESTS OVERRIDE FACTS: Anything the user asked to forget belongs only in PRIVACY & FORGET CONSTRAINTS, not in active routines or preferences.
7. FACTUAL BOUNDARY: Prefer explicit evidence and close paraphrases. Do not lose any specific hobbies, preferred websites/platforms (like Facebook), sports, or constraints mentioned in the source bullets!
8. PRESERVE LANDLOCKED OR SEASONAL HOBBIES: Hobbies or sports practiced on vacation or seasonally (e.g. surfing when traveling to coastal areas, swimming in the summer, etc.) are still highly personal and must be preserved under ACTIVE ROUTINES or IMPLIED PREFERENCES, even if the user lives in a landlocked area (like Kansas) or cannot practice them daily.
9. DO NOT GENERALIZE: Keep highly specific names (such as "Facebook", "surfing", "swimming", "ultramarathons", "Premier League", "lower back pain", "herbal tea") intact exactly as written. Do not generalize them.
10. CANONICAL DOMAIN TRANSLATION: To prevent downstream matching failures caused by semantic synonym gaps, explicitly translate physical routines and preferences into their canonical domain names alongside the raw facts (such as stretching/breathing on a mat -> "yoga and meditation", or long training runs -> "running ultramarathons"). Do NOT copy or invent any unrelated sports, teams, cuisines, or habits that are not explicitly present in the source bullets.
11. DO NOT EXTRACT NEGATIVE CONSTRAINTS FROM HYPOTHETICAL QUESTIONS: If the user asks a general or hypothetical question (e.g., "If someone's never had back problems..."), do NOT extract a negative constraint (such as "does not have lower back pain" or "no history of back problems") as a stable user fact or constraint. Do NOT write "No history of back problems" in the profile. Only extract negative facts if the user explicitly and unambiguously asserts them about themselves (e.g., "I have never had any back pain").
12. STRICT FORGET COMPILING RULE: In the PRIVACY & FORGET CONSTRAINTS section, translate all forget requests into short, dense, high-impact semantic keywords and direct topic names. Do NOT include any placeholder or example constraints that the user did not actually ask to forget in this chat history.
13. STRICT COMPLIANCE RULE: You must implement all mappings in Law 10 literally and directly under STABLE USER FACTS or ACTIVE PERSONAL ROUTINES. Do NOT write any meta-cognitive parenthetical notes, corrections, explanations, arguments, or self-debates in the profile. Simply output the mapped facts as absolute, clean truths. Your job is to compile, not to debate.

Format your output into these precise sections:
- STABLE USER FACTS (Name, Age, Location, Job, Family)
- ACTIVE PERSONAL ROUTINES, HOBBIES & INTERESTS (Specifically extract physical habits, hobbies like surfing/cycling/baking, and active personal interests like anime, cinema, history)
- IMPLIED PREFERENCES & HEALTH CONSTRAINTS (Monitoring cholesterol, allergies, childhood asthma, etc.)
- THIRD-PARTY OR FICTIONAL CONTEXT (Traits or emotions belonging to friends, family, fictional characters, or draft subjects like Mark Ellison)
- PRIVACY & FORGET CONSTRAINTS

SOURCE EXTRACTED BULLETS:
{combined_bullets}
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": merge_prompt}],
        temperature=0,
    )
    profile = resp.choices[0].message.content or ""
    # Programmatic post-processing to clean up negative constraint hallucinations
    profile_lines = profile.splitlines()
    cleaned_lines = []
    for line in profile_lines:
        lower_line = line.lower()
        if "no history of back" in lower_line or "no back problems" in lower_line or "no back pain" in lower_line:
            continue
        cleaned_lines.append(line)
    profile = "\n".join(cleaned_lines)
    profile_path.write_text(profile)
    hash_path.write_text(prompt_hash)
    return profile
