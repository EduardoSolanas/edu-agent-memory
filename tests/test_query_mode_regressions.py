from edumem.core.query_mode import (
    analyze_question_intent,
    build_system_prompt,
    is_date_interval_query,
    is_duration_query,
    is_list_query,
    is_ordering_query,
    is_background_query,
    is_summarization_query,
    is_knowledge_update_query,
    is_stated_duration_query,
    needs_second_pass,
)
from edumem.core.query_intent import classify_intent


def test_stated_duration_questions_are_answered_directly():
    question = "How long did I say the project is expected to take?"

    assert is_stated_duration_query(question)
    assert not is_duration_query(question)

    prompt = build_system_prompt(question)
    assert "STATED DURATION" in prompt
    assert "Answer that stated duration directly" in prompt
    assert "compute the difference" not in prompt


def test_true_interval_questions_still_compute_from_dates():
    question = "How many days passed between the kickoff and the launch?"

    assert not is_stated_duration_query(question)
    assert is_duration_query(question)

    prompt = build_system_prompt(question)
    assert "DURATION" in prompt
    assert "compute the difference" in prompt
    assert "Answer that stated duration directly" not in prompt


def test_ordering_prompt_spreads_items_across_full_timeline():
    question = (
        "List the order in which I brought up different aspects of the project, "
        "mentioning ONLY 3 items."
    )

    prompt = build_system_prompt(question)
    assert "ORDERING" in prompt
    # still keeps the existing MSGIDX-ordering guidance
    assert "LOWEST MSGIDX" in prompt
    # new guidance: spread across the entire timeline (lowest to highest MSGIDX)
    assert "HIGHEST MSGIDX" in prompt and "LOWEST MSGIDX" in prompt
    assert "Do NOT cluster" in prompt
    # new guidance: ensure later phases are represented
    assert "later phases" in prompt
    # new guidance: exact-count instruction
    assert "EXACTLY" in prompt


def test_event_pair_interval_wording_counts_as_a_date_interval():
    question = "How many days passed between when I planned peer review and when I completed final review?"

    assert is_duration_query(question)
    assert is_date_interval_query(question)
    assert needs_second_pass(question)


def test_stated_duration_wording_does_not_count_as_a_date_interval():
    question = "How long did I say the project is expected to take?"

    assert not is_date_interval_query(question)
    assert not needs_second_pass(question)


def test_non_temporal_between_question_does_not_trigger_second_pass():
    question = "Between my fetch call latency and autocomplete API response time, which is faster?"

    assert is_duration_query(question)
    assert not is_date_interval_query(question)
    assert not needs_second_pass(question)


def test_absence_rule_rejects_tangential_facts():
    # Fix A: tangential/related facts must NOT make an out-of-scope question
    # answerable. The strengthened ABSENCE rule must say so explicitly.
    prompt = build_system_prompt("How did user feedback influence the UI/UX?")
    lowered = prompt.lower()
    assert "tangential" in lowered or "related" in lowered
    assert "directly" in lowered
    # must not let the model synthesize from loosely-related facts
    assert "do not synthesize" in lowered or "do not infer" in lowered


def test_background_questions_require_direct_biographical_evidence():
    prompt = build_system_prompt(
        "Can you tell me about my background and previous development projects?"
    )
    lowered = prompt.lower()

    assert "personal background" in lowered or "prior work experience" in lowered
    assert "current project" in lowered
    assert "do not append" in lowered or "stop" in lowered


def test_duration_prompt_only_uses_newer_date_for_same_milestone():
    prompt = build_system_prompt(
        "How many weeks do I have between finishing the transaction "
        "management features and the final deployment deadline?"
    )
    lowered = prompt.lower()

    assert "most recently stated" in lowered
    assert "same milestone" in lowered or "same event" in lowered
    assert "different phase" in lowered or "do not replace one event" in lowered


def test_ordering_prompt_discourages_generic_planning_labels():
    prompt = build_system_prompt(
        "Can you walk me through the order in which I brought up different "
        "aspects of my app development and deployment across our conversations?"
    )
    lowered = prompt.lower()

    assert "generic" in lowered
    assert "planning" in lowered or "project scope" in lowered
    assert "concrete" in lowered
    assert "testing" in lowered or "deployment" in lowered


def test_multilingual_ordering_detection_works_for_spanish():
    question = "¿En qué orden hablamos de las distintas partes del proyecto?"

    profile = analyze_question_intent(question)

    assert profile.ordering is True
    assert is_ordering_query(question) is True


def test_multilingual_duration_detection_works_for_spanish():
    question = "¿Cuántos días pasaron entre el inicio del sprint y el lanzamiento?"

    profile = analyze_question_intent(question)

    assert profile.duration is True
    assert profile.date_interval is True
    assert is_duration_query(question) is True
    assert is_date_interval_query(question) is True
    assert needs_second_pass(question) is True


def test_multilingual_summary_and_list_detection_work_for_spanish():
    summary_question = "Resume los temas principales de nuestra conversación."
    list_question = "¿Qué bibliotecas y dependencias usamos?"

    summary_profile = analyze_question_intent(summary_question)
    list_profile = analyze_question_intent(list_question)

    assert summary_profile.summarization is True
    assert is_summarization_query(summary_question) is True
    assert list_profile.listing is True
    assert is_list_query(list_question) is True


def test_multilingual_current_state_detection_works_for_spanish():
    question = "¿Cuál es la versión actual de la API?"

    profile = analyze_question_intent(question)

    assert profile.knowledge_update is True
    assert is_knowledge_update_query(question) is True
def test_multilingual_background_detection_works_for_spanish():
    question = "Cual es mi experiencia laboral previa y mis proyectos anteriores?"

    profile = analyze_question_intent(question)

    assert profile.background is True
    assert is_background_query(question) is True

    prompt = build_system_prompt(question)
    lowered = prompt.lower()
    assert "background / prior-project questions" in lowered
    assert "personal background" in lowered
    assert "stop there" in lowered


def test_multilingual_procedural_intent_adapter_uses_shared_query_profile():
    intent = classify_intent("¿Cómo organizo el flujo de despliegue?")

    assert intent.category == "procedural"
    assert intent.confidence > 0.0

