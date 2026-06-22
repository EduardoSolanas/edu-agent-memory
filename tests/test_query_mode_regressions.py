from edumem.core.query_mode import (
    build_system_prompt,
    is_date_interval_query,
    is_duration_query,
    is_stated_duration_query,
)


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


def test_stated_duration_wording_does_not_count_as_a_date_interval():
    question = "How long did I say the project is expected to take?"

    assert not is_date_interval_query(question)
