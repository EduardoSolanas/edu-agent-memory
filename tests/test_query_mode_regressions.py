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


def test_event_pair_interval_wording_counts_as_a_date_interval():
    question = "How many days passed between when I planned peer review and when I completed final review?"

    assert is_duration_query(question)
    assert is_date_interval_query(question)


def test_stated_duration_wording_does_not_count_as_a_date_interval():
    question = "How long did I say the project is expected to take?"

    assert not is_date_interval_query(question)
