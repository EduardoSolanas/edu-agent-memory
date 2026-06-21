import os
import threading
import time

from tools import evaluate_beam_end_to_end as beam_eval


def test_deterministic_intent_signals_take_precedence_without_reranker():
    assert beam_eval._deterministic_intent("In what order did I deploy the services?") == "ordered"
    assert beam_eval._deterministic_intent("What is the deadline for integration testing?") == "timeline"
    assert beam_eval._deterministic_intent("How many days passed between launch and review?") == "timeline"
    assert beam_eval._deterministic_intent("What changed from the old port to the new port?") == "change"
    assert beam_eval._deterministic_intent("Which Python version is installed?") is None


def test_change_intent_requires_an_explicit_state_transition():
    assert beam_eval._deterministic_intent(
        "Tell me about my background and previous development projects."
    ) is None
    assert beam_eval._deterministic_intent("Compare the previous versus current API version.") == "change"
    assert beam_eval._deterministic_intent("What changed in the deployment configuration?") == "change"
    assert beam_eval._deterministic_intent("I switched from SQLite to PostgreSQL; what were both choices?") == "change"


def test_reranker_intent_requires_confidence_and_margin():
    confident = [
        {"index": 3, "score": 0.91},
        {"index": 1, "score": 0.40},
        {"index": 0, "score": 0.20},
        {"index": 2, "score": 0.10},
    ]
    assert beam_eval._intent_from_reranker_scores(confident) == "current"

    low_confidence = [dict(item) for item in confident]
    low_confidence[0]["score"] = 0.49
    assert beam_eval._intent_from_reranker_scores(low_confidence) is None

    narrow_margin = [dict(item) for item in confident]
    narrow_margin[1]["score"] = 0.85
    assert beam_eval._intent_from_reranker_scores(narrow_margin) is None


def test_worker_clients_are_independent_real_clients():
    source = beam_eval.LLMClient(model="qwen3.6", api_key="local-test-key", base_url="http://127.0.0.1:9/v1")
    answer, judge = beam_eval._new_worker_clients(source, source)

    assert answer is not source
    assert judge is not source
    assert answer is not judge
    assert (answer.model, answer.api_key, answer.base_url) == (source.model, source.api_key, source.base_url)

    answer.last_error_message = "answer-only"
    assert judge.last_error_message == ""
    assert source.last_error_message == ""


def test_start_rate_limiter_spaces_parallel_request_starts():
    limiter = beam_eval._StartRateLimiter(0.02)
    starts = []
    lock = threading.Lock()

    def run():
        limiter.wait()
        with lock:
            starts.append(time.monotonic())

    threads = [threading.Thread(target=run) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    starts.sort()
    assert starts[1] - starts[0] >= 0.015
    assert starts[2] - starts[1] >= 0.015


def test_grader_error_logging_is_bounded_unless_debug_enabled(capsys):
    previous = os.environ.pop("BEAM_DEBUG_GRADER_ERRORS", None)
    try:
        beam_eval._log_grader_error(ValueError("bad payload\nwith extra detail"))
        output = capsys.readouterr()
        assert "ValueError: bad payload" in output.out
        assert "with extra detail" not in output.out
        assert output.err == ""

        os.environ["BEAM_DEBUG_GRADER_ERRORS"] = "1"
        try:
            raise RuntimeError("debug detail")
        except RuntimeError as exc:
            beam_eval._log_grader_error(exc)
        output = capsys.readouterr()
        assert "Traceback" in output.err
    finally:
        if previous is None:
            os.environ.pop("BEAM_DEBUG_GRADER_ERRORS", None)
        else:
            os.environ["BEAM_DEBUG_GRADER_ERRORS"] = previous
