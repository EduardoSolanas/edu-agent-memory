import os
import pytest
import requests
from tools.evaluate_beam_end_to_end import _intent_from_question

pytestmark = pytest.mark.skipif(
    os.environ.get("BEAM_LIVE_RERANKER_TEST") != "1",
    reason="set BEAM_LIVE_RERANKER_TEST=1 to run the live reranker integration",
)

def test_reranker_intent_classification_integration():
    reranker_url = os.environ.get("EDUMEM_RERANKER_URL", "http://localhost:3002/rerank")
    try:
        response = requests.post(
            reranker_url,
            json={"query": "health check", "texts": ["health check"]},
            timeout=2,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        pytest.skip(f"live reranker unavailable: {type(exc).__name__}")

    # Test cases representing different BEAM query intents
    test_cases = [
        # Event Ordering / Sequence -> 'ordered'
        ("Can you walk me through the sequence of events of setting up the containers?", "ordered"),
        ("In what order did I brought up different Docker containers?", "ordered"),
        
        # Temporal Reasoning / Timeline -> 'timeline'
        ("How many days passed between the first deployment and the final launch?", "timeline"),
        ("What is the deadline for completing the integration tests?", "timeline"),
        
        # Contradiction / Change / State switch -> 'change'
        ("What did I switch the database port from and to?", "change"),
        ("Did I change my preference about using docker-compose?", "change"),
        
        # Information Extraction / Current state / General facts -> 'current'
        ("What version of Python is installed in the system?", "current"),
        ("Which libraries are used in this project?", "current")
    ]
    
    for question, expected_intent in test_cases:
        actual_intent = _intent_from_question(question)
        assert actual_intent == expected_intent, f"Failed for question: '{question}'. Expected '{expected_intent}', got '{actual_intent}'"
