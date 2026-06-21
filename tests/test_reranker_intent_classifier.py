import os
import pytest
from tools.evaluate_beam_end_to_end import _intent_from_question

def test_reranker_intent_classification_integration():
    # Set the environment variable for the reranker URL
    os.environ["EDUMEM_RERANKER_URL"] = "http://localhost:3002/rerank"
    
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
