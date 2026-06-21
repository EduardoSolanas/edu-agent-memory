from edumem.core.beam import BeamMemory


def test_ordered_recall_includes_topic_introductions_across_the_conversation(
    tmp_path, monkeypatch
):
    """Regression for BEAM 100K case 1, EO q4."""
    monkeypatch.setenv("EDUMEM_NO_EMBEDDINGS", "1")
    beam = BeamMemory(db_path=tmp_path / "beam.db", session_id="eo-q4")
    try:
        messages = [
            (
                4,
                "I need help implementing the core functionality, including user "
                "authentication, expense tracking, and data visualization.",
            ),
            (30, "I am setting up database migrations with Flask-Migrate."),
            (
                60,
                "I am working on transaction creation and need proper error handling "
                "for failures.",
            ),
            (90, "I optimized CRUD queries and dashboard response time."),
            (
                116,
                "Before deployment I need security hardening for authentication and "
                "authorization.",
            ),
        ]
        # The real case contains dozens of long user turns between the three
        # introductions. A 360-character excerpt per turn pushes the final
        # topic beyond the evaluator's 16K retrieved-memory budget.
        for message_index in range(6, 60, 2):
            messages.append(
                (
                    message_index,
                    f"I am reviewing implementation detail {message_index} for the "
                    + "database layer and its validation behavior. " * 12,
                )
            )
        for message_index in range(62, 116, 2):
            messages.append(
                (
                    message_index,
                    f"I am reviewing implementation detail {message_index} for the "
                    + "dashboard layer and its response behavior. " * 12,
                )
            )
        messages.sort(key=lambda item: item[0])
        for message_index, content in messages:
            beam.remember(
                content,
                source="beam_user",
                message_index=message_index,
                veracity="stated",
            )

        result = beam.memoria_retrieve(
            "In what order did I bring up different aspects of the project?",
            intent="ordered",
            top_k=30,
        )

        context = result["context"].lower()
        expected = [
            "core functionality",
            "transaction creation",
            "security hardening",
        ]
        positions = [context.index(item) for item in expected]
        assert positions == sorted(positions)
        assert positions[-1] < 16_000
        assert len(result["context"]) <= 16_000
        assert "msgidx:4" in context
        assert "msgidx:60" in context
        assert "msgidx:116" in context
    finally:
        beam.conn.close()
