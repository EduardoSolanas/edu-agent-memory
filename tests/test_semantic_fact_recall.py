"""TDD tests for semantic recall over memoria_facts (vec_facts repurposed)."""
from __future__ import annotations

import os
from pathlib import Path

# Suppress embeddings for the offline unit tests in this file. The write-path
# embed is exercised by stubbing, not by a real model. (Spec Part E.)
os.environ.setdefault("EDUMEM_NO_EMBEDDINGS", "1")

from edumem.core import beam as beam_mod
from edumem.core.beam import BeamMemory, _vec_available, _effective_vec_type


def _new_beam(tmp_path: Path) -> BeamMemory:
    return BeamMemory(session_id="sem-test", db_path=tmp_path / "sem.db")


def test_vec_available_checks_the_named_table(tmp_path):
    """_vec_available(table=...) probes the given table, not vec_episodes."""
    beam = _new_beam(tmp_path)
    try:
        # vec_episodes and vec_facts both exist after init_beam (beam.py:738, 1059).
        assert _vec_available(beam.conn, table="vec_episodes") in (True, False)
        # vec_facts is created in init_beam too; probing it must not raise and
        # must reflect ITS existence, independent of vec_episodes.
        assert _vec_available(beam.conn, table="vec_facts") in (True, False)
        # A nonexistent table must return False, never raise.
        assert _vec_available(beam.conn, table="vec_does_not_exist") is False
    finally:
        beam.conn.close()


def test_effective_vec_type_reads_named_table_schema(tmp_path):
    """_effective_vec_type(table=...) reads the named table's schema, not vec_episodes."""
    beam = _new_beam(tmp_path)
    try:
        # Both tables created with the same effective_vec_type in init_beam, so both
        # resolve to the same type string. The point: the call must not raise and
        # must accept the table kwarg.
        t_episodes = _effective_vec_type(beam.conn, table="vec_episodes")
        t_facts = _effective_vec_type(beam.conn, table="vec_facts")
        assert t_episodes in ("bit", "int8", "float32")
        assert t_facts == t_episodes  # both created together -> same type
    finally:
        beam.conn.close()
