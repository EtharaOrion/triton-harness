"""uuid5 entry ids must be stable across runs and distinct per context type."""

from __future__ import annotations

import uuid

import pytest

from taskgen.ids import CONTEXT_TYPES, NS, entry_id

ARGS = ('python-a2a-python', 'src/a2a/utils/task.py', '', 'apply_history_length')


def test_namespace_is_the_documented_uuid5():
    assert NS == uuid.uuid5(uuid.NAMESPACE_URL, 'triton.taskgen')


def test_entry_id_is_deterministic():
    assert entry_id(*ARGS, 'bm25') == entry_id(*ARGS, 'bm25')


def test_entry_id_matches_the_documented_derivation():
    want = str(uuid.uuid5(NS, '|'.join((*ARGS, 'bm25'))))
    assert entry_id(*ARGS, 'bm25') == want


def test_entry_id_is_a_v5_uuid():
    assert uuid.UUID(entry_id(*ARGS, 'bm25')).version == 5


def test_entry_id_differs_per_context_type():
    ids = {ct: entry_id(*ARGS, ct) for ct in CONTEXT_TYPES}
    assert len(set(ids.values())) == len(CONTEXT_TYPES) == 11


def test_entry_id_differs_per_function():
    a = entry_id('python-a2a-python', 'src/a2a/utils/task.py', '', 'f', 'bm25')
    b = entry_id('python-a2a-python', 'src/a2a/utils/task.py', '', 'g', 'bm25')
    assert a != b


def test_unknown_context_type_is_rejected():
    with pytest.raises(ValueError):
        entry_id(*ARGS, 'not-a-context-type')
