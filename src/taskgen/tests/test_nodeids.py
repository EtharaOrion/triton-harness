"""Linked tests -> pytest node ids, exactly the five frozen ones."""

from __future__ import annotations

from taskgen.nodeids import format_nodeid, linked_nodeids

from .conftest import FROZEN_NODEIDS


def test_yields_the_five_frozen_nodeids(repo, target):
    assert linked_nodeids(repo, target) == FROZEN_NODEIDS


def test_expected_is_five(repo, target):
    assert len(linked_nodeids(repo, target)) == 5


def test_nodeids_are_sorted_and_unique(repo, target):
    got = linked_nodeids(repo, target)
    assert got == sorted(set(got))


def test_class_segment_is_omitted_when_empty():
    assert format_nodeid('tests/t.py', '', 'test_x') == 'tests/t.py::test_x'


def test_class_segment_is_included_when_present():
    assert format_nodeid('tests/t.py', 'TestC', 'test_x') == 'tests/t.py::TestC::test_x'
