"""Linked test functions -> pytest node ids.

The node id is what the grader's allowlist is made of, so a wrong one is not a
cosmetic bug: `harbor_filter` would silently deselect it and the run would score
0 with a green-looking pytest. Two rules earn their own functions here:

  * the class segment is OMITTED for module-level tests -- `f.py::::test_x`
    matches nothing;
  * ids are sorted and de-duplicated, because the parser hands back a `set`.

Parametrised tests (`test_x[case]`) are NOT expanded: the parser sees the
`def`, not pytest's generated ids. `verify` (next wave) cross-checks the
allowlist against `pytest --collect-only` inside the image, which is where such
a mismatch must be caught.
"""

from __future__ import annotations

from pathlib import Path


def format_nodeid(relpath: str, class_name: str, name: str) -> str:
    """`<relpath>::<Class>::<name>`, dropping the class segment when it is empty."""
    parts = [relpath, class_name, name] if class_name else [relpath, name]
    return '::'.join(parts)


def linked_nodeids(repo: Path, target) -> list[str]:
    """Sorted, unique node ids of every test the parser linked to `target`."""
    return sorted({format_nodeid(t.relpath, t.class_name, t.name) for t in target.tests})


def expected_count(nodeids: list[str]) -> int:
    """EXPECTED for tests/test.sh -- the exact number of graded ids."""
    return len(nodeids)
