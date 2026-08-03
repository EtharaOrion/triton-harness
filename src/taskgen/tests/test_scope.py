"""Carve-scope resolution: harbor glob semantics + the fail-closed guards.

The glob engine is a PORT of `harbor-tasks/shared/tooling/manifest.py`
(`_glob_to_regex` / `GlobSet`), so one test drives the ported engine against the
harbor original over a vector table -- a port that silently diverges from the
manifests the shipped dataset was carved with is worse than no port at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taskgen.scope import (
    CarveScope,
    CarveScopeError,
    GlobSet,
    _glob_to_regex,
    resolve_carved_set,
)

TREE = {
    'pkg/__init__.py': '',
    'pkg/a.py': 'A = 1\n',
    'pkg/b.py': 'B = 1\n',
    'pkg/sub/c.py': 'C = 1\n',
    'pkg/sub/notes.txt': 'notes\n',
    'pkg/sub/deep/d.py': 'D = 1\n',
    'tests/test_a.py': 'def test_a(): pass\n',
    'tests/conftest.py': '',
    'README.md': '# readme\n',
}


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    for rel, text in TREE.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding='utf-8')
    return tmp_path


# --------------------------------------------------------------------------
# glob engine: ported, not re-invented
# --------------------------------------------------------------------------

GLOB_VECTORS = [
    ('pkg/**', 'pkg/a.py'),
    ('pkg/**', 'pkg/sub/deep/d.py'),
    ('pkg/**/*.py', 'pkg/a.py'),
    ('pkg/**/*.py', 'pkg/sub/deep/d.py'),
    ('pkg/**/*.py', 'pkg/sub/notes.txt'),
    ('pkg/*.py', 'pkg/sub/c.py'),
    ('pkg/?.py', 'pkg/a.py'),
    ('pkg/?.py', 'pkg/ab.py'),
    ('**/test_*.py', 'tests/test_a.py'),
    ('**/test_*.py', 'test_a.py'),
    ('*.md', 'README.md'),
    ('*.md', 'docs/README.md'),
]


@pytest.mark.parametrize('pattern,relpath', GLOB_VECTORS)
def test_glob_engine_matches_harbor_byte_for_byte(pattern, relpath):
    """The ported regex compiler must agree with harbor's on every vector."""
    from taskgen import _tooling_path  # noqa: F401  (sys.path -> harbor tooling)

    import manifest as harbor_manifest

    ours = bool(_glob_to_regex(pattern).match(relpath))
    theirs = bool(harbor_manifest._glob_to_regex(pattern).match(relpath))
    assert ours == theirs
    assert GlobSet([pattern]).matches(relpath) is ours


def test_globset_is_falsey_when_empty():
    assert not GlobSet([])
    assert GlobSet(['a']).matches('a')


# --------------------------------------------------------------------------
# resolve_carved_set
# --------------------------------------------------------------------------


def test_include_resolves_the_expected_set(tree):
    got = resolve_carved_set(tree, CarveScope.FILE, include=['pkg/**/*.py'])
    assert got == (
        'pkg/__init__.py',
        'pkg/a.py',
        'pkg/b.py',
        'pkg/sub/c.py',
        'pkg/sub/deep/d.py',
    )


def test_exclude_is_subtracted_from_include(tree):
    got = resolve_carved_set(
        tree, CarveScope.FOLDER, include=['pkg/**/*.py'], exclude=['pkg/sub/**']
    )
    assert got == ('pkg/__init__.py', 'pkg/a.py', 'pkg/b.py')


def test_result_is_sorted_deduped_and_a_tuple(tree):
    got = resolve_carved_set(tree, CarveScope.FILE, include=['pkg/**/*.py', 'pkg/a.py'])
    assert isinstance(got, tuple)
    assert list(got) == sorted(set(got))


def test_function_scope_returns_only_the_target(tree):
    got = resolve_carved_set(
        tree, CarveScope.FUNCTION, target_relpath='pkg/a.py', include=['pkg/**']
    )
    assert got == ('pkg/a.py',)


def test_function_scope_requires_a_target(tree):
    with pytest.raises(CarveScopeError):
        resolve_carved_set(tree, CarveScope.FUNCTION)


def test_function_scope_rejects_a_target_outside_the_repo(tree):
    with pytest.raises(CarveScopeError):
        resolve_carved_set(tree, CarveScope.FUNCTION, target_relpath='pkg/nope.py')


def test_empty_resolution_fails_closed(tree):
    with pytest.raises(CarveScopeError) as exc:
        resolve_carved_set(tree, CarveScope.FILE, include=['nothing/**'])
    assert 'zero files' in str(exc.value)


def test_missing_include_fails_closed(tree):
    with pytest.raises(CarveScopeError):
        resolve_carved_set(tree, CarveScope.FILE)


def test_carving_a_graded_test_file_fails_closed(tree):
    with pytest.raises(CarveScopeError) as exc:
        resolve_carved_set(tree, CarveScope.FILE, include=['**/*.py'], test_globs=['tests/**'])
    assert 'tests/test_a.py' in str(exc.value)


def test_carving_a_fingerprinted_file_fails_closed(tree):
    with pytest.raises(CarveScopeError) as exc:
        resolve_carved_set(
            tree,
            CarveScope.FILE,
            include=['pkg/**/*.py'],
            fingerprint_relpaths={'pkg/a.py'},
        )
    assert 'pkg/a.py' in str(exc.value)


def test_the_guard_also_runs_for_function_scope(tree):
    """No language may opt out of the guard by construction (plan §T2)."""
    with pytest.raises(CarveScopeError):
        resolve_carved_set(
            tree,
            CarveScope.FUNCTION,
            target_relpath='tests/test_a.py',
            test_globs=['tests/**'],
        )


def test_guard_passes_when_the_sets_are_disjoint(tree):
    got = resolve_carved_set(
        tree,
        CarveScope.FILE,
        include=['pkg/**/*.py'],
        exclude=['pkg/sub/**'],
        fingerprint_relpaths={'tests/test_a.py'},
        test_globs=['tests/**'],
    )
    assert got == ('pkg/__init__.py', 'pkg/a.py', 'pkg/b.py')


def test_resolution_is_deterministic(tree):
    a = resolve_carved_set(tree, CarveScope.FILE, include=['pkg/**'])
    b = resolve_carved_set(tree, CarveScope.FILE, include=['pkg/**'])
    assert a == b


def test_scope_enum_values():
    assert [s.value for s in CarveScope] == ['function', 'file', 'folder']
