"""File-level carve: skeleton-stub every OUTERMOST function, or delete the file.

The two properties that matter and that `ast.parse` cannot see (plan §(c), T4 AC):

  * the skeletonised module still really IMPORTS (module-level code runs);
  * `pytest --collect-only` over the skeletonised tree has zero collection
    errors -- otherwise one carved file turns into a repo-wide fake RED.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from taskgen.carve_file import (
    DELETED,
    CarveFileError,
    CarveFileMode,
    carve_files,
    default_stub_body,
    delete_whole_file,
    skeleton_stub_file,
)

STUB = 'raise NotImplementedError'

#: Deliberately hostile: a nested def, a closure inside a decorator factory, a
#: module-level class with methods, and a class defined inside a function body.
SOURCE = '''\
"""Module docstring."""

from __future__ import annotations

import os

CONST = 3
LOOKUP = {'k': os.sep}


def outer(x):
    """Outermost: its body swallows `inner`."""
    def inner(y):
        return y + 1

    return inner(x) + CONST


def make_decorator(n):
    """Outermost: two levels of nesting live inside this body."""
    def deco(fn):
        def wrapper(*a, **k):
            return fn(*a, **k) * n

        return wrapper

    return deco


class Widget:
    """A module-level class: its body is NOT a function body."""

    ATTR = 'widget-attribute'

    def __init__(self, v):
        self.v = v

    def double(self):
        return self.v * 2

    @property
    def tripled(self):
        return self.v * 3


def factory():
    class Inner:
        def method(self):
            return 42

    return Inner
'''

#: outer, make_decorator, Widget.__init__, Widget.double, Widget.tripled, factory
OUTERMOST_COUNT = 6


@pytest.fixture(scope='module')
def stubbed() -> str:
    return skeleton_stub_file(SOURCE, 'mypkg/core.py', stub_body=STUB)


def test_stubs_every_outermost_function_exactly_once(stubbed):
    assert stubbed.count(STUB) == OUTERMOST_COUNT


def test_nested_defs_are_swallowed_not_double_stubbed(stubbed):
    # They live INSIDE an outermost body, so replacing that body removes them
    # wholesale. If the outermost filter were missing they would be edited a
    # second time and either reappear or corrupt the file.
    for gone in ('def inner', 'def deco', 'def wrapper', 'class Inner', 'def method'):
        assert gone not in stubbed


def test_class_bodies_are_not_function_bodies(stubbed):
    assert 'class Widget:' in stubbed
    assert "ATTR = 'widget-attribute'" in stubbed


def test_signatures_and_decorators_survive(stubbed):
    for kept in (
        'def outer(x):',
        'def make_decorator(n):',
        'def __init__(self, v):',
        'def double(self):',
        '@property',
        'def tripled(self):',
        'def factory():',
    ):
        assert kept in stubbed


def test_imports_and_module_level_code_survive(stubbed):
    assert 'from __future__ import annotations' in stubbed
    assert 'import os' in stubbed
    assert 'CONST = 3' in stubbed
    assert "LOOKUP = {'k': os.sep}" in stubbed


def test_bodies_are_gone(stubbed):
    for gone in ('return self.v * 2', 'return self.v * 3', 'self.v = v', 'return inner(x) + CONST'):
        assert gone not in stubbed


def test_stub_is_shorter_and_different(stubbed):
    assert stubbed != SOURCE
    assert len(stubbed) < len(SOURCE)


def test_stub_parses(stubbed):
    import ast

    ast.parse(stubbed)


def test_a_file_with_no_functions_is_a_no_op_error():
    with pytest.raises(CarveFileError):
        skeleton_stub_file('X = 1\n', 'mypkg/consts.py', stub_body=STUB)


def test_unknown_language_fails_closed():
    with pytest.raises(CarveFileError):
        skeleton_stub_file(SOURCE, 'a.py', stub_body=STUB, language='cobol')


def test_default_stub_bodies_are_language_specific():
    assert default_stub_body('python') == 'raise NotImplementedError'
    assert 'panic' in default_stub_body('go')


def test_deterministic(stubbed):
    again = skeleton_stub_file(SOURCE, 'mypkg/core.py', stub_body=STUB)
    assert again == stubbed


# --------------------------------------------------------------------------
# the real properties: import + collect
# --------------------------------------------------------------------------


@pytest.fixture()
def skeleton_tree(tmp_path: Path, stubbed: str) -> Path:
    proj = tmp_path / 'proj'
    (proj / 'mypkg').mkdir(parents=True)
    (proj / 'tests').mkdir()
    (proj / 'mypkg' / '__init__.py').write_text('', encoding='utf-8')
    (proj / 'mypkg' / 'core.py').write_text(stubbed, encoding='utf-8')
    (proj / 'tests' / 'test_core.py').write_text(
        'from mypkg.core import Widget, outer\n\n\n'
        'def test_double():\n'
        '    assert Widget(2).double() == 4\n\n\n'
        'def test_outer():\n'
        '    assert outer(1) == 5\n',
        encoding='utf-8',
    )
    return proj


def test_skeletonised_module_really_imports(skeleton_tree):
    out = subprocess.run(
        [sys.executable, '-c', 'import mypkg.core; print(mypkg.core.CONST)'],
        cwd=skeleton_tree,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == '3'


def test_pytest_collect_only_has_zero_collection_errors(skeleton_tree):
    out = subprocess.run(
        [
            sys.executable, '-m', 'pytest', '--collect-only', '-q',
            '-o', 'addopts=', '-p', 'no:cacheprovider',
        ],
        cwd=skeleton_tree,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert 'error' not in out.stdout.lower()
    assert '2 tests collected' in out.stdout


# --------------------------------------------------------------------------
# carve_files / whole-file delete
# --------------------------------------------------------------------------


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / 'repo'
    (r / 'mypkg').mkdir(parents=True)
    (r / 'mypkg' / 'core.py').write_text(SOURCE, encoding='utf-8')
    (r / 'mypkg' / 'other.py').write_text(SOURCE, encoding='utf-8')
    return r


def test_delete_whole_file_returns_the_sentinel():
    assert delete_whole_file(SOURCE, 'mypkg/core.py') is DELETED


def test_carve_files_skeleton_mode(repo):
    res = carve_files(
        repo,
        ['mypkg/core.py'],
        mode=CarveFileMode.SKELETON,
        stub_body=STUB,
        language='python',
    )
    assert res.carved_relpaths == ('mypkg/core.py',)
    assert res.deleted_relpaths == ()
    assert res.overlay['mypkg/core.py'].count(STUB) == OUTERMOST_COUNT
    assert res.originals['mypkg/core.py'] == SOURCE


def test_carve_files_delete_mode(repo):
    res = carve_files(
        repo,
        ['mypkg/core.py', 'mypkg/other.py'],
        mode=CarveFileMode.DELETE,
        stub_body=STUB,
        language='python',
    )
    assert res.deleted_relpaths == ('mypkg/core.py', 'mypkg/other.py')
    assert res.overlay == {}
    assert set(res.originals) == {'mypkg/core.py', 'mypkg/other.py'}


def test_original_is_recoverable_byte_for_byte(repo):
    res = carve_files(
        repo, ['mypkg/core.py'], mode=CarveFileMode.DELETE, stub_body=STUB, language='python'
    )
    digest = hashlib.sha256(res.originals['mypkg/core.py'].encode('utf-8')).hexdigest()
    assert digest == res.original_sha256['mypkg/core.py']
    assert digest == hashlib.sha256((repo / 'mypkg/core.py').read_bytes()).hexdigest()


def test_per_file_shas_are_recorded(repo):
    res = carve_files(
        repo, ['mypkg/core.py'], mode=CarveFileMode.SKELETON, stub_body=STUB, language='python'
    )
    stub_sha = hashlib.sha256(res.overlay['mypkg/core.py'].encode('utf-8')).hexdigest()
    assert res.stubbed_sha256['mypkg/core.py'] == stub_sha
    assert res.stubbed_sha256 != res.original_sha256


def test_carve_files_never_writes_to_the_repo(repo):
    before = hashlib.sha256((repo / 'mypkg/core.py').read_bytes()).hexdigest()
    carve_files(
        repo, ['mypkg/core.py'], mode=CarveFileMode.SKELETON, stub_body=STUB, language='python'
    )
    assert hashlib.sha256((repo / 'mypkg/core.py').read_bytes()).hexdigest() == before


def test_carve_files_is_deterministic(repo):
    a = carve_files(
        repo, ['mypkg/core.py'], mode=CarveFileMode.SKELETON, stub_body=STUB, language='python'
    )
    b = carve_files(
        repo, ['mypkg/core.py'], mode=CarveFileMode.SKELETON, stub_body=STUB, language='python'
    )
    assert a.overlay == b.overlay
    assert a.carved_relpaths == b.carved_relpaths


def test_carve_files_fails_closed_on_a_missing_file(repo):
    with pytest.raises(CarveFileError):
        carve_files(
            repo, ['mypkg/ghost.py'], mode=CarveFileMode.SKELETON, stub_body=STUB,
            language='python',
        )


def test_carve_files_fails_closed_on_an_empty_set(repo):
    with pytest.raises(CarveFileError):
        carve_files(repo, [], mode=CarveFileMode.SKELETON, stub_body=STUB, language='python')


def test_partition_carveable_reports_why_the_parser_refused(tmp_path):
    """A parser refusal must not masquerade as 'this file has no functions'.

    Two very different causes -- a tree-sitter grammar that is not installed,
    and a language the carve layer does not support at all -- were both being
    swallowed into `carveable = False`, so the caller reported
    'none of the N glob-matched file(s) holds a <lang> function body'. That
    sends the reader hunting for a better --include when the real fix is
    installing a wheel or using --delete-whole-file.
    """
    from taskgen.carve import _partition_carveable

    (tmp_path / 'a.c').write_text('int main(void) { return 0; }\n', encoding='utf-8')
    keep, skipped, reasons = _partition_carveable(tmp_path, ('a.c',), 'c')

    assert not keep
    assert skipped == ('a.c',)
    assert reasons, 'the parser refusal reason must reach the caller'
    assert 'unsupported language' in reasons[0]
