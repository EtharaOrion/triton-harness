"""Carve scope: how much of the repository the task removes, resolved by globs.

`--carve-scope function` is today's behaviour (one function body inside one
file). `file` and `folder` widen it to a glob-selected set of whole files, which
is what the shipped `harbor-tasks/dataset` entries do.

The glob engine is a PORT of `harbor-tasks/shared/tooling/manifest.py`
(`_glob_to_regex` :31, `GlobSet` :58, `resolve_carved` :121). It is ported
rather than imported for one reason: `manifest.resolve_carved` only accepts a
`Manifest` loaded from a `*.carve.toml`, and taskgen resolves a carve set from
CLI arguments with no manifest anywhere. `tests/test_scope.py` drives both
implementations over a vector table so the port cannot silently diverge from the
semantics the shipped dataset was carved with -- `**` in particular means "any
number of directories", which `PurePath.match` does not give us.

Two guards, both FAIL-CLOSED, both raising rather than warning:

  * an empty carve set is refused (harbor `carve.py:49-52` does the same) --
    a task that carves nothing is a task whose oracle is a no-op;
  * the carve set may never intersect the graded-test files or the
    sha256-fingerprinted files. Carving a graded test shrinks the denominator
    and hands out free reward; carving a fingerprinted file makes the pinned
    floor unverifiable. The guard runs for `function` scope too -- where the
    intersection is trivially empty -- so that no language and no scope can opt
    out of it by construction.
"""

from __future__ import annotations

import enum
import re
from pathlib import Path

__all__ = [
    'CarveScope',
    'CarveScopeError',
    'DEFAULT_TEST_GLOBS',
    'GlobSet',
    'assert_carve_set_safe',
    'resolve_carved_set',
]

#: Where a language's graded tests live. The guard below refuses to carve them,
#: so these are the shapes a `--include` glob must never be allowed to sweep up.
#: Deliberately broad: a false positive costs an explicit `--include`, a false
#: negative hands the solver a shrunken denominator.
DEFAULT_TEST_GLOBS: dict[str, tuple[str, ...]] = {
    'python': ('tests/**', 'test/**', '**/test_*.py', '**/*_test.py', 'conftest.py'),
    'go': ('**/*_test.go',),
    # rust: the graded oracle is the integration harnesses under tests/ and the
    # .wast conformance corpus they drive. A --include glob must never sweep them.
    'rust': ('tests/**', 'benches/**', '**/*.wast'),
    # c-xs: the graded oracle is tests/conformance/, tests/regression/ and
    # tests/unit/. --include glob for the carve is 'src/runtime/**' and cannot
    # match here, but the guard is enforced anyway so no scope can opt out.
    'c': ('tests/**',),
    # cpp: the graded oracle is the repo's test tree, under all three spellings
    # C++ projects use for it -- cpp-Rux's capital `Tests/` is one repository's
    # layout, not the language's, and a guard that only knew that spelling let
    # a `--include` glob carve the tests of every other project.
    'cpp': ('Tests/**', 'test/**', 'tests/**'),
    # java-tamboui: the graded oracle is tamboui-widgets/src/test/java (49
    # files). --include for the carve is tamboui-widgets/src/main/java/** and
    # cannot match here (main and test live in different source roots), but
    # the guard is enforced anyway so no scope can opt out.
    'java': ('**/src/test/**',),
}


class CarveScopeError(RuntimeError):
    """A carve set that must not be shipped. Always raised, never downgraded."""


class CarveScope(enum.Enum):
    FUNCTION = 'function'
    FILE = 'file'
    FOLDER = 'folder'

    @classmethod
    def parse(cls, value: 'str | CarveScope') -> 'CarveScope':
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError:
            raise CarveScopeError(
                f'unknown carve scope {value!r}; expected one of '
                f'{", ".join(s.value for s in cls)}'
            ) from None


# --- ported verbatim from harbor manifest.py:31-68 -------------------------


def _glob_to_regex(pattern: str) -> re.Pattern:
    i = 0
    out = ['^']
    n = len(pattern)
    while i < n:
        c = pattern[i]
        if c == '*':
            if pattern.startswith('**/', i):
                out.append('(?:.*/)?')
                i += 3
                continue
            if pattern.startswith('**', i):
                out.append('.*')
                i += 2
                continue
            out.append('[^/]*')
            i += 1
        elif c == '?':
            out.append('[^/]')
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    out.append('$')
    return re.compile(''.join(out))


class GlobSet:
    def __init__(self, patterns) -> None:
        self.patterns = list(patterns)
        self._res = [_glob_to_regex(p) for p in self.patterns]

    def matches(self, relpath: str) -> bool:
        return any(r.match(relpath) for r in self._res)

    def __bool__(self) -> bool:
        return bool(self._res)


# --- resolution ------------------------------------------------------------


def _walk(repo: Path):
    """POSIX relpaths of every file under `repo`, harbor `resolve_carved` style."""
    for path in repo.rglob('*'):
        if not path.is_file():
            continue
        yield path.relative_to(repo).as_posix()


def _guard(
    carved: tuple[str, ...],
    fingerprint_relpaths,
    test_globs,
) -> None:
    fingerprints = {Path(p).as_posix() for p in fingerprint_relpaths}
    tests = GlobSet(test_globs)
    bad = sorted(
        {rel for rel in carved if rel in fingerprints}
        | {rel for rel in carved if tests.matches(rel)}
    )
    if bad:
        raise CarveScopeError(
            'refusing to carve graded/fingerprinted file(s): '
            + ', '.join(bad[:10])
            + (f' (+{len(bad) - 10} more)' if len(bad) > 10 else '')
            + '. Carving a graded test shrinks the reward denominator and '
            'carving a fingerprinted file makes the floor unverifiable.'
        )


def assert_carve_set_safe(carved, fingerprint_relpaths=(), test_globs=()) -> None:
    """Re-run the fail-closed guard once the graded set is actually known.

    `resolve_carved_set` runs it against the language's generic test globs,
    which is all it can know before the graded set is derived. The MEASURED
    graded test files are stricter, and a repo that keeps a test somewhere the
    generic globs miss would otherwise slip through.
    """
    _guard(tuple(carved), fingerprint_relpaths, test_globs)


def resolve_carved_set(
    repo: Path,
    scope: 'CarveScope | str',
    *,
    target_relpath: str | None = None,
    include=(),
    exclude=(),
    fingerprint_relpaths=frozenset(),
    test_globs=(),
) -> tuple[str, ...]:
    """POSIX relpaths to carve: sorted, deduplicated, guarded.

    `function` scope collapses to the single file that holds the target (the
    body-level carve inside it is `carve_fn.py`'s job). `file`/`folder` scope
    resolve `include` minus `exclude` against the tree, exactly as harbor's
    `resolve_carved` resolves a manifest's `[carve]` globs.
    """
    scope = CarveScope.parse(scope)
    repo = Path(repo)
    if not repo.is_dir():
        raise CarveScopeError(f'repo is not a directory: {repo}')

    if scope is CarveScope.FUNCTION:
        if not target_relpath:
            raise CarveScopeError('function scope requires target_relpath')
        rel = Path(target_relpath).as_posix()
        if not (repo / rel).is_file():
            raise CarveScopeError(f'function-scope target is not a file in {repo}: {rel}')
        carved = (rel,)
    else:
        if not include:
            raise CarveScopeError(
                f'{scope.value} scope requires at least one --include glob'
            )
        inc = GlobSet(include)
        exc = GlobSet(exclude)
        carved = tuple(
            sorted({rel for rel in _walk(repo) if inc.matches(rel) and not exc.matches(rel)})
        )

    if not carved:
        raise CarveScopeError(
            f'carve globs matched zero files under {repo} -- refusing an empty carve '
            f'(include={list(include)!r} exclude={list(exclude)!r})'
        )

    _guard(carved, fingerprint_relpaths, test_globs)
    return carved
