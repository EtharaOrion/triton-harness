"""Target selection: repo source -> one eligible function, deterministically.

Wraps the harness `PyParser` + `get_func_and_tests`, which between them define
MRG-Bench eligibility: a function is a candidate iff it calls at least one other
first-party function, carries a docstring/leading comment, and is reached by at
least one test function.

Nothing here knows about any particular repo or function. The caller either
names a target (`--file`/`--func`) or gets the first eligible one under a total
order on `(relpath, class_name, name)`, which is stable across runs because the
parser's `callee`/`test_funcs` sets are re-sorted here before anyone reads them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from . import _tooling_path  # noqa: F401  (side effect: sys.path + tree-sitter shim)

DEFAULT_EXCLUDE_DIRS = ('.venv', '.git', '__pycache__', 'node_modules', 'build', 'dist')


def _relpath(repo: Path, file_path: str) -> str:
    return Path(os.path.relpath(file_path, repo)).as_posix()


def _sort_key(repo: Path, fd) -> tuple[str, str, str]:
    return (_relpath(repo, fd.file_path), fd.class_name or '', fd.name)


@dataclass(frozen=True)
class Linked:
    """A test function linked to the target by the parser's call graph."""

    relpath: str
    class_name: str
    name: str


@dataclass
class Target:
    """One carve candidate, fully resolved and order-stabilised."""

    repo: Path
    relpath: str
    class_name: str
    name: str
    fd: object
    callees: list
    tests: list[Linked]

    @property
    def qualname(self) -> str:
        return f'{self.class_name}.{self.name}' if self.class_name else self.name

    @property
    def docstring(self) -> str:
        return self.fd.get_comment() if self.fd.comment_node else ''

    def callee_names(self) -> list[str]:
        return [c.name for c in self.callees]

    def callee_qualnames(self) -> list[str]:
        return [f'{c.class_name}.{c.name}' if c.class_name else c.name for c in self.callees]


def parse_repo(repo: Path, package_base: str = 'src/', exclude_dirs=DEFAULT_EXCLUDE_DIRS):
    """All eligible (callee-bearing, documented, test-linked) functions, sorted."""
    from parser.py_parser import PyParser, get_func_and_tests

    parser = PyParser(str(repo), exclude_dirs=list(exclude_dirs), package_base_path=package_base)
    eligible = get_func_and_tests(parser.funcs)
    return sorted(eligible, key=lambda fd: _sort_key(repo, fd))


def _to_target(repo: Path, fd) -> Target:
    callees = sorted(fd.callee, key=lambda c: _sort_key(repo, c))
    tests = sorted(
        (
            Linked(_relpath(repo, t.file_path), t.class_name or '', t.name)
            for t in fd.test_funcs
        ),
        key=lambda t: (t.relpath, t.class_name, t.name),
    )
    return Target(
        repo=repo,
        relpath=_relpath(repo, fd.file_path),
        class_name=fd.class_name or '',
        name=fd.name,
        fd=fd,
        callees=callees,
        tests=tests,
    )


def select_target(
    repo: Path,
    package_base: str = 'src/',
    file: str | None = None,
    func: str | None = None,
    cls: str | None = None,
    exclude_dirs=DEFAULT_EXCLUDE_DIRS,
) -> Target:
    """Pick the named target, or the first eligible one in `(file, class, name)` order."""
    repo = Path(repo).resolve()
    if not repo.is_dir():
        raise SystemExit(f'--repo is not a directory: {repo}')

    candidates = parse_repo(repo, package_base, exclude_dirs)
    if not candidates:
        raise SystemExit(
            f'no eligible functions in {repo}: none had a first-party callee, a '
            'docstring AND a linked test.'
        )

    if file is None and func is None and cls is None:
        return _to_target(repo, candidates[0])

    wanted = Path(file).as_posix() if file else None
    matches = [
        fd
        for fd in candidates
        if (wanted is None or _relpath(repo, fd.file_path) == wanted)
        and (func is None or fd.name == func)
        and (cls is None or (fd.class_name or '') == cls)
    ]
    if not matches:
        raise SystemExit(
            f'no eligible function matches file={file!r} func={func!r} cls={cls!r} '
            f'in {repo} ({len(candidates)} candidates were eligible)'
        )
    return _to_target(repo, matches[0])
