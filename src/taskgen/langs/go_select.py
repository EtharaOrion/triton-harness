"""Function-scope target selection for Go, on top of the harness `GOParser`.

`select.py` does this for python, where a target is `(relpath, class, name)` and
the graded set is a list of pytest node ids. Go needs a different module for two
structural reasons and two parser bugs.

STRUCTURAL. Go's grader selects by PACKAGE plus an anchored `-run` regex, not by
node id, so the package import path has to be reconstructed from `go.mod` and
the directory. And a Go method's identity includes its RECEIVER: `Run` is not a
name in a file, it is a name on a type.

PARSER BUG G3. `GOParser.get_class_name` (go_parser.py:65-77) defines an inner
`search_node(root, ...)` that then closes over the OUTER `node` instead of using
`root`, so it never descends into the receiver and `class_name` is always "".
`FunctionData.__hash__` is `file_path + name`, so `(*Alpha).Run` and
`(Beta).Run` in one file are indistinguishable to every set the parser builds --
including `callee` and `test_funcs`. This module derives the receiver from the
`method_declaration` node itself and treats `(relpath, receiver, name)` as the
key. Where that key is ambiguous it RAISES: the parser's own linkage is
receiver-blind, so a silent pick would attach one method's tests to the other's
body, which is a graded set that tests the wrong code.

PARSER BUG G4. `FunctionData.is_test` is `"test" in name.lower()`, which matches
`testFormatQualifiedName`, `BenchmarkFoo` and `Testify` alike; the spike
measured ~5% (588/11702) of linked `test_funcs` as unrunnable. `go test -run`
only ever runs `TestXxx(*testing.T)` declared in a `_test.go` file, so anything
else in the regex inflates EXPECTED and the floor can never be met.

Nothing here knows about any particular repo. The frozen S-GO target lives in
the tests.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .. import _tooling_path  # noqa: F401  (sys.path + tree-sitter shim; G5-safe)

__all__ = [
    'GoSelectError',
    'GoTarget',
    'GoTest',
    'find_go_function',
    'is_runnable_test',
    'is_runnable_test_signature',
    'module_path',
    'package_import_path',
    'parse_go_repo',
    'project_name',
    'receiver_type',
    'run_regex',
    'select_go_target',
]

#: Directories a Go build never compiles into the module under test.
DEFAULT_EXCLUDE_DIRS = ('.git', 'vendor', 'testdata', 'node_modules')

#: `go help test`: a test is `func TestXxx(*testing.T)` where Xxx does not start
#: with a lower-case letter. `Test` alone qualifies; `Testify` does not.
_TEST_NAME = re.compile(r'^Test([A-Z_0-9]|$)')
#: `*testing.T` and nothing that merely starts with it -- `*testing.TB` is the
#: assert-library interface, not a test signature.
_TESTING_T = re.compile(r'\*\s*testing\.T(?![A-Za-z0-9_])')

_MODULE_LINE = re.compile(r'^\s*module\s+(\S+)', re.MULTILINE)


class GoSelectError(RuntimeError):
    """A go target that cannot be graded: ambiguous, unknown or test-less."""


# --------------------------------------------------------------------------
# go.mod -> import paths
# --------------------------------------------------------------------------


def module_path(repo) -> str:
    """The module path declared by `go.mod`, e.g. github.com/multigres/multigres."""
    gomod = Path(repo) / 'go.mod'
    if not gomod.is_file():
        raise GoSelectError(f'no go.mod at {gomod}: not a Go module root')
    match = _MODULE_LINE.search(gomod.read_text(encoding='utf-8'))
    if not match:
        raise GoSelectError(f'{gomod} declares no `module` line')
    return match.group(1)


def project_name(repo) -> str:
    """The segment `GOParser` matches imports on.

    `extract_import_info` keeps an import only when `path.split('/')[2]` equals
    the project name, so for github.com/<org>/<project> that is <project>. Get
    it wrong and the parser resolves no cross-package callee at all.
    """
    parts = module_path(repo).split('/')
    return parts[2] if len(parts) >= 3 else parts[-1]


def package_import_path(repo, module: str, path) -> str:
    """Import path of the package containing `path` (a file or a directory)."""
    repo = Path(repo).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo / candidate
    directory = candidate.parent if candidate.suffix == '.go' else candidate
    rel = Path(os.path.relpath(directory, repo)).as_posix()
    if rel.startswith('..'):
        raise GoSelectError(f'{path} is outside the module root {repo}')
    return module if rel in ('.', '') else f'{module}/{rel}'


# --------------------------------------------------------------------------
# G3: the receiver the parser refuses to give us
# --------------------------------------------------------------------------


def _first_type_identifier(node):
    """Depth-first search for the receiver's named type.

    Written as an honest recursion over ALL children because the shape varies:
    `(l *Listener)`, `(l Listener)`, `(s *Store[K, V])` and `(p pkg.Alias)` are
    a pointer_type, a type_identifier, a generic_type and a qualified_type
    respectively. go_parser's version bails out after the first child and
    recurses on the wrong variable, which is exactly why it always returns "".
    """
    if node is None:
        return None
    if node.type == 'type_identifier':
        return node
    for child in node.children:
        found = _first_type_identifier(child)
        if found is not None:
            return found
    return None


def receiver_type(func_node) -> str:
    """`Listener` for `func (l *Listener) f()`; "" for a plain function.

    The pointer is dropped deliberately: Go forbids declaring both a value and a
    pointer method of the same name on one type, so the bare type name is
    already a unique key -- and it is what a reader means by "the receiver".
    """
    if func_node is None or func_node.type != 'method_declaration':
        return ''
    receiver = func_node.child_by_field_name('receiver')
    found = _first_type_identifier(receiver)
    return found.text.decode('utf-8') if found is not None else ''


# --------------------------------------------------------------------------
# G4: which linked "test" is actually a test
# --------------------------------------------------------------------------


def is_runnable_test_signature(relpath: str, name: str, params: str) -> bool:
    """Can `go test -run '^<name>$'` in this file's package actually run it?

    All three conditions are load-bearing. A `TestFoo` outside a `_test.go`
    file is compiled into the package but is not a test; a lower-case
    `testHelper` is a helper the parser's substring `is_test` mistakes for one;
    and `BenchmarkFoo(*testing.B)` / `TestMain(*testing.M)` are neither.
    """
    if not relpath.endswith('_test.go'):
        return False
    if not _TEST_NAME.match(name):
        return False
    return bool(_TESTING_T.search(params or ''))


def is_runnable_test(repo, fd) -> bool:
    params = fd.func_node.child_by_field_name('parameters')
    return is_runnable_test_signature(
        _relpath(repo, fd.file_path),
        fd.name,
        params.text.decode('utf-8') if params is not None else '',
    )


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def _relpath(repo, file_path: str) -> str:
    return Path(os.path.relpath(file_path, Path(repo).resolve())).as_posix()


@dataclass(frozen=True)
class GoTest:
    """One runnable, `-run`-selectable test, with the package that owns it."""

    relpath: str
    name: str
    package: str


@dataclass(frozen=True)
class GoTarget:
    """A carve candidate: a function or method, and the tests that reach it."""

    repo: Path
    relpath: str
    receiver: str
    name: str
    package: str
    fd: object
    tests: tuple[GoTest, ...]

    @property
    def qualname(self) -> str:
        """`(*Listener).assignConnectionID` -- the receiver is part of the name."""
        if not self.receiver:
            return self.name
        star = '*' if self._pointer_receiver else ''
        return f'({star}{self.receiver}).{self.name}'

    @property
    def _pointer_receiver(self) -> bool:
        node = self.fd.func_node.child_by_field_name('receiver')
        return node is not None and b'*' in node.text

    @property
    def test_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.tests)

    @property
    def packages(self) -> tuple[str, ...]:
        """Every package `go test` must be pointed at, sorted."""
        return tuple(sorted({t.package for t in self.tests}))

    @property
    def expected(self) -> int:
        """The pinned denominator: how many tests the graded run must select."""
        return len(self.tests)

    @property
    def run_regex(self) -> str:
        return run_regex(self.test_names)


def run_regex(names) -> str:
    """`^(TestA|TestB)$` -- anchored, sorted, deduplicated.

    Anchoring is not cosmetic: `-run TestA` is a substring match that would also
    select `TestAB`, inflating the observed count past the pinned floor.
    """
    unique = sorted(set(names))
    if not unique:
        raise GoSelectError(
            'refusing to build an empty -run regex: `go test` with no -run '
            'selects the WHOLE package, so the graded set would be the suite'
        )
    return '^(' + '|'.join(unique) + ')$'


def parse_go_repo(repo, project=None, scope_dir=None):
    """Every Go function/method under `scope_dir`, with the parser's linkage.

    `scope_dir` narrows the walk. The whole of go-multigres is 1231 files and
    about two minutes; a single package is under a second, and a function-scope
    target's linkage does not cross a package boundary in the frozen case.

    `get_func_and_tests` is called for its SIDE EFFECT, not its return value:
    `parse_project` builds the callee graph but leaves every `test_funcs` set
    empty, and it is `get_func_and_tests` that walks the test functions and
    pushes each one onto its callees. Skip it and every target looks test-less.
    The full list is returned rather than its eligible subset because that
    subset also drops anything without a doc comment, which is an MRG-Bench
    prompt-quality rule and not a gradeability one.
    """
    from parser.go_parser import GOParser, get_func_and_tests

    repo = Path(repo).resolve()
    root = repo if scope_dir in (None, '', '.') else repo / scope_dir
    if not root.is_dir():
        raise GoSelectError(f'no such directory: {root}')
    parser = GOParser(str(repo), project or project_name(repo))
    funcs = parser.parse_project(str(root))
    get_func_and_tests(funcs)
    return funcs


def _link_tests(repo, module, fd) -> tuple[GoTest, ...]:
    tests = {
        GoTest(
            relpath=_relpath(repo, t.file_path),
            name=t.name,
            package=package_import_path(repo, module, t.file_path),
        )
        for t in fd.test_funcs
        if is_runnable_test(repo, t)
    }
    return tuple(sorted(tests, key=lambda t: (t.package, t.relpath, t.name)))


def find_go_function(
    repo,
    project=None,
    *,
    file: str | None = None,
    func: str | None = None,
    receiver: str | None = None,
    scope_dir: str | None = None,
):
    """`(FunctionData, receiver)` for one unambiguous declaration.

    Identity WITHOUT gradeability, so that "which `Run` is this" stays separate
    from "can this `Run` be graded". `select_go_target` layers the graded set on
    top; carve and prompt rendering only ever need this half.
    """
    repo = Path(repo).resolve()
    wanted_file = Path(file).as_posix() if file else None
    if scope_dir is None and wanted_file:
        scope_dir = Path(wanted_file).parent.as_posix()

    funcs = parse_go_repo(repo, project, scope_dir)
    matches = [
        fd for fd in funcs
        if (wanted_file is None or _relpath(repo, fd.file_path) == wanted_file)
        and (func is None or fd.name == func)
    ]
    if not matches:
        raise GoSelectError(
            f'no go function matches file={file!r} func={func!r} under '
            f'{scope_dir or "."} ({len(funcs)} functions were parsed)'
        )

    keyed = {(receiver_type(fd.func_node), fd.name): fd for fd in matches}
    if receiver is not None:
        fd = keyed.get((receiver, func)) if func else None
        if fd is None:
            raise GoSelectError(
                f'no go function {func!r} on receiver {receiver!r} in {file}; '
                f'receivers present: {sorted({r for r, _ in keyed}) or ["<none>"]}'
            )
        recv = receiver
    else:
        receivers = sorted({r for r, n in keyed if func is None or n == func})
        if len(receivers) > 1:
            raise GoSelectError(
                f'ambiguous target {func!r} in {file}: declared on receivers '
                f'{", ".join(receivers)}. GOParser.get_class_name is broken '
                '(spike G3) so FunctionData cannot tell them apart -- pass '
                'receiver= rather than let the graded set attach to the wrong body'
            )
        fd = matches[0]
        recv = receiver_type(fd.func_node)
    return fd, recv


def select_go_target(
    repo,
    project=None,
    *,
    file: str | None = None,
    func: str | None = None,
    receiver: str | None = None,
    scope_dir: str | None = None,
) -> GoTarget:
    """The single GRADEABLE target matching `(file, receiver, func)`.

    Deliberately total: every failure mode raised here would otherwise become a
    task that grades the wrong code or grades nothing at all.
    """
    repo = Path(repo).resolve()
    module = module_path(repo)
    fd, recv = find_go_function(
        repo, project, file=file, func=func, receiver=receiver, scope_dir=scope_dir,
    )
    tests = _link_tests(repo, module, fd)
    if not tests:
        raise GoSelectError(
            f'{func!r} on receiver {recv or "<none>"} has no runnable linked test '
            '(a Go test is TestXxx(*testing.T) in a _test.go file, spike G4). An '
            'empty graded set is a free reward, so this fails closed. Note that '
            "when two methods in one file share a name, the parser's receiver-blind "
            'linkage (G3) strands one of them with no tests at all'
        )
    return GoTarget(
        repo=repo,
        relpath=_relpath(repo, fd.file_path),
        receiver=recv,
        name=fd.name,
        package=package_import_path(repo, module, fd.file_path),
        fd=fd,
        tests=tests,
    )
