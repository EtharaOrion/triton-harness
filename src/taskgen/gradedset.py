"""The graded set: which tests a carve is allowed to be judged by.

ONE RULE, stated once so no scope and no language can drift from it:

    graded = UNION over every CARVED NON-TEST function of its linked tests
             MINUS every test that lives in a CARVED file

Both halves are load-bearing, and both fail closed.

THE UNION is over carved FUNCTIONS, not carved files, and that distinction is
what makes `function` scope a special case of the same rule rather than a
separate code path. Function scope removes one body, so the union runs over one
function and reproduces `nodeids.linked_nodeids` exactly. File and folder scope
remove every body in the carve set, so the union runs over all of them. A rule
keyed on files instead would silently widen function scope to "every test of
every function that happens to share the target's file", which grades code the
solver was never asked to write.

THE MINUS drops a test whose own body was just stubbed out. Grading it is a free
reward: the skeleton carve replaced its body with `raise NotImplementedError`,
so it cannot fail for any reason connected to the task, and under
`--delete-whole-file` it does not exist at all. `scope.resolve_carved_set`
already refuses a carve set that intersects the graded test globs, but that
guard cannot see a test that lives *inside* a carved source file -- a
`if __name__` smoke test, a doctest-style helper -- so the rule subtracts them
here too. Belt and braces, on purpose: a shrunken denominator is the single
highest-value thing for a solver to attack.

FAIL CLOSED. An empty union raises. A task whose graded set is empty scores
reward = passed/0, which every plugin's floor rejects -- but only after the
entry has been written, built and shipped. Better here.

SELECTOR EMISSION diverges per language and nothing else does:

    python   pytest node ids, `<relpath>::<Class>::<name>`, class segment
             omitted when the test is module-level
    go       bare `TestXxx` names, plus the set of packages `go test` must be
             pointed at -- go selects by package + anchored `-run`, not by id
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .nodeids import format_nodeid

__all__ = [
    'CarvedFunction',
    'GradedSelection',
    'GradedSetError',
    'LinkedTest',
    'ParsedFunction',
    'collect_carved_functions',
    'derive_graded_set',
    'parse_linked',
    'select_carved',
    'whole_suite_selection',
]

#: Which selector dialect a language speaks. Unknown languages raise rather
#: than defaulting to pytest -- a wrong selector grades nothing and looks green.
SELECTOR_KIND = {
    'python': 'pytest-allowlist',
    'go': 'go-run',
    # A language with no tree-sitter parser cannot link tests to a carved
    # function, so it grades the WHOLE suite against a measured denominator.
    'rust': 'whole-suite',
    # c-xs is the second whole-suite language: --delete-whole-file over the
    # entire src/runtime directory, no parser, no per-test selectors. The
    # grader IS the reference test.sh (three make targets, 91 units).
    'c': 'whole-suite',
    # cpp-Rux is the third: --delete-whole-file over Compiler/{Semantic,Ir,
    # CodeGen}/**, no parser, no per-test selectors. The grader is the single
    # rux-tests doctest binary registered by Tests/Unit/CMakeLists.txt.
    'cpp': 'whole-suite',
    # java-tamboui is the fourth: --delete-whole-file over
    # tamboui-widgets/src/main/java (85 files, the whole dev.tamboui.widgets
    # subsystem). The grader is `:tamboui-widgets:test` (49 test files, 69
    # JUnit suite files, 823 tests), measured host-side against the intact tree.
    'java': 'whole-suite',
}


class GradedSetError(RuntimeError):
    """A graded set that must not be shipped. Always raised, never warned."""


@dataclass(frozen=True)
class LinkedTest:
    """One test the parser linked to a carved function."""

    relpath: str
    class_name: str = ''
    name: str = ''
    #: go only: the import path `go test` must be pointed at.
    package: str = ''

    @property
    def nodeid(self) -> str:
        return format_nodeid(self.relpath, self.class_name, self.name)

    @property
    def sort_key(self) -> tuple[str, str, str]:
        return (self.relpath, self.class_name, self.name)


@dataclass(frozen=True)
class CarvedFunction:
    """A non-test function whose body the carve removed."""

    relpath: str
    qualname: str
    name: str
    tests: tuple[LinkedTest, ...] = ()


@dataclass(frozen=True)
class ParsedFunction:
    """Every function the parser saw, before the carve set is applied.

    Parsing is the expensive half (a full python repo is ~16s, a full Go module
    ~2min) and the rule is the interesting half, so they are separate: `emit`
    parses once and selects once, and the rule can be exercised on a table.
    """

    relpath: str
    qualname: str
    name: str
    is_test: bool
    tests: tuple[LinkedTest, ...] = ()
    #: The parser's `FunctionData`, so the carve and the callee_* contexts get
    #: byte ranges without re-parsing. It is a node handle, not identity, hence
    #: compare=False -- including it would break determinism comparisons.
    fd: object = field(default=None, compare=False, repr=False)

    def as_carved(self) -> CarvedFunction:
        return CarvedFunction(
            relpath=self.relpath, qualname=self.qualname,
            name=self.name, tests=self.tests,
        )


@dataclass(frozen=True)
class GradedSelection:
    """The denominator, the selectors that produce it, and where they live."""

    kind: str
    selectors: tuple[str, ...]
    packages: tuple[str, ...]
    tests: tuple[LinkedTest, ...]
    carved_files: tuple[str, ...]
    carved_functions: tuple[str, ...]
    #: Whole-suite only: a denominator MEASURED against the intact tree rather
    #: than counted from selectors. None keeps python/go on len(selectors).
    _expected: int | None = None
    #: Whole-suite only: the literal grade-time command, carried to the plugin's
    #: test.sh because a parser-less language has no selector list to render one.
    test_command: str = ''
    #: Whole-suite only: explicit relpaths to sha256-fingerprint. python/go
    #: derive this from `tests` (one relpath per graded test); a whole-suite
    #: language with no per-test tests entry still needs a way to declare its
    #: grader-lock (c-xs pins 227 files under tests/ + Makefile), which this is.
    _fingerprint_relpaths: tuple[str, ...] = ()

    @property
    def expected(self) -> int:
        return self._expected if self._expected is not None else len(self.selectors)

    @property
    def fingerprint_relpaths(self) -> tuple[str, ...]:
        """The graded test FILES -- what the plugins sha256-lock."""
        if self._fingerprint_relpaths:
            return self._fingerprint_relpaths
        return tuple(sorted({t.relpath for t in self.tests}))

    def to_dict(self) -> dict:
        """The `tests/graded.json` record. Sorted throughout; no host state."""
        return {
            'kind': self.kind,
            'selectors': list(self.selectors),
            'packages': list(self.packages),
            'expected': self.expected,
            'carved_files': list(self.carved_files),
            'carved_functions': list(self.carved_functions),
            'fingerprint_relpaths': list(self.fingerprint_relpaths),
        }


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------


def derive_graded_set(language: str, carved_functions, carved_files) -> GradedSelection:
    """UNION of the carved functions' tests, MINUS tests in the carved files."""
    kind = SELECTOR_KIND.get(language)
    if kind is None:
        raise GradedSetError(
            f'no graded-set selector dialect for language {language!r}; known: '
            f'{", ".join(sorted(SELECTOR_KIND))}'
        )

    funcs = tuple(carved_functions)
    files = tuple(sorted({Path(f).as_posix() for f in carved_files}))
    if not funcs:
        raise GradedSetError(
            f'no carved non-test function found in {list(files)} -- there is '
            'nothing whose regeneration could be graded. Widen the carve set, or '
            'use --delete-whole-file if the intent was to remove data files.'
        )

    carved_set = set(files)
    linked: set[LinkedTest] = set()
    for fn in funcs:
        linked |= set(fn.tests)
    graded = sorted(
        (t for t in linked if Path(t.relpath).as_posix() not in carved_set),
        key=lambda t: t.sort_key,
    )

    if not graded:
        dropped = len(linked)
        raise GradedSetError(
            f'empty graded set: {len(funcs)} carved function(s) across {len(files)} '
            f'file(s) linked {dropped} test(s), and every one of them lives inside '
            'the carve set. A carved test is a stubbed test -- grading it would be '
            'a free reward -- so the union is empty and this fails closed.'
        )

    if kind == 'go-run':
        selectors = tuple(sorted({t.name for t in graded}))
    else:
        selectors = tuple(sorted({t.nodeid for t in graded}))
    packages = tuple(sorted({t.package for t in graded if t.package}))

    return GradedSelection(
        kind=kind,
        selectors=selectors,
        packages=packages,
        tests=tuple(graded),
        carved_files=files,
        carved_functions=tuple(sorted(f'{f.relpath}::{f.qualname}' for f in funcs)),
    )


def whole_suite_selection(
    language: str,
    *,
    expected: int,
    test_command: str,
    carved_files=(),
    fingerprint_relpaths=(),
) -> GradedSelection:
    """The graded set for a language with no parser: the WHOLE test suite.

    python and go derive a denominator by parsing which tests link to the carved
    function. A language without a tree-sitter parser (rust, c) cannot, so it
    grades the entire integration suite against a denominator MEASURED on the
    intact tree and passed in here, never len(selectors). There are no per-test
    selectors; the anti-shrink defence is the equality floor over `expected`
    plus, when the plugin opts in via `fingerprint_relpaths`, an sha256 lock
    over the grader-owned files (c-xs pins 227 = tests/** + Makefile). rust
    leaves it empty; the wast-corpus floor and harness-count guard in its
    test.sh cover the same role.
    """
    kind = SELECTOR_KIND.get(language)
    if kind != 'whole-suite':
        raise GradedSetError(
            f'whole_suite_selection is only for whole-suite languages; {language!r} '
            f'has dialect {kind!r}'
        )
    if not isinstance(expected, int) or isinstance(expected, bool) or expected < 1:
        raise GradedSetError(
            f'whole-suite expected must be a positive int, got {expected!r}'
        )
    if not test_command.strip():
        raise GradedSetError('whole-suite grading needs a non-empty test command')
    return GradedSelection(
        kind=kind,
        selectors=(),
        packages=(),
        tests=(),
        carved_files=tuple(sorted({Path(f).as_posix() for f in carved_files})),
        carved_functions=(),
        _expected=int(expected),
        test_command=test_command,
        _fingerprint_relpaths=tuple(
            sorted({Path(p).as_posix() for p in fingerprint_relpaths})
        ),
    )


# --------------------------------------------------------------------------
# collectors: repo + carve set -> the carved functions, with their linkage
# --------------------------------------------------------------------------


def _relpath(repo, file_path: str) -> str:
    return Path(os.path.relpath(file_path, Path(repo).resolve())).as_posix()


def select_carved(parsed, carved_relpaths, *, only=None) -> tuple[CarvedFunction, ...]:
    """The pure half: which parsed functions the carve set actually removed.

    Test functions are excluded here rather than at parse time, because the
    MINUS half of the rule needs to know a carved file's tests exist in order
    to drop them.
    """
    carved = {Path(r).as_posix() for r in carved_relpaths}
    out = [
        fn.as_carved()
        for fn in parsed
        if fn.relpath in carved
        and not fn.is_test
        and (only is None or fn.name in only)
    ]
    return tuple(sorted(out, key=lambda f: (f.relpath, f.qualname)))


def parse_linked(
    language: str,
    repo,
    *,
    package_base: str = 'src/',
    project: str | None = None,
    scope_dirs=None,
) -> tuple[ParsedFunction, ...]:
    """Every function the parser can see, with `test_funcs` linkage populated."""
    if language == 'python':
        return _parse_python(repo, package_base)
    if language == 'go':
        return _parse_go(repo, project, scope_dirs)
    raise GradedSetError(
        f'no carved-function collector for language {language!r}; known: python, go'
    )


def collect_carved_functions(
    language: str,
    repo,
    carved_relpaths,
    *,
    package_base: str = 'src/',
    project: str | None = None,
    only=None,
) -> tuple[CarvedFunction, ...]:
    """Every non-test function declared in `carved_relpaths`, with its tests.

    `only` restricts to named functions, which is how `function` scope reuses
    this collector rather than owning a second implementation of the rule.
    """
    scope_dirs = sorted({Path(r).parent.as_posix() for r in carved_relpaths})
    parsed = parse_linked(
        language, repo, package_base=package_base, project=project,
        scope_dirs=scope_dirs,
    )
    return select_carved(parsed, carved_relpaths, only=only)


def _parse_python(repo, package_base) -> tuple[ParsedFunction, ...]:
    from parser.py_parser import PyParser, get_func_and_tests

    from .select import DEFAULT_EXCLUDE_DIRS

    repo = Path(repo).resolve()
    parser = PyParser(
        str(repo),
        exclude_dirs=list(DEFAULT_EXCLUDE_DIRS),
        package_base_path=package_base,
    )
    funcs = parser.funcs
    # Called for its SIDE EFFECT: parse populates the callee graph but leaves
    # every `test_funcs` set empty, and it is this pass that walks the test
    # functions and pushes each one onto its callees.
    get_func_and_tests(funcs)

    out = []
    for fd in funcs:
        qual = f'{fd.class_name}.{fd.name}' if fd.class_name else fd.name
        tests = tuple(sorted(
            {
                LinkedTest(_relpath(repo, t.file_path), t.class_name or '', t.name)
                for t in fd.test_funcs
            },
            key=lambda t: t.sort_key,
        ))
        out.append(ParsedFunction(
            relpath=_relpath(repo, fd.file_path), qualname=qual, name=fd.name,
            is_test=bool(fd.is_test_func), tests=tests, fd=fd,
        ))
    return tuple(sorted(out, key=lambda f: (f.relpath, f.qualname)))


def _parse_go(repo, project, scope_dirs) -> tuple[ParsedFunction, ...]:
    from .langs import go_select as GS

    repo = Path(repo).resolve()
    module = GS.module_path(repo)
    # Narrow the parse to the directories actually carved: the whole of
    # go-multigres is 1231 files and about two minutes, one package is instant.
    dirs = sorted({Path(d).as_posix() for d in (scope_dirs or ['.'])}) or ['.']
    seen: dict[tuple[str, str, str], ParsedFunction] = {}

    for scope_dir in dirs:
        for fd in GS.parse_go_repo(repo, project, scope_dir):
            rel = _relpath(repo, fd.file_path)
            recv = GS.receiver_type(fd.func_node)
            qual = f'({recv}).{fd.name}' if recv else fd.name
            tests = tuple(sorted(
                {
                    LinkedTest(
                        _relpath(repo, t.file_path), '', t.name,
                        GS.package_import_path(repo, module, t.file_path),
                    )
                    for t in fd.test_funcs
                    if GS.is_runnable_test(repo, t)
                },
                key=lambda t: t.sort_key,
            ))
            seen[(rel, recv, fd.name)] = ParsedFunction(
                relpath=rel, qualname=qual, name=fd.name,
                # G4: `is_test` on FunctionData is a substring match, so a
                # `testHelper` would count. Only a runnable Go test is a test.
                is_test=GS.is_runnable_test(repo, fd),
                tests=tests, fd=fd,
            )
    return tuple(sorted(seen.values(), key=lambda f: (f.relpath, f.qualname)))
