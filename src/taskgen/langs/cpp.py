"""The cpp plugin: whole-suite grading, measured floor, cmake+ninja+g++-14, doctest.

cpp-Rux is the third whole-suite language after rust and c. The reference carve
is `Compiler/{Semantic,Ir,CodeGen}/**/*.{h,cpp}` (~35 files, the whole compiler
middle-end + backend); with those subsystems gone nothing linkable remains that
the doctest harness can reach, so per-test selectors are meaningless and the
graded surface collapses to "the whole ctest suite" -- which for Rux is exactly
ONE registered ctest (`add_test(NAME UnitTests COMMAND rux-tests)`) that runs
the single doctest binary covering 170 TEST_CASEs across ~2900 assertions.

FLOOR MODE: equality. The graded unit is a doctest TEST_CASE, and doctest
reports per-case pass/fail counts in a single trailing summary line -- observed
cases == EXPECTED cases is a real assertion (a crash mid-suite prints a
truncated summary or none at all, both of which the grader catches ahead of
the floor). This is the same argument as rust and c (spike G1 applies only to
Go's whole-binary-abort-on-panic semantics).

TOOLCHAIN: cmake 4.4+, g++-14 and ninja from Kitware+Ubuntu apt. harbor-base
ships cmake 3.28.3 + gcc 13.3.0 + no ninja + no clang; Rux declares
`cmake_minimum_required(VERSION 3.30...4.3)` and `CMAKE_CXX_STANDARD 26`, and
its AArch64 backend shells out to `clang` at link time -- all four gaps have to
be closed at BUILD time, which is the only time apt.kitware.com is reachable.
Grading itself is `--network=none` (task.toml pins it), so the apt install must
land in a graded-image layer before the leak gate. That is what pre-leakgate
was designed for, and cpp's install goes there for the same reason rust's
rustup does: everything must be present before leakscan runs.

GRADER FINGERPRINT. The reference asset writes `/opt/harbor/tests.sha256` at
image build time and asserts it aggregate-style at grade time; taskgen bakes the
same lock INLINE via `fingerprint_gate_block`, per-file across every file under
`Tests/**`. Per-file is STRICTLY STRONGER than the reference aggregate lock: a
tampered test binary that recompiles from a mutated Tests/Unit/*.cpp would fail
the per-file check regardless of any file-count trick.

NO REPO-MAGIC NUMBERS IN PLUGIN SOURCE. 170 (doctest TEST_CASEs) is the ONLY
number that would need threading, and it comes from `graded.expected` -- the
denominator measured host-side in phase 1 by `measure.py` running the intact
tree's build+rux-tests once and reading the `[doctest] test cases:` summary.
Assertion count (~2900) is DELIBERATELY not threaded: the Tests/** per-file
fingerprint prevents any mutation of the doctest bodies that would change the
assertion tally, so `nassert > 0` (structural) plus the fingerprint gate
subsumes the reference's `nassert >= 2900` gate. This is Route (b) in the
plan's design; Route (a) (thread the assertion count) would need
`measure.parse_measure_json` to tolerate extra keys, which it currently does
not, and offers no additional guarantee once the fingerprint is per-file.

BUILD PARALLELISM. The reference test.sh uses UNBOUNDED
`cmake --build ... --parallel`, which on a 64-core aarch64 grading box spawns
64 concurrent cc1plus. Each C++26 -O3 TU peaks near 1.5 GB (measured), so 64x
easily exceeds any reasonable cgroup memory cap and kills the build with
`g++-14: fatal error: Killed signal terminated program cc1plus`.
CMAKE_BUILD_PARALLEL_LEVEL is NOT honored by `--parallel`. This grader
therefore hardcodes `--parallel 4` (4 * ~1.5 GB ~ 6 GB, comfortably under the
task.toml-declared 8 GB `memory_mb`). This is a DELIBERATE, DOCUMENTED
improvement over the reference; a faithful unbounded grader would OOM the
delivered task at its own declared memory on a large-core host.

DOCKERFILE INVARIANTS. `git init` is NOT synthesised (unlike python); the
base's ban on it stays in force. No vendored assets (unlike rust's wabt), no
dep_warm stage (deps arrive via apt), and no strings-target leak assert
(unlike rust the shipped image never compiles carved sources; the carved
Compiler/* subtree is gone, so there is no build/ tree in which mangled
symbols could survive).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Mapping

from ..depplan import (
    DepPlan,
    FlagValue,
    HarnessValue,
    TestValue,
    canonicalize,
    validate,
)
from . import base as B
from .base import DepWarmSpec, EnvSpec, GradedSet, ToolchainSpec

__all__ = [
    'BAKED_CAPABILITIES',
    'CPP_MEASURE_DEP_PLAN',
    'CPP_RUX_HARNESS',
    'CppPlugin',
    'GRADER_FINGERPRINT_GLOBS',
    'Harness',
    'MAX_BUILD_PARALLEL',
    'MEASURE_NO_WARM_COMMENT',
    'REQUIRED_BUILD_FLAGS',
    'REQUIRED_PLAN_SLOTS',
    'REQUIRED_TEST_INVOCATION_KEYS',
    'TEST_COMMAND',
    'read_harness',
]

#: cpp-Rux's own harness as a `DepPlan.harness` section -- the fixed point of
#: the refactor: rendered, it reproduces the scripts this plugin used to
#: hardcode byte for byte. Every value is a fact about the cpp-Rux REPOSITORY,
#: which is exactly why none of it belongs in the plugin.
#:
#: `harness_files` are the files whose PRESENCE the graded verifier refuses to
#: score without: they ARE the doctest oracle (Main.cpp registers the driver,
#: the *Tests.cpp files own the TEST_CASE bodies, doctest.h is the framework
#: header). Per-file fingerprinting protects their CONTENTS; this list is the
#: presence check. `registered_tests` is how many tests the build registers
#: with ctest -- the structural net that used to be a bare `-eq 1`.
CPP_RUX_HARNESS: tuple[tuple[str, HarnessValue], ...] = (
    ('build_artifacts', ('Bin/Tests/Unit/rux-tests',)),
    ('harness_files', (
        'Tests/Unit/CMakeLists.txt',
        'Tests/Unit/Main.cpp',
        'Tests/Unit/SemanticTests.cpp',
        'Tests/Unit/CodegenTests.cpp',
        'Tests/Unit/OptimizerTests.cpp',
        'Tests/Unit/GoldenDiagnosticsTests.cpp',
        'Tests/Unit/ThirdParty/doctest.h',
    )),
    ('harness_files_note', 'doctest driver, doctest.h and the four *Tests.cpp bodies'),
    ('project_label', 'Rux'),
    ('registered_test_names', ('UnitTests',)),
    ('registered_tests', 1),
    ('stale_paths', ('Build', 'Bin')),
)

#: What the plugin locks against tampering: the repo's test tree, under any of
#: the three spellings C++ projects use for it. The aggregate-hash reference
#: lock (`/opt/harbor/tests.sha256`) is replaced with an inlined PER-FILE lock
#: built host-side at generate time from the intact tree -- strictly stronger,
#: because a solver can no longer swap one test file for another of the same
#: aggregate size + hash chain.
#:
#: THREE globs rather than cpp-Rux's one `Tests/**`: these are expanded by
#: `emit.plan_carve` BEFORE any environment is resolved, so they cannot come
#: from the plan, and a plugin that only knew one repository's capital-T
#: layout refused every project that spells the directory `test/` or `tests/`.
#: Only the globs that actually match contribute, so a repo using one spelling
#: locks exactly the files it always did.
GRADER_FINGERPRINT_GLOBS: tuple[str, ...] = ('Tests/**', 'test/**', 'tests/**')

#: The ceiling this grader puts on `cmake --build --parallel N` (see the module
#: docstring): a per-LANGUAGE memory fact, not a per-repo one, so it is enforced
#: against whatever the plan asks for rather than read out of it.
MAX_BUILD_PARALLEL = 4

#: apt+kitware install block. Transcribed VERBATIM from the reference
#: environment/Dockerfile (rux-lang/Rux, MIT) -- rendered here as a module
#: constant so the shipped Dockerfile, the measure Dockerfile and any test
#: that inspects the install text all see byte-identical content.
#: Per-language base carrying cmake/g++-14/ninja/clang already.
BASE_IMAGE = '426628337772.dkr.ecr.ap-south-2.amazonaws.com/triton/base-cpp@sha256:3e2450a37081e6fa480f2a09de01620d9dba3578d49b281629f36c6dbd9ce074'

_INSTALL_BLOCK = (
    '# The base ships cmake 4.4.2, g++-14 14.2.0, ninja 1.12.1 and clang 19.1.7 --\n'
    '# exactly the set this block used to apt-install on every task build, at the\n'
    '# cost of two package-index refreshes, a Kitware key import and an unpinned\n'
    '# package resolution inside a network the grade step forbids.\n'
    '# Make the C++26-capable compiler the default so a plain `g++`/`cc` picks\n'
    '# up 14 rather than the distro default. `cc`/`c++` are already MASTER\n'
    '# alternatives, so they cannot be attached as slaves of the gcc group.\n'
    'RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 \\\n'
    '      --slave /usr/bin/g++ g++ /usr/bin/g++-14 \\\n'
    ' && update-alternatives --install /usr/bin/cc  cc  /usr/bin/gcc-14 100 \\\n'
    ' && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-14 100\n'
    '# Assert under a login shell: the base re-exports its own PATH from\n'
    "# /etc/profile.d, which is how harbor's test.sh resolves these binaries.\n"
    'RUN set -eux; \\\n'
    '    c="$(bash -lc \'cmake --version | head -1\')"; \\\n'
    '    g="$(bash -lc \'g++-14 --version | head -1\')"; \\\n'
    '    case "$c" in *4.4.2*) ;; *) echo "TOOLCHAIN PIN FAILED (login shell): $c" >&2; exit 42;; esac; \\\n'
    '    case "$g" in *14.2.0*) ;; *) echo "TOOLCHAIN PIN FAILED (login shell): $g" >&2; exit 42;; esac; \\\n'
    '    bash -lc \'ninja --version && clang --version | head -1\''
)

_LOGS_DEFAULT = '${VERIFIER_DIR:-' + B.LOGS_DIR + '}'
_MEASURE_LOGS_DEFAULT = '${MEASURE_DIR:-' + B.LOGS_DIR + '}'

#: The measure image's dep-warm slot. cpp has no warm stage (the toolchain is
#: baked and the repo vendors doctest), so this comment IS the slot's content --
#: it is not `DepWarmSpec.render()`, which speaks about a COPY that never
#: happens here.
MEASURE_NO_WARM_COMMENT = '# no warmed dependencies (cpp installs its toolchain via apt above)'

#: What the base image already provides, as `--resolve-env` states it to a
#: model. Sorted and version-exact, so the same prompt is built on every run.
#: Declared, never probed: the same four facts the pin assert in `render_gap`
#: checks at build time.
BAKED_CAPABILITIES: tuple[str, ...] = (
    'apt-get (build time only; the graded run has no network)',
    'clang 19.1.7 (the AArch64 backend shells out to it at link time)',
    'cmake 4.4.2',
    'g++-14 14.2.0 (C++26-capable)',
    'ninja 1.12.1',
)

#: Every `build_flags` key `render_gap` interpolates. Enumerated once, read by
#: BOTH the renderer and `CppPlugin.validate_dep_plan`, so a flag added to the
#: rendered text without being added here is the only way the two can disagree.
REQUIRED_BUILD_FLAGS: tuple[str, ...] = (
    'clang_version', 'cmake_version', 'ninja_version',
)

#: Every `test_invocation` key the cpp measure path needs. `render_gap` does not
#: interpolate them -- the measure script runs the three steps itself -- but a
#: plan that does not state what it was resolved FOR is a plan nobody can check
#: against the image it produced, so they are required at the same gate.
#: THREE keys rather than c's one: cpp's graded command is `configure && build
#: && test`, and `&&` is a shell metacharacter `depplan.validate` refuses, so
#: the one shell string is carried as the three argv vectors it really is.
REQUIRED_TEST_INVOCATION_KEYS: tuple[str, ...] = ('build', 'configure', 'test')

#: The same requirements as the resolver is told them, before it answers. Kept
#: adjacent to the checks above so the ask and the rejection cannot drift apart.
#: Leak-safe: slot names, shapes and toolchain versions only -- never a repo
#: path, a target name or a source body.
REQUIRED_PLAN_SLOTS: tuple[str, ...] = (
    'build_flags["clang_version"] must be a non-empty string: the clang version '
    'the base image ships, e.g. "19.1.7"',
    'build_flags["cmake_version"] must be a non-empty string: the version of the '
    'package_manager binary the base image ships, e.g. "4.4.2"; the build-time '
    'pin assert matches it verbatim',
    'build_flags["ninja_version"] must be a non-empty string: the ninja version '
    'the base image ships, e.g. "1.12.1"',
    'install_commands must NOT compile the graded sources -- test_invocation '
    'configures, builds and runs the suite itself -- so for a cmake-driven repo '
    'this list is USUALLY EMPTY. Add a step only to PREPARE the environment '
    '(fetch or vendor a dependency); never a bare "cmake --build". A build step '
    'that cannot succeed makes the whole plan unbuildable, not merely suboptimal',
    'apt_packages lists ONLY system libraries the sources need that the base '
    'image lacks; the compiler, the generator and cmake are already baked in, so '
    'naming them here buys nothing and costs a network fetch',
    'package_manager must be cmake: the gap runs its version pin against that '
    'binary and the repo is configured through it',
    'test_invocation["build"] must be a non-empty list of argv tokens: the '
    'command that compiles the configured tree, e.g. ["cmake", "--build", '
    '"Build"]. Leave "--parallel" out unless the repo needs a specific value: '
    f'the grader caps it at {MAX_BUILD_PARALLEL} and adds it for you, because '
    'a C++ translation unit peaks near 1.5 GB and an unbounded --parallel is '
    'OOM-killed on a big-core host',
    'test_invocation["configure"] must be a non-empty list of argv tokens: the '
    'command that configures the build tree, e.g. ["cmake", "-S", ".", "-B", '
    '"Build"]',
    'test_invocation["test"] must be a non-empty list of argv tokens: the '
    'command that runs the whole graded suite, e.g. ["ctest", "--test-dir", '
    '"Build"]',
    'toolchain_version must carry at least major.minor.patch, e.g. "14.2.0": it '
    'is the C++ compiler version, the build-time pin asserts it verbatim, and '
    'its major component selects the gcc-N/g++-N binaries',
    'harness is REQUIRED for cpp and describes THIS repository\'s own test '
    'harness; it is what the graded verifier script is rendered from. State it '
    'from the build manifests and the TEST FILE PATHS you were given, and from '
    'nothing else. harness["build_artifacts"] is a list holding EXACTLY ONE '
    'repo-relative path: the doctest executable the build produces and the '
    'verifier runs directly, e.g. ["Bin/Tests/Unit/unit-tests"]. Read it off '
    'the target\'s RUNTIME_OUTPUT_DIRECTORY and its target name; a wrong path '
    'makes the whole plan unmeasurable, not merely suboptimal',
    'harness["stale_paths"] lists the build-output directories the verifier '
    'wipes before it configures, e.g. ["Build", "Bin"]. A stale still-passing '
    'binary left in the image would let the suite report green with zero '
    'regenerated code, so name EVERY directory the configure and build steps '
    'write into. Empty is allowed only for a build that writes nowhere',
    'harness["registered_tests"] is an INTEGER: how many tests the configured '
    'build registers with ctest (one per add_test in the CMake files). It is '
    'cross-checked against an independent host-side scan of the intact tree, '
    'and a disagreement REFUSES the plan -- because a suite you under-count is '
    'a suite that is partly ungraded, and a shrunken denominator still passes '
    'every gate. harness["registered_test_names"] names them, in declaration '
    'order, and must have exactly that many entries (or be empty). This '
    'verifier grades ONE doctest binary, so registered_tests must equal the '
    'length of build_artifacts: a repo registering several test executables '
    'is REFUSED loudly rather than graded on the first one',
    'harness["harness_files"] lists the repo-relative files whose PRESENCE the '
    'verifier refuses to score without -- the test driver, the framework '
    'header and the files owning the test bodies. The per-file fingerprint '
    'protects their contents; this is the belt-and-braces existence check. '
    'Empty is allowed. harness["harness_files_note"] is an optional short '
    'phrase naming them in prose, e.g. "the driver, doctest.h and the test '
    'bodies"; harness["project_label"] is the project\'s own name, e.g. the '
    'argument of the CMake project() call. Both are prose only and may be ""',
)

#: The one package manager cpp's gap has prose (and a version pin) for. `make`
#: is schema-legal for cpp but this gap runs `cmake --version` and configures a
#: build tree; rendering it for `make` would emit an assert that cannot pass.
_SUPPORTED_MANAGERS: frozenset[str] = frozenset({'cmake'})


def _flag_str(build_flags: tuple[tuple[str, FlagValue], ...], key: str) -> str:
    """A build flag the gap's rendered text needs, or a refusal to render at all."""
    for name, value in build_flags:
        if name == key:
            if isinstance(value, str) and value:
                return value
            break
    raise B.LangError(
        f'the cpp gap needs build_flags[{key!r}]: it must be a non-empty string. '
        'A plan without it would render a Dockerfile pin with a hole in it, and '
        'a hole in a version assert is how a base swap goes unnoticed'
    )


def _test_tokens(
    test_invocation: tuple[tuple[str, TestValue], ...], key: str,
) -> tuple[str, ...]:
    """A test_invocation entry the cpp measure path needs, as argv tokens."""
    for name, value in test_invocation:
        if name == key:
            tokens = (value,) if isinstance(value, str) else tuple(value)
            if tokens and all(token.strip() for token in tokens):
                return tokens
            break
    raise B.LangError(
        f'the cpp plan needs test_invocation[{key!r}]: it must be a non-empty '
        'list of argv tokens. The graded command is configure, build and run, '
        'and a plan that does not state all three cannot be checked against the '
        'image it produced'
    )


def _compiler_major(toolchain_version: str) -> str:
    """The `N` in the `gcc-N`/`g++-N` the alternatives group points at.

    A version whose first component is not a number cannot name a binary, and a
    Dockerfile that installs an alternative for `gcc-` is a build failure with
    no diagnostic, so an unusable version is refused before it is rendered.
    """
    major = toolchain_version.partition('.')[0]
    if not major.isdigit():
        raise B.LangError(
            f'toolchain_version {toolchain_version!r} has no numeric major '
            'component, so there is no gcc-N to make the default compiler; the '
            'cpp gap cannot render an alternatives group without one'
        )
    return major


def _parallelism(build_tokens: tuple[str, ...]) -> tuple[tuple[str, ...], int]:
    """The build argv with this grader's OOM ceiling enforced, and that ceiling.

    The cap is the plugin's, not the plan's (module docstring: 64 concurrent
    cc1plus at ~1.5 GB each is OOM-killed at the 8 GB the task declares). A plan
    that says nothing gets it added; a plan that asks for more is REFUSED rather
    than silently narrowed, because a resolver told its value was honoured while
    it was not is a resolver that cannot repair anything.
    """
    if '--parallel' not in build_tokens:
        return (*build_tokens, '--parallel', str(MAX_BUILD_PARALLEL)), MAX_BUILD_PARALLEL
    index = build_tokens.index('--parallel')
    value = build_tokens[index + 1] if index + 1 < len(build_tokens) else ''
    if not value.isdigit() or not 1 <= int(value) <= MAX_BUILD_PARALLEL:
        raise B.LangError(
            f'test_invocation["build"] asks for `--parallel {value}`; the cpp '
            f'grader caps build parallelism at {MAX_BUILD_PARALLEL}. Each C++ '
            'translation unit peaks near 1.5 GB and the shipped task declares '
            '8 GB, so an unbounded or larger --parallel is OOM-killed on a '
            'big-core host. Drop the flag and the renderer adds the cap itself'
        )
    return build_tokens, int(value)


def _configure_lines(argv: tuple[str, ...], log_var: str) -> list[str]:
    """The configure argv as a shell continuation: tool line, then one -D each.

    Cache variables get a line apiece because a configure line is the one place
    a reader diffs a task against upstream CI, and a fifteen-token single line
    is unreadable. The split point is the first `-D`, so a plan that declares no
    cache variables renders one line and nothing else.
    """
    split = max(
        next((i for i, token in enumerate(argv) if token.startswith('-D')), len(argv)),
        1,
    )
    return [
        ' '.join(argv[:split]) + ' \\',
        *[f'      {token} \\' for token in argv[split:]],
        f'      > "${{{log_var}}}" 2>&1',
    ]


def _werror_var(configure: tuple[str, ...]) -> str:
    """The repo's warnings-as-errors cache variable, if its configure sets one.

    The rendered rationale for building the regenerated code under -Werror is
    only true of a repo that asks for it, so it is emitted only when the plan's
    own configure argv turns such a switch ON. It cannot move into the plan as
    prose: `DepPlan` refuses `;` and `()` in every token, and weakening that
    gate to carry a comment would be trading the metacharacter guarantee for a
    sentence.
    """
    for token in configure:
        name, _, value = token.removeprefix('-D').partition('=')
        if token.startswith('-D') and value == 'ON' and name.endswith('WERROR'):
            return name
    return ''


_ENGLISH: tuple[str, ...] = (
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
    'nine', 'ten',
)


def _english(n: int) -> str:
    return _ENGLISH[n] if 0 <= n < len(_ENGLISH) else str(n)


def _dir_phrase(directories) -> str:
    """`Build/ and Bin/` -- how the rendered comments name a set of directories."""
    return ' and '.join(f'{d}/' for d in directories)


def _slot_error(key: str, want: str) -> B.LangError:
    return B.LangError(
        f'the cpp harness needs harness[{key!r}]: {want}. Without it the '
        'rendered verifier would either grade nothing or grade a suite nobody '
        'described, and a denominator nobody described is not a floor'
    )


def _tuple_slot(values: Mapping[str, HarnessValue], key: str,
                want: str, *, required: bool = False) -> tuple[str, ...]:
    raw = values.get(key)
    if raw is None:
        if required:
            raise _slot_error(key, want)
        return ()
    if isinstance(raw, (bool, int)):
        raise _slot_error(key, f'{want} (a list of strings, not a scalar)')
    tokens = (raw,) if isinstance(raw, str) else tuple(raw)
    if required and not tokens:
        raise _slot_error(key, want)
    return tokens


def _str_slot(values: Mapping[str, HarnessValue], key: str, want: str) -> str:
    raw = values.get(key)
    if raw is None or raw == '':
        return ''
    if not isinstance(raw, str):
        raise _slot_error(key, f'{want} (a single string)')
    return raw


def _int_slot(values: Mapping[str, HarnessValue], key: str, want: str) -> int:
    raw = values.get(key)
    if isinstance(raw, bool) or raw is None:
        raise _slot_error(key, want)
    if isinstance(raw, int):
        parsed = raw
    elif isinstance(raw, str) and raw.strip().isdigit():
        parsed = int(raw)
    else:
        raise _slot_error(key, f'{want} (a whole number)')
    if parsed < 1:
        raise _slot_error(key, f'{want} (at least 1; zero would grade nothing)')
    return parsed


@dataclass(frozen=True)
class Harness:
    """A cpp test harness as data: build it, find its binary, decide its cases.

    Everything the two rendered scripts used to hardcode about the cpp-Rux REPO,
    read off `DepPlan.harness`. `read_harness` is the only constructor, so a
    slot that is missing, mistyped or inconsistent with its siblings becomes a
    `LangError` before any container exists rather than a shell script that
    looks for a binary nobody builds and reports a floor of zero.
    """

    build_artifacts: tuple[str, ...]
    stale_paths: tuple[str, ...]
    harness_files: tuple[str, ...]
    harness_files_note: str
    project_label: str
    registered_tests: int
    registered_test_names: tuple[str, ...]

    @property
    def test_binary(self) -> str:
        return self.build_artifacts[0]

    @property
    def test_binary_name(self) -> str:
        return self.test_binary.rsplit('/', 1)[-1]


def read_harness(plan: DepPlan) -> Harness:
    """`plan.harness` as the record the two renderers read, or a `LangError`.

    Every check here is reachable from the refine loop, so every message names
    the slot and says what a correct value looks like: a resolver that gets one
    wrong must be able to repair it without being shown the repository.
    """
    values = {key: value for key, value in canonicalize(plan).harness}
    harness = Harness(
        build_artifacts=_tuple_slot(
            values, 'build_artifacts',
            'the repo-relative path of the doctest executable the build '
            'produces and the verifier runs directly',
            required=True,
        ),
        stale_paths=_tuple_slot(
            values, 'stale_paths',
            'the build-output directories wiped before the configure step, so '
            'no stale binary can report a cached pass',
        ),
        harness_files=_tuple_slot(
            values, 'harness_files',
            'the repo-relative files whose presence the verifier refuses to '
            'score without',
        ),
        harness_files_note=_str_slot(
            values, 'harness_files_note',
            'a short prose phrase naming the harness files',
        ),
        project_label=_str_slot(
            values, 'project_label', "the project's own name",
        ),
        registered_tests=_int_slot(
            values, 'registered_tests',
            'how many tests the configured build registers with ctest',
        ),
        registered_test_names=_tuple_slot(
            values, 'registered_test_names',
            'the names of the tests the build registers with ctest, in '
            'declaration order',
        ),
    )
    names = harness.registered_test_names
    if names and len(names) != harness.registered_tests:
        raise B.LangError(
            f'harness["registered_test_names"] has {len(names)} entries but '
            f'registered_tests is {harness.registered_tests}; the two describe '
            'the same ctest registry, and a list shorter than the count is a '
            'registration nobody named'
        )
    if len(harness.build_artifacts) != harness.registered_tests:
        raise B.LangError(
            f'harness["registered_tests"] is {harness.registered_tests} but '
            f'build_artifacts names {len(harness.build_artifacts)} executable(s). '
            'Every ctest registration must name the binary this verifier runs, '
            'or the registrations it does not name are graded by nobody and the '
            'denominator silently shrinks to the part of the suite that was '
            'described'
        )
    if len(harness.build_artifacts) != 1:
        raise B.LangError(
            'the cpp verifier grades ONE doctest binary: it runs it directly '
            'and reads a single `[doctest] test cases:` summary. This plan '
            f'describes {len(harness.build_artifacts)} graded binaries, whose '
            'cases this renderer cannot sum -- so it REFUSES rather than '
            'pinning a floor over the first one and leaving the rest ungraded'
        )
    return harness


def _cpp_measure_dep_plan() -> DepPlan:
    """cpp's own environment as the record a resolver would have to produce.

    The fixed point of the exercise: fed to `CppPlugin.render_gap`, this plan
    reproduces the gap bytes `render_measure_dockerfile` hardcodes today.

    `apt_packages` and `install_commands` are EMPTY and must stay empty. cmake,
    g++-14, ninja and clang are all BAKED into the per-language base (that is
    the whole point of the base swap the install block's comment describes), and
    these two fields mean "what the gap INSTALLS" -- listing a baked component
    would make the gap emit an `apt-get` line today's bytes do not contain, and
    that line would want network in the measure build.

    `test_invocation` carries the graded command as three argv vectors rather
    than one string: `&&` is a shell metacharacter and `depplan.validate` refuses
    it, correctly -- a plan describes tokens, not a command line.
    """
    plan = DepPlan(
        lang='cpp',
        toolchain_version='14.2.0',
        package_manager='cmake',
        manifest_files=('CMakeLists.txt',),
        apt_packages=(),
        install_commands=(),
        build_flags=(
            ('clang_version', '19.1.7'),
            ('cmake_version', '4.4.2'),
            ('ninja_version', '1.12.1'),
        ),
        test_invocation=(
            ('build', (
                'cmake', '--build', 'Build', '--config', 'Release',
                '--parallel', '4',
            )),
            ('configure', (
                'cmake', '-S', '.', '-B', 'Build', '-G', 'Ninja',
                '-DCMAKE_BUILD_TYPE=Release', '-DRUX_WERROR=ON',
                '-DRUX_BUILD_TESTS=ON',
            )),
            ('test', (
                'ctest', '--test-dir', 'Build', '--output-on-failure',
                '-C', 'Release',
            )),
        ),
        harness=CPP_RUX_HARNESS,
        needs_git_metadata=False,
    )
    validate(plan)
    return canonicalize(plan)


#: cpp's canonical, validated environment plan. Module-level so a test can
#: assert the rendered gap against it without re-deriving the facts it states.
CPP_MEASURE_DEP_PLAN: DepPlan = _cpp_measure_dep_plan()


def _test_command(plan: DepPlan) -> str:
    """`configure && build && test` as the one prose line instruction.md shows.

    Derived from the plan rather than spelled again: the three argv vectors are
    already the authority the two scripts render from, and a second spelling of
    them is how the instruction the model reads starts describing a command the
    grader does not run.
    """
    build, _ = _parallelism(_test_tokens(plan.test_invocation, 'build'))
    steps = (
        _test_tokens(plan.test_invocation, 'configure'),
        build,
        _test_tokens(plan.test_invocation, 'test'),
    )
    return ' && '.join(' '.join(step) for step in steps)


#: The graded build+run command as prose, for instruction.md and the graded set.
#: cpp-Rux's, because it is derived from cpp-Rux's plan -- a resolved plan
#: replaces it through `test_command_from_plan` before an entry is written.
TEST_COMMAND = _test_command(CPP_MEASURE_DEP_PLAN)


class CppPlugin(B.LangPlugin):
    """cmake+ninja+g++-14+doctest, whole-suite equality floor, measured denominator."""

    name: ClassVar[str] = 'cpp'
    toml_family: ClassVar[str] = 'A'
    floor_mode: ClassVar[str] = 'equality'
    parser_backed: ClassVar[bool] = False
    synthesizes_git: ClassVar[bool] = False

    #: See `emit.plan_carve`: whole-suite plugins can declare a set of intact-
    #: tree globs to fingerprint. Empty for rust; ('tests/**', 'Makefile') for
    #: c; the three spellings of a C++ test tree here.
    grader_fingerprint_globs: ClassVar[tuple[str, ...]] = GRADER_FINGERPRINT_GLOBS

    #: The same facts `toolchain_spec().install_block` asserts at build time,
    #: as the capability list `--resolve-env` shows the model. apt is listed
    #: because principle 5 allows apt and ONLY apt, and a model not told so
    #: reaches for curl.
    baked_capabilities: ClassVar[tuple[str, ...]] = BAKED_CAPABILITIES

    #: What `render_gap` reads and `depplan.validate` cannot know about. Stated
    #: to the model in the prompt, enforced by `validate_dep_plan` before a
    #: build; the two read the same tuple.
    required_plan_slots: ClassVar[tuple[str, ...]] = REQUIRED_PLAN_SLOTS

    test_command: ClassVar[str] = TEST_COMMAND

    def test_command_from_plan(self, dep_plan: DepPlan | None) -> str:
        """What the rendered scripts ACTUALLY run, for instruction.md to quote."""
        plan, _harness = self._harness(dep_plan)
        return _test_command(plan)

    # --- axis 1 -----------------------------------------------------------

    def toolchain_spec(self) -> ToolchainSpec:
        """apt+kitware install to close the four gaps described in the module docstring."""
        return ToolchainSpec(
            base_image=BASE_IMAGE,
            install_block=_INSTALL_BLOCK,
            env={
                'CC': 'gcc-14',
                'CXX': 'g++-14',
                'DEBIAN_FRONTEND': 'noninteractive',
            },
            workdir=B.WORKDIR,
        )

    # --- axis 2 -----------------------------------------------------------

    def dep_warm_spec(self) -> DepWarmSpec:
        """No warm stage: dependencies arrive via apt at build time (see toolchain)."""
        return DepWarmSpec(stage_block='', files_needed=(), copy_paths=())

    # --- axes 3-6 ---------------------------------------------------------

    def _harness(self, dep_plan: DepPlan | None) -> tuple[DepPlan, Harness]:
        """The plan the harness comes from, and the harness itself.

        `dep_plan=None` is the pre-resolution path (`--no-resolve-env`, and
        every caller written before the harness was plan-driven). It falls back
        to cpp-Rux's own canonical plan, which is exactly the environment those
        callers were hardcoded against -- so the fallback is not a guess, it is
        the same bytes under a name.
        """
        plan = canonicalize(CPP_MEASURE_DEP_PLAN if dep_plan is None else dep_plan)
        return plan, read_harness(plan)

    def assert_repo_agrees(self, plan: DepPlan, *, registered_tests: int) -> None:
        """`validate_dep_plan` plus what only the INTACT tree can answer."""
        self._assert_registry_agrees(read_harness(plan), int(registered_tests))

    @staticmethod
    def _assert_registry_agrees(harness: Harness, registered_tests: int) -> None:
        """THE UNDER-ENUMERATION GUARD, and the reason `-eq 1` could be relaxed.

        `CTEST_RAN -eq 1` used to be the net that caught a plan describing less
        of the suite than the repo has. Once the count comes from the plan the
        net is gone, and a plan that under-enumerates is the dangerous failure:
        a shrunken denominator is SELF-CONSISTENT at the wrong number, so every
        downstream gate -- measure, the equality floor, RED/GREEN -- passes
        while part of the suite is ungraded and never missed.

        The replacement is an INDEPENDENT signal: `emit._cpp_grader_metadata`
        counts the `add_test()` registrations in the intact tree's own CMake
        files, host-side, without asking the resolver anything. The plan and the
        repository have to agree, and a disagreement REFUSES loudly rather than
        pinning a floor over whichever number happened to be smaller.
        """
        if harness.registered_tests != registered_tests:
            raise B.LangError(
                f'the plan says the build registers {harness.registered_tests} '
                f'ctest test(s), but the intact tree registers '
                f'{registered_tests} (add_test calls counted host-side in its '
                'CMake files). A plan that under-enumerates the suite pins a '
                'floor over only the part it described, and a shrunken '
                'denominator is self-consistent at the wrong number -- every '
                'later gate would pass while the rest of the suite went '
                'ungraded. Restate harness["registered_tests"] (and the '
                'build_artifacts it must match) from the repository'
            )

    def render_test_sh(
        self,
        graded: GradedSet,
        *,
        expected: int | None = None,
        fingerprint: Mapping[str, str] | None = None,
        registered_tests: int | None = None,
        carve_root: str | None = None,
        dep_plan: DepPlan | None = None,
    ) -> str:
        expected = graded.expected if expected is None else int(expected)
        fingerprint = graded.fingerprint_sha256 if fingerprint is None else fingerprint
        if registered_tests is None:
            raise B.LangError(
                'cpp plugin needs registered_tests threaded from the intact tree '
                'at generate time; emit._render_test_sh supplies it (no '
                'repo-magic numbers in the plugin source)'
            )
        plan, harness = self._harness(dep_plan)
        self._assert_registry_agrees(harness, int(registered_tests))

        configure = _test_tokens(plan.test_invocation, 'configure')
        build, parallel = _parallelism(_test_tokens(plan.test_invocation, 'build'))
        ctest = _test_tokens(plan.test_invocation, 'test')
        manager = plan.package_manager
        major = _compiler_major(plan.toolchain_version)
        cc, cxx = f'gcc-{major}', f'g++-{major}'
        binary, bin_name = harness.test_binary, harness.test_binary_name
        stale = harness.stale_paths
        registered = harness.registered_tests
        names = harness.registered_test_names
        plural = '' if registered == 1 else 's'
        be = 'is' if registered == 1 else 'are'
        oracle = _dir_phrase(
            sorted({path.split('/', 1)[0] for path in fingerprint})
        ) or 'The graded test'
        carved_glob = f'{carve_root}/*' if carve_root else 'the carved tree'

        if registered == 1 and harness.project_label and names:
            registry = (
                f'{harness.project_label} registers `{names[0]}`, '
                f'the single {bin_name} binary'
            )
        elif names:
            registry = 'the build registers ' + ', '.join(f'`{n}`' for n in names)
        else:
            registry = f'the build registers {_english(registered)} ctest test{plural}'
        single = f' (the single {bin_name} binary)' if registered == 1 else ''

        presence: list[str] = []
        if harness.harness_files:
            note = harness.harness_files_note
            presence = [
                '',
                '# Belt and braces on top of the fingerprint: the '
                f'{len(harness.harness_files)} named harness files',
                *([f'# ({note}) must all', '# still be present.'] if note
                  else ['# must all still be present.']),
                'for f in \\',
                ' \\\n'.join(f'        {rel}' for rel in harness.harness_files) + '; do',
                '    [ -f "$f" ] || fail "graded harness file missing: $f"',
                'done',
            ]

        cleanup: list[str] = []
        if stale:
            them = 'them' if len(stale) > 1 else 'it'
            cleanup = [
                '',
                '# --- clean previous artifacts ---------------------------------------',
                f'# {_dir_phrase(stale)} {"are" if len(stale) > 1 else "is"} removed '
                'unconditionally. The image ships without',
                f'# {them}, but a stale still-passing {bin_name} binary inside the same',
                '# container would let ctest report green with zero regenerated code.',
                f'# Removing {them} makes a cached PASS structurally impossible.',
                f'rm -rf {" ".join(stale)}',
            ]

        werror = _werror_var(configure)
        strictness = [
            f'# {werror}=ON matches upstream CI exactly (macOS.yml). Regenerated',
            '# code must be clean under -Wall -Wextra -Wpedantic -Wshadow -Werror;',
            f'# the oracle sources compile warning-free under {cxx} with these flags,',
            '# so the gate is known to be satisfiable.',
        ] if werror else []

        return '\n'.join([
            '#!/usr/bin/env bash',
            f'# Harbor verifier -- cpp ({expected} doctest TEST_CASEs, equality floor).',
            '#',
            '# EQUALITY floor. doctest reports per-case pass/fail counts in one',
            '# trailing summary line; a crash mid-suite prints a truncated summary or',
            '# none at all, both of which the grader catches ahead of the floor.',
            f'# The denominator ({expected}) was measured once against the intact tree',
            '# in phase 1 (measure.py) and pinned in graded.lock.json.',
            '#',
            f'# BUILD PARALLELISM: `{manager} --build ... --parallel {parallel}`, '
            'NOT the reference',
            '# unbounded `--parallel`. On a big-core aarch64 host the unbounded form',
            '# spawns one cc1plus per core, each peaking ~1.5 GB on C++26 -O3, and is',
            f'# OOM-killed at any reasonable cgroup memory cap. {parallel} processes '
            '* ~1.5 GB',
            f'# ~ {parallel * 3 // 2} GB sits comfortably under the task.toml-declared '
            '8 GB memory_mb.',
            '#',
            "# Harbor ignores this script's exit code; /logs/verifier/reward.json is",
            '# the single source of truth and is written on EVERY path.',
            '',
            'set -uo pipefail',
            '',
            f'REPO=${{REPO:-{B.WORKDIR}}}',
            B.reward_emitter_block(_LOGS_DEFAULT),
            f'CONFIGURE_LOG="${{VERIFIER_DIR}}/{manager}-configure.log"',
            f'BUILD_LOG="${{VERIFIER_DIR}}/{manager}-build.log"',
            'CTEST_LOG="${VERIFIER_DIR}/ctest.log"',
            'DOCTEST_LOG="${VERIFIER_DIR}/doctest.log"',
            '',
            f'# COMPILED is set to 1.0 only after {manager} --build actually links the',
            f'# {bin_name} binary in THIS run. Kept as a shell variable so the',
            '# fail-closed helper quotes the current value on every early exit --',
            '# a build that failed halfway must not be reported as compiled=1 by an',
            '# earlier optimistic setting.',
            'COMPILED=0.0',
            '',
            B.fail_closed_preamble(expected, compiled='"${COMPILED}"'),
            '',
            'cd "${REPO}" || fail "no ${REPO}"',
            '',
            '# --- integrity guards -----------------------------------------------',
            f'# {oracle} tree is the doctest oracle. The per-file lock below was',
            '# captured host-side from the intact tree at generate time and refuses',
            '# to grade if any pinned file changed. Strictly stronger than the',
            '# reference aggregate lock: a solver cannot swap one test file for',
            '# another of matching aggregate hash.',
            B.fingerprint_gate_block(fingerprint, repo_var='${REPO}'),
            *presence,
            *cleanup,
            '',
            f'export CC={cc} CXX={cxx}',
            f'echo "VERIFIER: $({manager} --version | head -1), '
            f'$({cxx} --version | head -1)'
            ', ninja $(ninja --version)"',
            '',
            '# --- configure ------------------------------------------------------',
            *strictness,
            *_configure_lines(configure, 'CONFIGURE_LOG'),
            'CONFIGURE_STATUS=$?',
            'if [ "${CONFIGURE_STATUS}" -ne 0 ]; then',
            f'    echo "--- tail of {manager}-configure.log ---" >&2',
            '    tail -40 "${CONFIGURE_LOG}" >&2',
            f'    echo "VERIFIER: {manager} configure failed (expected while '
            f'{carved_glob} is'
            ' carved)" >&2',
            '    emit 0.0 0 "${EXPECTED}" 0.0 0.0',
            '    exit 0',
            'fi',
            'echo "VERIFIER: configure OK"',
            '',
            f'# --- build (parallel {parallel}, load-bearing) '.ljust(70, '-'),
            f'{" ".join(build)} \\',
            '      > "${BUILD_LOG}" 2>&1',
            'BUILD_STATUS=$?',
            'if [ "${BUILD_STATUS}" -ne 0 ]; then',
            f'    echo "--- tail of {manager}-build.log ---" >&2',
            '    tail -60 "${BUILD_LOG}" >&2',
            '    echo "VERIFIER: build failed (compiled=0, tests cannot run)" >&2',
            '    emit 0.0 0 "${EXPECTED}" 0.0 0.0',
            '    exit 0',
            'fi',
            '',
            '# The test binary must have been produced by THIS build.',
            f'TEST_BIN={binary}',
            '[ -x "${TEST_BIN}" ] || fail "${TEST_BIN} was not produced by the build"',
            '',
            'COMPILED=1.0',
            'echo "VERIFIER: build OK (compiled=1.0)"',
            '',
            '# --- ctest ----------------------------------------------------------',
            '# ctest exits non-zero as soon as any doctest case fails; under graded',
            '# reward that is the ordinary partial-credit path, NOT an error. What',
            '# ctest IS used for is the structural check that exactly '
            f'{_english(registered)} test{plural} {be}',
            f'# registered ({registry}).',
            f'{" ".join(ctest)} \\',
            '      > "${CTEST_LOG}" 2>&1',
            'CTEST_STATUS=$?',
            'tail -30 "${CTEST_LOG}"',
            '',
            "# CTest wording varies: 3.x prints \"100% tests passed, 0 tests failed",
            "# out of 1\"; 4.x drops the middle clause. Match both and read the",
            '# percentage explicitly rather than inferring success from clause count.',
            "CTEST_SUMMARY=$(grep -E '^[0-9]+% tests passed' \"${CTEST_LOG}\" | tail -1)",
            '[ -n "${CTEST_SUMMARY}" ] \\',
            '    || fail "could not parse the ctest pass summary -- suite did not run"',
            "CTEST_PCT=$(echo \"${CTEST_SUMMARY}\""
            " | sed -n 's/^\\([0-9]\\+\\)% tests passed.*/\\1/p')",
            "CTEST_RAN=$(echo \"${CTEST_SUMMARY}\""
            " | sed -n 's/.*out of \\([0-9]\\+\\).*/\\1/p')",
            '[ -n "${CTEST_RAN}" ] || fail "could not parse the ctest test count"',
            # The registry size comes from the PLAN and is cross-checked against
            # the intact tree host-side (`_assert_registry_agrees`), so this
            # stays a structural assertion about the harness shape rather than
            # a repo-magic number the plugin invented.
            f'[ "${{CTEST_RAN}}" -eq {registered} ] \\',
            f'    || fail "ctest registered ${{CTEST_RAN}} tests, expected exactly '
            f'{registered}{single}"',
            'echo "VERIFIER: ctest exit=${CTEST_STATUS} pct=${CTEST_PCT}%'
            ' registered-tests=${CTEST_RAN}"',
            '',
            '# --- doctest cases --------------------------------------------------',
            '# Run the doctest binary DIRECTLY so the reward depends on counted,',
            '# per-case tallies printed by the harness rather than a summary line.',
            '# Non-zero exit is exactly what a partially passing suite produces, so',
            '# DOCTEST_STATUS gates the BINARY bar but not the fractional reward.',
            '"${TEST_BIN}" > "${DOCTEST_LOG}" 2>&1',
            'DOCTEST_STATUS=$?',
            'tail -5 "${DOCTEST_LOG}"',
            '',
            "SUMMARY=$(grep -m1 '^\\[doctest\\] test cases:' \"${DOCTEST_LOG}\")",
            "ASSERTS=$(grep -m1 '^\\[doctest\\] assertions:' \"${DOCTEST_LOG}\")",
            '',
            '# No summary means the binary crashed or aborted before finishing; no',
            '# trustworthy denominator, so it scores 0 rather than a shrunken fraction.',
            '[ -n "${SUMMARY}" ] \\',
            '    || fail "no doctest test-case summary (exit ${DOCTEST_STATUS})'
            ' -- suite did not run to completion"',
            '',
            "CASES=$(echo \"${SUMMARY}\""
            " | sed -n 's/.*test cases: *\\([0-9]\\+\\).*/\\1/p')",
            "CASES_PASSED=$(echo \"${SUMMARY}\""
            " | sed -n 's/.*| *\\([0-9]\\+\\) passed.*/\\1/p')",
            "CASES_FAILED=$(echo \"${SUMMARY}\""
            " | sed -n 's/.*| *\\([0-9]\\+\\) failed.*/\\1/p')",
            "NASSERT=$(echo \"${ASSERTS}\""
            " | sed -n 's/.*assertions: *\\([0-9]\\+\\).*/\\1/p')",
            "ASSERT_PASSED=$(echo \"${ASSERTS}\""
            " | sed -n 's/.*| *\\([0-9]\\+\\) passed.*/\\1/p')",
            "ASSERT_FAILED=$(echo \"${ASSERTS}\""
            " | sed -n 's/.*| *\\([0-9]\\+\\) failed.*/\\1/p')",
            '',
            'CASES=${CASES:-0}',
            'CASES_PASSED=${CASES_PASSED:-0}',
            'CASES_FAILED=${CASES_FAILED:-0}',
            'NASSERT=${NASSERT:-0}',
            'ASSERT_PASSED=${ASSERT_PASSED:-0}',
            'ASSERT_FAILED=${ASSERT_FAILED:-0}',
            '',
            'echo "VERIFIER: doctest exit=${DOCTEST_STATUS} cases=${CASES}'
            ' passed=${CASES_PASSED} failed=${CASES_FAILED}"',
            'echo "VERIFIER: assertions=${NASSERT} passed=${ASSERT_PASSED}'
            ' failed=${ASSERT_FAILED}"',
            '',
            '# --- structural anti-gaming gates (no repo-magic literals) ----------',
            '# nassert > 0: the doctest suite actually executed assertions. A rebuilt',
            f'# binary with every REQUIRE stripped would print `test cases: {expected}'
            f' | {expected}',
            '# passed` with zero assertions -- the fingerprint gate above catches',
            '# that by hash, and this catches it structurally too.',
            f'# CTEST_PCT == 100: the {"single" if registered == 1 else _english(registered)}'
            f' registered ctest{plural} '
            f'{"was" if registered == 1 else "were"} green (the DOCTEST_STATUS',
            '# gate below in the BINARY reproduces the all-or-nothing semantics).',
            '[ "${NASSERT}" -gt 0 ] \\',
            '    || fail "doctest ran no assertions -- the test suite is empty or a'
            ' compiled-out no-op"',
            '[ "${CTEST_PCT:-0}" -eq 100 ] \\',
            '    || fail "ctest reported ${CTEST_PCT}% (need 100% since exactly '
            f'{registered} test{plural}'
            f' {be} registered)"',
            '[ "${CASES_PASSED}" -le "${EXPECTED}" ] \\',
            '    || fail "nonsensical tally: ${CASES_PASSED} passed out of ${EXPECTED}"',
            '',
            '# CASES is the observed test-case count; EQUALITY floor asserts it equals',
            '# the pinned denominator. A suite that ran fewer than all EXPECTED cases',
            '# (because cases were removed, filtered, or the binary died partway) fails',
            '# closed here rather than scoring a flattering fraction of a shrunken total.',
            'PASSED=${CASES_PASSED}',
            'TOTAL=${CASES}',
            '',
            B.floor_gate_block(
                self.floor_mode, expected,
                passed_var='PASSED', total_var='TOTAL', compiled_var='COMPILED',
            ),
            '',
            '# Extend reward.json with the doctest assertion tallies and failure',
            '# counts so the shape matches the reference grader (verify.py reads the',
            '# 5 canonical keys and ignores extras). BINARY additionally requires',
            '# doctest exit==0 -- a case that aborts a REQUIRE mid-way lowers the',
            '# assertion count without lowering the case count, and only exit==0',
            '# proves the whole binary ran to a natural end.',
            'if [ "${BINARY}" = "1.0" ] && [ "${DOCTEST_STATUS}" -ne 0 ]; then',
            '    BINARY=0.0',
            'fi',
            '',
            "printf '{\"reward\": %s, \"tests_passed\": %s, \"tests_total\": %s, "
            '"binary": %s, "compiled": %s, "assertions_total": %s, '
            '"assertions_passed": %s, "assertions_failed": %s, "cases_failed": %s, '
            "\"ctest_pct\": %s, \"ctest_registered\": %s, \"doctest_exit\": %s}\\n' \\",
            '    "${REWARD}" "${PASSED}" "${EXPECTED}" "${BINARY}" "${COMPILED}" \\',
            '    "${NASSERT}" "${ASSERT_PASSED}" "${ASSERT_FAILED}" "${CASES_FAILED}" \\',
            '    "${CTEST_PCT}" "${CTEST_RAN}" "${DOCTEST_STATUS}" \\',
            '    > "${VERIFIER_DIR}/reward.json"',
            'echo "reward.json (extended) = $(cat "${VERIFIER_DIR}/reward.json")"',
            '',
            'exit 0',
            '',
        ])

    def measure_test_sh(
        self,
        *,
        graded: GradedSet | None = None,
        dep_plan: DepPlan | None = None,
        **kwargs,
    ) -> str:
        """Phase 1: build the intact tree, run the test binary, count doctest cases.

        Floor-FREE by construction. `measure` writes tests_total = the doctest
        `test cases: N` summary; a build that broke halfway registers as 0 and
        `parse_measure_json` rejects zero. Keeps the capped `--parallel` for the
        same OOM reasons documented in render_test_sh (measure runs on the SAME
        host).
        """
        del graded, kwargs
        plan, harness = self._harness(dep_plan)
        configure = _test_tokens(plan.test_invocation, 'configure')
        build, _parallel = _parallelism(_test_tokens(plan.test_invocation, 'build'))
        manager = plan.package_manager
        major = _compiler_major(plan.toolchain_version)
        return '\n'.join([
            '#!/usr/bin/env bash',
            '# Harbor MEASURE (phase 1) -- cpp. Floor-FREE by construction; the pinned',
            '# denominator is what THIS run measures against the intact tree.',
            '',
            'set -uo pipefail',
            '',
            f'REPO=${{REPO:-{B.WORKDIR}}}',
            B.measure_emitter_block(_MEASURE_LOGS_DEFAULT),
            f'CONFIGURE_LOG="${{MEASURE_DIR}}/{manager}-configure.log"',
            f'BUILD_LOG="${{MEASURE_DIR}}/{manager}-build.log"',
            'DOCTEST_LOG="${MEASURE_DIR}/doctest.log"',
            '',
            'cd "${REPO}" || { echo "no ${REPO}" >&2; measure 0 \'\'; exit 0; }',
            '',
            f'export CC=gcc-{major} CXX=g++-{major}',
            '',
            *_configure_lines(configure, 'CONFIGURE_LOG'),
            'if [ $? -ne 0 ]; then',
            '    tail -40 "${CONFIGURE_LOG}" >&2',
            '    echo "measure: intact configure failed" >&2',
            '    measure 0 \'\'',
            '    exit 0',
            'fi',
            '',
            f'{" ".join(build)} \\',
            '      > "${BUILD_LOG}" 2>&1',
            'if [ $? -ne 0 ]; then',
            '    tail -60 "${BUILD_LOG}" >&2',
            '    echo "measure: intact build failed" >&2',
            '    measure 0 \'\'',
            '    exit 0',
            'fi',
            '',
            f'TEST_BIN={harness.test_binary}',
            'if [ ! -x "${TEST_BIN}" ]; then',
            '    echo "measure: no ${TEST_BIN} produced by intact build" >&2',
            '    measure 0 \'\'',
            '    exit 0',
            'fi',
            '',
            '"${TEST_BIN}" > "${DOCTEST_LOG}" 2>&1 || true',
            'tail -5 "${DOCTEST_LOG}"',
            "SUMMARY=$(grep -m1 '^\\[doctest\\] test cases:' \"${DOCTEST_LOG}\")",
            "CASES=$(echo \"${SUMMARY}\""
            " | sed -n 's/.*test cases: *\\([0-9]\\+\\).*/\\1/p')",
            'CASES=${CASES:-0}',
            'echo "MEASURE: doctest cases=${CASES}"',
            '',
            'measure "${CASES}" \'\'',
            'exit 0',
            '',
        ])

    # --- axis 7 -----------------------------------------------------------

    def post_restore_block(self) -> str:
        return self.post_restore_for(None)

    def post_restore_for(self, dep_plan: DepPlan | None) -> str:
        """Invalidate stale build output so GREEN rebuilds honestly.

        Mirrors the reference solve.sh which wipes the build-output directories
        after copying carved/ back. The grader's own `rm -rf` at the top of
        render_test_sh would catch this too, but doing it in the oracle keeps
        the post-restore state identical to what the reference solve.sh
        produces (`diff -r` between the two states should be empty). WHICH
        directories is the plan's answer, not this method's -- `Build`/`Bin`
        are cpp-Rux's names for them, not C++'s.
        """
        _plan, harness = self._harness(dep_plan)
        if not harness.stale_paths:
            return ''
        return 'rm -rf ' + ' '.join(
            f'"${{REPO}}/{path}"' for path in harness.stale_paths
        )

    # --- axis 8 + the image ----------------------------------------------

    def validate_dep_plan(self, plan: DepPlan) -> None:
        """Every precondition `render_gap` has, checked before a container exists.

        The same set of checks `render_gap` makes, hoisted to where they cost
        nothing and can still be repaired: same helpers, same messages, so a
        plan accepted here cannot then fail to render. Enumerated from the
        renderer below -- `toolchain_version`'s major, the manager the pin runs,
        the three version flags and the three argv vectors -- because a slot the
        renderer reads and this gate does not is exactly the crash this exists
        to prevent.
        """
        validate(plan)
        plan = canonicalize(plan)
        if plan.lang != self.name:
            raise B.LangError(
                f'the cpp gap cannot render a {plan.lang!r} plan; a gap is the '
                'one part of a Dockerfile that is language-specific by definition'
            )
        _compiler_major(plan.toolchain_version)
        if plan.package_manager not in _SUPPORTED_MANAGERS:
            raise B.LangError(
                f'the cpp gap has no prose for package_manager '
                f'{plan.package_manager!r}; expected one of '
                f'{", ".join(sorted(_SUPPORTED_MANAGERS))}'
            )
        for key in REQUIRED_BUILD_FLAGS:
            _flag_str(plan.build_flags, key)
        for key in REQUIRED_TEST_INVOCATION_KEYS:
            _test_tokens(plan.test_invocation, key)
        _parallelism(_test_tokens(plan.test_invocation, 'build'))
        read_harness(plan)

    def _gap_body(self, plan: DepPlan) -> str:
        """cpp's toolchain bytes, rendered from a plan instead of a literal.

        The part BOTH the measure and the shipped render take; the measure-only
        no-warm note is `render_gap`'s (see `LangPlugin._gap_body`).

        Same seam as c, one size up. What stays FIXED is the plugin's own
        SCAFFOLDING -- the update-alternatives group that makes the C++26
        compiler the default, the login-shell pin assert, the ENV/WORKDIR
        placement tail -- because those are how harbor resolves a compiler at
        grade time, not facts about this repo. What comes from the PLAN is every
        version those lines assert, the manager they run, and the apt/install
        lines a repo needing system libraries would add.

        The apt and install blocks are unreachable for `CPP_MEASURE_DEP_PLAN`
        (both fields are empty by construction, because the base BAKES the
        toolchain) and are rendered from the plan for the case where a resolved
        plan does declare them. They are the only lines here that are not in
        today's image.

        `CC`/`CXX` are re-derived from the plan's compiler major rather than
        taken from `toolchain_spec()`: they name the very binaries the
        alternatives group installs, so a plan that moved the compiler and left
        those ENVs behind would ship an image whose environment contradicts its
        own alternatives.
        """
        validate(plan)
        plan = canonicalize(plan)
        if plan.lang != self.name:
            raise B.LangError(
                f'the cpp gap cannot render a {plan.lang!r} plan; a gap is the '
                'one part of a Dockerfile that is language-specific by definition'
            )
        if plan.package_manager not in _SUPPORTED_MANAGERS:
            raise B.LangError(
                f'the cpp gap has no prose for package_manager '
                f'{plan.package_manager!r}; expected one of '
                f'{", ".join(sorted(_SUPPORTED_MANAGERS))}'
            )

        major = _compiler_major(plan.toolchain_version)
        cc, cxx = f'gcc-{major}', f'g++-{major}'
        manager = plan.package_manager
        manager_version = _flag_str(plan.build_flags, 'cmake_version')
        lines = [
            f'# The base ships {manager} {manager_version}, {cxx} '
            f'{plan.toolchain_version}, ninja '
            f'{_flag_str(plan.build_flags, "ninja_version")} and clang '
            f'{_flag_str(plan.build_flags, "clang_version")} --',
            '# exactly the set this block used to apt-install on every task build, at the',
            '# cost of two package-index refreshes, a Kitware key import and an unpinned',
            '# package resolution inside a network the grade step forbids.',
            '# Make the C++26-capable compiler the default so a plain `g++`/`cc` picks',
            f'# up {major} rather than the distro default. `cc`/`c++` are already MASTER',
            '# alternatives, so they cannot be attached as slaves of the gcc group.',
            f'RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/{cc} 100 \\',
            f'      --slave /usr/bin/g++ g++ /usr/bin/{cxx} \\',
            f' && update-alternatives --install /usr/bin/cc  cc  /usr/bin/{cc} 100 \\',
            f' && update-alternatives --install /usr/bin/c++ c++ /usr/bin/{cxx} 100',
            '# Assert under a login shell: the base re-exports its own PATH from',
            "# /etc/profile.d, which is how harbor's test.sh resolves these binaries.",
            'RUN set -eux; \\',
            f'    c="$(bash -lc \'{manager} --version | head -1\')"; \\',
            f'    g="$(bash -lc \'{cxx} --version | head -1\')"; \\',
            f'    case "$c" in *{manager_version}*) ;; *) echo "TOOLCHAIN PIN '
            'FAILED (login shell): $c" >&2; exit 42;; esac; \\',
            f'    case "$g" in *{plan.toolchain_version}*) ;; *) echo "TOOLCHAIN '
            'PIN FAILED (login shell): $g" >&2; exit 42;; esac; \\',
            "    bash -lc 'ninja --version && clang --version | head -1'",
        ]
        if plan.apt_packages:
            lines.append(
                'RUN apt-get update && apt-get install -y --no-install-recommends '
                + ' '.join(plan.apt_packages)
                + ' && rm -rf /var/lib/apt/lists/*'
            )
        lines += [
            ' '.join((f'RUN {command.tool}', *command.args)).rstrip()
            for command in plan.install_commands
        ]

        spec = self.toolchain_spec()
        return replace(
            spec,
            install_block='\n'.join(lines),
            env={**spec.env, 'CC': cc, 'CXX': cxx},
        ).render()

    def render_gap(self, plan: DepPlan) -> str:
        """The measure image's gap: the shared body plus the no-warm note.

        The note is the one line the shipped image must NOT carry, because it
        COPYs its dependencies from a warm stage two blocks later.
        """
        return '\n'.join([self._gap_body(plan), '', MEASURE_NO_WARM_COMMENT])

    def render_measure_dockerfile(
        self, env: EnvSpec, *, dep_plan: DepPlan | None = None,
    ) -> str:
        """The stripped Dockerfile for the never-ship measure image.

        Same toolchain (apt+kitware) as the shipped image; the intact tree lands
        directly (no repo/ prefix) because measure.py points repoctx at --repo.
        NO leak gate, NO tripwire scan, NO carve-receipt assert: on the intact
        tree all three would fire by construction (they exist to catch carved
        bytes, and the intact tree IS carved bytes).

        `dep_plan` swaps the hardcoded gap for `render_gap(dep_plan)` in the
        SAME slot and touches nothing else. `dep_plan=None` is what emit.py
        passes and renders the bytes it always did.
        """
        base_image = self.toolchain_spec().base_image
        gap = (
            '\n'.join([self.toolchain(), '', MEASURE_NO_WARM_COMMENT])
            if dep_plan is None
            else self.render_gap(dep_plan)
        )
        return '\n'.join([
            '# syntax=docker/dockerfile:1.7',
            f'# Harbor MEASURE image -- {env.repo_name} ({self.name}). NEVER SHIP.',
            '#',
            '# Built by measure.py phase 1 to count the intact doctest suite. Contains',
            '# the intact tree by construction -- an escaped measure image is not a',
            '# partial leak, it is the whole answer. `measure_image_tag` marks it as',
            '# never-ship and `measure.py` deletes it in a finally block.',
            '',
            f'FROM {base_image} AS measure',
            '',
            gap,
            '',
            f'# The measure phase points {env.repo_context} at the INTACT repo directly,',
            f'# not a staging tree, so there is no repo/ prefix to copy from. That is',
            f'# the only structural difference from the shipped Dockerfile.',
            f'COPY --from={env.repo_context} . {env.workdir}/',
            f'RUN mkdir -p {env.logs_dir}',
            '',
            '# --- measure script (COPYed, not carved-tree, so it lives in a layer) ---',
            'COPY measure.sh /opt/harbor/tests/measure.sh',
            'RUN chmod 0555 /opt/harbor/tests/measure.sh',
            '',
        ])


B.register(CppPlugin())
