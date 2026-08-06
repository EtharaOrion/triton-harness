"""The c plugin: whole-suite grading, measured floor, gcc + make from harbor-base.

c-xs is the second whole-suite language after rust. The carve is the entire
`src/runtime` directory (58 files, ~26.9 kLOC of C11); nothing linkable remains
in that subtree, so per-test selectors are meaningless and the graded surface
collapses to "the whole three-corpus suite".

FLOOR MODE: equality. Each `./xs foo.xs` invocation and each `_test` binary is
its own process, so a panic or abort in one program cannot abort the others
(unlike Go, spike G1). observed==EXPECTED is a real assertion.

TOOLCHAIN. harbor-base ships gcc 13.3.0 + GNU Make 4.3 and the repo's whole
dependency set is `-lm -lpthread -ldl` (all in libc6-dev). No toolchain
upgrade, no vendored tarballs, no network at build or grade time.

GRADER FINGERPRINT. The reference asset writes /opt/harbor/grader.sha256 into
the image at build time and asserts it at grade time; taskgen bakes the same
227-entry lock INLINE via `fingerprint_gate_block` (see the extension in
`gradedset.py` for how a whole-suite plugin declares its lockable files).
The tls-count and per-corpus counts likewise land as bash literals rather
than as image-side files, computed HOST-SIDE at generate time from the intact
tree by `emit._c_grader_metadata`.

DOCKERFILE INVARIANTS. `git init` is NOT synthesised (unlike python); the
base's ban on it stays in force. No vendored assets, no dep_warm stage, and
no strings-target leak assert: unlike rust, the shipped image never compiles
carved sources (the src/runtime tree is gone), so there is no target/ tree in
which mangled symbols could survive.
"""

from __future__ import annotations

import re
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
    'CORPUS_KINDS',
    'CPlugin',
    'C_MEASURE_DEP_PLAN',
    'C_XS_HARNESS',
    'CorpusSpec',
    'GRADER_FINGERPRINT_GLOBS',
    'Harness',
    'MEASURE_NO_WARM_COMMENT',
    'REQUIRED_BUILD_FLAGS',
    'REQUIRED_PLAN_SLOTS',
    'REQUIRED_TEST_INVOCATION_KEYS',
    'TEST_COMMAND',
    'UNIT_NAMES',
    'read_harness',
]

#: The 13 unit harness names from the Makefile / reference test.sh.
UNIT_NAMES: tuple[str, ...] = (
    'lexer', 'parser', 'sema', 'value', 'gc', 'utf8', 'bigint',
    'regex', 'msgpack', 'strbuf', 'limits', 'bytecode_buf', 'self',
)

#: The three hermetic make targets. Deliberately NOT `make test`:
#: tests/run-all.sh sweeps tests/*.xs, which includes test_http_client.xs --
#: that opens real outbound HTTP connections and would break --network=none.
TEST_COMMAND = 'make test-conformance test-regression test-unit'

#: The two ways a graded unit can be decided, and the only two `read_harness`
#: accepts. `runner`: one runner binary is applied to each file a corpus glob
#: matches, and its exit status is the verdict. `binary`: each unit IS an
#: executable the build produces, and running it is the verdict.
CORPUS_KINDS: tuple[str, ...] = ('binary', 'runner')

#: c-xs's own harness as a `DepPlan.harness` section -- the fixed point of the
#: refactor: rendered, it reproduces the scripts this plugin used to hardcode
#: byte for byte. Every value is a fact about the c-xs REPOSITORY, which is
#: exactly why none of it belongs in the plugin.
C_XS_HARNESS: tuple[tuple[str, HarnessValue], ...] = (
    ('binary_env_var', 'XS'),
    ('binary_names', UNIT_NAMES),
    ('binary_suffix', '_test'),
    ('build_artifacts', ('xs',)),
    ('build_command', ('make',)),
    ('build_parallel', True),
    ('clean_command', ('make', 'clean')),
    ('corpus_dirs', ('tests/conformance', 'tests/regression', 'tests/unit')),
    ('corpus_kinds', ('runner', 'runner', 'binary')),
    ('corpus_names', ('conformance', 'regression', 'unit')),
    ('corpus_patterns', ('*.xs', '*.xs', '*_test.c')),
    ('corpus_vars', ('CONF', 'REG', 'UNIT')),
    ('exclude_names', ('test_http_client',)),
    ('runner_argv', ('./xs',)),
    ('stale_paths', ('build/obj',)),
    ('vendored_dirs', ('src/tls',)),
    ('vendored_labels', ('BearSSL',)),
)

#: What the plugin locks against tampering: every file under tests/ and the
#: top-level Makefile. The Makefile is oracle because it enumerates the 58
#: carved files in RUNTIME_SRCS/ASYNC_SRCS and defines the three graded
#: targets; letting a solver rewrite it would let them redefine the task.
GRADER_FINGERPRINT_GLOBS: tuple[str, ...] = ('tests/**', 'Makefile')

#: What harbor-base already provides, as `--resolve-env` states it to a model.
#: Sorted and version-exact, so the same prompt is built on every run.
BAKED_CAPABILITIES: tuple[str, ...] = (
    'GNU Make 4.3',
    'apt-get (build time only; the graded run has no network)',
    'gcc 13.3.0 (aarch64)',
    'libc6-dev, providing -lm -lpthread -ldl',
)

#: The measure image's dep-warm slot. c has no warm stage at all (its whole
#: dependency set is libc), so this comment IS the slot's content -- it is not
#: `DepWarmSpec.render()`, which speaks about a COPY that never happens here.
MEASURE_NO_WARM_COMMENT = '# no warmed dependencies (harbor-base ships gcc + make + libc6-dev)'


def _c_measure_dep_plan() -> DepPlan:
    """c's own environment as the record a resolver would have to produce.

    The fixed point of the exercise: fed to `CPlugin.render_gap`, this plan
    reproduces the gap bytes `render_measure_dockerfile` hardcodes today.

    `apt_packages` is EMPTY and must stay empty. libc6-dev is baked into
    harbor-base, and this field means "packages the gap INSTALLS" -- listing a
    baked package would make the gap emit an `apt-get` line today's bytes do not
    contain, and that line would need network in the measure build.

    `toolchain_version` is the full gcc triple, but the image's pin asserts
    major.minor only: harbor-base may move the patch level, and a pin that
    failed the build on gcc 13.3.1 would be a false alarm, not a caught drift.
    """
    plan = DepPlan(
        lang='c',
        toolchain_version='13.3.0',
        package_manager='make',
        manifest_files=('Makefile',),
        apt_packages=(),
        install_commands=(),
        build_flags=(
            ('link_libraries', '-lm -lpthread -ldl'),
            ('make_version', '4.3'),
        ),
        test_invocation=(('command', tuple(TEST_COMMAND.split())),),
        harness=C_XS_HARNESS,
        needs_git_metadata=False,
    )
    validate(plan)
    return canonicalize(plan)


#: c's canonical, validated environment plan. Module-level so a test can assert
#: the rendered gap against it without re-deriving the facts it states.
C_MEASURE_DEP_PLAN: DepPlan = _c_measure_dep_plan()

_LOGS_DEFAULT = '${VERIFIER_DIR:-' + B.LOGS_DIR + '}'
_MEASURE_LOGS_DEFAULT = '${MEASURE_DIR:-' + B.LOGS_DIR + '}'

#: How the gap's prose names each package manager `depplan` allows for c.
_MANAGER_LABELS: Mapping[str, str] = {'make': 'GNU Make', 'cmake': 'CMake'}

#: Every `build_flags` key `render_gap` interpolates. Enumerated once, read by
#: BOTH the renderer and `CPlugin.validate_dep_plan`, so a flag added to the
#: rendered text without being added here is the only way the two can disagree.
REQUIRED_BUILD_FLAGS: tuple[str, ...] = ('link_libraries', 'make_version')

#: Every `test_invocation` key the c measure path needs. `render_gap` itself
#: does not interpolate one -- the measure script runs the graded targets -- but
#: a plan that does not state the command it was resolved FOR is a plan nobody
#: can check against the image it produced, so it is required at the same gate.
REQUIRED_TEST_INVOCATION_KEYS: tuple[str, ...] = ('command',)

#: The same requirements as the resolver is told them, before it answers. Kept
#: adjacent to the checks above so the ask and the rejection cannot drift apart.
REQUIRED_PLAN_SLOTS: tuple[str, ...] = (
    'build_flags["link_libraries"] must be a non-empty string: the linker flags '
    'the repo builds with, e.g. "-lm -lpthread -ldl"',
    'build_flags["make_version"] must be a non-empty string: the GNU make '
    'version the base image ships, e.g. "4.3"',
    'install_commands must NOT compile the graded sources -- test_invocation '
    'builds and runs the suite itself -- so for a make-driven repo this list '
    'is USUALLY EMPTY. Add a step only to PREPARE the environment (fetch or '
    'vendor a dependency); never a bare "make" or "make all" unless a build '
    'file at the build root declares that default target and the plan needs '
    'its output first. A build step that cannot succeed makes the whole plan '
    'unbuildable, not merely suboptimal',
    'apt_packages lists ONLY system libraries the sources need that the base '
    'image lacks; the toolchain above is already baked in, so naming it here '
    'buys nothing and costs a network fetch',
    'package_manager must be one of: cmake, make',
    'test_invocation["command"] must be a non-empty list of argv tokens: the '
    'command that runs the whole graded suite, e.g. ["make", "test"]',
    'toolchain_version must carry at least major.minor, e.g. "13.3.0": the gcc '
    'pin asserted at build time is derived from its first two components',
    'harness is REQUIRED for c and describes THIS repository\'s own test '
    'harness; it is what the graded verifier script is rendered from. State it '
    'from the build manifests and the TEST FILE PATHS you were given, and from '
    'nothing else. Build keys: harness["build_command"] is the argv that builds '
    'the repo from clean (parallelism is added by the renderer, so never spell '
    '-j yourself); harness["build_parallel"] is true when that build is safe '
    'under -j; harness["clean_command"] is the argv that wipes previous output; '
    'harness["build_artifacts"] lists the repo-relative files the build MUST '
    'produce, and may be [] when it produces no single named binary; '
    'harness["stale_paths"] lists build-output directories that must be gone '
    'once the clean step has run',
    'harness["corpus_names"], ["corpus_vars"], ["corpus_kinds"], '
    '["corpus_dirs"] and ["corpus_patterns"] are five PARALLEL lists of the '
    'SAME length, one entry each per graded corpus, read positionally. '
    'corpus_names are lowercase identifiers (they become reward.json keys and '
    'log filenames); corpus_vars are UPPERCASE shell prefixes; corpus_dirs are '
    'repo-relative directories; corpus_patterns are basename globs; '
    'corpus_kinds is "runner" (ONE runner binary is applied to each file the '
    'glob matches and its exit status is the verdict) or "binary" (each unit '
    'IS its own executable that the build produces from its own source file, '
    'and running it is the verdict). At MOST ONE corpus may be "binary", and '
    'all "runner" corpora must share a single corpus_pattern',
    'ENUMERATE EVERY CORPUS. This is the single most consequential thing you '
    'do here: the corpora you declare ARE the denominator the benchmark scores '
    'against, so a corpus you leave out is not graded, is never missed, and '
    'silently shrinks the task. Before answering, walk the TEST FILE PATHS list '
    'you were given and account for EVERY entry -- each path must either be '
    'matched by exactly one corpus\'s dir + pattern, or be a helper that is not '
    'itself a runnable unit (a shared header, a stub, a fixture, a data file), '
    'or be named in exclude_names. In particular: a directory of per-test C '
    'SOURCE files, each with its own main(), which the build compiles into one '
    'executable per file, IS a graded corpus of kind "binary" -- do not skip it '
    'because its members are sources rather than data. Likewise, if the build '
    'file exposes several test targets, they are usually several corpora, and '
    'an aggregate target that chains them is still several corpora here',
    'harness["runner_argv"] is required whenever a corpus is "runner": the '
    'argv each matched file is passed to as a final argument. '
    'harness["binary_suffix"], ["binary_names"] and ["binary_env_var"] are '
    'required whenever a corpus is "binary": what the executable\'s name adds '
    'to the unit name (a unit "parse" whose program is built as "parse_check" '
    'has suffix "_check"), then EVERY unit name in that corpus in the order '
    'the repository declares them, then the environment variable that must '
    'point at the first build artifact while a unit runs, or "" if none does',
    'harness["exclude_names"] lists basename stems that must NEVER be graded '
    'because they need the network, an external service or a browser -- the '
    'graded run has no network at all, so one of these caught by a corpus glob '
    'turns a sound floor into a flaky one. Empty is fine and is the common case',
)


def _flag_str(build_flags: tuple[tuple[str, FlagValue], ...], key: str) -> str:
    """A build flag the gap's rendered text needs, or a refusal to render at all."""
    for name, value in build_flags:
        if name == key:
            if isinstance(value, str) and value:
                return value
            break
    raise B.LangError(
        f'the c gap needs build_flags[{key!r}]: it must be a non-empty string. '
        'A plan without it would render a Dockerfile comment with a hole in it, '
        'and a hole in the toolchain description is how a base swap goes unnoticed'
    )


def _test_tokens(
    test_invocation: tuple[tuple[str, TestValue], ...], key: str,
) -> tuple[str, ...]:
    """A test_invocation entry the c measure path needs, as argv tokens."""
    for name, value in test_invocation:
        if name == key:
            tokens = (value,) if isinstance(value, str) else tuple(value)
            if tokens and all(token.strip() for token in tokens):
                return tokens
            break
    raise B.LangError(
        f'the c plan needs test_invocation[{key!r}]: it must be a non-empty list '
        'of argv tokens naming the command that runs the graded suite. A plan '
        'that does not state what it was resolved to RUN cannot be checked '
        'against the image it produced'
    )


def _major_minor(toolchain_version: str) -> str:
    """The `major.minor` the image's gcc pin asserts, or a refusal to render.

    A one-component version would widen the pin from `*13.3*` to `*13*`, which
    matches every 13.x the base ever ships -- an assert that cannot fail is not
    an assert, so an under-specified version is rejected instead of narrowed.
    """
    pin = '.'.join(toolchain_version.split('.')[:2])
    if pin.count('.') != 1:
        raise B.LangError(
            f'toolchain_version {toolchain_version!r} has no major.minor to pin '
            'gcc against; a one-component version would make the pin assert '
            'match every 13.x the base image ever ships'
        )
    return pin


@dataclass(frozen=True)
class CorpusSpec:
    """One graded corpus: where its units live and how each one is decided."""

    name: str
    var: str
    kind: str
    directory: str
    pattern: str

    @property
    def noun(self) -> str:
        return 'harnesses' if self.kind == 'binary' else 'programs'


@dataclass(frozen=True)
class Harness:
    """A c test harness as data: build it, discover its units, decide each one.

    Everything the two rendered scripts used to hardcode about the c-xs REPO,
    read off `DepPlan.harness`. `read_harness` is the only constructor, so a
    slot that is missing, mistyped or inconsistent with its siblings becomes a
    `LangError` before any container exists rather than a shell script that
    greps an empty directory and reports a floor of zero.
    """

    build_command: tuple[str, ...]
    build_parallel: bool
    clean_command: tuple[str, ...]
    build_artifacts: tuple[str, ...]
    stale_paths: tuple[str, ...]
    vendored: tuple[tuple[str, str], ...]
    corpora: tuple[CorpusSpec, ...]
    runner_argv: tuple[str, ...]
    binary_suffix: str
    binary_names: tuple[str, ...]
    binary_env_var: str
    exclude_names: tuple[str, ...]

    @property
    def runner_corpora(self) -> tuple[CorpusSpec, ...]:
        return tuple(c for c in self.corpora if c.kind == 'runner')

    @property
    def binary_corpus(self) -> CorpusSpec | None:
        for corpus in self.corpora:
            if corpus.kind == 'binary':
                return corpus
        return None

    @property
    def runner(self) -> str:
        return ' '.join(self.runner_argv)

    @property
    def artifact_paths(self) -> tuple[str, ...]:
        return tuple(f'./{a}' for a in self.build_artifacts)

    @property
    def build_argv(self) -> str:
        parallel = ' -j"$(nproc)"' if self.build_parallel else ''
        return ' '.join(self.build_command) + parallel

    @property
    def unit_build_argv(self) -> str:
        """The unit-harness build: `-k` so one link failure does not hide twelve."""
        tool, *rest = self.build_command
        parallel = [' -j"$(nproc)"'] if self.build_parallel else []
        return ' '.join([tool, *rest, '-k']) + ''.join(parallel)

    def counts(self, corpus_counts: Mapping[str, int]) -> tuple[tuple[CorpusSpec, int], ...]:
        missing = [c.name for c in self.corpora if c.name not in corpus_counts]
        if missing:
            raise B.LangError(
                f'no host-side count for corpus {", ".join(missing)}; emit '
                '._c_grader_metadata must glob every corpus the plan declares'
            )
        return tuple((c, int(corpus_counts[c.name])) for c in self.corpora)


_VAR_RE = re.compile(r'^[A-Z][A-Z0-9_]*$')
_NAME_RE = re.compile(r'^[a-z][a-z0-9_]*$')

_ENGLISH: tuple[str, ...] = (
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
    'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
    'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
)


def _english(n: int) -> str:
    return _ENGLISH[n] if 0 <= n < len(_ENGLISH) else str(n)


def _slot_error(key: str, want: str) -> B.LangError:
    return B.LangError(
        f'the c harness needs harness[{key!r}]: {want}. Without it the rendered '
        'verifier would either grade nothing or grade a suite nobody described, '
        'and a denominator nobody described is not a floor'
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


def _str_slot(values: Mapping[str, HarnessValue], key: str,
              want: str, *, required: bool = False) -> str:
    raw = values.get(key)
    if raw is None or raw == '':
        if required:
            raise _slot_error(key, want)
        return ''
    if not isinstance(raw, str):
        raise _slot_error(key, f'{want} (a single string)')
    return raw


def _bool_slot(values: Mapping[str, HarnessValue], key: str) -> bool:
    raw = values.get(key)
    if raw is None:
        return False
    if not isinstance(raw, bool):
        raise _slot_error(key, 'it must be true or false')
    return raw


def _corpora(values: Mapping[str, HarnessValue]) -> tuple[CorpusSpec, ...]:
    names = _tuple_slot(
        values, 'corpus_names',
        'the ordered list of graded corpora, one entry per test directory or '
        'filename pattern the suite is split across',
        required=True,
    )
    parallel = {
        'corpus_vars': (
            'one UPPERCASE shell-variable prefix per corpus, same order and '
            'same length as corpus_names'
        ),
        'corpus_kinds': (
            f'one of {", ".join(CORPUS_KINDS)} per corpus, same order as '
            'corpus_names: "runner" if a runner binary is applied to each file '
            'the corpus glob matches, "binary" if each unit is itself an '
            'executable the build produces'
        ),
        'corpus_dirs': (
            'the repo-relative directory holding each corpus, same order as '
            'corpus_names'
        ),
        'corpus_patterns': (
            'the filename glob that selects each corpus\'s units inside its '
            'directory, same order as corpus_names'
        ),
    }
    columns = {
        key: _tuple_slot(values, key, want, required=True)
        for key, want in parallel.items()
    }
    for key, column in columns.items():
        if len(column) != len(names):
            raise B.LangError(
                f'harness[{key!r}] has {len(column)} entries but corpus_names '
                f'has {len(names)}; the corpus slots are read POSITIONALLY, so '
                'a short column silently re-pairs every corpus after it with '
                "someone else's directory"
            )
    for name in names:
        if not _NAME_RE.match(name):
            raise B.LangError(
                f'corpus name {name!r} must be lowercase letters, digits and '
                'underscores: it is spelled into reward.json keys, into a log '
                'filename and into a grep pattern'
            )
    for var in columns['corpus_vars']:
        if not _VAR_RE.match(var):
            raise B.LangError(
                f'corpus var {var!r} must be an UPPERCASE shell identifier; it '
                'becomes EXPECT_<var>, <var>_PASS and <var>_RAN in the verifier'
            )
    for kind in columns['corpus_kinds']:
        if kind not in CORPUS_KINDS:
            raise B.LangError(
                f'corpus kind {kind!r} is not one of {", ".join(CORPUS_KINDS)}; '
                'the verifier has exactly two ways to decide a unit and cannot '
                'invent a third'
            )
    for label, column in (('name', names), ('var', columns['corpus_vars'])):
        if len(set(column)) != len(column):
            raise B.LangError(
                f'two corpora share a {label}; each corpus owns its own '
                'counters and its own log, so a duplicate would have one '
                'corpus overwrite the other and shrink the denominator'
            )
    corpora = tuple(
        CorpusSpec(
            name=name,
            var=columns['corpus_vars'][i],
            kind=columns['corpus_kinds'][i],
            directory=columns['corpus_dirs'][i],
            pattern=columns['corpus_patterns'][i],
        )
        for i, name in enumerate(names)
    )
    if len([c for c in corpora if c.kind == 'binary']) > 1:
        raise B.LangError(
            'more than one corpus declares kind "binary", but the harness has a '
            'single binary_suffix and a single binary_names list; split them or '
            'describe them as one corpus'
        )
    patterns = {c.pattern for c in corpora if c.kind == 'runner'}
    if len(patterns) > 1:
        raise B.LangError(
            f'the runner corpora declare {len(patterns)} different '
            f'corpus_patterns ({", ".join(sorted(patterns))}), but they share '
            'one sweep function in the rendered verifier; give them one common '
            'glob or describe the odd one out as its own binary corpus'
        )
    return corpora


def read_harness(plan: DepPlan) -> Harness:
    """`plan.harness` as the record the two renderers read, or a `LangError`.

    Every check here is reachable from the refine loop, so every message names
    the slot and says what a correct value looks like: a resolver that gets one
    wrong must be able to repair it without being shown the repository.
    """
    values = {key: value for key, value in canonicalize(plan).harness}
    corpora = _corpora(values)

    vendored_dirs = _tuple_slot(
        values, 'vendored_dirs',
        'the repo-relative directories holding vendored dependencies that are '
        'NOT part of the carve and must stay intact',
    )
    vendored_labels = _tuple_slot(
        values, 'vendored_labels',
        'one human name per vendored_dirs entry, naming the dependency',
    )
    if len(vendored_labels) != len(vendored_dirs):
        raise B.LangError(
            f'harness["vendored_labels"] has {len(vendored_labels)} entries but '
            f'vendored_dirs has {len(vendored_dirs)}; they are read as parallel '
            'columns and the label names the dependency the guard protects'
        )

    harness = Harness(
        build_command=_tuple_slot(
            values, 'build_command',
            'the argv that builds the repo from clean. Parallelism is added by '
            'the renderer, so do not spell -j here',
            required=True,
        ),
        build_parallel=_bool_slot(values, 'build_parallel'),
        clean_command=_tuple_slot(
            values, 'clean_command',
            'the argv that wipes previous build output',
        ),
        build_artifacts=_tuple_slot(
            values, 'build_artifacts',
            'the repo-relative files the build must produce. Empty is allowed '
            'for a repo whose build produces no single named binary',
        ),
        stale_paths=_tuple_slot(
            values, 'stale_paths',
            'the build-output directories that must be gone after the clean step',
        ),
        vendored=tuple(zip(vendored_dirs, vendored_labels)),
        corpora=corpora,
        runner_argv=_tuple_slot(
            values, 'runner_argv',
            'the argv of the binary each "runner" corpus unit is passed to',
            required=any(c.kind == 'runner' for c in corpora),
        ),
        binary_suffix=_str_slot(
            values, 'binary_suffix',
            'what a "binary" corpus unit\'s executable name adds to its unit '
            'name: a unit "parse" built as "parse_check" has suffix "_check"',
            required=any(c.kind == 'binary' for c in corpora),
        ),
        binary_names=_tuple_slot(
            values, 'binary_names',
            'every unit name in the "binary" corpus, in the order the repo '
            'declares them',
            required=any(c.kind == 'binary' for c in corpora),
        ),
        binary_env_var=_str_slot(
            values, 'binary_env_var',
            'the environment variable a "binary" corpus unit needs pointing at '
            'the first build artifact',
        ),
        exclude_names=_tuple_slot(
            values, 'exclude_names',
            'basename stems that must NEVER appear in a graded corpus: a test '
            'needing the network, an external service or a browser',
        ),
    )
    if harness.binary_env_var and not harness.build_artifacts:
        raise B.LangError(
            f'harness["binary_env_var"] is {harness.binary_env_var!r} but '
            'build_artifacts is empty; the variable is set to the first build '
            'artifact, and there is none to point it at'
        )
    if harness.binary_env_var and not _VAR_RE.match(harness.binary_env_var):
        raise B.LangError(
            f'harness["binary_env_var"] {harness.binary_env_var!r} must be an '
            'UPPERCASE shell identifier; it is exported to each unit binary'
        )
    return harness


class CPlugin(B.LangPlugin):
    """gcc + GNU Make from harbor-base, whole-suite equality floor, measured denominator."""

    name: ClassVar[str] = 'c'
    toml_family: ClassVar[str] = 'A'
    floor_mode: ClassVar[str] = 'equality'
    parser_backed: ClassVar[bool] = False
    synthesizes_git: ClassVar[bool] = False

    #: See `emit.plan_carve`: whole-suite plugins can declare a set of intact-
    #: tree globs to fingerprint. Empty for rust; non-empty here.
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

    # --- axis 1 -----------------------------------------------------------

    def toolchain_spec(self) -> ToolchainSpec:
        """No-op install: harbor-base already ships gcc 13.3.0 and GNU Make 4.3."""
        install = (
            '# harbor-base already ships gcc 13.3.0 and GNU Make 4.3 on aarch64.\n'
            '# The c-xs repo compiles with -lm -lpthread -ldl only (libc6-dev), so\n'
            '# there is nothing to install here; the version echo makes the layer\n'
            '# non-empty and prints the exact toolchain grades will use.\n'
            'RUN gcc --version | head -1 && make --version | head -1\n'
            '# gcc is baked into the base image, so there is no shim to reorder --\n'
            '# but an UNASSERTED compiler is how a base swap moves the published\n'
            '# numbers with nothing failing. Run as a login shell because that is\n'
            "# how harbor's test.sh resolves the compiler at grade time.\n"
            'RUN set -eux; \\\n'
            '    g="$(bash -lc \'gcc --version | head -1\')"; \\\n'
            '    case "$g" in \\\n'
            '      *13.3*) echo "TOOLCHAIN PIN OK (login shell): $g" ;; \\\n'
            '      *) echo "TOOLCHAIN PIN FAILED (login shell): got $g want gcc 13.3" >&2; exit 42 ;; \\\n'
            '    esac'
        )
        return ToolchainSpec(
            base_image='426628337772.dkr.ecr.ap-south-2.amazonaws.com/triton/base-c@sha256:3adb49fa8da50672faa73062b9c27a5f5c5d9e79c5806f17aa4655c70f715497',
            install_block=install,
            env={
                'LC_ALL': 'C.UTF-8',
                'MAKEFLAGS': '-j4',
            },
            workdir=B.WORKDIR,
        )

    # --- axis 2 -----------------------------------------------------------

    def dep_warm_spec(self) -> DepWarmSpec:
        """No warm stage: the repo's whole dependency set is stdlib (-lm -lpthread -ldl)."""
        return DepWarmSpec(stage_block='', files_needed=(), copy_paths=())

    # --- axes 3-6 ---------------------------------------------------------

    def _harness(self, dep_plan: DepPlan | None) -> tuple[DepPlan, Harness]:
        """The plan the harness comes from, and the harness itself.

        `dep_plan=None` is the pre-resolution path (`--no-resolve-env`, and
        every caller written before the harness was plan-driven). It falls back
        to c-xs's own canonical plan, which is exactly the environment those
        callers were hardcoded against -- so the fallback is not a guess, it is
        the same bytes under a name.
        """
        plan = canonicalize(C_MEASURE_DEP_PLAN if dep_plan is None else dep_plan)
        return plan, read_harness(plan)

    def _oracle_phrase(self) -> str:
        """`tests/**`, `Makefile` as the fingerprint comment names them in prose."""
        phrases = []
        for glob in self.grader_fingerprint_globs:
            trimmed = glob.removesuffix('/**').removesuffix('**')
            phrases.append(f'{trimmed}/ tree' if trimmed != glob else glob)
        return ' and the '.join(phrases)

    def render_test_sh(
        self,
        graded: GradedSet,
        *,
        expected: int | None = None,
        fingerprint: Mapping[str, str] | None = None,
        corpus_counts: Mapping[str, int] | None = None,
        vendored_counts: Mapping[str, int] | None = None,
        carve_root: str | None = None,
        dep_plan: DepPlan | None = None,
    ) -> str:
        expected = graded.expected if expected is None else int(expected)
        fingerprint = graded.fingerprint_sha256 if fingerprint is None else fingerprint

        if corpus_counts is None or vendored_counts is None:
            raise B.LangError(
                'c plugin needs corpus_counts and vendored_counts threaded from '
                'the intact tree at generate time; emit._render_test_sh supplies '
                'them (no magic numbers in the plugin source)'
            )
        plan, harness = self._harness(dep_plan)
        counted = harness.counts(corpus_counts)
        total = sum(n for _, n in counted)
        if total != expected:
            raise B.LangError(
                'c plugin sub-counts do not sum to expected: '
                f'{"+".join(str(n) for _, n in counted)} != {expected}'
            )
        if total < 1:
            raise B.LangError(
                'the resolved harness discovers 0 graded units against the '
                'intact tree, so the denominator would be zero and every '
                'solution would score 0.0. Widen corpus_dirs/corpus_patterns to '
                "the repo's real test layout, or refuse the repo outright"
            )
        corpora = harness.corpora
        runners = harness.runner_corpora
        binary = harness.binary_corpus
        counts = {c.name: n for c, n in counted}
        if binary is not None and len(harness.binary_names) != counts[binary.name]:
            raise B.LangError(
                f'harness["binary_names"] has {len(harness.binary_names)} entries '
                f'but the intact {binary.directory}/ holds {counts[binary.name]} '
                f'{binary.pattern} files -- the plan\'s harness list drifted from '
                'the repo'
            )

        artifacts = harness.artifact_paths
        artifact_str = ' '.join(artifacts) or 'the build'
        clean_str = ' '.join(harness.clean_command)
        tool = harness.build_command[0]
        manager = plan.package_manager
        suffix = harness.binary_suffix
        carved = carve_root or 'the carved sources'

        subjects = [
            *([f'Each {harness.runner} invocation'] if runners else []),
            *([f'each {binary.name} {suffix} binary'] if binary is not None else []),
        ]
        note = ' and '.join(subjects)
        blocks: list[list[str]] = [[
            f'# EQUALITY floor. {note[:1].upper()}{note[1:]} is',
            '# its own process, so a crash in one cannot abort the others (unlike',
            '# Go, spike G1). observed==EXPECTED is a real assertion.',
        ]]
        if runners:
            command = _test_tokens(plan.test_invocation, 'command')
            oracle = (plan.manifest_files or (tool,))[0]
            recipes = '/'.join(
                next((t for t in command[1:] if t.endswith(c.name)), c.name)
                for c in runners
            )
            blocks.append([
                f"# The {oracle}'s {recipes} recipes are for-loops",
                '# that `exit 1` on the FIRST failing program -- useless for a fraction.',
                "# This script therefore runs each program with the recipe's own command",
                '# and records per-program exit status. That is a faithful replication,',
                f'# not a re-definition: the {oracle} is inside the fingerprint lock, so',
                '# the recipes cannot drift out from under it.',
            ])
        blocks.append([
            "# Harbor ignores this script's exit code; /logs/verifier/reward.json is",
            '# the single source of truth and is written on EVERY path.',
        ])

        lines: list[str] = [
            '#!/usr/bin/env bash',
            f'# Harbor verifier -- c ({expected} = '
            + ' + '.join(f'{n} {c.name}' for c, n in counted)
            + ', equality floor).',
        ]
        for block in blocks:
            lines.append('#')
            lines += block
        lines += [
            '',
            'set -uo pipefail',
            '',
            f'REPO=${{REPO:-{B.WORKDIR}}}',
            B.reward_emitter_block(_LOGS_DEFAULT),
            'BUILD_LOG="${VERIFIER_DIR}/build.log"',
        ]
        if binary is not None:
            lines.append(
                f'{binary.var}_BUILD_LOG="${{VERIFIER_DIR}}/{binary.name}-build.log"'
            )
        lines += [f'{c.var}_LOG="${{VERIFIER_DIR}}/{c.name}.log"' for c in corpora]
        lines += [
            '',
            f'# COMPILED is set to 1.0 only after {artifact_str} actually relinks in THIS run.',
            '# Kept as a shell variable so the fail-closed helper quotes the current',
            '# value on every early exit -- a build that failed halfway must not be',
            '# reported as compiled=1 by an earlier optimistic setting.',
            'COMPILED=0.0',
            '',
            B.fail_closed_preamble(expected, compiled='"${COMPILED}"'),
            '',
            'cd "${REPO}" || fail "no ${REPO}"',
            '',
            *[f'EXPECT_{c.var}={n}' for c, n in counted],
            f'EXPECT_TOTAL={expected}',
        ]
        if binary is not None:
            lines.append(f'{binary.var}_NAMES="{" ".join(harness.binary_names)}"')
        lines += [
            '',
            f'echo "VERIFIER: $(gcc --version | head -1) / '
            f'$({manager} --version | head -1) / $(uname -m)"',
            '',
            '# --- integrity guards -----------------------------------------------',
            f'# The {self._oracle_phrase()} ARE the oracle. The {len(fingerprint)}-entry lock',
            '# below was captured host-side from the intact tree at generate time',
            '# and refuses to grade if any pinned file changed.',
            B.fingerprint_gate_block(fingerprint, repo_var='${REPO}'),
        ]

        for directory, label in harness.vendored:
            if directory not in vendored_counts:
                raise B.LangError(
                    f'no host-side file count for vendored dir {directory!r}; '
                    'emit._c_grader_metadata must scan every vendored_dirs entry'
                )
            n = int(vendored_counts[directory])
            var = directory.rsplit('/', 1)[-1].upper()
            lines += [
                '',
                f'# Vendored {label} is not part of the carve; a solver that deleted or',
                '# stubbed it could make the link succeed without regenerating anything.',
                '# The pristine count is baked in from the host-side scan at generate time.',
                f'{var}_NOW=$(find {directory} -type f | wc -l | tr -d " ")',
                f'[ "${{{var}_NOW}}" = "{n}" ] \\',
                f'    || fail "{directory} file count changed: found ${{{var}_NOW}}, '
                f'expected {n} (vendored {label} must stay intact)"',
                f'echo "VERIFIER: {directory} intact -- ${{{var}_NOW}} files (pinned at {n})"',
            ]

        if runners:
            lines += [
                '',
                '# Corpus is asserted EXACTLY, not as a lower bound: this IS the denominator.',
                *[
                    f'{c.var}_N=$(ls {c.directory}/{c.pattern} 2>/dev/null '
                    '| wc -l | tr -d " ")'
                    for c in runners
                ],
            ]
            for c in runners:
                lines += [
                    f'[ "${{{c.var}_N}}" = "${{EXPECT_{c.var}}}" ] \\',
                    f'    || fail "{c.name} corpus truncated: ${{{c.var}_N}}, '
                    f'expected ${{EXPECT_{c.var}}}"',
                ]
            globs = ' '.join(f'{c.directory}/{c.pattern}' for c in runners)
            corpus_desc = ' + '.join(f'${{{c.var}_N}} {c.name}' for c in runners)
            excluded = ''
            lines.append('')
            if harness.exclude_names:
                needles = harness.exclude_names
                grep = (
                    f'grep -q "{needles[0]}"' if len(needles) == 1
                    else f'grep -qE "{"|".join(needles)}"'
                )
                ext = runners[0].pattern[1:] if runners[0].pattern.startswith('*') else ''
                lines += [
                    '# The network-dependent harness must not have been dragged '
                    'into either graded glob.',
                    f'if ls {globs} 2>/dev/null | {grep}; then',
                    '    fail "network-dependent '
                    + ', '.join(f'{n}{ext}' for n in needles)
                    + ' leaked into the graded globs"',
                    'fi',
                ]
                excluded = ', ' + ', '.join(
                    n.removeprefix('test_') for n in needles
                ) + ' excluded'
            lines.append(f'echo "VERIFIER: corpus = {corpus_desc}{excluded}"')

        lines += [
            '',
            '# --- genuine rebuild -------------------------------------------------',
        ]
        if harness.clean_command:
            wipes = ' and '.join([*harness.stale_paths, *artifacts]) or 'previous output'
            head = f'# `{clean_str}` wipes {wipes}'
            if binary is not None:
                lines += [
                    head + '; the unit-test binaries under',
                    f'# {binary.directory}/ are removed by hand (clean does not '
                    'touch them). Nothing',
                ]
            else:
                lines.append(head + '. Nothing')
            lines += [
                "# stale from image build time or from the model's own iteration can",
                '# masquerade as a pass.',
                f'{clean_str} > "${{VERIFIER_DIR}}/clean.log" 2>&1 || true',
                *[f'rm -f {a}' for a in artifacts],
            ]
            if binary is not None:
                lines.append(f'rm -f {binary.directory}/*{suffix}')
            lines += [
                f'[ -e {a} ] && fail "{a} survived {clean_str} -- '
                'refusing to trust the build"'
                for a in artifacts
            ]
            lines += [
                f'[ -d {s} ] && fail "{s} survived {clean_str} -- '
                'refusing to trust the build"'
                for s in harness.stale_paths
            ]
        lines += [
            '',
            'echo "VERIFIER: clean tree confirmed; starting full rebuild from source"',
            f'{harness.build_argv} > "${{BUILD_LOG}}" 2>&1',
            'BUILD_STATUS=$?',
            f'echo "VERIFIER: {tool} exit=${{BUILD_STATUS}}"',
            'if [ "${BUILD_STATUS}" -ne 0 ]; then',
            '    echo "--- tail of build.log ---" >&2',
            '    tail -25 "${BUILD_LOG}" >&2',
            f'    echo "VERIFIER: build failed (expected while {carved} is '
            'missing or incomplete)" >&2',
            '    emit 0.0 0 "${EXPECTED}" 0.0 0.0',
            '    exit 0',
            'fi',
            *[
                f'[ -x {a} ] || fail "build reported success but {a} was not produced"'
                for a in artifacts
            ],
            '',
            'COMPILED=1.0',
            *(
                [
                    f'echo "VERIFIER: rebuilt {a} ($(stat -c%s {a}) bytes), compiled=1.0"'
                    for a in artifacts
                ]
                or ['echo "VERIFIER: build succeeded, compiled=1.0"']
            ),
        ]

        if binary is not None:
            n_units = len(harness.binary_names)
            lines.append('')
            if n_units == 1:
                lines.append(
                    f'# Build the one {binary.name} harness; it is scored on its own merits.'
                )
            else:
                lines += [
                    f'# Build the {n_units} {binary.name} harnesses. -k so that '
                    'one failing to link does not',
                    f'# hide the results of the other {_english(n_units - 1)}; '
                    'each is scored on its own merits.',
                ]
            lines += [
                f'{binary.var}_TARGETS=""',
                f'for t in ${{{binary.var}_NAMES}}; do {binary.var}_TARGETS='
                f'"${{{binary.var}_TARGETS}} {binary.directory}/${{t}}{suffix}"; done',
                '# shellcheck disable=SC2086',
                f'{harness.unit_build_argv} ${{{binary.var}_TARGETS}} > '
                f'"${{{binary.var}_BUILD_LOG}}" 2>&1',
                f'echo "VERIFIER: {binary.name} harness build exit=$? '
                '(per-harness results counted below)"',
            ]

        lines += [
            '',
            '# --- graded run: per-program exit status ----------------------------',
        ]
        if runners:
            lines += [
                'run_corpus() {',
                '    local label=$1 dir=$2 log=$3',
                '    local passed=0 ran=0 f',
                '    : > "${log}"',
                f'    for f in "${{dir}}"/{runners[0].pattern}; do',
                '        [ -f "${f}" ] || continue',
                '        ran=$((ran + 1))',
                f'        if {harness.runner} "${{f}}" >/dev/null 2>>"${{log}}"; then',
                '            passed=$((passed + 1))',
                '            echo "[${label}] PASS ${f}" >> "${log}"',
                '        else',
                '            echo "[${label}] FAIL ${f} (exit $?)" >> "${log}"',
                '        fi',
                '    done',
                '    echo "${passed} ${ran}"',
                '}',
                '',
            ]
            for c in runners:
                lines += [
                    f'read -r {c.var}_PASS {c.var}_RAN <<EOF',
                    f'$(run_corpus {c.name} {c.directory} "${{{c.var}_LOG}}")',
                    'EOF',
                ]

        if binary is not None:
            env = (
                f'{harness.binary_env_var}="$(pwd)/{harness.build_artifacts[0]}" '
                if harness.binary_env_var else ''
            )
            lines += [
                '',
                f'{binary.var}_PASS=0',
                f'{binary.var}_RAN=0',
                f': > "${{{binary.var}_LOG}}"',
                f'for t in ${{{binary.var}_NAMES}}; do',
                f'    bin="{binary.directory}/${{t}}{suffix}"',
                f'    {binary.var}_RAN=$(({binary.var}_RAN + 1))',
                '    if [ ! -x "${bin}" ]; then',
                f'        echo "[{binary.name}:${{t}}] FAIL -- harness binary '
                f'missing (did not link)" >> "${{{binary.var}_LOG}}"',
                '        continue',
                '    fi',
                f'    one="${{VERIFIER_DIR}}/{binary.name}-${{t}}.log"',
                f'    {env}"./${{bin}}" > "${{one}}" 2>&1',
                '    rc=$?',
                f'    cat "${{one}}" >> "${{{binary.var}_LOG}}"',
                '    if [ "${rc}" -ne 0 ]; then',
                f'        echo "[{binary.name}:${{t}}] FAIL (exit ${{rc}})" '
                f'>> "${{{binary.var}_LOG}}"',
                f'    elif grep -qiE "\\[{binary.name}:[a-z_]+\\] .*(fail|FAILED)" '
                '"${one}"; then',
                f'        echo "[{binary.name}:${{t}}] FAIL -- reported a failure '
                f'despite exit 0" >> "${{{binary.var}_LOG}}"',
                '    else',
                f'        {binary.var}_PASS=$(({binary.var}_PASS + 1))',
                f'        echo "[{binary.name}:${{t}}] PASS" >> "${{{binary.var}_LOG}}"',
                '    fi',
                'done',
            ]

        lines += [
            '',
            f'PASSED=$(({" + ".join(f"{c.var}_PASS" for c in corpora)}))',
            f'RAN=$(({" + ".join(f"{c.var}_RAN" for c in corpora)}))',
            'TOTAL=${RAN}',
            '',
            'echo "VERIFIER: '
            + ', '.join(
                f'{c.name} ${{{c.var}_PASS}}/${{{c.var}_RAN}}' for c in corpora
            )
            + '"',
            'echo "VERIFIER: graded total ${PASSED}/${RAN}"',
            '',
            '# Structural anti-gaming: exact sweep counts. A sweep that visited fewer',
            '# programs than the corpus scores 0 rather than a fraction of a shrunken total.',
        ]
        for c in corpora:
            lines += [
                f'[ "${{{c.var}_RAN}}" -eq "${{EXPECT_{c.var}}}" ] \\',
                f'    || fail "ran ${{{c.var}_RAN}} {c.name} {c.noun}, '
                f'expected ${{EXPECT_{c.var}}}"',
            ]
        lines += [
            '',
            B.floor_gate_block(
                self.floor_mode, expected,
                passed_var='PASSED', total_var='TOTAL', compiled_var='COMPILED',
            ),
            '',
            '# Extend reward.json with sub-counts so the shape matches the reference',
            '# grader (verify.py reads the 5 canonical keys and ignores extras).',
            'printf \'{"reward": %s, "tests_passed": %s, "tests_total": %s, '
            '"binary": %s, "compiled": %s'
            + ''.join(
                f', "{c.name}_passed": %s, "{c.name}_total": %s' for c in corpora
            )
            + '}\\n\' \\',
            '    "${REWARD}" "${PASSED}" "${EXPECTED}" "${BINARY}" "${COMPILED}" \\',
            '    '
            + ' '.join(
                f'"${{{c.var}_PASS}}" "${{EXPECT_{c.var}}}"' for c in corpora
            )
            + ' \\',
            '    > "${VERIFIER_DIR}/reward.json"',
            'echo "reward.json (extended) = $(cat "${VERIFIER_DIR}/reward.json")"',
            '',
            'exit 0',
            '',
        ]
        return '\n'.join(lines)

    def measure_test_sh(
        self,
        *,
        graded: GradedSet | None = None,
        dep_plan: DepPlan | None = None,
        **kwargs,
    ) -> str:
        """Phase 1: build and sweep every corpus the plan declares, count runs.

        Floor-FREE by construction. `measure` writes tests_total = the count of
        units the sweep actually attempted, so a build that broke halfway -- or
        a harness whose globs match nothing -- registers as fewer-than-expected
        and `parse_measure_json` rejects zero, which sends the refine loop back
        to the resolver instead of shipping a task nobody can score.
        """
        _plan, harness = self._harness(dep_plan)
        corpora = harness.corpora
        runners = harness.runner_corpora
        binary = harness.binary_corpus
        artifacts = harness.artifact_paths

        lines: list[str] = [
            '#!/usr/bin/env bash',
            '# Harbor MEASURE (phase 1) -- c. Floor-FREE by construction; the pinned',
            '# denominator is what THIS run measures against the intact tree.',
            '',
            'set -uo pipefail',
            '',
            f'REPO=${{REPO:-{B.WORKDIR}}}',
            B.measure_emitter_block(_MEASURE_LOGS_DEFAULT),
            'BUILD_LOG="${MEASURE_DIR}/build.log"',
        ]
        if binary is not None:
            lines += [
                f'{binary.var}_BUILD_LOG="${{MEASURE_DIR}}/{binary.name}-build.log"',
                f'{binary.var}_NAMES="{" ".join(harness.binary_names)}"',
            ]
        lines += [
            '',
            'cd "${REPO}" || { echo "no ${REPO}" >&2; measure 0 \'\'; exit 0; }',
            '',
            f'{harness.build_argv} > "${{BUILD_LOG}}" 2>&1',
            'BUILD_STATUS=$?',
            'if [ "${BUILD_STATUS}" -ne 0 ]; then',
            '    tail -20 "${BUILD_LOG}" >&2',
            '    echo "measure: intact build failed (exit ${BUILD_STATUS})" >&2',
            '    measure 0 \'\'',
            '    exit 0',
            'fi',
        ]
        for artifact in artifacts:
            lines += [
                f'if [ ! -x {artifact} ]; then',
                f'    echo "measure: no {artifact} produced by intact build" >&2',
                '    measure 0 \'\'',
                '    exit 0',
                'fi',
            ]
        if binary is not None:
            lines += [
                '',
                f'{binary.var}_TARGETS=""',
                f'for t in ${{{binary.var}_NAMES}}; do {binary.var}_TARGETS='
                f'"${{{binary.var}_TARGETS}} {binary.directory}/${{t}}'
                f'{harness.binary_suffix}"; done',
                '# shellcheck disable=SC2086',
                f'{harness.unit_build_argv} ${{{binary.var}_TARGETS}} > '
                f'"${{{binary.var}_BUILD_LOG}}" 2>&1',
            ]
        if runners:
            lines += [
                '',
                'run_dir() {',
                '    local dir=$1 passed=0 ran=0 f',
                f'    for f in "${{dir}}"/{runners[0].pattern}; do',
                '        [ -f "${f}" ] || continue',
                '        ran=$((ran + 1))',
                f'        {harness.runner} "${{f}}" >/dev/null 2>&1 '
                '&& passed=$((passed + 1))',
                '    done',
                '    echo "${passed} ${ran}"',
                '}',
                '',
            ]
            for c in runners:
                lines += [
                    f'read -r {c.var}_PASS {c.var}_RAN <<EOF',
                    f'$(run_dir {c.directory})',
                    'EOF',
                ]
        if binary is not None:
            env = (
                f'{harness.binary_env_var}="$(pwd)/{harness.build_artifacts[0]}" '
                if harness.binary_env_var else ''
            )
            lines += [
                '',
                f'{binary.var}_PASS=0',
                f'{binary.var}_RAN=0',
                f'for t in ${{{binary.var}_NAMES}}; do',
                f'    bin="{binary.directory}/${{t}}{harness.binary_suffix}"',
                f'    {binary.var}_RAN=$(({binary.var}_RAN + 1))',
                '    if [ -x "${bin}" ]; then',
                f'        {env}"./${{bin}}" >/dev/null 2>&1 \\',
                f'            && {binary.var}_PASS=$(({binary.var}_PASS + 1))',
                '    fi',
                'done',
            ]
        lines += [
            '',
            f'RAN=$(({" + ".join(f"{c.var}_RAN" for c in corpora)}))',
            f'PASSED=$(({" + ".join(f"{c.var}_PASS" for c in corpora)}))',
            'echo "MEASURE: '
            + ', '.join(
                f'{c.var.lower()} ${{{c.var}_PASS}}/${{{c.var}_RAN}}' for c in corpora
            )
            + ' -- total ran ${RAN}"',
            '',
            'measure "${RAN}" \'\'',
            'exit 0',
            '',
        ]
        return '\n'.join(lines)

    # --- axis 7 -----------------------------------------------------------

    def post_restore_block(self) -> str:
        """Nothing: `make clean` in test.sh handles cache invalidation on its own."""
        return ''

    # --- axis 8 + the image ----------------------------------------------

    def validate_dep_plan(self, plan: DepPlan) -> None:
        """Every precondition `render_gap` has, checked before a container exists.

        A real model returned a plan that cleared `depplan.validate` -- correct
        schema, allowlisted manager, no metacharacter -- and still exploded
        inside `render_gap`, because the generic validator is lang-agnostic and
        c's gap interpolates two `build_flags` the schema has never heard of.
        Reached from the refine loop, that crash escaped a handler that only
        knew `MeasureError` and aborted the whole generate.

        This is the same set of checks `render_gap` makes, hoisted to where they
        cost nothing and can still be repaired: same helpers, same messages, so
        a plan accepted here cannot then fail to render.
        """
        validate(plan)
        plan = canonicalize(plan)
        if plan.lang != self.name:
            raise B.LangError(
                f'the c gap cannot render a {plan.lang!r} plan; a gap is the one '
                'part of a Dockerfile that is language-specific by definition'
            )
        _major_minor(plan.toolchain_version)
        if plan.package_manager not in _MANAGER_LABELS:
            raise B.LangError(
                f'the c gap has no prose for package_manager '
                f'{plan.package_manager!r}; expected one of '
                f'{", ".join(sorted(_MANAGER_LABELS))}'
            )
        for key in REQUIRED_BUILD_FLAGS:
            _flag_str(plan.build_flags, key)
        for key in REQUIRED_TEST_INVOCATION_KEYS:
            _test_tokens(plan.test_invocation, key)
        read_harness(plan)

    def _gap_body(self, plan: DepPlan) -> str:
        """c's toolchain + dependency bytes, rendered from a plan instead of a literal.

        c is the smallest possible gap and that is exactly why it is the one to
        prove the seam on: harbor-base already ships gcc and GNU Make, and the
        repo links only against libc, so NOTHING is installed and the whole gap
        is a version echo and a pin assert. If the seam cannot reproduce those
        bytes it cannot reproduce anyone's.

        This is the part BOTH renders take. The measure-only "nothing was
        warmed" note is `render_gap`'s, because the shipped image does warm
        dependencies -- see `LangPlugin._gap_body`.

        The apt and install blocks below are unreachable for
        `C_MEASURE_DEP_PLAN` (both fields are empty by construction) and are
        rendered from the plan for the case where a resolved plan does declare
        them. They are the only lines here that are not in today's image.

        The ENV/WORKDIR tail comes from `toolchain_spec()` rather than being
        respelled: those are placement, not provisioning, and the plan has no
        business moving them.
        """
        validate(plan)
        plan = canonicalize(plan)
        if plan.lang != self.name:
            raise B.LangError(
                f'the c gap cannot render a {plan.lang!r} plan; a gap is the one '
                'part of a Dockerfile that is language-specific by definition'
            )

        pin = _major_minor(plan.toolchain_version)
        manager = plan.package_manager
        lines = [
            f'# harbor-base already ships gcc {plan.toolchain_version} and '
            f'{_MANAGER_LABELS[manager]} {_flag_str(plan.build_flags, "make_version")} '
            'on aarch64.',
            f'# The c-xs repo compiles with '
            f'{_flag_str(plan.build_flags, "link_libraries")} only (libc6-dev), so',
            '# there is nothing to install here; the version echo makes the layer',
            '# non-empty and prints the exact toolchain grades will use.',
            f'RUN gcc --version | head -1 && {manager} --version | head -1',
            '# gcc is baked into the base image, so there is no shim to reorder --',
            '# but an UNASSERTED compiler is how a base swap moves the published',
            '# numbers with nothing failing. Run as a login shell because that is',
            "# how harbor's test.sh resolves the compiler at grade time.",
            'RUN set -eux; \\',
            '    g="$(bash -lc \'gcc --version | head -1\')"; \\',
            '    case "$g" in \\',
            f'      *{pin}*) echo "TOOLCHAIN PIN OK (login shell): $g" ;; \\',
            f'      *) echo "TOOLCHAIN PIN FAILED (login shell): got $g want gcc {pin}"'
            ' >&2; exit 42 ;; \\',
            '    esac',
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

        return replace(
            self.toolchain_spec(), install_block='\n'.join(lines),
        ).render()

    def render_gap(self, plan: DepPlan) -> str:
        """The measure image's gap: the shared body plus the no-warm note.

        The note is the one line the shipped image must NOT carry -- it COPYs
        from a warm stage -- so it lives here rather than in `_gap_body`. The
        manager is re-read off the canonical plan for the same reason the body
        interpolates it: a plan that moved the manager and left this comment
        naming the old one would describe an image it did not build.
        """
        body = self._gap_body(plan)
        manager = canonicalize(plan).package_manager
        return '\n'.join([
            body,
            '',
            f'# no warmed dependencies (harbor-base ships gcc + {manager} + libc6-dev)',
        ])

    def render_measure_dockerfile(
        self, env: EnvSpec, *, dep_plan: DepPlan | None = None,
    ) -> str:
        """The stripped Dockerfile for the never-ship measure image.

        Same toolchain as the shipped image; the intact tree lands directly
        (no repo/ prefix) because measure.py points repoctx at --repo. NO
        leak gate, NO tripwire scan, NO carve-receipt assert: on the intact
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
            '# Built by measure.py phase 1 to count the intact test suite. Contains',
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


B.register(CPlugin())
