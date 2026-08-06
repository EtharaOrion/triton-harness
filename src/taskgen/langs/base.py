"""The language-plugin contract: eight axes, one reward schema, one leak gate.

A `LangPlugin` is everything that differs between python, go, java, rust, c and
cpp when a carved repository is turned into a harbor task. The eight axes of
plan §(a), and where each one lives:

    1  toolchain            `toolchain_spec()` -> `toolchain()` Dockerfile snippet
    2  dep-warm             `dep_warm_spec()`  -> `dep_warm()`  COPY --from=warm
    3  graded build + test  `render_test_sh()`
    4  reward parser        `render_test_sh()`, via `reward_emitter_block()`
    5  fingerprint + floor  `GradedSet.expected/floor_mode/fingerprint_sha256`
    6  cache-defeat         `render_test_sh()` (`-count=1`, `--rerun`, `make clean`)
    7  run-time solution    `render_solve_sh()` + `render_dockerfile()`
    8  task.toml family     `toml_family`

Three of those are load-bearing enough to state as rules rather than as slots:

COMMON REWARD SCHEMA. Every language writes the SAME five keys to
`/logs/verifier/reward.json` -- reward, tests_passed, tests_total, binary,
compiled -- so `verify.py` grades any language without a per-language parser.
`reward.json` is authoritative; `reward.txt` carries the binary bar, which is
harbor's convention (`three_state_gate.sh:48` reads `reward.txt`). Every path
through a rendered `test.sh` writes a reward, starting with a zero before the
suite runs, so a crash can never read as a pass.

FLOOR MODE IS A FIELD. `equality` (observed total == EXPECTED) is harbor's
shipped defence against a shrunken denominator. It is NOT universal: a Go panic
aborts the whole test binary, so only the first selected test emits an event and
an equality floor mis-fires as "refusing partial credit" (spike gotcha G1). Go
therefore uses `pinned-denominator`: reward = passed/EXPECTED, never
passed/observed, with a scope-growth guard (observed <= EXPECTED) and the
test-file sha256 fingerprint as the anti-shrink defence. A plugin declares which
one it is; nothing infers it.

THE ORACLE IS NEVER BAKED. `render_solve_sh()` restores from a RUN-TIME MOUNT at
/opt/harbor/solution, and `render_dockerfile()` refuses to emit a line that
copies it, the intact tree, the carve receipt or the carve tooling into a layer.
`docker save` reads every layer, not just the last, so "the last layer is clean"
is not a security property.

Everything here is pure: no clock, no host paths, no environment lookups. The
rendered strings land in files that `diff -r` compares between two generation
runs, so a dict iteration order would be a bug.
"""

from __future__ import annotations

import abc
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal, Mapping

from ..depplan import DepPlan

__all__ = [
    'DepWarmSpec',
    'EnvSpec',
    'FLOOR_MODES',
    'GradedSet',
    'LANGS',
    'LangError',
    'LangPlugin',
    'MEASURE_KEYS',
    'REWARD_KEYS',
    'RewardCounts',
    'SOLUTION_MOUNT',
    'ToolchainSpec',
    'available',
    'fail_closed_preamble',
    'fingerprint_gate_block',
    'floor_gate_block',
    'get',
    'measure_emitter_block',
    'register',
    'reward_emitter_block',
]

FloorMode = Literal['equality', 'pinned-denominator']
TomlFamily = Literal['A', 'B', 'java-hybrid']

#: Harbor's shipped defence is `equality`; `pinned-denominator` exists for the
#: languages whose runner cannot be trusted to report a total (spike G1).
FLOOR_MODES: tuple[str, ...] = ('equality', 'pinned-denominator')
TOML_FAMILIES: tuple[str, ...] = ('A', 'B', 'java-hybrid')

#: The COMMON schema. verify.py reads exactly these keys, for every language.
REWARD_KEYS: tuple[str, ...] = (
    'reward', 'tests_passed', 'tests_total', 'binary', 'compiled',
)
#: What the floor-FREE measure phase writes instead (measure.py reads these).
MEASURE_KEYS: tuple[str, ...] = ('tests_total', 'graded')

#: Standardised run-time mount. The image never contains this path.
SOLUTION_MOUNT = '/opt/harbor/solution'
WORKDIR = '/opt/harbor/repo'
TESTS_DIR = '/opt/harbor/tests'
LOGS_DIR = '/logs/verifier'

#: BuildKit named contexts. `repoctx` is the tree staging.py carved on the HOST.
REPO_CONTEXT = 'repoctx'
TRIP_CONTEXT = 'trip'
ENTRY_CONTEXT = 'entry'
TOOLING_CONTEXT = 'tooling'
#: A COPY source stage only -- never the shipped base (invariant 7).
WARM_STAGE = 'warm'

#: Languages the plan schedules; `get()` explains rather than raising ImportError.
PLANNED_LANGS: tuple[str, ...] = ('python', 'go', 'java', 'rust', 'c', 'cpp', 'csharp')


class LangError(RuntimeError):
    """A plugin, graded set or rendered asset that must not be shipped."""


# --------------------------------------------------------------------------
# specs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolchainSpec:
    """Axis 1: how the language's compiler/runtime gets into the image."""

    #: No default. A plugin that forgets to name its base would otherwise
    #: silently inherit whichever image the default happened to hold, which is
    #: how a task lands on the wrong toolchain without anything failing.
    base_image: str
    install_block: str = ''
    env: Mapping[str, str] = field(default_factory=dict)
    workdir: str = WORKDIR

    def render(self) -> str:
        lines = []
        if self.install_block:
            lines.append(self.install_block.rstrip('\n'))
        # Sorted, because a dict's insertion order would make the Dockerfile
        # depend on how the plugin happened to spell its literal.
        lines += [f'ENV {k}={v}' for k, v in sorted(self.env.items())]
        lines.append(f'WORKDIR {self.workdir}')
        return '\n'.join(lines)


@dataclass(frozen=True)
class DepWarmSpec:
    """Axis 2: dependencies warmed in a SEPARATE image, consumed by COPY only.

    `files_needed` is the manifest set the warm build is allowed to see -- go.mod,
    Cargo.lock, build.gradle. Never the repo: a warm stage that has the sources
    has the answer, and the shipped image would inherit it.
    """

    stage_block: str = ''
    files_needed: tuple[str, ...] = ()
    copy_paths: tuple[tuple[str, str], ...] = ()
    stage: str = WARM_STAGE
    manifest_dir: str = ''

    def render(self) -> str:
        if not self.copy_paths:
            return '# no warmed dependencies for this language'
        return '\n'.join(
            f'COPY --from={self.stage} {src} {dst}'
            for src, dst in sorted(self.copy_paths)
        )

    def render_stage(self, env: EnvSpec, base_image: str, toolchain: str) -> str:
        """The `FROM ... AS warm` stage the COPY above refers to.

        Without this the COPY names a stage that was never declared, and docker
        falls back to treating `warm` as an IMAGE REFERENCE -- it tries to pull
        `docker.io/library/warm:latest` and the build dies on authorization.

        The manifests are taken from the CARVED context, never from the source
        checkout, and only the declared `files_needed` are copied: a warm stage
        that can see the tree has the answer, and `COPY --from=warm` would
        carry it into the shipped image (invariant 7).
        """
        if not self.copy_paths:
            return ''
        lines = [f'FROM {base_image} AS {self.stage}', '', toolchain, '']
        if self.files_needed:
            sources = ' '.join(f'repo/{name}' for name in sorted(self.files_needed))
            lines += [
                f'RUN mkdir -p {self.manifest_dir}',
                f'COPY --from={env.repo_context} {sources} {self.manifest_dir}/',
                '',
            ]
        lines.append(self.stage_block)
        return '\n'.join(lines).rstrip('\n') + '\n'


@dataclass(frozen=True)
class RewardCounts:
    """Axis 4: the common schema, as data rather than as a printf."""

    reward: float
    tests_passed: int
    tests_total: int
    binary: float
    compiled: float

    def to_dict(self) -> dict:
        return {
            'reward': float(self.reward),
            'tests_passed': int(self.tests_passed),
            'tests_total': int(self.tests_total),
            'binary': float(self.binary),
            'compiled': float(self.compiled),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass(frozen=True)
class GradedSet:
    """Axis 5: the denominator, how it is enforced, and what pins it.

    `expected` is the FLOOR, measured once against the intact tree by
    `measure.py` and then pinned in `graded.lock.json`. It is never re-derived
    at grade time -- that is the circularity the two-phase build exists to
    break.
    """

    expected: int
    floor_mode: str = 'equality'
    fingerprint_sha256: Mapping[str, str] = field(default_factory=dict)
    kind: str = 'whole-suite'
    selectors: tuple[str, ...] = ()
    packages: tuple[str, ...] = ()
    test_command: str = ''

    def __post_init__(self) -> None:
        if not isinstance(self.expected, int) or isinstance(self.expected, bool):
            raise LangError(f'expected must be an int, got {self.expected!r}')
        if self.expected < 1:
            raise LangError(
                f'expected={self.expected}: a denominator of zero is not a floor '
                '(reward = passed/EXPECTED would be undefined, and every carve '
                'would grade green for free)'
            )
        if self.floor_mode not in FLOOR_MODES:
            raise LangError(
                f'unknown floor_mode {self.floor_mode!r}; expected one of '
                f'{", ".join(FLOOR_MODES)}. "ge" is deliberately absent: it lets a '
                'solver delete or skip tests and still clear the floor.'
            )
        object.__setattr__(
            self, 'fingerprint_sha256',
            dict(sorted((str(k), str(v)) for k, v in dict(self.fingerprint_sha256).items())),
        )
        object.__setattr__(self, 'selectors', tuple(self.selectors))
        object.__setattr__(self, 'packages', tuple(self.packages))

    @property
    def fingerprint_relpaths(self) -> tuple[str, ...]:
        return tuple(self.fingerprint_sha256)


@dataclass(frozen=True)
class EnvSpec:
    """Where things live in the image, and which named context supplies them."""

    repo_name: str
    workdir: str = WORKDIR
    tests_dir: str = TESTS_DIR
    logs_dir: str = LOGS_DIR
    repo_context: str = REPO_CONTEXT
    trip_context: str = TRIP_CONTEXT
    entry_context: str = ENTRY_CONTEXT
    tooling_context: str = TOOLING_CONTEXT
    solution_mount: str = SOLUTION_MOUNT
    tripwire_file: str = 'tripwires.txt'
    stage: str = 'graded'


# --------------------------------------------------------------------------
# shared bash: the parts every language's test.sh renders identically
# --------------------------------------------------------------------------


def reward_emitter_block(logs_dir: str = LOGS_DIR) -> str:
    """The COMMON reward writer. Every plugin emits through this, unmodified.

    reward.json is AUTHORITATIVE and carries all five keys; reward.txt carries
    the BINARY bar only, which is what harbor's own gate reads.
    """
    return (
        f'VERIFIER_DIR={logs_dir}\n'
        'mkdir -p "${VERIFIER_DIR}"\n'
        '\n'
        '# emit <reward> <passed> <total> <binary> <compiled>\n'
        'emit() {\n'
        "    printf '{\"reward\": %s, \"tests_passed\": %s, \"tests_total\": %s, "
        '"binary": %s, "compiled": %s}\\n\' \\\n'
        '        "$1" "$2" "$3" "$4" "$5" > "${VERIFIER_DIR}/reward.json"\n'
        '    printf \'%s\\n\' "$4" > "${VERIFIER_DIR}/reward.txt"\n'
        '    echo "reward.json = $(cat "${VERIFIER_DIR}/reward.json")"\n'
        '}'
    )


def fail_closed_preamble(expected: int, compiled: str = '0.0') -> str:
    """A zero on disk BEFORE anything can go wrong, and a `fail` that keeps it."""
    return (
        f'EXPECTED={int(expected)}\n'
        '\n'
        '# Fail closed: the zero lands first, so an early exit, a crashed\n'
        '# toolchain or a killed container can never be read as a pass.\n'
        f'emit 0.0 0 "${{EXPECTED}}" 0.0 {compiled}\n'
        '\n'
        'fail() {\n'
        '    echo "FATAL: $*" >&2\n'
        f'    emit 0.0 0 "${{EXPECTED}}" 0.0 {compiled}\n'
        '    exit 0\n'
        '}'
    )


def fingerprint_gate_block(
    fingerprint_sha256: Mapping[str, str], repo_var: str = '${REPO:-/opt/harbor/repo}'
) -> str:
    """Axis 5's anti-shrink half: a graded test file that changed is a fail.

    The floor pins how MANY tests run; the fingerprint pins WHICH ones. Without
    it, `pinned-denominator` would reward a solver who rewrote the tests.
    """
    items = sorted(dict(fingerprint_sha256).items())
    if not items:
        return '# no fingerprinted files for this graded set'
    lines = [
        '# Fingerprint lock: a graded test file that changed is a shrunken',
        '# denominator by another name.',
        'check_sha256() {',
        f'    _got=$(sha256sum "{repo_var}/$1" 2>/dev/null | cut -d" " -f1)',
        '    [ "${_got}" = "$2" ] || fail "fingerprint mismatch for $1: '
        '${_got:-<missing>} != $2"',
        '}',
    ]
    lines += [f"check_sha256 '{rel}' '{digest}'" for rel, digest in items]
    return '\n'.join(lines)


def floor_gate_block(
    floor_mode: str,
    expected: int,
    *,
    passed_var: str = 'PASSED',
    total_var: str = 'TOTAL',
    compiled_var: str = 'COMPILED',
) -> str:
    """Axis 5's denominator half. `equality` and `pinned-denominator` differ ONLY here."""
    if floor_mode not in FLOOR_MODES:
        raise LangError(f'unknown floor_mode {floor_mode!r}')
    p, t, c = '${%s}' % passed_var, '${%s}' % total_var, '${%s:-1.0}' % compiled_var
    reward = (
        f'REWARD=$(awk -v p="{p}" -v e="${{EXPECTED}}" '
        "'BEGIN{printf \"%.4f\", (e > 0 ? p / e : 0)}')"
    )
    binary = (
        f'if [ "{p}" -eq "${{EXPECTED}}" ]; then BINARY=1.0; else BINARY=0.0; fi'
    )
    emit = f'emit "${{REWARD}}" "{p}" "{t}" "${{BINARY}}" "{c}"'

    if floor_mode == 'equality':
        guard = (
            '# EQUALITY floor: the graded set must be exactly the size it was\n'
            '# measured at. ">=" would let a solver delete or skip tests and still\n'
            '# clear the floor.\n'
            f'[ "{t}" -ne "${{EXPECTED}}" ] && fail "graded set is {t}, pinned at '
            '${EXPECTED} -- the denominator moved"'
        )
    else:
        guard = (
            '# PINNED-DENOMINATOR floor (spike G1): a panic aborts the whole test\n'
            '# binary, so only the first selected test emits an event and the\n'
            '# OBSERVED total is not a denominator. Divide by the pinned EXPECTED,\n'
            '# never by what was observed; the fingerprint lock above is the\n'
            '# anti-shrink defence. What remains to guard is scope-growth.\n'
            f'[ "{t}" -gt "${{EXPECTED}}" ] && fail "scope-growth: observed {t} > '
            'pinned ${EXPECTED} -- the graded set grew under the floor"'
        )
    return '\n'.join([guard, reward, binary, emit])


def measure_emitter_block(logs_dir: str = LOGS_DIR) -> str:
    """Phase 1's writer. Floor-FREE by construction: it counts, it asserts nothing.

    A floor cannot be enforced by the very run that is supposed to measure it,
    which is why this exists as a separate emitter instead of a flag on the
    graded one (plan A2).
    """
    return (
        f'MEASURE_DIR={logs_dir}\n'
        'mkdir -p "${MEASURE_DIR}"\n'
        '\n'
        '# measure <total> <json-array-body>\n'
        'measure() {\n'
        "    printf '{\"tests_total\": %s, \"graded\": [%s]}\\n' \\\n"
        '        "$1" "$2" > "${MEASURE_DIR}/measure.json"\n'
        '    cat "${MEASURE_DIR}/measure.json"\n'
        '}'
    )


# --------------------------------------------------------------------------
# the plugin
# --------------------------------------------------------------------------


class LangPlugin(abc.ABC):
    """One language's answers to the eight axes.

    A Wave 2 plugin implements four methods -- `toolchain_spec`,
    `dep_warm_spec`, `render_test_sh`, `measure_test_sh` -- and declares three
    class attributes. `render_solve_sh` and `render_dockerfile` are inherited
    because their content is a SECURITY invariant, not a language detail: a
    plugin that could spell them freely could spell them wrong.
    """

    name: ClassVar[str] = ''
    toml_family: ClassVar[str] = ''
    floor_mode: ClassVar[str] = ''
    parser_backed: ClassVar[bool] = False
    solution_mount: ClassVar[str] = SOLUTION_MOUNT

    #: Opt-in to synthesising VCS metadata over the ALREADY-CARVED tree, for a
    #: build system that refuses to resolve a version without it (python's
    #: uv-dynamic-versioning is the case in hand). Declaring it is necessary but
    #: not sufficient: `_assert_dockerfile_invariants` still demands the
    #: compensating audit of plan (e).8, because `git add -A` over a tree that
    #: is not provably carved is leak route C.
    synthesizes_git: ClassVar[bool] = False

    #: What the BASE IMAGE already ships, phrased for `--resolve-env`'s prompt.
    #: Declared, not probed from a container: the plugin already had to know it
    #: to write `toolchain_spec().install_block`, and a second derivation is how
    #: the prompt and the image start disagreeing about what is installed.
    baked_capabilities: ClassVar[tuple[str, ...]] = ()

    def __init__(self) -> None:
        if not self.name:
            raise LangError(f'{type(self).__name__}: name is required')
        if self.toml_family not in TOML_FAMILIES:
            raise LangError(
                f'{type(self).__name__}: unknown toml_family {self.toml_family!r}; '
                f'expected one of {", ".join(TOML_FAMILIES)}'
            )
        if self.floor_mode not in FLOOR_MODES:
            raise LangError(
                f'{type(self).__name__}: unknown floor_mode {self.floor_mode!r}; '
                f'expected one of {", ".join(FLOOR_MODES)}'
            )

    # --- axes 1-2 ---------------------------------------------------------

    @abc.abstractmethod
    def toolchain_spec(self) -> ToolchainSpec:
        """Axis 1, as data."""

    @abc.abstractmethod
    def dep_warm_spec(self) -> DepWarmSpec:
        """Axis 2, as data. Warmed from manifest files only, never from the repo."""

    def toolchain(self) -> str:
        """Axis 1 as a Dockerfile snippet."""
        return self.toolchain_spec().render()

    def dep_warm(self) -> str:
        """Axis 2 as a Dockerfile snippet: COPY --from=warm, never FROM warm."""
        return self.dep_warm_spec().render()

    # --- axes 3-6 ---------------------------------------------------------

    @abc.abstractmethod
    def render_test_sh(
        self,
        graded: GradedSet,
        *,
        expected: int | None = None,
        fingerprint: Mapping[str, str] | None = None,
    ) -> str:
        """The graded verifier: build, run, parse, emit the COMMON schema.

        `expected` and `fingerprint` override the graded set when they are
        being read out of `graded.lock.json` (phase 2 of the two-phase build).
        """

    @abc.abstractmethod
    def measure_test_sh(self, **kwargs) -> str:
        """The phase-1 verifier: counts the suite, asserts NO floor."""

    # --- axis 7 -----------------------------------------------------------

    def post_restore_block(self) -> str:
        """Hook for languages that must invalidate something after a restore."""
        return ''

    def render_solve_sh(self, carved_relpaths) -> str:
        """The oracle, restoring from the RUN-TIME MOUNT -- never from a layer."""
        rels = tuple(sorted({Path(r).as_posix() for r in carved_relpaths}))
        if not rels:
            raise LangError('refusing to render an oracle that restores nothing')
        restores = '\n'.join(f"restore '{rel}'" for rel in rels)
        post = self.post_restore_block()
        post_block = f'\n{post}\n' if post else ''
        return f"""#!/usr/bin/env bash
# Oracle -- restores the carved files of a {self.name} task.
#
# The oracle is NEVER baked into a layer. harbor gives the agent a shell, and
# `docker save` reads EVERY layer rather than just the last one, so "the final
# filesystem is clean" is not a security property. solution/ reaches the
# container only for the seconds the gate needs it, mounted read-only at
# {self.solution_mount}.
#
# The restore is a copy of the INTACT original, not a patch, so the
# post-condition is byte equality rather than "it looks about right".
set -uo pipefail

REPO=${{HARBOR_REPO:-{WORKDIR}}}
SOLUTION=${{HARBOR_SOLUTION:-{self.solution_mount}}}
CARVED=${{SOLUTION}}/carved
WANT={len(rels)}
RESTORED=0

[ -d "${{CARVED}}" ] || {{
    echo "FATAL: no oracle at ${{CARVED}} -- it is mounted at run time, never baked" >&2
    exit 1
}}

restore() {{
    src="${{CARVED}}/$1"
    dst="${{REPO}}/$1"
    [ -f "${{src}}" ] || {{ echo "FATAL: the oracle is missing $1" >&2; exit 1; }}
    mkdir -p "$(dirname "${{dst}}")"
    cp -a "${{src}}" "${{dst}}"
    RESTORED=$((RESTORED + 1))
    echo "restored $1"
}}

{restores}
{post_block}
[ "${{RESTORED}}" -eq "${{WANT}}" ] || {{
    echo "FATAL: restored ${{RESTORED}} file(s), promised ${{WANT}}" >&2
    exit 1
}}
echo "SOLVE OK (${{RESTORED}} file(s) restored from ${{SOLUTION}})"
"""

    # --- axis 8 + the image ----------------------------------------------

    def pre_leakgate_blocks(self, env: EnvSpec) -> tuple[str, ...]:
        """Blocks between the carved-tree COPY and the leak gate.

        The slot exists because two requirements pull in opposite directions:
        dependency resolution has to run AFTER the repo lands (python's uv
        resolves against the project's own pyproject, and its editable install
        must point into the carved tree), while the leak scan has to run AFTER
        that resolution -- `.git/objects` and a built virtualenv are exactly the
        places a stray answer would come to rest. `extra_dockerfile_blocks`
        runs after the gate and is therefore the wrong slot for anything the
        gate ought to be inspecting.
        """
        return ()

    def extra_dockerfile_blocks(self, env: EnvSpec) -> tuple[str, ...]:
        """Per-language additions -- e.g. the compiled-artifact scan of §(e).5."""
        return ()

    def extra_ctx_assets(self) -> tuple[tuple[Path, str], ...]:
        """Static plugin assets staged beside leakscan.sh in the tooling context.

        Returns (host_absolute_path, dest_filename) pairs. `emit._copy_leakscan`
        copies each into the entry's tooling/ directory so the shipped
        Dockerfile can bind-mount them from the `tooling` named context. The
        default is (): most languages need only harbor's own leakscan.sh.

        Rust uses this for the wabt tarball: wast2json is a hard runtime
        dependency of the graded harness and is not on any distro repo at the
        pinned version.
        """
        return ()

    def render_gap(self, plan: DepPlan) -> str:
        """The language GAP: the toolchain/dependency provisioning bytes, from a plan.

        A rendered Dockerfile is two things welded together. Most of it is FIXED
        SCAFFOLDING that every language shares and that no resolver may touch --
        the COPY of the tree, the leak gate, the carve-receipt assert, the floor.
        The rest is the GAP: how this language's compiler, its system packages
        and its dependencies get into the image. Only the gap varies per repo,
        and only the gap is a candidate for being RESOLVED rather than written by
        hand -- which is why it is a method taking a `DepPlan` instead of a
        free-form string a resolver could hand back.

        The contract is deliberately narrow: `plan` in, Dockerfile INSTRUCTIONS
        out, no trailing newline, and nothing else. The plan has already passed
        `depplan.validate`, so no argument here can carry a shell metacharacter
        and no tool outside the per-language allowlist can be named.

        The default raises `NotImplementedError` rather than returning '': a
        plugin whose gap is genuinely empty must say so by emitting the fixed
        lines it emits today, because silently rendering nothing would turn a
        missing implementation into an image with no toolchain in it.
        """
        raise NotImplementedError(
            f'the {self.name!r} plugin does not render its gap from a DepPlan; '
            'its toolchain and dependency bytes are still hardcoded in '
            'render_measure_dockerfile. Implement render_gap (and prove it '
            'byte-identical to those bytes) before threading a plan through'
        )

    def render_measure_dockerfile(
        self, env: EnvSpec, *, dep_plan: DepPlan | None = None,
    ) -> str:
        """The stripped Dockerfile for the never-ship measure image (phase 1).

        Only a plugin with `parser_backed = False` needs one: parser-backed
        languages derive their denominator from the parser, not from a run
        against the intact tree, so the two-phase build does not apply. This
        default raises for that reason.

        A whole-suite plugin renders a MINIMAL Dockerfile here: toolchain,
        intact COPY, whatever is needed to compile the tests, and a COPY of
        measure.sh -- NO leak gate, NO strings-assert, NO tripwire scan (all
        three would fire on the intact tree by construction).

        `dep_plan` is the seam for a RESOLVED gap, and it is opt-in for a
        reason. `dep_plan is None` -- the only thing `emit.py` ever passes --
        renders exactly the bytes it rendered before this parameter existed, so
        the shipped path cannot move because a resolver was wired up somewhere
        else. When a plan IS passed, the plugin substitutes `render_gap(plan)`
        for its hardcoded gap and leaves every line of fixed scaffolding alone;
        for a plan describing the environment the plugin already hardcodes, the
        two must come out byte-identical (`tests/test_render_gap.py`), which is
        what makes the substitution reviewable rather than a leap of faith.
        """
        raise LangError(
            f'{self.name!r} does not render a measure dockerfile; whole-suite '
            'languages must implement render_measure_dockerfile so measure.py '
            'can build an intact-tree image to count the suite against '
            '(parser_backed languages derive the denominator without one)'
        )

    def render_dockerfile(self, env: EnvSpec) -> str:
        """The shipped image: the staged carved tree, and nothing that describes it."""
        blocks = [
            '# syntax=docker/dockerfile:1.7',
            f'# Harbor task image -- {env.repo_name} ({self.name})',
            '#',
            f'# The `{env.repo_context}` build context is the tree staging.py carved on the',
            '# HOST. The intact tree never enters ANY build context, so no layer holds',
            '# the answer and `docker save` has nothing to recover (plan A1).',
            '',
            self.dep_warm_spec().render_stage(
                env, self.toolchain_spec().base_image, self.toolchain(),
            ) or None,
            f'FROM {self.toolchain_spec().base_image} AS {env.stage}',
            '',
            self.toolchain(),
            '',
            '# Dependencies arrive by COPY from a SEPARATE warm image. `FROM warm` would',
            '# inherit every layer that build happened to touch (invariant 7).',
            self.dep_warm(),
            '',
            f'COPY --from={env.repo_context} repo/ {env.workdir}',
            f'COPY --from={env.entry_context} tests/ {env.tests_dir}/',
            f'RUN mkdir -p {env.logs_dir}',
            '',
            *self.pre_leakgate_blocks(env),
            '',
            '# Content leak gate: harbor shared/tooling/leakscan.sh, unmodified, over',
            '# the whole filesystem. The tripwires arrive on a BuildKit bind mount, so',
            '# they leave no layer of their own.',
            f'RUN --mount=type=bind,from={env.trip_context},'
            f'source={env.tripwire_file},target=/tmp/.harbor-tripwires \\',
            f'    --mount=type=bind,from={env.tooling_context},'
            'source=leakscan.sh,target=/usr/local/bin/leakscan.sh \\',
            f'    TRIPWIRE_FILE=/tmp/.harbor-tripwires LEAKSCAN_LABEL={env.repo_name} \\',
            '    bash /usr/local/bin/leakscan.sh',
            '',
            '# Nothing that DESCRIBES the carve may ship either: the receipt names every',
            '# removed path, the manifest names the include globs, and the oracle is the',
            '# answer outright (invariant 6).',
            'RUN test "$(find / -name carve_receipt.json -not -path \'/proc/*\' | wc -l)" = "0" \\',
            ' && test ! -e /opt/harbor-tooling/carve.py \\',
            f' && test ! -e {self.solution_mount}',
        ]
        blocks += ['', *self.extra_dockerfile_blocks(env)]
        text = '\n'.join(b for b in blocks if b is not None).rstrip('\n') + '\n'
        _assert_dockerfile_invariants(
            text, self.solution_mount, synthesizes_git=self.synthesizes_git,
        )
        return text


#: What plan (e).8 demands of an image that synthesises git over the carved
#: tree. `--reflog` is the load-bearing flag: plain `--all` walks refs only and
#: would miss an object kept alive by a reflog entry alone, and `fsck` then
#: proves nothing is hiding outside both.
GIT_AUDIT_MARKERS: tuple[str, ...] = (
    'git rev-list --objects --all --reflog',
    'git rev-list --count --all',
    'git fsck',
)


def _assert_dockerfile_invariants(
    text: str, solution_mount: str, *, synthesizes_git: bool = False,
) -> None:
    """Refuse to hand back a Dockerfile that undoes the leak fix.

    A plugin supplies free-form snippets, so the composed result is checked
    rather than trusted: these are the failure modes that shipped once already.

    The token scan reads INSTRUCTIONS, not comments. A comment builds nothing,
    and the comments here necessarily quote the very anti-patterns they warn
    about -- scanning them would make documenting the rule a violation of it.
    The solution-mount check keeps the whole text in scope: naming that path
    anywhere in a Dockerfile is how it got baked in the first time.
    """
    instructions = '\n'.join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith('#')
    )
    banned = [
        ('repos-src', 'copies the pristine source tree into a layer'),
        ('repo-src', 'copies the pristine source tree into a layer'),
        (f'FROM {WARM_STAGE}', 'inherits every layer of the dep-warm image'),
    ]
    if not synthesizes_git:
        banned.append(
            ('git init', 'puts the carved history in .git/objects (leak route C)')
        )
    for token, why in banned:
        if token in instructions:
            raise LangError(f'rendered Dockerfile {why}: contains {token!r}')

    if synthesizes_git and 'git init' in instructions:
        missing = [m for m in GIT_AUDIT_MARKERS if m not in instructions]
        if missing:
            raise LangError(
                'rendered Dockerfile synthesises git but ships no audit proving the '
                f'repository carries version metadata only; missing: {missing}. '
                'Without it `git checkout -- <carve_root>` is a one-command solve '
                'even without `docker save` (plan (e).8)'
            )
    allowed = f'test ! -e {solution_mount}'
    if text.replace(allowed, '').count(solution_mount):
        raise LangError(
            f'rendered Dockerfile references {solution_mount} somewhere other than the '
            'absence assertion -- the oracle must reach the container by run-time mount'
        )


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

#: Populated by each plugin module at import time; empty until Wave 2 lands.
LANGS: dict[str, LangPlugin] = {}


def register(plugin, *, overwrite: bool = False) -> LangPlugin:
    if not isinstance(plugin, LangPlugin):
        raise LangError(
            f'{plugin!r} is not a LangPlugin; a plugin must subclass it so the '
            'solve.sh and Dockerfile invariants are inherited rather than retyped'
        )
    if plugin.name in LANGS and not overwrite:
        raise LangError(
            f'{plugin.name!r} is already registered ({type(LANGS[plugin.name]).__name__}); '
            'pass overwrite=True if that is deliberate'
        )
    LANGS[plugin.name] = plugin
    return plugin


def available() -> tuple[str, ...]:
    return tuple(sorted(LANGS))


def get(name: str) -> LangPlugin:
    """The plugin for `name`, importing its module on first use."""
    if name in LANGS:
        return LANGS[name]
    if name in PLANNED_LANGS:
        try:
            importlib.import_module(f'{__package__}.{name}')
        except ImportError as exc:
            raise LangError(
                f'the {name!r} plugin lands in Wave 2 of the plan (T6-T12); Wave 1 '
                f'ships only the contract in {__name__}. Registered so far: '
                f'{", ".join(available()) or "(none)"}'
            ) from exc
        if name in LANGS:
            return LANGS[name]
        raise LangError(f'{name!r} module imported but registered no plugin')
    raise LangError(
        f'unknown language {name!r}; registered: {", ".join(available()) or "(none)"}; '
        f'planned: {", ".join(PLANNED_LANGS)}'
    )
