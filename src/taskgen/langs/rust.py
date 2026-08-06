"""The rust plugin: whole-suite grading, measured floor, vendored offline build.

Rust is the first plugin whose graded set is NOT derived by a tree-sitter
parser. python and go link individual tests to a carved function; rust here
deletes the WHOLE `src/` tree of an integration crate, so nothing linkable
remains and the "which tests are graded" question collapses to "the whole
integration suite". The denominator therefore has to be MEASURED against the
intact tree once (measure.py, phase 1) and PINNED in `graded.lock.json` so the
shipped test.sh grades against a number that is not derivable from the carved
image (which by construction cannot see it). The plan's two-phase build exists
for exactly this shape.

FLOOR MODE: equality. Rust's libtest reports one `test result:` line per test
binary and does not abort a whole binary on a panic the way Go does (spike G1),
so `observed == EXPECTED` is a real assertion rather than a mis-fire.

TOOLCHAIN: rustc 1.87.0 via rustup. harbor-base ships an older rustc (1.79 per
the reference Dockerfile); the crate under test declares edition 2024 /
rust-version 1.87. wabt (`wast2json`) 1.0.41 is the second unpinned dependency:
tests/util/spectest.rs shells out to it for every .wast file and always passes
`--enable-custom-page-sizes`, which Ubuntu's 1.0.34 rejects. Both come in via
the pre-leakgate blocks so the leak scan sees the warmed target/ directory.

VENDORING via empty-lib-stub. `cargo vendor` resolves dependencies from
Cargo.toml + Cargo.lock; it does NOT need `src/` to contain any real code. A
one-byte `src/lib.rs` therefore reproduces the intact tree's vendor set exactly
(same 12 crates measured in the spike) while keeping the carved sources out of
every layer. The stub warms `target/deps/*` for the vendored crates so the
graded run does not rebuild serde from scratch; the spacewasm-specific
fingerprints and incremental artifacts are then explicitly deleted.

STRINGS-TARGET LEAK ASSERT. rustc leaves mangled symbol names and debug-info
paths in target/*.o. `strings target/... | grep -E '<crate>|<workdir>/src'`
would recover the carved API without touching src/ at all, so the assert makes
that fail-fast at build time rather than a nasty surprise after `docker save`.

DOCKERFILE INVARIANTS. `git init` is NOT synthesised (unlike python), so the
base's ban on it stays in force. The empty-lib-stub uses `mkdir src; :>src/lib.rs`
which contains neither `repos-src`/`repo-src` nor `git init` -- verify the
composed Dockerfile against `_assert_dockerfile_invariants` in the test file.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import ClassVar, Mapping

from ..depplan import DepPlan, FlagValue, TestValue, canonicalize, validate
from . import base as B
from .base import DepWarmSpec, EnvSpec, GradedSet, ToolchainSpec

__all__ = [
    'BAKED_CAPABILITIES',
    'EXPECTED_HARNESSES',
    'HARNESSES',
    'INTACT_VENDOR_BLOCK',
    'INTEGRITY_HARNESS_FILES',
    'MEASURE_NO_WARM_COMMENT',
    'MIN_WAST',
    'BASE_IMAGE',
    'REQUIRED_BUILD_FLAGS',
    'REQUIRED_PLAN_SLOTS',
    'REQUIRED_TEST_INVOCATION_KEYS',
    'RUST_MEASURE_DEP_PLAN',
    'RUST_VERSION',
    'RustPlugin',
    'SHIMS_PATH',
    'TEST_COMMAND',
    'TOOLCHAIN_SOURCE',
    'WABT_VERSION',
]

RUST_VERSION = '1.87.0'

#: Where the base image's mise keeps the shims that resolve the selected
#: toolchain. The zz- profile.d file prepends exactly this, so a base that moved
#: its shims would leave the pin asserting a rustc nothing on PATH points at.
SHIMS_PATH = '/opt/mise/shims'

#: The CDN a `rustup toolchain install` would reach, named in the comment that
#: explains why this image does not: the toolchain is baked, so no task build
#: pays a download it cannot repeat under `--network=none`.
TOOLCHAIN_SOURCE = 'static.rust-lang.org'

#: Per-language base carrying rustc/cargo and wasm2wat already, so no task build
#: reaches static.rust-lang.org for a compiler the image can simply hold.
BASE_IMAGE = '426628337772.dkr.ecr.ap-south-2.amazonaws.com/triton/base-rust@sha256:9bb61cf220efe0c7c8236841c3cb985aa2ce744c414e1fbeaab858832daae911'
WABT_VERSION = '1.0.41'

#: The four graded integration harnesses. They survive the src/ carve because
#: every symbol they touch is either a compilable test-side helper (in
#: tests/util/) or a .wast file lifted from the official spec conformance
#: corpus. Deleting src/ makes them fail to link; restoring src/ makes them
#: pass again. That is the whole point of the task.
HARNESSES: tuple[str, ...] = (
    'core_integration',
    'regression_integration',
    'statistics_integration',
    'custom_page_sizes_integration',
)

#: Files whose presence the graded verifier refuses to score without. The
#: harness .rs files ARE the oracle (backed by ~92 .wast files under tests/);
#: a solver who deleted or rewrote them to make the suite trivially pass would
#: score 0, not a fraction of a shrunken corpus.
INTEGRITY_HARNESS_FILES: tuple[str, ...] = (
    'tests/util/mod.rs',
    'tests/util/spectest.rs',
    'tests/util/inspector.rs',
) + tuple(f'tests/{h}.rs' for h in HARNESSES)

#: Floor on the .wast conformance corpus, mirrored from the reference test.sh.
#: The intact tree has ~92; this refuses to grade a truncated set.
MIN_WAST = 90

#: The graded cargo invocation, in one place so the plugin's render_test_sh,
#: measure_test_sh and _render_grading all say the same thing. `--skip memory_max`
#: excludes ONE case in custom_page_sizes_integration that declares two ~4 GiB
#: linear memories and is killed by the cgroup OOM killer at 7 GiB; grading it
#: would make the reward a function of host RAM.
TEST_COMMAND = (
    'cargo test --offline '
    + ' '.join(f'--test {h}' for h in HARNESSES)
    + ' -- --skip memory_max'
)

_LOGS_DEFAULT = '${VERIFIER_DIR:-' + B.LOGS_DIR + '}'
_MEASURE_LOGS_DEFAULT = '${MEASURE_DIR:-' + B.LOGS_DIR + '}'

#: The measure image's dep-warm slot. rust has no warm stage at all (see
#: `dep_warm_spec`: `cargo vendor` needs a resolvable [lib] target, so vendoring
#: happens inside the graded image), so this comment IS the slot's content.
MEASURE_NO_WARM_COMMENT = '# no warmed dependencies (rust vendors in the graded image)'

#: The measure image's vendor step, verbatim and FIXED. It is scaffolding, not
#: provisioning a resolver may vary: it exists because the measure image holds
#: the INTACT tree, where the shipped image's empty-lib-stub would half-carve
#: src/ and break every compile. It also cannot live in `render_gap`'s string --
#: it has to run AFTER the repo COPY, since `cargo vendor` reads Cargo.toml,
#: and the gap slot sits above that COPY.
INTACT_VENDOR_BLOCK = '\n'.join([
    '# Vendor against the INTACT src/. No empty-lib-stub here: overwriting',
    '# src/lib.rs while the other 30 .rs files remain would half-carve the',
    '# tree and every test would fail to compile. The intact src/ is the',
    '# real one, so cargo vendor resolves and downloads normally.',
    'RUN set -eux; \\',
    '    CARGO_NET_OFFLINE=false cargo vendor > /tmp/vendor-config.toml; \\',
    '    mkdir -p .cargo; \\',
    '    cp /tmp/vendor-config.toml .cargo/config.toml; \\',
    '    rm -f /tmp/vendor-config.toml',
])

#: What the base image already provides, as `--resolve-env` states it to a
#: model. Sorted and version-exact, so the same prompt is built on every run.
#: Declared, never probed: the same facts the pin assert in `render_gap` checks
#: at build time.
BAKED_CAPABILITIES: tuple[str, ...] = (
    'apt-get (build time only; the graded run has no network)',
    f'cargo {RUST_VERSION}, with CARGO_NET_OFFLINE=true already exported',
    'mise, whose shims resolve the selected toolchain under a login shell',
    f'rustc {RUST_VERSION} (edition 2024 capable)',
)

#: Every `build_flags` key `render_gap` interpolates. Enumerated once, read by
#: BOTH the renderer and `RustPlugin.validate_dep_plan`, so a flag added to the
#: rendered text without being added here is the only way the two can disagree.
REQUIRED_BUILD_FLAGS: tuple[str, ...] = ('shims_path', 'toolchain_source')

#: Every `test_invocation` key the rust measure path needs. `render_gap` does
#: not interpolate it -- the measure script runs the graded command itself --
#: but a plan that does not state what it was resolved FOR is a plan nobody can
#: check against the image it produced, so it is required at the same gate.
REQUIRED_TEST_INVOCATION_KEYS: tuple[str, ...] = ('command',)

#: The same requirements as the resolver is told them, before it answers. Kept
#: adjacent to the checks above so the ask and the rejection cannot drift apart.
#: Leak-safe: slot names, shapes and toolchain versions only -- never a repo
#: path, a crate name or a source body.
REQUIRED_PLAN_SLOTS: tuple[str, ...] = (
    'build_flags["shims_path"] must be a non-empty string: the directory the '
    f'base image keeps its mise shims in, e.g. "{SHIMS_PATH}". The gap writes a '
    'zz- profile.d file prepending it to PATH, which is the ONLY reason a login '
    'shell resolves the pinned toolchain rather than the base default',
    'build_flags["toolchain_source"] must be a non-empty string: the CDN a '
    f'toolchain install would otherwise reach, e.g. "{TOOLCHAIN_SOURCE}". The '
    'gap names it in the comment explaining why this image downloads nothing',
    'install_commands must NOT build or test the graded sources -- '
    'test_invocation runs the suite itself, and the image vendors its crates in '
    'a later fixed step -- so for a cargo repo this list is USUALLY EMPTY. Add a '
    'step only to PREPARE the environment; never a bare "cargo build". A build '
    'step that cannot succeed makes the whole plan unbuildable, not merely '
    'suboptimal',
    'apt_packages lists ONLY system libraries the sources need that the base '
    'image lacks; rustc, cargo and mise are already baked in, so naming them '
    'here buys nothing and costs a network fetch',
    'package_manager must be cargo: the gap pins the toolchain cargo drives and '
    'the measure image vendors its dependencies with it',
    'test_invocation["command"] must be a non-empty list of argv tokens: the '
    'command that runs the whole graded suite, e.g. ["cargo", "test", '
    '"--offline"]',
    'toolchain_version must carry at least major.minor.patch, e.g. "1.87.0": it '
    'is the rustc version, and the build-time login-shell pin matches it '
    'verbatim against `rustc --version`',
)

#: The one package manager rust's gap has prose for. `depplan.PACKAGE_MANAGERS`
#: already closes rust to exactly this, so the check is a belt on a brace -- but
#: it is the check that would fire first if that enum ever widened.
_SUPPORTED_MANAGERS: frozenset[str] = frozenset({'cargo'})


def _flag_str(build_flags: tuple[tuple[str, FlagValue], ...], key: str) -> str:
    """A build flag the gap's rendered text needs, or a refusal to render at all."""
    for name, value in build_flags:
        if name == key:
            if isinstance(value, str) and value:
                return value
            break
    raise B.LangError(
        f'the rust gap needs build_flags[{key!r}]: it must be a non-empty '
        'string. A plan without it would render a Dockerfile pin with a hole in '
        'it, and a hole in a toolchain pin is how a base swap goes unnoticed'
    )


def _test_tokens(
    test_invocation: tuple[tuple[str, TestValue], ...], key: str,
) -> tuple[str, ...]:
    """A test_invocation entry the rust measure path needs, as argv tokens."""
    for name, value in test_invocation:
        if name == key:
            tokens = (value,) if isinstance(value, str) else tuple(value)
            if tokens and all(token.strip() for token in tokens):
                return tokens
            break
    raise B.LangError(
        f'the rust plan needs test_invocation[{key!r}]: it must be a non-empty '
        'list of argv tokens naming the command that runs the graded suite. A '
        'plan that does not state what it was resolved to RUN cannot be checked '
        'against the image it produced'
    )


def _rustc_version(toolchain_version: str) -> str:
    """The rustc version the login-shell pin matches verbatim, or a refusal.

    The pin is `case "$v" in *"<version>"*)`, so a version that is not the exact
    string `rustc --version` prints either matches nothing -- turning every
    build into an exit 42 -- or, for something as short as a bare major, matches
    far too much and passes on a toolchain nobody asked for.
    """
    parts = toolchain_version.split('.')
    if len(parts) < 3 or not all(part.isdigit() for part in parts[:2]):
        raise B.LangError(
            f'toolchain_version {toolchain_version!r} must carry at least '
            'major.minor.patch, e.g. "1.87.0"; the rust gap matches it verbatim '
            'against what `rustc --version` prints under a login shell, and a '
            'looser pin either never matches or matches a toolchain nobody asked '
            'for'
        )
    return toolchain_version


def _rust_measure_dep_plan() -> DepPlan:
    """rust's own environment as the record a resolver would have to produce.

    The fixed point of the exercise: fed to `RustPlugin.render_gap`, this plan
    reproduces the gap bytes `render_measure_dockerfile` hardcodes today.

    `apt_packages` and `install_commands` are EMPTY and must stay empty. rustc,
    cargo and mise are all BAKED into the per-language base -- that is exactly
    why the install block SELECTS a toolchain instead of reaching
    `static.rust-lang.org` -- and these two fields mean "what the gap INSTALLS".
    Listing a baked component would make the gap emit an `apt-get` line today's
    bytes do not contain, and that line would want network in the measure build.

    What is NOT described here: the wabt install, the workspace prune, the
    vendor step and the strings leak-assert. Those are fixed scaffolding (and
    for the shipped image, the leak proof), they run BELOW the repo COPY rather
    than in the gap slot, and they stay hardcoded.
    """
    plan = DepPlan(
        lang='rust',
        toolchain_version=RUST_VERSION,
        package_manager='cargo',
        manifest_files=('Cargo.lock', 'Cargo.toml'),
        apt_packages=(),
        install_commands=(),
        build_flags=(
            ('shims_path', SHIMS_PATH),
            ('toolchain_source', TOOLCHAIN_SOURCE),
        ),
        test_invocation=(('command', tuple(TEST_COMMAND.split())),),
        needs_git_metadata=False,
    )
    validate(plan)
    return canonicalize(plan)


#: rust's canonical, validated environment plan. Module-level so a test can
#: assert the rendered gap against it without re-deriving the facts it states.
RUST_MEASURE_DEP_PLAN: DepPlan = _rust_measure_dep_plan()


class RustPlugin(B.LangPlugin):
    """rustc 1.87 + wabt 1.0.41, whole-suite equality floor, measured denominator."""

    name: ClassVar[str] = 'rust'
    toml_family: ClassVar[str] = 'A'
    floor_mode: ClassVar[str] = 'equality'
    parser_backed: ClassVar[bool] = False
    synthesizes_git: ClassVar[bool] = False

    #: The same facts `toolchain_spec().install_block` asserts at build time,
    #: as the capability list `--resolve-env` shows the model. apt is listed
    #: because principle 5 allows apt and ONLY apt, and a model not told so
    #: reaches for curl.
    baked_capabilities: ClassVar[tuple[str, ...]] = BAKED_CAPABILITIES

    #: What `render_gap` reads and `depplan.validate` cannot know about. Stated
    #: to the model in the prompt, enforced by `validate_dep_plan` before a
    #: build; the two read the same tuple.
    required_plan_slots: ClassVar[tuple[str, ...]] = REQUIRED_PLAN_SLOTS

    rust_version: ClassVar[str] = RUST_VERSION
    wabt_version: ClassVar[str] = WABT_VERSION
    test_command: ClassVar[str] = TEST_COMMAND

    # --- axis 1 -----------------------------------------------------------

    def toolchain_spec(self) -> ToolchainSpec:
        """Select the rustc the base bakes, and prove it under a login shell.

        harbor-base ships rustc 1.79 per the reference Dockerfile, but the
        crate under test declares edition 2024 / rust-version 1.87. The install
        is `rustup toolchain install ... --profile minimal`, which reaches
        static.rust-lang.org -- allowed at BUILD time (harbor-base blocks
        github.com but not the rust CDN), forbidden at grade time (network=none
        in task.toml).

        `CARGO_NET_OFFLINE=true` is deliberately NOT set here: harbor-base
        already exports it, and the vendor RUN below overrides it locally for
        the one step that needs the network. Baking it as ENV would silently
        make it un-overridable and break the vendor step.
        """
        install = (
            f'# rustc {self.rust_version} is baked into the base; select it rather\n'
            f'# than reaching {TOOLCHAIN_SOURCE} on every task build.\n'
            '# profile.d is sourced in sorted order and the base re-exports its own\n'
            '# PATH ahead of the mise shims, so only a zz- file keeps the pin in front\n'
            "# for the login shells harbor's test.sh and solve.sh run as.\n"
            'RUN set -eux; \\\n'
            f'    printf \'export PATH="{SHIMS_PATH}:$PATH"\\n\' > /etc/profile.d/zz-harbor-toolchain-pin.sh; \\\n'
            '    chmod 0644 /etc/profile.d/zz-harbor-toolchain-pin.sh\n'
            'RUN set -eux; \\\n'
            '    v="$(bash -lc \'rustc --version\')"; \\\n'
            '    case "$v" in \\\n'
            f'        *"{self.rust_version}"*) echo "TOOLCHAIN PIN OK (login shell): $v" ;; \\\n'
            f'        *) echo "TOOLCHAIN PIN FAILED (login shell): got $v want {self.rust_version}" >&2; exit 42 ;; \\\n'
            '    esac'
        )
        return ToolchainSpec(
            base_image=BASE_IMAGE,
            install_block=install,
            env={
                'CARGO_TERM_COLOR': 'never',
                'RUST_BACKTRACE': '1',
            },
            workdir=B.WORKDIR,
        )

    # --- axis 2 -----------------------------------------------------------

    def dep_warm_spec(self) -> DepWarmSpec:
        """No warm stage: vendoring runs in the graded image (see pre_leakgate).

        Rust cannot warm dependencies the way go does. `cargo vendor` needs a
        resolvable `[lib]` target in Cargo.toml -- which for spacewasm means
        SOME `src/lib.rs` on disk. A warm stage that carried a real one would
        have the answer (invariant 7); an empty-stub one carried alongside the
        real repo would still need the carved Cargo.toml in the warm image.
        The pattern that works is: vendor in the graded image, WITH the carved
        tree already in place, using an empty `src/lib.rs` stub that lives for
        the duration of that one RUN and is then deleted.
        """
        return DepWarmSpec(
            stage_block='',
            files_needed=(),
            copy_paths=(),
        )

    # --- axes 3-6 ---------------------------------------------------------

    def render_test_sh(
        self,
        graded: GradedSet,
        *,
        expected: int | None = None,
        fingerprint: Mapping[str, str] | None = None,
    ) -> str:
        expected = graded.expected if expected is None else int(expected)
        fingerprint = graded.fingerprint_sha256 if fingerprint is None else fingerprint
        test_cmd = graded.test_command or TEST_COMMAND

        harness_check = '\n'.join(
            [
                'for f in \\',
                ' \\\n'.join(f'        {rel}' for rel in INTEGRITY_HARNESS_FILES) + '; do',
                '    [ -f "$f" ] || fail "graded harness file missing: $f"',
                'done',
            ]
        )

        return '\n'.join([
            '#!/usr/bin/env bash',
            f'# Harbor verifier -- rust, {expected} pinned test(s) across '
            f'{len(HARNESSES)} harnesses.',
            '#',
            '# EQUALITY floor. Rust libtest reports one `test result:` line per test',
            '# binary and does NOT abort the whole binary on a panic (that is a Go',
            '# hazard, spike G1), so observed==EXPECTED is a real assertion. The',
            f'# denominator ({expected}) was measured once against the intact tree in',
            '# phase 1 (measure.py) and pinned in graded.lock.json.',
            '#',
            "# Harbor ignores this script's exit code; /logs/verifier/reward.json is",
            '# the single source of truth and is written on EVERY path.',
            '',
            'set -uo pipefail',
            '',
            f'REPO=${{REPO:-{B.WORKDIR}}}',
            B.reward_emitter_block(_LOGS_DEFAULT),
            'CARGO_LOG="${VERIFIER_DIR}/cargo-test.log"',
            '',
            B.fail_closed_preamble(expected, compiled='0.0'),
            '',
            'cd "${REPO}" || fail "no ${REPO}"',
            '',
            '# Integrity guards. The 7 harness files ARE the oracle (backed by the',
            '# .wast conformance corpus); refusing to score if any of them was deleted',
            '# or rewritten stops a solver from making the suite trivially pass.',
            harness_check,
            '',
            'WAST_COUNT=$(find tests -name \'*.wast\' | wc -l | tr -d \' \')',
            f'[ "${{WAST_COUNT}}" -ge {MIN_WAST} ] \\',
            f'    || fail "wast corpus truncated: found ${{WAST_COUNT}}, expected >={MIN_WAST}"',
            '',
            'command -v wast2json >/dev/null 2>&1 || fail "wast2json missing from image"',
            '',
            B.fingerprint_gate_block(fingerprint, repo_var='${REPO}'),
            '',
            f'echo "== cargo test: {expected} pinned test(s) across '
            f'{len(HARNESSES)} harnesses (offline) =="',
            'echo "rustc $(rustc --version), wast2json $(wast2json --version), '
            '${WAST_COUNT} .wast files"',
            '',
            # --offline defeats the network; -count/rerun is not a rust concept.
            # A cached PASS still recompiles on `cargo test` because src/ was
            # replaced by the oracle, invalidating the incremental cache.
            f'{test_cmd} > "${{CARGO_LOG}}" 2>&1',
            'STATUS=$?',
            'tail -40 "${CARGO_LOG}"',
            '',
            '# Count the "test result:" lines. Each harness prints exactly one; a',
            '# missing summary means that harness failed to LINK, which is different',
            '# from "linked but tests failed" and the reward schema keeps them apart.',
            'SUMMARIES=$(grep -c \'^test result:\' "${CARGO_LOG}" || true)',
            'SUMMARIES=${SUMMARIES:-0}',
            '',
            'if [ "${SUMMARIES}" -eq 0 ]; then',
            '    echo "no \'test result:\' lines -- crate did not build (compiled=0)"',
            '    emit 0.0 0 "${EXPECTED}" 0.0 0.0',
            '    exit 0',
            'fi',
            '',
            '# Every harness that printed a summary linked successfully.',
            'COMPILED=1.0',
            '',
            '# libtest layout: `<n> passed; <n> failed; <n> ignored; ...`. The count',
            '# for each keyword sits in the field immediately BEFORE its label, so',
            '# awk peeks at $(i+1) and sums $i across every "test result:" line.',
            'sum_field() {',
            '    awk -v want="$1" \'',
            '        /^test result:/ {',
            '            for (i = 1; i < NF; i++)',
            '                if ($(i+1) == want) s += $i',
            '        }',
            '        END { printf "%d", s + 0 }\' "${CARGO_LOG}"',
            '}',
            'PASSED=$(sum_field \'passed;\')',
            'FAILED=$(sum_field \'failed;\')',
            'IGNORED=$(sum_field \'ignored;\')',
            'PASSED=${PASSED:-0}',
            'FAILED=${FAILED:-0}',
            'IGNORED=${IGNORED:-0}',
            'TOTAL=$((PASSED + FAILED))',
            '',
            'echo "cargo exit=${STATUS} harnesses=${SUMMARIES}/'
            + str(len(HARNESSES))
            + ' passed=${PASSED} failed=${FAILED} ignored=${IGNORED} '
            'total=${TOTAL}/${EXPECTED} compiled=${COMPILED}"',
            '',
            '# Structural anti-gaming gates (equality floor is the third).',
            f'[ "${{SUMMARIES}}" -eq {len(HARNESSES)} ] \\',
            f'    || fail "only ${{SUMMARIES}}/{len(HARNESSES)} harnesses produced a '
            'summary -- refusing partial credit"',
            '[ "${IGNORED}" -eq 0 ] \\',
            '    || fail "${IGNORED} tests were #[ignore]d -- refusing to grade a '
            'suite that opted out"',
            '',
            B.floor_gate_block(
                self.floor_mode, expected,
                passed_var='PASSED', total_var='TOTAL', compiled_var='COMPILED',
            ),
            'exit 0',
            '',
        ])

    def measure_test_sh(self, *, graded: GradedSet | None = None, **kwargs) -> str:
        """Phase 1: run the WHOLE integration suite and count `passed + failed`.

        Floor-FREE by construction: it counts, it asserts nothing. A floor
        cannot be enforced by the run that is supposed to discover it. Unlike
        python (`--collect-only`) and go (`-list`), rust's libtest has no
        "select + count without running" mode, so this actually runs the tests
        -- against the intact tree, where they must all pass.
        """
        test_cmd = (graded.test_command if graded and graded.test_command else TEST_COMMAND)
        return '\n'.join([
            '#!/usr/bin/env bash',
            '# Harbor MEASURE (phase 1) -- rust. Floor-FREE by construction; the',
            '# pinned denominator is what THIS run measures.',
            '',
            'set -uo pipefail',
            '',
            f'REPO=${{REPO:-{B.WORKDIR}}}',
            B.measure_emitter_block(_MEASURE_LOGS_DEFAULT),
            'MEASURE_LOG="${MEASURE_DIR}/cargo-measure.log"',
            '',
            'cd "${REPO}" || { echo "no ${REPO}" >&2; exit 1; }',
            '',
            f'{test_cmd} > "${{MEASURE_LOG}}" 2>&1',
            'STATUS=$?',
            'tail -40 "${MEASURE_LOG}"',
            'echo "cargo exit=${STATUS}"',
            '',
            'sum_field() {',
            '    awk -v want="$1" \'',
            '        /^test result:/ {',
            '            for (i = 1; i < NF; i++)',
            '                if ($(i+1) == want) s += $i',
            '        }',
            '        END { printf "%d", s + 0 }\' "${MEASURE_LOG}"',
            '}',
            'PASSED=$(sum_field \'passed;\')',
            'FAILED=$(sum_field \'failed;\')',
            'PASSED=${PASSED:-0}',
            'FAILED=${FAILED:-0}',
            'TOTAL=$((PASSED + FAILED))',
            '',
            'measure "${TOTAL}" \'\'',
            'exit 0',
            '',
        ])

    # --- axis 7 -----------------------------------------------------------

    def post_restore_block(self) -> str:
        """Nothing: the oracle restore into src/ invalidates cargo's incremental.

        cargo detects src/ changes by mtime and re-builds -- the incremental
        cache under target/debug/incremental gets stale-marked automatically.
        No `cargo clean` needed; the test.sh doesn't ship one either (`--offline`
        does not defeat caching, but the src/ mtime change does).
        """
        return ''

    # --- axis 8 + the image ----------------------------------------------

    def extra_ctx_assets(self) -> tuple[tuple[Path, str], ...]:
        """Nothing. The base carries wabt, so the tooling context holds only
        leakscan.sh. The vendored tarball it used to stage was named for one
        architecture, which pinned every rust task image to arm64.
        """
        return ()

    def pre_leakgate_blocks(self, env: EnvSpec) -> tuple[str, ...]:
        """The four RUN blocks that turn a bare rustc image into a graded one.

        Order matters: wabt first (so wast2json is present before any test
        script or vendor step references it), then the workspace prune (so
        `cargo vendor` sees the trimmed member list), then the empty-lib-stub
        vendor (so target/deps/* is warm for the vendored crates), then the
        strings-target leak assert (which needs target/ to exist).

        Everything runs BEFORE the leak gate because the leak scan is what
        should be catching a mistake here: a stray carved symbol in target/,
        a leftover incremental fingerprint, a receipt fragment. Putting these
        after the gate would hide exactly the things the gate is looking for.
        """
        wabt_install = '\n'.join([
            f'# wabt {self.wabt_version}: wast2json is a HARD runtime dependency of the graded',
            "# harness (tests/util/spectest.rs shells out to it for every .wast file).",
            f'# The base ships the wabt suite at {self.wabt_version}, so this ASSERTS the',
            '# dependency rather than installing it. Asserted under a login shell because',
            "# that is how harbor's test.sh resolves it at grade time.",
            'RUN set -eux; \\',
            "    v=\"$(bash -lc 'wast2json --version')\"; \\",
            f'    case \"$v\" in *{self.wabt_version}*) echo \"WABT OK (login shell): $v\" ;; \\',
            f'      *) echo \"TOOLCHAIN PIN FAILED (login shell): wast2json $v want {self.wabt_version}\" >&2; exit 42 ;; \\',
            '    esac',
        ])

        # Prune out-of-scope workspace members so cargo vendor covers only the
        # graded package. crates/* pull wasi-common/wasm-smith/clap which are
        # not carved and not graded.
        prune = '\n'.join([
            '# Prune out-of-scope workspace members: crates/* and fuzz/ are neither',
            '# carved nor graded, and their dependency graphs would bloat the vendor',
            '# set and the offline runtime.',
            'RUN rm -rf crates fuzz \\',
            ' && python3 - <<\'PY\'',
            'import pathlib',
            f'p = pathlib.Path("{env.workdir}/Cargo.toml")',
            's = p.read_text()',
            's = s.replace(\'members = ["crates/*"]\', \'members = []\')',
            's = s.replace(\'exclude = ["fuzz"]\', \'exclude = []\')',
            'p.write_text(s)',
            'PY',
        ])

        # The empty-lib-stub trick. cargo vendor needs a resolvable [lib]
        # target; it does NOT need that target to contain any code. A one-byte
        # src/lib.rs lets vendor run, warms target/deps for the resolved
        # crates, and is deleted before the RUN ends. Nothing carved is
        # disclosed -- src/ ends up empty on disk.
        vendor = '\n'.join([
            '# Vendor via empty-lib-stub. cargo vendor needs a resolvable [lib] target,',
            '# which the carve deleted along with the rest of src/. It does NOT need',
            '# that target to hold real code: vendoring is dependency RESOLUTION driven',
            '# by Cargo.toml + Cargo.lock, and a one-byte src/lib.rs reproduces the',
            '# intact vendor set exactly. The same stub warms target/deps/* for the',
            '# resolved crates so the graded run does not rebuild them from scratch;',
            '# the carved-crate fingerprints and incremental artifacts are then',
            '# explicitly deleted (see the strings-assert RUN below).',
            "# CARGO_NET_OFFLINE=true is inherited from harbor-base; it is relaxed HERE",
            '# (and only here) so the lockfile can be re-resolved for the pruned',
            '# workspace and the crates fetched.',
            'RUN set -eux; \\',
            '    CRATE=$(sed -n \'/^\\[package\\]/,/^\\[/p\' Cargo.toml \\',
            '            | sed -n \'s/^name[[:space:]]*=[[:space:]]*"\\([^"]*\\)".*/\\1/p\' \\',
            '            | head -1); \\',
            '    echo "vendoring for crate: ${CRATE}"; \\',
            '    mkdir -p src; \\',
            '    : > src/lib.rs; \\',
            f'    CARGO_NET_OFFLINE=false cargo vendor {env.workdir}/vendor > /tmp/vendor-config.toml; \\',
            f'    mkdir -p {env.workdir}/.cargo; \\',
            f'    cp /tmp/vendor-config.toml {env.workdir}/.cargo/config.toml; \\',
            '    rm -f /tmp/vendor-config.toml; \\',
            '    (cargo test --offline --no-run 2>&1 | tail -5) || true; \\',
            '    cargo clean --offline -p "${CRATE}" 2>/dev/null || true; \\',
            '    rm -f src/lib.rs; \\',
            '    test -z "$(ls -A src)"; \\',
            '    rm -rf target/debug/incremental; \\',
            '    find target/debug/.fingerprint -maxdepth 1 -name "${CRATE}-*" -exec rm -rf {} + 2>/dev/null || true; \\',
            '    find target/debug/deps -maxdepth 1 -name "${CRATE}-*" -exec rm -rf {} + 2>/dev/null || true; \\',
            '    find target/debug/deps -maxdepth 1 -name "lib${CRATE}-*" -exec rm -rf {} + 2>/dev/null || true',
        ])

        # target/ is the one place where a build could have left a structural
        # fossil of the carved API. `strings <target-obj> | grep <crate>` would
        # recover mangled symbol names, private method names, unit-test names;
        # `grep <workdir>/src` would recover every carved file path. Both fail
        # closed here.
        strings_assert = '\n'.join([
            '# strings-target leak assert. target/ is the one place where compiling',
            '# the crate could have left a structural fossil of the carved API. Require',
            '# that the crate name and every "<workdir>/src" debug-info path appear',
            '# NOWHERE in target/: the first catches every mangled symbol, the second',
            '# catches every debug-info path that would enumerate the carved file list.',
            '# Vendored crates reference "<workdir>/vendor/<crate>/src/..." and so do',
            '# not trip this. This is BELT-AND-BRACES: the generic leak scan below is',
            '# what catches content leaks, but a symbol-only leak in an .o would slip',
            '# past a content grep and be recoverable with `strings`.',
            'RUN set -eu; \\',
            '    CRATE=$(sed -n \'/^\\[package\\]/,/^\\[/p\' Cargo.toml \\',
            '            | sed -n \'s/^name[[:space:]]*=[[:space:]]*"\\([^"]*\\)".*/\\1/p\' \\',
            '            | head -1); \\',
            f'    hits=$(find {env.workdir}/target -type f -print0 2>/dev/null \\',
            '             | xargs -0 -r strings -a 2>/dev/null \\',
            f'             | grep -E "${{CRATE}}|{env.workdir}/src" || true); \\',
            '    if [ -n "$hits" ]; then \\',
            '        echo "LEAK: rustc artifacts under target/ reveal carved internals:" >&2; \\',
            '        echo "$hits" | sort -u | head -40 >&2; \\',
            '        exit 1; \\',
            '    fi; \\',
            f'    echo "ASSERT: target/ carries no ${{CRATE}} symbols and no {env.workdir}/src paths"',
        ])

        return (wabt_install, '', prune, '', vendor, '', strings_assert)

    def validate_dep_plan(self, plan: DepPlan) -> None:
        """Every precondition `render_gap` has, checked before a container exists.

        The same set of checks `render_gap` makes, hoisted to where they cost
        nothing and can still be repaired: same helpers, same messages, so a
        plan accepted here cannot then fail to render. Enumerated from the
        renderer below -- the rustc version the pin matches, the manager whose
        toolchain it pins, the two rendered flags and the graded command --
        because a slot the renderer reads and this gate does not is exactly the
        crash this exists to prevent.
        """
        validate(plan)
        plan = canonicalize(plan)
        self._reject_foreign(plan)
        _rustc_version(plan.toolchain_version)
        for key in REQUIRED_BUILD_FLAGS:
            _flag_str(plan.build_flags, key)
        for key in REQUIRED_TEST_INVOCATION_KEYS:
            _test_tokens(plan.test_invocation, key)

    def _gap_body(self, plan: DepPlan) -> str:
        """rust's toolchain bytes, rendered from a plan instead of a literal.

        The part BOTH the measure and the shipped render take; the measure-only
        no-warm note is `render_gap`'s (see `LangPlugin._gap_body`).

        The SCAFFOLDING stays fixed and stays hardcoded: the zz- profile.d file
        that keeps the mise shims ahead of the base's own PATH, the login-shell
        pin assert's shape, the ENV/WORKDIR placement tail -- and, below the
        repo COPY where this string cannot reach, the wabt install, the
        workspace prune, `INTACT_VENDOR_BLOCK` and the strings leak-assert.
        Those are how harbor resolves a baked toolchain and proves no carved
        byte survives, not facts about this repo. What comes from the PLAN is
        the rustc version the pin matches, the shims directory it prepends, the
        CDN the comment says it avoids, and the apt/install lines a repo needing
        system libraries would add.

        The apt and install blocks are unreachable for `RUST_MEASURE_DEP_PLAN`
        (both fields are empty by construction, because the base BAKES the
        toolchain) and are rendered from the plan for the case where a resolved
        plan does declare them. They are the only lines here that are not in
        today's image.
        """
        validate(plan)
        plan = canonicalize(plan)
        self._reject_foreign(plan)

        version = _rustc_version(plan.toolchain_version)
        shims = _flag_str(plan.build_flags, 'shims_path')
        lines = [
            f'# rustc {version} is baked into the base; select it rather',
            f'# than reaching {_flag_str(plan.build_flags, "toolchain_source")} '
            'on every task build.',
            '# profile.d is sourced in sorted order and the base re-exports its own',
            '# PATH ahead of the mise shims, so only a zz- file keeps the pin in front',
            "# for the login shells harbor's test.sh and solve.sh run as.",
            'RUN set -eux; \\',
            f'    printf \'export PATH="{shims}:$PATH"\\n\' > '
            '/etc/profile.d/zz-harbor-toolchain-pin.sh; \\',
            '    chmod 0644 /etc/profile.d/zz-harbor-toolchain-pin.sh',
            'RUN set -eux; \\',
            '    v="$(bash -lc \'rustc --version\')"; \\',
            '    case "$v" in \\',
            f'        *"{version}"*) echo "TOOLCHAIN PIN OK (login shell): $v" ;; \\',
            f'        *) echo "TOOLCHAIN PIN FAILED (login shell): got $v want '
            f'{version}" >&2; exit 42 ;; \\',
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
        the warmed cargo registry from a separate stage.
        """
        return '\n'.join([self._gap_body(plan), '', MEASURE_NO_WARM_COMMENT])

    def _reject_foreign(self, plan: DepPlan) -> None:
        if plan.lang != self.name:
            raise B.LangError(
                f'the rust gap cannot render a {plan.lang!r} plan; a gap is the '
                'one part of a Dockerfile that is language-specific by definition'
            )
        if plan.package_manager not in _SUPPORTED_MANAGERS:
            raise B.LangError(
                f'the rust gap has no prose for package_manager '
                f'{plan.package_manager!r}; expected one of '
                f'{", ".join(sorted(_SUPPORTED_MANAGERS))}'
            )

    def render_measure_dockerfile(
        self, env: EnvSpec, *, dep_plan: DepPlan | None = None,
    ) -> str:
        """The stripped Dockerfile for the never-ship measure image.

        Same toolchain + wabt + prune as the shipped image, but the vendor
        step is SIMPLER: src/ contains the real crate here, so the empty-lib
        stub of the shipped image is inapplicable (would overwrite lib.rs
        without touching the other 30 files, leaving a broken half-carve).
        `cargo vendor` runs directly against the intact tree.

        NO leak gate / NO strings-assert / NO tripwire scan / NO carve-receipt
        check. On the INTACT tree the strings-assert and tripwires WOULD fire
        (they exist to catch carved bytes); the measure image is built ONLY to
        count `tests_total` and is deleted in `measure.py`'s finally block.

        `dep_plan` swaps the hardcoded gap for `render_gap(dep_plan)` in the
        SAME slot and touches nothing else -- notably NOT the wabt install, the
        prune or `INTACT_VENDOR_BLOCK` below, which are scaffolding and which
        run after the repo COPY the gap slot sits above. `dep_plan=None` is what
        emit.py passes and renders the bytes it always did.
        """
        base_image = self.toolchain_spec().base_image
        gap = (
            '\n'.join([self.toolchain(), '', MEASURE_NO_WARM_COMMENT])
            if dep_plan is None
            else self.render_gap(dep_plan)
        )
        pre = self.pre_leakgate_blocks(env)
        wabt_install, _sep1, prune, _sep2, _shipped_vendor, _sep3, _strings = pre

        return '\n'.join([
            '# syntax=docker/dockerfile:1.7',
            f'# Harbor MEASURE image -- {env.repo_name} ({self.name}). NEVER SHIP.',
            '#',
            '# Built by measure.py phase 1 to count the intact test suite. Contains',
            '# the intact tree by construction -- an escaped measure image is the',
            '# whole answer, not a partial leak. `measure_image_tag` marks it as',
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
            f'# --- wabt (bind-mounted from the {env.tooling_context} context) ---',
            wabt_install,
            '',
            '# --- prune out-of-scope workspace members ---',
            prune,
            '',
            '# --- vendor deps against the intact tree ---',
            INTACT_VENDOR_BLOCK,
            '',
            '# --- measure script (COPYed, not carved-tree, so it lives in a layer) ---',
            f'COPY measure.sh /opt/harbor/tests/measure.sh',
            f'RUN chmod 0555 /opt/harbor/tests/measure.sh',
            '',
        ])


B.register(RustPlugin())
