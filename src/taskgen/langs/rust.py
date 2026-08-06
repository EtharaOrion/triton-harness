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

from pathlib import Path
from typing import ClassVar, Mapping

from . import base as B
from .base import DepWarmSpec, EnvSpec, GradedSet, ToolchainSpec

__all__ = [
    'EXPECTED_HARNESSES',
    'HARNESSES',
    'INTEGRITY_HARNESS_FILES',
    'MIN_WAST',
    'BASE_IMAGE',
    'RUST_VERSION',
    'RustPlugin',
    'TEST_COMMAND',
    'WABT_VERSION',
]

RUST_VERSION = '1.87.0'

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


class RustPlugin(B.LangPlugin):
    """rustc 1.87 + wabt 1.0.41, whole-suite equality floor, measured denominator."""

    name: ClassVar[str] = 'rust'
    toml_family: ClassVar[str] = 'A'
    floor_mode: ClassVar[str] = 'equality'
    parser_backed: ClassVar[bool] = False
    synthesizes_git: ClassVar[bool] = False

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
            '# than reaching static.rust-lang.org on every task build.\n'
            '# profile.d is sourced in sorted order and the base re-exports its own\n'
            '# PATH ahead of the mise shims, so only a zz- file keeps the pin in front\n'
            "# for the login shells harbor's test.sh and solve.sh run as.\n"
            'RUN set -eux; \\\n'
            '    printf \'export PATH="/opt/mise/shims:$PATH"\\n\' > /etc/profile.d/zz-harbor-toolchain-pin.sh; \\\n'
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

    def render_measure_dockerfile(self, env: EnvSpec) -> str:
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
        """
        base_image = self.toolchain_spec().base_image
        toolchain = self.toolchain()
        pre = self.pre_leakgate_blocks(env)
        wabt_install, _sep1, prune, _sep2, _shipped_vendor, _sep3, _strings = pre

        intact_vendor = '\n'.join([
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
            toolchain,
            '',
            '# no warmed dependencies (rust vendors in the graded image)',
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
            intact_vendor,
            '',
            '# --- measure script (COPYed, not carved-tree, so it lives in a layer) ---',
            f'COPY measure.sh /opt/harbor/tests/measure.sh',
            f'RUN chmod 0555 /opt/harbor/tests/measure.sh',
            '',
        ])


B.register(RustPlugin())
