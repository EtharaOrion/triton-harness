"""The python plugin: today's proven entry, re-expressed on the plugin base.

This is a MIGRATION, not a design. The behaviour it renders is the one WAVE 0
proved in a container -- GREEN-baseline 5/5, RED-stub 0/5, GREEN-oracle 5/5,
three stable repeats -- so every flag below is load-bearing and was measured,
not chosen:

    -o addopts=''        the project pins `--dist loadgroup`, which pytest
                         REFUSES to start with once xdist is disabled
    -p no:xdist          loadgroup scheduling corrupts per-test attribution
    -p no:cacheprovider  a .pytest_cache in the image is state between runs
    -p harbor_filter     selection by DESELECTION. Passing node ids on argv
                         makes pytest run them in the order GIVEN, which is not
                         this suite's collection order, and the suite has
                         order-dependent state (hence its own xdist_group
                         markers). Reordering produces failures that have
                         nothing to do with the carved code.
    HARBOR_SELECTED_OUT  set explicitly: harbor_filter swallows OSError when
                         writing its count, so a path typo would read as
                         `selected=0` -- a silent RED -- instead of an error.

TWO THINGS DELIBERATELY CHANGED.

The intact tree no longer enters a build context. The old Dockerfile did
`COPY --from=repo-src . ${HARBOR_REPO}` and then overwrote one file with its
stub; `docker save` reads every layer, so the answer was recoverable from the
earlier one (plan A1). The carve now happens on the HOST and the image only ever
sees the carved tree.

And `git init` now runs over that already-carved tree. It cannot simply be
dropped: uv-dynamic-versioning declares `vcs = "git"`, so without git metadata
`uv sync` cannot resolve a version at all. Running it BEFORE the carve is leak
route C -- the answer in `.git/objects`, recoverable with one
`git checkout --` and no docker tooling whatsoever. Running it after, plus the
audit in `pre_leakgate_blocks`, is what the reference asset settled on.

Python is `compiled=1.0` unconditionally: there is no build step to fail, so
folding "it compiled" into "the tests passed" would erase a distinction the
common schema exists to keep.
"""

from __future__ import annotations

from typing import ClassVar, Mapping

from . import base as B
from .base import DepWarmSpec, EnvSpec, GradedSet, ToolchainSpec

__all__ = ['PythonPlugin', 'UV_VERSION', 'PYTHON_VERSION', 'BASE_IMAGE']

#: Matches emit.UV_VERSION. An unpinned resolver is a reproducibility hole.
UV_VERSION = '0.12.1'

#: The runtime this task is graded on, selected from the several the base bakes.
#: Pinning here rather than inheriting the base default is what stops a rebuild
#: of the base from silently moving the interpreter under the benchmark.
PYTHON_VERSION = '3.10.20'

#: Per-language base: every baked runtime plus uv and mise, so no task build has
#: to reach the network to patch its own toolchain.
BASE_IMAGE = '426628337772.dkr.ecr.ap-south-2.amazonaws.com/triton/base-python@sha256:dd204a8e01253845e88953524f4e3078583b95f6652d5c099cce18491d33d707'

_ALLOWLIST = f'{B.TESTS_DIR}/allowlist.txt'
_LOGS_DEFAULT = '${VERIFIER_DIR:-' + B.LOGS_DIR + '}'


class PythonPlugin(B.LangPlugin):
    """uv + pytest, graded through the harbor_filter allowlist, equality floor."""

    name: ClassVar[str] = 'python'
    toml_family: ClassVar[str] = 'A'
    floor_mode: ClassVar[str] = 'equality'
    parser_backed: ClassVar[bool] = True
    synthesizes_git: ClassVar[bool] = True

    uv_version: ClassVar[str] = UV_VERSION
    python_version: ClassVar[str] = PYTHON_VERSION

    # --- axis 1 -----------------------------------------------------------

    def toolchain_spec(self) -> ToolchainSpec:
        """Select one baked runtime, and prove the selection survives a login shell.

        A build-time assertion alone is not sound. The base re-exports its own
        PATH from /etc/profile.d/harbor-toolchain.sh, ahead of the mise shims;
        profile.d is sourced in sorted order, so only a zz- file keeps the pin in
        front. harbor's test.sh and solve.sh run as login shells, so an
        assertion that is not itself a login shell can pass while the graded run
        executes on the base default -- observed, not theorised.
        """
        install = f'''# The base bakes several runtimes and ships uv {self.uv_version}; select one.
RUN set -eux; mise use -g python@{self.python_version}
RUN set -eux; \\
    printf 'export PATH="/opt/mise/shims:$PATH"\\n' > /etc/profile.d/zz-harbor-toolchain-pin.sh; \\
    chmod 0644 /etc/profile.d/zz-harbor-toolchain-pin.sh
RUN set -eux; \\
    v="$(bash -lc 'python3 --version')"; \\
    case "$v" in \\
        *"{self.python_version}"*) echo "TOOLCHAIN PIN OK (login shell): $v" ;; \\
        *) echo "TOOLCHAIN PIN FAILED (login shell): got $v want {self.python_version}" >&2; exit 42 ;; \\
    esac
RUN uv --version'''
        return ToolchainSpec(
            base_image=BASE_IMAGE,
            install_block=install,
            env={
                'HARBOR_REPO': B.WORKDIR,
                'UV_PROJECT_ENVIRONMENT': f'{B.WORKDIR}/.venv',
                'UV_LINK_MODE': 'copy',
                'UV_COMPILE_BYTECODE': '1',
                'PYTHONDONTWRITEBYTECODE': '1',
            },
            workdir=B.WORKDIR,
        )

    # --- axis 2 -----------------------------------------------------------

    def dep_warm_spec(self) -> DepWarmSpec:
        """`uv sync` against the CARVED tree, plus the two gates that make it sound.

        `copy_paths` is empty on purpose, and that is the interesting part.
        Every other language warms its dependencies in a separate image and
        pulls them in by COPY. Python cannot: the environment is a `.venv`
        INSIDE the project and the project is installed EDITABLE into it, so a
        venv built elsewhere would either carry absolute paths to a tree that
        does not exist or -- worse -- require the repo to be present in the warm
        image, which is exactly what invariant 7 forbids. So the sync runs in
        the graded image, over the carved tree, where the only sources it can
        possibly bake are already carved.
        """
        stage = f'''# Trap: uv-dynamic-versioning declares `vcs = "git"`, so uv cannot resolve a
# version without git metadata. This synthesises it over the ALREADY-CARVED
# tree -- doing it before the carve is leak route C, the answer in
# .git/objects, recoverable with a single `git checkout --`. Signing is
# force-disabled: an inherited signing config would hang or fail the commit.
RUN rm -rf .git .venv \\
 && git init -q -b main \\
 && git -c user.email=harbor@localhost -c user.name=Harbor \\
        -c commit.gpgsign=false -c tag.gpgsign=false \\
        -c core.hooksPath=/dev/null add -A \\
 && git -c user.email=harbor@localhost -c user.name=Harbor \\
        -c commit.gpgsign=false -c core.hooksPath=/dev/null \\
        commit -q --no-gpg-sign -m 'harbor: synthetic baseline' \\
 && git -c user.email=harbor@localhost -c user.name=Harbor \\
        -c tag.gpgsign=false tag -a v0.0.0 -m 'harbor synthetic tag' \\
 && git describe --tags

# The ONLY step that touches the network. --locked forbids uv from silently
# re-resolving the lockfile, which would make the image depend on the day it
# was built.
RUN uv sync --locked --all-extras

# Prove the install is EDITABLE. The whole benchmark rests on it: a solver's
# regenerated function must take effect with no reinstall, which is impossible
# if uv baked a non-editable copy into site-packages.
RUN uv run --no-sync --offline python - <<'PY'
import pathlib, sysconfig
site = pathlib.Path(sysconfig.get_paths()["purelib"])
pth = sorted(site.glob("*.pth")) + sorted(site.glob("__editable__*"))
assert pth, f"NOT EDITABLE: no editable marker in {{site}}"
print("editable install confirmed ->", [p.name for p in pth])
PY'''
        return DepWarmSpec(
            stage_block=stage,
            files_needed=('pyproject.toml', 'uv.lock'),
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

        return '\n'.join([
            '#!/usr/bin/env bash',
            f'# Harbor verifier -- python, {expected} graded test(s).',
            '#',
            '# EQUALITY floor. Unlike go, nothing here aborts the process on the first',
            '# failure, so the observed selection IS a denominator and a `>=` floor',
            '# would let a solver delete or skip tests and still clear it.',
            '#',
            "# Harbor ignores this script's exit code; /logs/verifier/reward.json is",
            '# the single source of truth and is written on EVERY path.',
            '',
            'set -uo pipefail',
            '',
            f'REPO=${{REPO:-{B.WORKDIR}}}',
            f'ALLOWLIST=${{ALLOWLIST:-{_ALLOWLIST}}}',
            B.reward_emitter_block(_LOGS_DEFAULT),
            'REPORT=${VERIFIER_DIR}/pytest.log',
            'SELECTED=${VERIFIER_DIR}/selected_count.txt',
            'JUNIT=${VERIFIER_DIR}/junit.xml',
            '# PLAN 8b: the JUnit report is ALSO persisted as results.xml under a',
            '# run-time results dir. Created here, inside the container, at run time --',
            '# never baked into the image and never an emitted asset.',
            'RESULTS_DIR=${VERIFIER_DIR}/results',
            'RESULTS_XML=${RESULTS_DIR}/results.xml',
            'mkdir -p "${RESULTS_DIR}"',
            'rm -f "${SELECTED}" "${JUNIT}" "${RESULTS_XML}"',
            '',
            B.fail_closed_preamble(expected, compiled='1.0'),
            '',
            'cd "${REPO}" || fail "no ${REPO}"',
            '[ -f "${ALLOWLIST}" ] || fail "no allowlist at ${ALLOWLIST}"',
            '[ "$(grep -c . "${ALLOWLIST}")" = "${EXPECTED}" ] \\',
            '    || fail "allowlist is not ${EXPECTED} ids"',
            '',
            B.fingerprint_gate_block(fingerprint, repo_var='${REPO}'),
            '',
            f'echo "== grading ${{EXPECTED}} allowlisted test(s) (offline, no xdist) =="',
            '',
            'HARBOR_ALLOWLIST="${ALLOWLIST}" \\',
            'HARBOR_SELECTED_OUT="${SELECTED}" \\',
            f'PYTHONPATH={B.TESTS_DIR} \\',
            'uv run --no-sync --offline pytest \\',
            "    -o addopts='' \\",
            '    -p no:xdist \\',
            '    -p no:cacheprovider \\',
            '    -p harbor_filter \\',
            '    --junitxml="${JUNIT}" \\',
            '    -q -rf 2>&1 | tee -a "${REPORT}"',
            '',
            'STATUS=${PIPESTATUS[0]}',
            '',
            _SELECTED_BLOCK,
            _JUNIT_PASSED_BLOCK,
            _RESULTS_XML_BLOCK,
            '',
            '# Python always "compiles": there is no build step that can fail, and',
            '# folding that into the test result would erase a distinction the',
            '# common schema exists to keep.',
            'COMPILED=1.0',
            '',
            'echo "== pytest exit=${STATUS} passed=${PASSED} '
            'selected=${TOTAL}/${EXPECTED} =="',
            '',
            B.floor_gate_block(self.floor_mode, expected),
            'exit 0',
            '',
        ])

    def measure_test_sh(self, *, graded: GradedSet | None = None, **kwargs) -> str:
        """Phase 1: COLLECT the allowlist against the intact tree and count it.

        `--collect-only` runs no test, so what is measured is the size of the
        selection rather than its outcome -- which is what a denominator has to
        be. It also catches a mistyped node id here, where it is a build-time
        error, rather than in the graded run where it would surface as a silent
        RED with green-looking pytest output.
        """
        return '\n'.join([
            '#!/usr/bin/env bash',
            '# Harbor MEASURE (phase 1) -- python. Floor-FREE by construction: it',
            '# counts, it asserts nothing. A floor cannot be enforced by the run that',
            '# makes it.',
            '',
            'set -uo pipefail',
            '',
            f'REPO=${{REPO:-{B.WORKDIR}}}',
            f'ALLOWLIST=${{ALLOWLIST:-{_ALLOWLIST}}}',
            'COLLECTED=$(mktemp)',
            '',
            B.measure_emitter_block(_LOGS_DEFAULT),
            '',
            'cd "${REPO}" || { echo "no ${REPO}" >&2; exit 1; }',
            '',
            'HARBOR_ALLOWLIST="${ALLOWLIST}" \\',
            'HARBOR_SELECTED_OUT="${MEASURE_DIR}/selected_count.txt" \\',
            f'PYTHONPATH={B.TESTS_DIR} \\',
            'uv run --no-sync --offline pytest \\',
            "    -o addopts='' -p no:xdist -p no:cacheprovider -p harbor_filter \\",
            '    --collect-only -q 2>/dev/null \\',
            '  | sed -n \'/::/p\' | sort -u > "${COLLECTED}"',
            '',
            'TOTAL=$(grep -c . "${COLLECTED}")',
            'GRADED=$(grep . "${COLLECTED}" | sed \'s/.*/"&"/\' | paste -sd, -)',
            '',
            'measure "${TOTAL}" "${GRADED}"',
            'exit 0',
            '',
        ])

    # --- axis 7 -----------------------------------------------------------

    def post_restore_block(self) -> str:
        """Drop stale bytecode so imports resolve to the restored source."""
        return (
            '# A cached .pyc would keep importing the CARVED module after the\n'
            '# restore, turning a correct oracle into a red run.\n'
            "find \"${REPO}\" -name '__pycache__' -type d -prune -exec rm -rf {} + "
            '2>/dev/null || true'
        )

    # --- the image --------------------------------------------------------

    def pre_leakgate_blocks(self, env: EnvSpec) -> tuple[str, ...]:
        """Resolve deps over the carved tree, then prove git holds metadata only.

        Ordered before the leak gate so that the scan sees `.git/objects` and
        the built `.venv` -- the two places a python answer would come to rest.
        """
        return (
            self.dep_warm_spec().stage_block,
            '',
            '# The git repo must be version metadata and NOTHING else. `--reflog` is',
            '# load-bearing: plain `--all` walks refs only and would miss an object',
            '# kept alive by a reflog entry alone; fsck then proves nothing is hiding',
            '# outside both.',
            'RUN set -eux; \\',
            f'    cd {env.workdir}; \\',
            '    test "$(git rev-list --count --all)" = "1"; \\',
            '    test "$(git stash list | wc -l)" = "0"; \\',
            '    test "$(git rev-list --objects --all --reflog | wc -l)" -gt 0; \\',
            '    test -z "$(git fsck --unreachable --dangling --no-progress 2>/dev/null)"; \\',
            '    git describe --tags; \\',
            '    echo "git carries version metadata only: one commit over the '
            'carved tree, no stash, no unreachable or dangling object"',
        )


# --------------------------------------------------------------------------
# bash fragments
# --------------------------------------------------------------------------

_SELECTED_BLOCK = '\n'.join([
    '# harbor_filter writes the count it actually selected. A run trimmed to a',
    '# passing subset still exits 0, so the exit status alone cannot detect a',
    '# shrunken denominator -- this is what the equality gate compares.',
    'TOTAL=$( [ -f "${SELECTED}" ] && tr -d \'[:space:]\' < "${SELECTED}" || echo 0 )',
    'TOTAL=${TOTAL:-0}',
])

_JUNIT_PASSED_BLOCK = '\n'.join([
    'PASSED=$(python3 - "${JUNIT}" <<\'PY\'',
    'import sys, xml.etree.ElementTree as ET',
    'try:',
    '    root = ET.parse(sys.argv[1]).getroot()',
    'except Exception:',
    '    print(0)',
    '    raise SystemExit',
    'suites = [root] if root.tag == "testsuite" else list(root)',
    'total = failed = errored = skipped = 0',
    'for s in suites:',
    '    total += int(s.get("tests", 0))',
    '    failed += int(s.get("failures", 0))',
    '    errored += int(s.get("errors", 0))',
    '    skipped += int(s.get("skipped", 0))',
    'print(max(total - failed - errored - skipped, 0))',
    'PY',
    ')',
    'PASSED=${PASSED:-0}',
])

#: PLAN 8b (D8) -- persist the JUnit report as `results.xml`.
#:
#: PURELY ADDITIVE REPORTING. It is a copy, taken AFTER ${PASSED} has already
#: been read off ${JUNIT}: nothing below it reads results.xml, so RED is still
#: 0.0 and GREEN is still 1.0 on exactly the arithmetic that was proven in a
#: container. It is written at RUN TIME under ${VERIFIER_DIR}/results (i.e.
#: /logs/verifier/results), which is a mounted log dir, not an image layer, so
#: image hygiene and the N-way byte-identity of the emitted test.sh are both
#: untouched. A missing ${JUNIT} (pytest died before writing one) is not fatal:
#: `cp` is guarded and the reward path already handled that case above.
#:
#: THE OTHER FIVE LANGUAGES ARE NOT IMPLEMENTED -- documented follow-up only
#: (PLAN 8b matrix). Nothing below is faked or half-wired anywhere in this tree:
#:
#:   go    `go test -json` is structured but not JUnit; convert with
#:         `gotestsum --junitfile results.xml`. Effort: small (adds a tool to
#:         the image).
#:   java  Maven Surefire already writes JUnit XML to
#:         target/surefire-reports/TEST-*.xml; collect/merge them into
#:         results.xml. Effort: trivial.
#:   rust  `cargo test`'s output is human-oriented; `cargo nextest run
#:         --profile ci` emits JUnit. Effort: small (adds nextest).
#:   c     ctest `--output-junit results.xml` (CMake >= 3.21), or wrap the
#:         custom .xs corpus runner. Effort: medium.
#:   cpp   doctest `--reporters=junit --out=results.xml`, or the same ctest
#:         flag. Effort: medium.
_RESULTS_XML_BLOCK = '\n'.join([
    '',
    '# PLAN 8b: persist the JUnit report as results.xml. Read-only side effect --',
    '# PASSED was already computed above, and nothing below consumes this file, so',
    '# the reward/binary contract is byte-for-byte the one that was proven.',
    'if [ -f "${JUNIT}" ]; then',
    '    cp "${JUNIT}" "${RESULTS_XML}" || echo "warn: could not write ${RESULTS_XML}" >&2',
    'fi',
])


B.register(PythonPlugin())
