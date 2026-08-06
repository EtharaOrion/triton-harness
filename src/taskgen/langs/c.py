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

from dataclasses import replace
from typing import ClassVar, Mapping

from ..depplan import DepPlan, FlagValue, canonicalize, validate
from . import base as B
from .base import DepWarmSpec, EnvSpec, GradedSet, ToolchainSpec

__all__ = [
    'BAKED_CAPABILITIES',
    'CPlugin',
    'C_MEASURE_DEP_PLAN',
    'GRADER_FINGERPRINT_GLOBS',
    'MEASURE_NO_WARM_COMMENT',
    'TEST_COMMAND',
    'UNIT_NAMES',
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


def _flag_str(build_flags: tuple[tuple[str, FlagValue], ...], key: str) -> str:
    """A build flag the gap's rendered text needs, or a refusal to render at all."""
    for name, value in build_flags:
        if name == key:
            if isinstance(value, str) and value:
                return value
            break
    raise B.LangError(
        f'the c gap needs build_flags[{key!r}] as a non-empty string; a plan '
        'without it would render a Dockerfile comment with a hole in it, and a '
        'hole in the toolchain description is how a base swap goes unnoticed'
    )


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

    test_command: ClassVar[str] = TEST_COMMAND
    unit_names: ClassVar[tuple[str, ...]] = UNIT_NAMES

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

    def render_test_sh(
        self,
        graded: GradedSet,
        *,
        expected: int | None = None,
        fingerprint: Mapping[str, str] | None = None,
        tls_count: int | None = None,
        corpus_counts: Mapping[str, int] | None = None,
    ) -> str:
        expected = graded.expected if expected is None else int(expected)
        fingerprint = graded.fingerprint_sha256 if fingerprint is None else fingerprint

        if tls_count is None or corpus_counts is None:
            raise B.LangError(
                'c plugin needs tls_count and corpus_counts threaded from the '
                'intact tree at generate time; emit._render_test_sh supplies them '
                '(no magic numbers in the plugin source)'
            )
        expect_conf = int(corpus_counts['conformance'])
        expect_reg = int(corpus_counts['regression'])
        expect_unit = int(corpus_counts['unit'])
        if expect_conf + expect_reg + expect_unit != expected:
            raise B.LangError(
                f'c plugin sub-counts do not sum to expected: '
                f'{expect_conf}+{expect_reg}+{expect_unit} != {expected}'
            )
        if len(self.unit_names) != expect_unit:
            raise B.LangError(
                f'unit_names has {len(self.unit_names)} entries but the intact '
                f'tests/unit/ has {expect_unit} *_test.c files -- the plugin\'s '
                'harness list drifted from the repo'
            )
        tls = int(tls_count)
        unit_names_str = ' '.join(self.unit_names)

        return '\n'.join([
            '#!/usr/bin/env bash',
            f'# Harbor verifier -- c ({expected} = {expect_conf} conformance + '
            f'{expect_reg} regression + {expect_unit} unit, equality floor).',
            '#',
            '# EQUALITY floor. Each ./xs invocation and each unit _test binary is',
            '# its own process, so a crash in one cannot abort the others (unlike',
            '# Go, spike G1). observed==EXPECTED is a real assertion.',
            '#',
            '# The Makefile\'s test-conformance/test-regression recipes are for-loops',
            '# that `exit 1` on the FIRST failing program -- useless for a fraction.',
            '# This script therefore runs each program with the recipe\'s own command',
            '# and records per-program exit status. That is a faithful replication,',
            '# not a re-definition: the Makefile is inside the fingerprint lock, so',
            '# the recipes cannot drift out from under it.',
            '#',
            "# Harbor ignores this script's exit code; /logs/verifier/reward.json is",
            '# the single source of truth and is written on EVERY path.',
            '',
            'set -uo pipefail',
            '',
            f'REPO=${{REPO:-{B.WORKDIR}}}',
            B.reward_emitter_block(_LOGS_DEFAULT),
            'BUILD_LOG="${VERIFIER_DIR}/build.log"',
            'UNIT_BUILD_LOG="${VERIFIER_DIR}/unit-build.log"',
            'CONF_LOG="${VERIFIER_DIR}/conformance.log"',
            'REG_LOG="${VERIFIER_DIR}/regression.log"',
            'UNIT_LOG="${VERIFIER_DIR}/unit.log"',
            '',
            '# COMPILED is set to 1.0 only after ./xs actually relinks in THIS run.',
            '# Kept as a shell variable so the fail-closed helper quotes the current',
            '# value on every early exit -- a build that failed halfway must not be',
            '# reported as compiled=1 by an earlier optimistic setting.',
            'COMPILED=0.0',
            '',
            B.fail_closed_preamble(expected, compiled='"${COMPILED}"'),
            '',
            'cd "${REPO}" || fail "no ${REPO}"',
            '',
            f'EXPECT_CONF={expect_conf}',
            f'EXPECT_REG={expect_reg}',
            f'EXPECT_UNIT={expect_unit}',
            f'EXPECT_TOTAL={expected}',
            f'UNIT_NAMES="{unit_names_str}"',
            '',
            'echo "VERIFIER: $(gcc --version | head -1) / $(make --version | head -1) / $(uname -m)"',
            '',
            '# --- integrity guards -----------------------------------------------',
            '# The tests/ tree and the Makefile ARE the oracle. The 227-entry lock',
            '# below was captured host-side from the intact tree at generate time',
            '# and refuses to grade if any pinned file changed.',
            B.fingerprint_gate_block(fingerprint, repo_var='${REPO}'),
            '',
            '# Vendored BearSSL is not part of the carve; a solver that deleted or',
            '# stubbed it could make the link succeed without regenerating anything.',
            '# The pristine count is baked in from the host-side scan at generate time.',
            'TLS_NOW=$(find src/tls -type f | wc -l | tr -d " ")',
            f'[ "${{TLS_NOW}}" = "{tls}" ] \\',
            f'    || fail "src/tls file count changed: found ${{TLS_NOW}}, expected {tls} '
            '(vendored BearSSL must stay intact)"',
            f'echo "VERIFIER: src/tls intact -- ${{TLS_NOW}} files (pinned at {tls})"',
            '',
            '# Corpus is asserted EXACTLY, not as a lower bound: this IS the denominator.',
            'CONF_N=$(ls tests/conformance/*.xs 2>/dev/null | wc -l | tr -d " ")',
            'REG_N=$(ls tests/regression/*.xs 2>/dev/null | wc -l | tr -d " ")',
            '[ "${CONF_N}" = "${EXPECT_CONF}" ] \\',
            '    || fail "conformance corpus truncated: ${CONF_N}, expected ${EXPECT_CONF}"',
            '[ "${REG_N}" = "${EXPECT_REG}" ] \\',
            '    || fail "regression corpus truncated: ${REG_N}, expected ${EXPECT_REG}"',
            '',
            '# The network-dependent harness must not have been dragged into either graded glob.',
            'if ls tests/conformance/*.xs tests/regression/*.xs 2>/dev/null | grep -q "test_http_client"; then',
            '    fail "network-dependent test_http_client.xs leaked into the graded globs"',
            'fi',
            'echo "VERIFIER: corpus = ${CONF_N} conformance + ${REG_N} regression, http_client excluded"',
            '',
            '# --- genuine rebuild -------------------------------------------------',
            '# `make clean` wipes build/obj and ./xs; the unit-test binaries under',
            '# tests/unit/ are removed by hand (clean does not touch them). Nothing',
            '# stale from image build time or from the model\'s own iteration can',
            '# masquerade as a pass.',
            'make clean > "${VERIFIER_DIR}/clean.log" 2>&1 || true',
            'rm -f ./xs',
            'rm -f tests/unit/*_test',
            '[ -e ./xs ] && fail "./xs survived make clean -- refusing to trust the build"',
            '[ -d build/obj ] && fail "build/obj survived make clean -- refusing to trust the build"',
            '',
            'echo "VERIFIER: clean tree confirmed; starting full rebuild from source"',
            'make -j"$(nproc)" > "${BUILD_LOG}" 2>&1',
            'BUILD_STATUS=$?',
            'echo "VERIFIER: make exit=${BUILD_STATUS}"',
            'if [ "${BUILD_STATUS}" -ne 0 ]; then',
            '    echo "--- tail of build.log ---" >&2',
            '    tail -25 "${BUILD_LOG}" >&2',
            '    echo "VERIFIER: build failed (expected while src/runtime is missing or incomplete)" >&2',
            '    emit 0.0 0 "${EXPECTED}" 0.0 0.0',
            '    exit 0',
            'fi',
            '[ -x ./xs ] || fail "build reported success but ./xs was not produced"',
            '',
            'COMPILED=1.0',
            'echo "VERIFIER: rebuilt ./xs ($(stat -c%s ./xs) bytes), compiled=1.0"',
            '',
            '# Build the 13 unit harnesses. -k so that one failing to link does not',
            '# hide the results of the other twelve; each is scored on its own merits.',
            'UNIT_TARGETS=""',
            'for t in ${UNIT_NAMES}; do UNIT_TARGETS="${UNIT_TARGETS} tests/unit/${t}_test"; done',
            '# shellcheck disable=SC2086',
            'make -k -j"$(nproc)" ${UNIT_TARGETS} > "${UNIT_BUILD_LOG}" 2>&1',
            'echo "VERIFIER: unit harness build exit=$? (per-harness results counted below)"',
            '',
            '# --- graded run: per-program exit status ----------------------------',
            'run_corpus() {',
            '    local label=$1 dir=$2 log=$3',
            '    local passed=0 ran=0 f',
            '    : > "${log}"',
            '    for f in "${dir}"/*.xs; do',
            '        [ -f "${f}" ] || continue',
            '        ran=$((ran + 1))',
            '        if ./xs "${f}" >/dev/null 2>>"${log}"; then',
            '            passed=$((passed + 1))',
            '            echo "[${label}] PASS ${f}" >> "${log}"',
            '        else',
            '            echo "[${label}] FAIL ${f} (exit $?)" >> "${log}"',
            '        fi',
            '    done',
            '    echo "${passed} ${ran}"',
            '}',
            '',
            'read -r CONF_PASS CONF_RAN <<EOF',
            '$(run_corpus conformance tests/conformance "${CONF_LOG}")',
            'EOF',
            'read -r REG_PASS REG_RAN <<EOF',
            '$(run_corpus regression tests/regression "${REG_LOG}")',
            'EOF',
            '',
            'UNIT_PASS=0',
            'UNIT_RAN=0',
            ': > "${UNIT_LOG}"',
            'for t in ${UNIT_NAMES}; do',
            '    bin="tests/unit/${t}_test"',
            '    UNIT_RAN=$((UNIT_RAN + 1))',
            '    if [ ! -x "${bin}" ]; then',
            '        echo "[unit:${t}] FAIL -- harness binary missing (did not link)" >> "${UNIT_LOG}"',
            '        continue',
            '    fi',
            '    one="${VERIFIER_DIR}/unit-${t}.log"',
            '    XS="$(pwd)/xs" "./${bin}" > "${one}" 2>&1',
            '    rc=$?',
            '    cat "${one}" >> "${UNIT_LOG}"',
            '    if [ "${rc}" -ne 0 ]; then',
            '        echo "[unit:${t}] FAIL (exit ${rc})" >> "${UNIT_LOG}"',
            '    elif grep -qiE "\\[unit:[a-z_]+\\] .*(fail|FAILED)" "${one}"; then',
            '        echo "[unit:${t}] FAIL -- reported a failure despite exit 0" >> "${UNIT_LOG}"',
            '    else',
            '        UNIT_PASS=$((UNIT_PASS + 1))',
            '        echo "[unit:${t}] PASS" >> "${UNIT_LOG}"',
            '    fi',
            'done',
            '',
            'PASSED=$((CONF_PASS + REG_PASS + UNIT_PASS))',
            'RAN=$((CONF_RAN + REG_RAN + UNIT_RAN))',
            'TOTAL=${RAN}',
            '',
            'echo "VERIFIER: conformance ${CONF_PASS}/${CONF_RAN}, regression ${REG_PASS}/${REG_RAN}, '
            'unit ${UNIT_PASS}/${UNIT_RAN}"',
            'echo "VERIFIER: graded total ${PASSED}/${RAN}"',
            '',
            '# Structural anti-gaming: exact sweep counts. A sweep that visited fewer',
            '# programs than the corpus scores 0 rather than a fraction of a shrunken total.',
            '[ "${CONF_RAN}" -eq "${EXPECT_CONF}" ] \\',
            '    || fail "ran ${CONF_RAN} conformance programs, expected ${EXPECT_CONF}"',
            '[ "${REG_RAN}" -eq "${EXPECT_REG}" ] \\',
            '    || fail "ran ${REG_RAN} regression programs, expected ${EXPECT_REG}"',
            '[ "${UNIT_RAN}" -eq "${EXPECT_UNIT}" ] \\',
            '    || fail "ran ${UNIT_RAN} unit harnesses, expected ${EXPECT_UNIT}"',
            '',
            B.floor_gate_block(
                self.floor_mode, expected,
                passed_var='PASSED', total_var='TOTAL', compiled_var='COMPILED',
            ),
            '',
            '# Extend reward.json with sub-counts so the shape matches the reference',
            '# grader (verify.py reads the 5 canonical keys and ignores extras).',
            'printf \'{"reward": %s, "tests_passed": %s, "tests_total": %s, "binary": %s, '
            '"compiled": %s, "conformance_passed": %s, "conformance_total": %s, '
            '"regression_passed": %s, "regression_total": %s, "unit_passed": %s, '
            '"unit_total": %s}\\n\' \\',
            '    "${REWARD}" "${PASSED}" "${EXPECTED}" "${BINARY}" "${COMPILED}" \\',
            '    "${CONF_PASS}" "${EXPECT_CONF}" "${REG_PASS}" "${EXPECT_REG}" '
            '"${UNIT_PASS}" "${EXPECT_UNIT}" \\',
            '    > "${VERIFIER_DIR}/reward.json"',
            'echo "reward.json (extended) = $(cat "${VERIFIER_DIR}/reward.json")"',
            '',
            'exit 0',
            '',
        ])

    def measure_test_sh(self, *, graded: GradedSet | None = None, **kwargs) -> str:
        """Phase 1: build and run the three corpora against the INTACT tree, count runs.

        Floor-FREE by construction. `measure` writes tests_total = the count of
        programs the sweep actually attempted, so a build that broke halfway
        registers as fewer-than-expected and `parse_measure_json` rejects zero.
        """
        unit_names_str = ' '.join(self.unit_names)
        return '\n'.join([
            '#!/usr/bin/env bash',
            '# Harbor MEASURE (phase 1) -- c. Floor-FREE by construction; the pinned',
            '# denominator is what THIS run measures against the intact tree.',
            '',
            'set -uo pipefail',
            '',
            f'REPO=${{REPO:-{B.WORKDIR}}}',
            B.measure_emitter_block(_MEASURE_LOGS_DEFAULT),
            'BUILD_LOG="${MEASURE_DIR}/build.log"',
            'UNIT_BUILD_LOG="${MEASURE_DIR}/unit-build.log"',
            f'UNIT_NAMES="{unit_names_str}"',
            '',
            'cd "${REPO}" || { echo "no ${REPO}" >&2; measure 0 \'\'; exit 0; }',
            '',
            'make -j"$(nproc)" > "${BUILD_LOG}" 2>&1',
            'BUILD_STATUS=$?',
            'if [ "${BUILD_STATUS}" -ne 0 ]; then',
            '    tail -20 "${BUILD_LOG}" >&2',
            '    echo "measure: intact build failed (exit ${BUILD_STATUS})" >&2',
            '    measure 0 \'\'',
            '    exit 0',
            'fi',
            'if [ ! -x ./xs ]; then',
            '    echo "measure: no ./xs produced by intact build" >&2',
            '    measure 0 \'\'',
            '    exit 0',
            'fi',
            '',
            'UNIT_TARGETS=""',
            'for t in ${UNIT_NAMES}; do UNIT_TARGETS="${UNIT_TARGETS} tests/unit/${t}_test"; done',
            '# shellcheck disable=SC2086',
            'make -k -j"$(nproc)" ${UNIT_TARGETS} > "${UNIT_BUILD_LOG}" 2>&1',
            '',
            'run_dir() {',
            '    local dir=$1 passed=0 ran=0 f',
            '    for f in "${dir}"/*.xs; do',
            '        [ -f "${f}" ] || continue',
            '        ran=$((ran + 1))',
            '        ./xs "${f}" >/dev/null 2>&1 && passed=$((passed + 1))',
            '    done',
            '    echo "${passed} ${ran}"',
            '}',
            '',
            'read -r CONF_PASS CONF_RAN <<EOF',
            '$(run_dir tests/conformance)',
            'EOF',
            'read -r REG_PASS REG_RAN <<EOF',
            '$(run_dir tests/regression)',
            'EOF',
            '',
            'UNIT_PASS=0',
            'UNIT_RAN=0',
            'for t in ${UNIT_NAMES}; do',
            '    bin="tests/unit/${t}_test"',
            '    UNIT_RAN=$((UNIT_RAN + 1))',
            '    if [ -x "${bin}" ]; then',
            '        XS="$(pwd)/xs" "./${bin}" >/dev/null 2>&1 \\',
            '            && UNIT_PASS=$((UNIT_PASS + 1))',
            '    fi',
            'done',
            '',
            'RAN=$((CONF_RAN + REG_RAN + UNIT_RAN))',
            'PASSED=$((CONF_PASS + REG_PASS + UNIT_PASS))',
            'echo "MEASURE: conf ${CONF_PASS}/${CONF_RAN}, reg ${REG_PASS}/${REG_RAN}, '
            'unit ${UNIT_PASS}/${UNIT_RAN} -- total ran ${RAN}"',
            '',
            'measure "${RAN}" \'\'',
            'exit 0',
            '',
        ])

    # --- axis 7 -----------------------------------------------------------

    def post_restore_block(self) -> str:
        """Nothing: `make clean` in test.sh handles cache invalidation on its own."""
        return ''

    # --- axis 8 + the image ----------------------------------------------

    def render_gap(self, plan: DepPlan) -> str:
        """c's toolchain + dependency bytes, rendered from a plan instead of a literal.

        c is the smallest possible gap and that is exactly why it is the one to
        prove the seam on: harbor-base already ships gcc and GNU Make, and the
        repo links only against libc, so NOTHING is installed and the whole gap
        is a version echo, a pin assert and a "nothing was warmed" comment. If
        the seam cannot reproduce those bytes it cannot reproduce anyone's.

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

        pin = '.'.join(plan.toolchain_version.split('.')[:2])
        if pin.count('.') != 1:
            raise B.LangError(
                f'toolchain_version {plan.toolchain_version!r} has no major.minor '
                'to pin gcc against; a one-component version would make the pin '
                'assert match every 13.x the base image ever ships'
            )
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

        toolchain = replace(
            self.toolchain_spec(), install_block='\n'.join(lines),
        ).render()
        return '\n'.join([
            toolchain,
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
