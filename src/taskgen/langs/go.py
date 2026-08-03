"""The go plugin: an offline `go test -json` run graded on a PINNED denominator.

Go is the language that forced `floor_mode` to become a field. Harbor's shipped
defence everywhere else is `selected == EXPECTED`, and for go it is WRONG in a
way that only shows up on the runs that matter: a panic aborts the whole test
binary, so a stubbed function that panics produces ONE event for the first
selected test and silence for the rest (spike G1, measured 1/4). An equality
floor reads that as "the denominator moved" and refuses to grade, which turns
the RED leg -- the leg that proves the task is not already solved -- into a
fail-closed error instead of a 0.

So the denominator is PINNED. `reward = TOP_PASS / EXPECTED`, never over what
was observed. That gives up the free anti-shrink property equality had, so it
is bought back explicitly, twice over:

    sha256 lock    every graded `_test.go` must hash to what was measured
    count lock     the graded package must still hold the same number of
                   `_test.go` files, so a helper cannot be deleted to change
                   what compiles

and what remains -- a graded set that GREW -- is guarded directly
(`observed > EXPECTED` fails).

Two more spike facts are load-bearing here. Only TOP-LEVEL tests count: go's
`-json` reports `"Test":"TestX/sub"` for subtests, and counting those would push
the numerator past a denominator that only ever counted the parents. And the
graded run is offline by ENVIRONMENT (`GOPROXY=off`, `GOTOOLCHAIN=local`), not
by convention: a `go` directive in go.mod would otherwise try to download a
toolchain, which offline is a build failure and online is an unpinned compiler.
"""

from __future__ import annotations

from typing import ClassVar, Mapping

from . import base as B
from .base import DepWarmSpec, EnvSpec, GradedSet, ToolchainSpec

__all__ = ['GoPlugin', 'GO_VERSION', 'GOROOT', 'MODCACHE']

GO_VERSION = '1.26.5'
GOROOT = '/opt/go1.26'
MODCACHE = '/opt/go/pkg/mod'
GOCACHE = '/opt/go/cache'
MODWARM = '/opt/modwarm'
#: The warm build is the ONLY step allowed a proxy.
WARM_PROXY = 'https://proxy.golang.org,direct'
DEFAULT_TIMEOUT = '10m'


class GoPlugin(B.LangPlugin):
    """go1.26, module cache warmed in a separate image, pinned-denominator floor."""

    name: ClassVar[str] = 'go'
    toml_family: ClassVar[str] = 'A'
    floor_mode: ClassVar[str] = 'pinned-denominator'
    parser_backed: ClassVar[bool] = True

    go_version: ClassVar[str] = GO_VERSION
    goroot: ClassVar[str] = GOROOT
    timeout: ClassVar[str] = DEFAULT_TIMEOUT

    # --- axis 1 -----------------------------------------------------------

    def toolchain_spec(self) -> ToolchainSpec:
        """The pinned tarball, unpacked to a versioned prefix.

        Not `apt install golang`: the distro's go is whatever the base image's
        release froze, and the benchmark's whole premise is that two runs a year
        apart compile the same way.
        """
        install = f'''# go{self.go_version}, from the pinned tarball. The archive is unpacked to a
# VERSIONED prefix so that a second toolchain could coexist rather than
# overwrite -- silently overwriting is how a pin stops being a pin.
RUN set -eux; \\
    arch="$(uname -m)"; \\
    case "${{arch}}" in \\
      x86_64|amd64) goarch=amd64 ;; \\
      aarch64|arm64) goarch=arm64 ;; \\
      *) echo "unsupported architecture: ${{arch}}" >&2; exit 1 ;; \\
    esac; \\
    curl -fsSL "https://go.dev/dl/go{self.go_version}.linux-${{goarch}}.tar.gz" \\
      -o /tmp/go.tar.gz; \\
    mkdir -p {self.goroot}; \\
    tar -C {self.goroot} --strip-components=1 -xzf /tmp/go.tar.gz; \\
    rm -f /tmp/go.tar.gz; \\
    {self.goroot}/bin/go version'''
        return ToolchainSpec(
            base_image='harbor-base:local',
            install_block=install,
            env={
                'PATH': f'{self.goroot}/bin:/usr/local/bin:/usr/bin:/bin',
                'GOROOT': self.goroot,
                # `local` forbids go from fetching another toolchain for a
                # `go 1.2x` directive: offline that is a build failure, online
                # it is an unpinned compiler.
                'GOTOOLCHAIN': 'local',
                # The graded run is offline by environment, not by hope.
                'GOPROXY': 'off',
                'GOFLAGS': '-mod=mod',
                'GOMODCACHE': MODCACHE,
                'GOCACHE': GOCACHE,
                'CGO_ENABLED': '0',
            },
            workdir=B.WORKDIR,
        )

    # --- axis 2 -----------------------------------------------------------

    def dep_warm_spec(self) -> DepWarmSpec:
        """`go mod download all` over go.mod/go.sum ONLY, in a scratch directory.

        The repo is never copied into the warm image. A warm stage that has the
        sources has the answer, and `COPY --from=warm` would carry whatever it
        touched into the shipped image (invariant 7).
        """
        stage = f'''# Dependencies are warmed from the MANIFEST, never from the repo: a warm
# stage that can see the sources has the answer in a layer.
RUN cd {MODWARM} \\
 && GOPROXY={WARM_PROXY} GOFLAGS=-mod=mod GOMODCACHE={MODCACHE} \\
    go mod download all'''
        return DepWarmSpec(
            stage_block=stage,
            files_needed=('go.mod', 'go.sum'),
            copy_paths=((MODCACHE, MODCACHE),),
            manifest_dir=MODWARM,
        )

    # --- axes 3-6 ---------------------------------------------------------

    def render_test_sh(
        self,
        graded: GradedSet,
        *,
        expected: int | None = None,
        fingerprint: Mapping[str, str] | None = None,
        test_file_counts: Mapping[str, int] | None = None,
    ) -> str:
        expected = graded.expected if expected is None else int(expected)
        fingerprint = graded.fingerprint_sha256 if fingerprint is None else fingerprint
        packages = ' '.join(sorted(graded.packages)) or './...'
        run = _run_regex(graded.selectors)

        return '\n'.join([
            '#!/usr/bin/env bash',
            f'# Harbor verifier -- go, {expected} graded test(s).',
            '#',
            '# PINNED-DENOMINATOR floor (spike G1). A go panic aborts the whole test',
            '# binary, so a stubbed function produces one event and then silence:',
            '# the OBSERVED total is not a denominator. reward = passed/EXPECTED,',
            '# with the sha256 + file-count locks below as the anti-shrink half and',
            '# an explicit scope-GROWTH guard as the other direction.',
            '#',
            '# Harbor ignores this script\'s exit code; /logs/verifier/reward.json is',
            '# the single source of truth and is written on EVERY path.',
            '',
            'set -uo pipefail',
            '',
            f'REPO=${{REPO:-{B.WORKDIR}}}',
            B.reward_emitter_block(_LOGS_DEFAULT),
            _JSON_LOG,
            '',
            B.fail_closed_preamble(expected, compiled='0.0'),
            '',
            'cd "${REPO}" || fail "no ${REPO}"',
            '',
            B.fingerprint_gate_block(fingerprint, repo_var='${REPO}'),
            '',
            _count_lock_block(test_file_counts),
            '',
            f'echo "== go test: {expected} pinned test(s) in {packages} (offline) =="',
            '',
            '# -count=1 defeats the test cache: a cached PASS would grade the',
            '# PREVIOUS build, which after a restore is the carved one.',
            '# -json is the only output with a per-test verdict.',
            f'go test -short -count=1 -timeout={self.timeout} -json \\',
            f"    -run '{run}' \\",
            f'    {packages} > "${{JSON_LOG}}" 2>&1',
            'GO_STATUS=$?',
            '',
            _COMPILED_BLOCK,
            '',
            _TOPLEVEL_COUNT_BLOCK,
            '',
            'echo "== go exit=${GO_STATUS} top-level pass=${TOP_PASS} '
            'observed=${TOTAL}/${EXPECTED} compiled=${COMPILED} =="',
            '',
            B.floor_gate_block(
                self.floor_mode, expected,
                passed_var='TOP_PASS', total_var='TOTAL', compiled_var='COMPILED',
            ),
            'exit 0',
            '',
        ])

    def measure_test_sh(self, *, graded: GradedSet | None = None, **kwargs) -> str:
        """Phase 1: `-list` the selected tests and COUNT them. No floor exists yet.

        `-list` compiles the package but runs nothing, so the measured number is
        the size of the selection rather than the outcome of it -- which is
        exactly what a denominator has to be.
        """
        packages = ' '.join(sorted(graded.packages)) if graded else './...'
        run = _run_regex(graded.selectors) if graded and graded.selectors else '.*'
        return '\n'.join([
            '#!/usr/bin/env bash',
            '# Harbor MEASURE (phase 1) -- go. Floor-FREE by construction: it counts,',
            '# it asserts nothing. A floor cannot be enforced by the run that makes it.',
            '',
            'set -uo pipefail',
            '',
            f'REPO=${{REPO:-{B.WORKDIR}}}',
            'LIST=$(mktemp)',
            '',
            B.measure_emitter_block(_LOGS_DEFAULT),
            '',
            'cd "${REPO}" || { echo "no ${REPO}" >&2; exit 1; }',
            '',
            f"go test -count=1 -list '{run}' {packages or './...'} > \"${{LIST}}\" 2>&1",
            '',
            '# `-list` prints one test name per line plus a trailing `ok <pkg> <t>`',
            '# summary; only the names are the selection.',
            "NAMES=$(grep -E '^(Test|Example)' \"${LIST}\" | sort -u)",
            'TOTAL=$(printf \'%s\\n\' "${NAMES}" | grep -c . )',
            'GRADED=$(printf \'%s\\n\' "${NAMES}" | grep . | sed \'s/.*/"&"/\' '
            '| paste -sd, -)',
            '',
            'measure "${TOTAL}" "${GRADED}"',
            'exit 0',
            '',
        ])

    # --- axis 7 -----------------------------------------------------------

    def post_restore_block(self) -> str:
        """Drop cached TEST RESULTS after the oracle restores the source.

        `-count=1` already refuses a cached result at grade time; this makes the
        restore itself sound even if a future caller drops that flag.
        """
        return (
            '# Go caches test RESULTS by input hash. `-count=1` in test.sh is the\n'
            '# primary defence; clearing the cache here makes the restore sound on\n'
            '# its own rather than by depending on a flag in another file.\n'
            'go clean -testcache >/dev/null 2>&1 || true'
        )

    # --- the image --------------------------------------------------------

    def extra_dockerfile_blocks(self, env: EnvSpec) -> tuple[str, ...]:
        """Prove the toolchain is the pinned one and the cache is actually warm.

        There is deliberately no `go build` here. Compiling the carved repo at
        build time would leave object files carrying that package's EXPORT DATA
        in a layer (invariant 5); the honest full rebuild happens at grade time.
        """
        return (
            '# The toolchain is the pin, or the build fails now rather than',
            '# producing subtly different binaries later.',
            'RUN set -eux; \\',
            f'    go version | grep -q "go{self.go_version}"; \\',
            '    [ "$(go env GOTOOLCHAIN)" = "local" ]; \\',
            '    [ "$(go env GOPROXY)" = "off" ]; \\',
            f'    test -d {MODCACHE}; \\',
            f'    test "$(find {MODCACHE} -maxdepth 2 -type d | wc -l)" -gt 1; \\',
            '    echo "go toolchain pinned, module cache warm, proxy off"',
        )


# --------------------------------------------------------------------------
# bash fragments
# --------------------------------------------------------------------------

#: Overridable so the reward arithmetic can be exercised outside a container,
#: exactly as render_solve_sh's HARBOR_REPO/HARBOR_SOLUTION already are. In the
#: image nothing sets it and it resolves to harbor's own path.
_LOGS_DEFAULT = '${VERIFIER_DIR:-' + B.LOGS_DIR + '}'
_JSON_LOG = 'JSON_LOG=${VERIFIER_DIR}/go-test.json'

#: A top-level test is one whose `Test` field holds no `/`. Subtests report as
#: `TestX/case`, and counting those would push the numerator past a denominator
#: that only ever counted parents.
_TOP = r'"Test":"[^"/]*"'

_TOPLEVEL_COUNT_BLOCK = '\n'.join([
    '# TOP-LEVEL tests only. `"Test":"TestX/sub"` is a subtest; a package-level',
    '# event has no Test field at all. Both are excluded by requiring a Test',
    '# name that contains no slash.',
    f"""TOP_PASS=$(grep '"Action":"pass"' "${{JSON_LOG}}" | grep -c '{_TOP}')""",
    'TOP_PASS=${TOP_PASS:-0}',
    # `|` as the sed delimiter, because the class being captured contains `/`.
    'TOTAL=$(grep \'"Action":"run"\' "${JSON_LOG}" '
    '| sed -n \'s|.*"Test":"\\([^"/]*\\)".*|\\1|p\' '
    '| sort -u | grep -c .)',
    'TOTAL=${TOTAL:-0}',
])

_COMPILED_BLOCK = '\n'.join([
    '# COMPILED is its own bar: a task whose package does not build is a',
    '# different failure from one whose tests fail, and the reward schema keeps',
    '# them apart. go reports a build failure as a `[build failed]` summary line',
    "# or, since 1.24, a `build-fail` action in the -json stream.",
    'if grep -qE \'\\[build failed\\]|"Action":"build-fail"|^# \' "${JSON_LOG}"; then',
    '    COMPILED=0.0',
    'else',
    '    COMPILED=1.0',
    'fi',
])


def _run_regex(selectors) -> str:
    """`^(TestA|TestB)$`, sorted and anchored.

    Anchoring is not cosmetic: `-run TestA` is a substring match that also
    selects `TestAB`, which is scope growth the floor would then reject.
    """
    names = sorted(set(selectors or ()))
    if not names:
        raise B.LangError(
            'refusing to render a go test.sh with an empty -run: `go test` with '
            'no selection runs the WHOLE package, so the graded set would '
            'silently become the entire suite'
        )
    return '^(' + '|'.join(names) + ')$'


def _count_lock_block(counts: Mapping[str, int] | None) -> str:
    """Anti-shrink, package level: the graded package keeps its `_test.go` files.

    The sha256 gate pins the graded FILE. This pins the graded PACKAGE, so a
    solver cannot delete a helper `_test.go` to change what the graded file
    compiles against.
    """
    items = sorted(dict(counts or {}).items())
    if not items:
        return (
            '# no _test.go count lock pinned for this graded set (measure.py '
            'supplies\n# one; its absence is stated rather than silently skipped)'
        )
    lines = [
        '# Anti-shrink, package level. The sha256 gate above pins the graded',
        '# FILE; this pins the graded PACKAGE, so a helper _test.go cannot be',
        '# deleted to change what compiles.',
        'check_test_files() {',
        '    _n=$(find "${REPO}/$1" -maxdepth 1 -name \'*_test.go\' | grep -c .)',
        '    [ "${_n}" -eq "$2" ] || fail "$1 holds ${_n} _test.go file(s), '
        'pinned at $2"',
        '}',
    ]
    lines += [f"check_test_files '{rel}' {int(n)}" for rel, n in items]
    return '\n'.join(lines)


B.register(GoPlugin())
