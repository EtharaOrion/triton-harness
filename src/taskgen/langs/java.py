"""The java plugin: whole-suite grading, measured floor, gradle + JDK25, JUnit 5.

java-tamboui is the fourth whole-suite language after rust, c and cpp. The
carve is `tamboui-widgets/src/main/java/**/*.java` (85 files, the entire
`dev.tamboui.widgets` subsystem). Nothing linkable to the widgets remains, so
per-test selectors are meaningless and the graded surface collapses to "the
whole `:tamboui-widgets:test` suite" -- for tamboui-widgets that is 49 test
files that JUnit fans out into 69 suite files and 823 tests (measured once
against the intact tree; all three counts are stable under the fingerprint
lock).

FLOOR MODE: equality. Each JUnit test is a separate method invocation and a
failure in one does not abort the JVM; a compile failure or a crashed JVM
produces zero JUnit XML at all (caught by `SUITES > 0` ahead of the floor).
observed==EXPECTED is a real assertion -- the same argument as rust / c / cpp
(spike G1 applies only to Go's whole-binary-abort-on-panic semantics).

TOOLCHAIN: JDK 25 via `apt-get install openjdk-25-jdk-headless`. harbor-base
ships JDK 21 and gradle 9.2.1; settings.gradle.kts hard-fails on any launcher
JVM older than 25, so a newer JDK has to be installed here rather than assumed.
The reference Dockerfile confirms `openjdk-25-jdk-headless` is in Ubuntu noble
apt and is NOT one of the blackholed hosts, so it is reachable at BUILD time
(grading itself is `--network=none`). The install block is DELIBERATELY
byte-identical across warm / graded / measure so BuildKit reuses the JDK layer
across all three images (see the reference's own comment on this).

DEP-WARM: separate `FROM harbor-base:local AS warm` stage that COPYs the
CARVED repoctx (widget src/main/java is empty by construction), warms the
gradle-home cache via `:buildSrc:build` + a resolve-only init script that
`.resolve()`s every configuration in every project, then scrubs the cache of
anything derived from widget code and asserts nothing under
`$GRADLE_USER_HOME` names `dev/tamboui/widgets`. Only the scrubbed
`/opt/gradle-home` is COPYed forward into the graded stage; the warm stage's
other layers are not part of the shipped image (`docker save` reads only the
final target's layers, but the widget-absence assert is defence-in-depth
against a future stage-graph change). The resolve-only pass is Option A of
the Oracle's decision matrix; Options B (measure-side warmup) and C (warm
inside graded) were rejected as either invasive or impossible for a carved
tree that never sees the widget answer.

GRADER FINGERPRINT. `('tamboui-widgets/src/test/**',)` expands host-side at
generate time to every file under widgets' test root (49 .java + resources
/ metadata) and every path gets an sha256 baked inline via
`fingerprint_gate_block`. The reference asset instead ships a
`test-tree.sha256` file into `/opt/harbor-tooling` at image build time; the
inline per-file lock is strictly stronger -- a solver cannot swap one test
file for another of matching aggregate hash, and the fingerprint document
never lands on disk as a rewritable file inside the container.

NO REPO-MAGIC NUMBERS IN PLUGIN SOURCE (mostly). 823 (JUnit test methods)
threads through `graded.expected` -- the denominator measured host-side in
phase 1 by `measure.py` running the intact tree's `:tamboui-widgets:test`
once and summing `<testsuite tests=>` across every XML. The 49 test files
lock in via the fingerprint dict, so the plugin does not repeat that count
either. The one exception is `EXPECTED_SUITES = 69`: the JUnit `tests`
attribute counts leaf methods, so `SUITES == 69` is a structural check that
JUnit fanned out every declared class (a compile failure that skipped some
classes would drop the suite count without necessarily dropping the test
count if `@Disabled` masked them). It is threaded as a plugin ClassVar with
a docstring rather than sprinkled through the rendered bash.

BUILD PARALLELISM. Gradle honours its own `org.gradle.parallel=true` +
`--parallel` heuristics on both configuration and execution; the 8 GB
`memory_mb` in the reference task.toml sizes the JVM heap
(`org.gradle.jvmargs=-Xmx1g`) plus test-worker JVMs. No `--parallel N`
override needed here (unlike cpp, where cc1plus TU peak memory forced
--parallel 4).

DOCKERFILE INVARIANTS. `git init` is NOT synthesised (unlike python); the
base's ban stays in force. No vendored tarballs (unlike rust's wabt), no
`FROM warm` (COPY --from=warm only, per invariant 7). The warm stage's
`dev/tamboui/widgets` assert is the java-side analog of rust's
strings-target leak assert -- a fossil scan on the one place where a build
step could have left carved bytes.
"""

from __future__ import annotations

from typing import ClassVar, Mapping

from . import base as B
from .base import DepWarmSpec, EnvSpec, GradedSet, ToolchainSpec

__all__ = [
    'EXPECTED_SUITES',
    'GRADER_FINGERPRINT_GLOBS',
    'JAVA_HOME',
    'JavaPlugin',
    'TEST_COMMAND',
    'WIDGETS_MODULE',
]

#: JDK 25's ARM64 install path on Ubuntu noble (the harbor-base flavour). The
#: reference Dockerfile bakes the same literal.
JAVA_HOME = '/usr/lib/jvm/java-25-openjdk-arm64'

#: Where harbor-base already exports the Gradle home. Written into the
#: toolchain env dict for redundancy (base already exports it) so the plugin
#: source is self-describing.
GRADLE_USER_HOME = '/opt/gradle-home'

#: The graded module and its default test path. Only tamboui-widgets is
#: carved; every other module is intact in repoctx and configures normally.
WIDGETS_MODULE = 'tamboui-widgets'

#: The graded test command as prose. Rendered into instruction.md so the model
#: sees exactly what test.sh runs. `--rerun` defeats gradle's build-cache
#: stale-green (spike caught this returning a cached PASS with zero test runs).
#: `--offline` refuses the network; `--no-daemon` matches the reference and
#: keeps each container invocation self-contained.
TEST_COMMAND = (
    f'gradle --offline --no-daemon --console=plain :{WIDGETS_MODULE}:test --rerun'
)

#: What the plugin locks against tampering: every file under
#: tamboui-widgets/src/test/ (49 .java files plus any resources / metadata).
#: The whole test tree IS the grader; a solver who edited one to make the
#: suite trivially pass would trip the per-file sha256 check inline in
#: `render_test_sh` and score zero. Strictly stronger than the reference
#: asset's aggregate-hash file at /opt/harbor-tooling/test-tree.sha256.
GRADER_FINGERPRINT_GLOBS: tuple[str, ...] = (
    f'{WIDGETS_MODULE}/src/test/**',
)

#: The intact-tree suite count -- how many `TEST-*.xml` files JUnit writes for
#: `:tamboui-widgets:test` when every source is present. 49 test SOURCE files
#: fan out into 69 SUITE files (nested @Nested classes and parameterised
#: providers produce more than one <testsuite> per source). The equality floor
#: pins the test count (823) but not the suite count, so a compile failure
#: that dropped ONE nested class while its declared tests still counted as
#: @Disabled would sneak past. Gating on `SUITES == EXPECTED_SUITES` closes
#: that hole structurally.
#:
#: This is the one repo-magic number the plugin source declares as a literal.
#: 823 threads through `graded.expected` because measure.py can pin it in
#: `graded.lock.json`; the plan's measure schema (`{"tests_total":N,"graded":[]}`)
#: does not carry a second scalar, so pinning the suite count would need a new
#: primitive. It is safe as a ClassVar because it is a property of the pinned
#: test tree, not of the host: the fingerprint gate refuses to grade if any
#: pinned file changed, so `EXPECTED_SUITES` cannot drift under it.
EXPECTED_SUITES = 69

_LOGS_DEFAULT = '${VERIFIER_DIR:-' + B.LOGS_DIR + '}'
_MEASURE_LOGS_DEFAULT = '${MEASURE_DIR:-' + B.LOGS_DIR + '}'

#: The apt+gradle-properties toolchain install block. RAW string, byte-identical
#: across warm / graded / measure so BuildKit dedupes the ~700 MB JDK layer.
#: Uses LITERAL paths (not $JAVA_HOME) because ToolchainSpec.render emits the
#: install_block BEFORE the ENV lines, so shell expansion of $JAVA_HOME here
#: would resolve to empty.
_INSTALL_BLOCK = (
    '# JDK 25: settings.gradle.kts hard-fails on any launcher JVM older than\n'
    '# 25 ("This project ... requires Java 25+ JDK for building"). harbor-base\n'
    '# ships JDK 21 and gradle 9.2.1, so a newer JDK must be added here.\n'
    '# openjdk-25-jdk-headless 25.0.x comes from Ubuntu noble-updates and is\n'
    '# NOT one of the blackholed hosts; grading itself is --network=none, so\n'
    "# the install must land in a build-time layer before the leak gate.\n"
    '# Byte-identical between warm / graded / measure Dockerfiles so BuildKit\n'
    '# reuses this ~700 MB layer across all three images.\n'
    'RUN set -eux; \\\n'
    '    apt-get update -qq; \\\n'
    '    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \\\n'
    '        openjdk-25-jdk-headless; \\\n'
    '    rm -rf /var/lib/apt/lists/*; \\\n'
    f'    test -x {JAVA_HOME}/bin/javac\n'
    "# Point Gradle's daemon JVM at 25 too. settings.gradle.kts's\n"
    '# JavaVersion.current() gate reads the LAUNCHER JVM (the one running\n'
    '# gradle itself, not org.gradle.java.home), so PATH-precedence handles\n'
    '# that; org.gradle.java.home pins the COMPILE JVM for --release=8 too.\n'
    '# $GRADLE_USER_HOME is already exported by harbor-base at /opt/gradle-home;\n'
    '# the properties file travels forward as part of the cache COPY.\n'
    'RUN set -eux; \\\n'
    f'    mkdir -p {GRADLE_USER_HOME}; \\\n'
    f"    printf 'org.gradle.java.home=%s\\n' {JAVA_HOME} \\\n"
    f'        >> {GRADLE_USER_HOME}/gradle.properties'
)


class JavaPlugin(B.LangPlugin):
    """gradle 9 + JDK25 + JUnit5, whole-suite equality floor, measured denominator."""

    name: ClassVar[str] = 'java'
    #: java-hybrid rather than A/B because the reference task.toml declares
    #: no [verifier].command (harbor mounts tests/ at run time); the emitted
    #: task.toml follows suit so verify.py mounts the plugin-rendered test.sh.
    toml_family: ClassVar[str] = 'java-hybrid'
    floor_mode: ClassVar[str] = 'equality'
    parser_backed: ClassVar[bool] = False
    synthesizes_git: ClassVar[bool] = False

    #: See `emit.plan_carve`: whole-suite plugins can declare intact-tree
    #: globs to fingerprint. Empty for rust; ('tests/**','Makefile') for c;
    #: ('Tests/**',) for cpp; ('tamboui-widgets/src/test/**',) for java.
    grader_fingerprint_globs: ClassVar[tuple[str, ...]] = GRADER_FINGERPRINT_GLOBS

    test_command: ClassVar[str] = TEST_COMMAND
    widgets_module: ClassVar[str] = WIDGETS_MODULE
    expected_suites: ClassVar[int] = EXPECTED_SUITES
    java_home: ClassVar[str] = JAVA_HOME
    gradle_user_home: ClassVar[str] = GRADLE_USER_HOME

    # --- axis 1 -----------------------------------------------------------

    def toolchain_spec(self) -> ToolchainSpec:
        """JDK 25 apt install + gradle.properties, byte-identical across stages."""
        return ToolchainSpec(
            base_image='harbor-base:local',
            install_block=_INSTALL_BLOCK,
            env={
                'DEBIAN_FRONTEND': 'noninteractive',
                'GRADLE_USER_HOME': GRADLE_USER_HOME,
                'JAVA_HOME': JAVA_HOME,
                # ENV lines are sorted by ToolchainSpec.render, so PATH lands
                # after JAVA_HOME and can reference it. Bake JAVA_HOME's literal
                # anyway rather than $JAVA_HOME so a base image that unsets it
                # cannot silently break the prepend.
                'PATH': f'{JAVA_HOME}/bin:${{PATH}}',
            },
            workdir=B.WORKDIR,
        )

    # --- axis 2 -----------------------------------------------------------

    def dep_warm_spec(self) -> DepWarmSpec:
        """Resolve-only warm stage: no widget bytecode possible, gradle-home clean.

        `files_needed=()` because the framework's single-line COPY concat
        mangles directory layout for the ~30 build files gradle needs across
        modules and buildSrc. The stage_block below COPYs the whole carved
        repoctx instead -- which by construction has NO widget src/main/java
        files -- and drives Gradle through a resolve-only pass that never
        produces widget bytecode. The scrub + widget-absence assert at the
        end is the load-bearing leak proof.

        `copy_paths=(('/opt/gradle-home', '/opt/gradle-home'),)` makes the
        framework emit `COPY --from=warm /opt/gradle-home /opt/gradle-home`
        into the graded stage (invariant 7: COPY --from=warm only, never
        FROM warm).
        """
        workdir = B.WORKDIR
        # Kotlin init script defined inline. `<<'INIT_EOF'` disables shell
        # expansion so ${project.path} et al survive verbatim into the file.
        init_script_lines = [
            'allprojects {',
            '    tasks.register("harborResolveAll") {',
            '        doLast {',
            '            configurations.matching { it.isCanBeResolved }.forEach {',
            '                try { it.resolve() }',
            '                catch (e: Exception) {',
            '                    println("SKIP ${project.path} :: ${it.name} :: ${e.message}")',
            '                }',
            '            }',
            '        }',
            '    }',
            '}',
            'gradle.rootProject {',
            '    tasks.named("harborResolveAll").configure {',
            '        dependsOn(subprojects.map { it.tasks.named("harborResolveAll") })',
            '    }',
            '}',
        ]
        init_body = '\n'.join(init_script_lines)

        stage_block = '\n'.join([
            '# The carved repoctx into the warm stage. tamboui-widgets/src/main/java',
            '# is empty by construction (85 files carved), so no widget code can be',
            '# compiled here. Every other module is intact -- resolve-only needs the',
            '# whole project tree to CONFIGURE the build graph.',
            f'COPY --from={B.REPO_CONTEXT} repo/ {workdir}/',
            '',
            '# Assert the carve landed as expected: no widget main sources, buildSrc',
            '# convention plugins present, widgets test tree intact (49 files, per',
            '# the fingerprint the graded stage will lock).',
            'RUN set -eux; \\',
            f'    test "$(find {workdir}/{WIDGETS_MODULE}/src/main/java '
            "-name '*.java' 2>/dev/null | wc -l)\" -eq 0; \\",
            f'    test "$(find {workdir}/buildSrc/src/main '
            "-type f 2>/dev/null | wc -l)\" -gt 0",
            '',
            '# Resolve-only init script. `harborResolveAll` iterates every project',
            '# (including buildSrc) and calls .resolve() on every resolvable',
            '# configuration -- downloads every jar the later --offline run will',
            "# ask for, without compiling anything derived from widget sources.",
            "# `<<'HARBOR_INIT_EOF'` disables shell expansion so Kotlin's ${...}",
            '# survives into the file verbatim.',
            f'RUN cat > /tmp/harbor-resolve.init.gradle.kts '
            "<<'HARBOR_INIT_EOF'",
            init_body,
            'HARBOR_INIT_EOF',
            '',
            '# Warm the cache. Three passes matching the Oracle design:',
            "#   1) :buildSrc:assemble compiles buildSrc's Kotlin convention",
            "#      plugins into $GRADLE_USER_HOME's instrumented-jar cache. NOT",
            "#      :buildSrc:build (which triggers :buildSrc:compileTestJava,",
            "#      whose testCompileClasspath resolves junit-jupiter WITHOUT a",
            "#      version -- java-tamboui's buildSrc references it as a plain",
            "#      dependency instead of via a BOM, and gradle fails the whole",
            "#      task rather than skip the version-less coord).",
            "#   2) harborResolveAll .resolve()s every configuration in every",
            "#      subproject (compileClasspath, testCompileClasspath,",
            "#      testRuntimeClasspath, annotationProcessor, ...) so no config",
            "#      the graded run touches is unresolved.",
            "#   3) --stop kills the daemon before scrub so no daemon-owned lock",
            "#      files under $GRADLE_USER_HOME/daemon would survive.",
            'RUN set -eux; \\',
            f'    cd {workdir}; \\',
            '    gradle --version; \\',
            '    gradle --no-daemon --console=plain :buildSrc:assemble; \\',
            '    gradle --no-daemon --console=plain '
            '-I /tmp/harbor-resolve.init.gradle.kts harborResolveAll; \\',
            '    gradle --no-daemon --stop || true; \\',
            '    rm -f /tmp/harbor-resolve.init.gradle.kts',
            '',
            '# Scrub the cache. build-cache-1 holds task OUTPUTS keyed by input',
            '# hash -- for :tamboui-widgets:compileJava that would be the answer',
            "# outright. Dropped wholesale; the graded run's inputs (carved) are",
            '# gone anyway. daemon/workers/notifications/kotlin-profile are scratch.',
            'RUN set -eux; \\',
            f'    rm -rf {GRADLE_USER_HOME}/caches/build-cache-1 \\',
            f'           {GRADLE_USER_HOME}/daemon \\',
            f'           {GRADLE_USER_HOME}/workers \\',
            f'           {GRADLE_USER_HOME}/notifications \\',
            f'           {GRADLE_USER_HOME}/kotlin-profile',
            '',
            '# The load-bearing leak assert. Nothing anywhere under gradle-home',
            "# may name 'dev/tamboui/widgets': bytecode holds FQNs as UTF8",
            '# constants, jars hold entry names in the central directory, and',
            '# either would be a decompile-back-to-source leak of the worst kind.',
            '# The resolve-only path above should never produce such artifacts',
            '# (no widget sources to compile), but this assert stops a future',
            '# stage-graph change from silently smuggling them through.',
            'RUN set -eu; \\',
            f'    hits=$(find {GRADLE_USER_HOME} '
            "-path '*dev/tamboui/widgets*' -print 2>/dev/null || true); \\",
            '    if [ -n "$hits" ]; then \\',
            f'        echo "LEAK: {GRADLE_USER_HOME} holds dev.tamboui.widgets '
            'artifacts:" >&2; \\',
            '        echo "$hits" >&2; \\',
            '        exit 1; \\',
            '    fi; \\',
            f'    echo "ASSERT: {GRADLE_USER_HOME} carries no dev.tamboui.widgets '
            'symbols or paths"',
        ]) + '\n'

        return DepWarmSpec(
            stage_block=stage_block,
            files_needed=(),
            copy_paths=((f'{GRADLE_USER_HOME}', f'{GRADLE_USER_HOME}'),),
        )

    # --- axes 3-6 ---------------------------------------------------------

    def pre_leakgate_blocks(self, env: EnvSpec) -> tuple[str, ...]:
        """The offline-sufficiency proof + project-cache scrub. Mandatory.

        Runs AFTER the carved-tree COPY and the warm-cache COPY but BEFORE the
        leak gate. If the warm-stage resolve missed a jar, this block fails
        the whole IMAGE BUILD with a specific "Could not resolve ..." message
        naming the missing artifact, rather than silently deferring the
        failure to the graded --offline run where it would surface as a
        compile error inside a black-box container.

        The project-cache scrub at the end (`rm -rf .gradle build */build`)
        resets Gradle's per-project cache so the graded run is genuinely
        from-scratch offline -- a stale-cached-artifact PASS after the
        oracle restore would otherwise be a subtle false green.
        """
        workdir = env.workdir
        proof = '\n'.join([
            '# Offline-sufficiency proof: prove the warm cache covers every',
            '# configuration the graded :tamboui-widgets:test task needs, at',
            '# BUILD time (network refused via --offline), so a genuine cache',
            '# gap fails the image build rather than surfacing later as a',
            '# black-box compile error inside the graded container.',
            'RUN set -eux; \\',
            f'    cd {workdir}; \\',
            '    for cfg in testRuntimeClasspath testCompileClasspath '
            'compileClasspath annotationProcessor testAnnotationProcessor; do \\',
            '        echo "=== --offline :dependencies --configuration $cfg ==="; \\',
            '        gradle --offline --no-daemon --console=plain '
            f':{WIDGETS_MODULE}:dependencies --configuration "$cfg" \\',
            '            > "/tmp/deps-${cfg}.log" 2>&1 \\',
            '            || { tail -60 "/tmp/deps-${cfg}.log" >&2; \\',
            '                 echo "OFFLINE PROOF FAILED for ${cfg}" >&2; \\',
            '                 exit 1; }; \\',
            '        if grep -q "Could not resolve" "/tmp/deps-${cfg}.log"; then \\',
            '            grep "Could not resolve" "/tmp/deps-${cfg}.log" >&2; \\',
            '            echo "warm cache is missing jars for ${cfg}" >&2; \\',
            '            exit 1; \\',
            '        fi; \\',
            '        rm -f "/tmp/deps-${cfg}.log"; \\',
            '    done; \\',
            '    gradle --no-daemon --stop || true; \\',
            '    echo "OFFLINE PROOF OK: warm cache covers every graded configuration"',
        ])

        # A separate RUN so the daemon-stop from the proof commits before the
        # rm-rf touches .gradle.
        reset = '\n'.join([
            '# Project-cache reset: ./.gradle and per-project build/ hold state',
            "# from the resolve above (dependency resolution results and the",
            "# offline-proof's transient outputs). Wiping them ensures the graded",
            '# run is genuinely from-scratch offline; a stale cached artifact PASS',
            "# after the oracle restore would otherwise be a subtle false green.",
            'RUN set -eux; \\',
            f'    cd {workdir}; \\',
            '    find . -path \'*/src/*\' -prune -o -type d -name build -prune -print0 \\',
            '        | xargs -0 -r rm -rf; \\',
            '    rm -rf .gradle',
        ])

        return (proof, '', reset)

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
        results_dir = f'{WIDGETS_MODULE}/build/test-results/test'

        # Python heredoc for the JUnit XML tally. `<<'PY_EOF'` disables shell
        # expansion so nothing in the parser needs escaping.
        xml_parser_lines = [
            "import glob, os, sys, xml.etree.ElementTree as ET",
            "results = sys.argv[1]",
            "suites = tests = failures = errors = skipped = 0",
            "for path in sorted(glob.glob(os.path.join(results, 'TEST-*.xml'))):",
            "    try:",
            "        root = ET.parse(path).getroot()",
            "    except ET.ParseError:",
            "        continue",
            "    suites += 1",
            "    tests += int(root.get('tests', 0))",
            "    failures += int(root.get('failures', 0))",
            "    errors += int(root.get('errors', 0))",
            "    skipped += int(root.get('skipped', 0))",
            "print(f'{suites} {tests} {failures} {errors} {skipped}')",
        ]

        return '\n'.join([
            '#!/usr/bin/env bash',
            f'# Harbor verifier -- java (equality floor, {expected} JUnit tests '
            f'across {EXPECTED_SUITES} suites).',
            '#',
            '# EQUALITY floor. Each JUnit test is a separate method invocation,',
            '# a per-test failure does not abort the JVM, and a compile failure',
            '# produces zero JUnit XML (caught by `SUITES > 0` ahead of the floor).',
            f'# The denominator ({expected}) was measured once against the intact',
            '# tree in phase 1 (measure.py) and pinned in graded.lock.json.',
            '#',
            "# Harbor ignores this script's exit code; /logs/verifier/reward.json",
            '# is the single source of truth and is written on EVERY path.',
            '',
            'set -uo pipefail',
            '',
            f'REPO=${{REPO:-{B.WORKDIR}}}',
            f'RESULTS={results_dir}',
            B.reward_emitter_block(_LOGS_DEFAULT),
            'GRADLE_LOG="${VERIFIER_DIR}/gradle-test.log"',
            'TREE_CHECK_LOG="${VERIFIER_DIR}/test-tree-check.log"',
            '',
            '# COMPILED is set to 1.0 only after JUnit XML appears in THIS run.',
            '# Kept as a shell variable so the fail-closed helper quotes the',
            '# current value on every early exit -- a build that failed halfway',
            '# must not be reported as compiled=1 by an earlier optimistic setting.',
            'COMPILED=0.0',
            '',
            B.fail_closed_preamble(expected, compiled='"${COMPILED}"'),
            '',
            'cd "${REPO}" || fail "no ${REPO}"',
            '',
            f'EXPECTED_SUITES={EXPECTED_SUITES}',
            '',
            '# --- integrity guards -----------------------------------------------',
            '# tamboui-widgets/src/test IS the grader. The per-file lock below',
            '# was captured host-side from the intact tree at generate time and',
            '# refuses to grade if any pinned file changed. Strictly stronger',
            "# than the reference asset's aggregate-hash file at",
            '# /opt/harbor-tooling/test-tree.sha256.',
            B.fingerprint_gate_block(fingerprint, repo_var='${REPO}'),
            '',
            '# --- clean previous artifacts ---------------------------------------',
            '# JUnit XML from a previous run in the same container would let',
            "# gradle's build cache report green with 0 regenerated code.",
            '# --rerun below defeats task-level caching; the explicit rm handles',
            '# the file-level cache that --rerun does not touch.',
            'rm -rf "${RESULTS}"',
            '',
            'echo "VERIFIER: $(java -version 2>&1 | head -1)"',
            'echo "VERIFIER: $(gradle --version 2>&1 | grep -E \'^Gradle\' | head -1)"',
            '',
            '# --- graded run -----------------------------------------------------',
            '# --offline: refuse the network; --no-daemon: no long-lived daemon',
            '# owning stale state; --rerun: defeat gradle build-cache stale-green.',
            f'{test_cmd} > "${{GRADLE_LOG}}" 2>&1',
            'STATUS=$?',
            'tail -40 "${GRADLE_LOG}"',
            'echo "VERIFIER: gradle exit=${STATUS}"',
            '',
            '# --- parse JUnit XML ------------------------------------------------',
            "# The XML under ${RESULTS} is the source of truth, NOT gradle's",
            '# console summary. Sum tests/failures/errors/skipped across every',
            "# <testsuite>; passed = tests - failures - errors - skipped.",
            "TALLY=$(python3 - \"${RESULTS}\" <<'PY_EOF'",
            *xml_parser_lines,
            'PY_EOF',
            ')',
            'if [ -z "${TALLY}" ]; then',
            '    echo "VERIFIER: could not read JUnit XML" >&2',
            '    emit 0.0 0 "${EXPECTED}" 0.0 0.0',
            '    exit 0',
            'fi',
            'read -r SUITES TESTS FAILURES ERRORS SKIPPED <<EOF',
            '${TALLY}',
            'EOF',
            '',
            'SUITES=${SUITES:-0}',
            'TESTS=${TESTS:-0}',
            'FAILURES=${FAILURES:-0}',
            'ERRORS=${ERRORS:-0}',
            'SKIPPED=${SKIPPED:-0}',
            '',
            'echo "VERIFIER: suites=${SUITES} tests=${TESTS} '
            'failures=${FAILURES} errors=${ERRORS} skipped=${SKIPPED}"',
            '',
            '# No XML at all means the test task never ran a single test, which',
            "# for this project means main or test sources failed to compile.",
            'if [ "${SUITES}" -eq 0 ]; then',
            '    echo "VERIFIER: no JUnit XML produced -- '
            'compilation failed, tests cannot run" >&2',
            '    emit 0.0 0 "${EXPECTED}" 0.0 0.0',
            '    exit 0',
            'fi',
            '',
            '# Tests actually ran, therefore both main and test sources compiled.',
            'COMPILED=1.0',
            '',
            '# --- structural anti-gaming gates -----------------------------------',
            "# JUnit's `tests` attribute counts leaf methods (the pinned EXPECTED),",
            '# and `SUITES == EXPECTED_SUITES` checks that every declared suite fanned',
            '# out. A partial-compile that skipped one nested @Nested class while its',
            '# tests were @Disabled would drop the suite count without necessarily',
            "# dropping the test count -- this gate catches that shape.",
            '[ "${SUITES}" -eq "${EXPECTED_SUITES}" ] \\',
            '    || fail "suite count moved: got ${SUITES}, expected '
            '${EXPECTED_SUITES} -- the graded suite set is not what was measured"',
            '',
            '# The graded pass count is (tests - failures - errors - skipped).',
            '# A @Disabled test is NOT a passed test: subtracting skips from the',
            '# denominator would let a solver @Disable their way to a perfect',
            "# score on a shrunken suite. The denominator stays the pinned",
            "# EXPECTED; a skipped test is simply a test that did not pass.",
            'PASSED=$((TESTS - FAILURES - ERRORS - SKIPPED))',
            '[ "${PASSED}" -ge 0 ] \\',
            '    || fail "nonsensical tally: ${PASSED} passed of ${EXPECTED}"',
            'TOTAL=${TESTS}',
            '',
            'echo "VERIFIER: PASSED=${PASSED}/${EXPECTED} (tests=${TESTS} '
            'failures=${FAILURES} errors=${ERRORS} skipped=${SKIPPED})"',
            '',
            B.floor_gate_block(
                self.floor_mode, expected,
                passed_var='PASSED', total_var='TOTAL', compiled_var='COMPILED',
            ),
            '',
            "# BINARY additionally requires gradle exit==0 AND zero failures/errors:",
            "# a solver who patched around a runtime failure until the test suite",
            "# tolerated it (via a JUnit assumption or a caught Throwable) would",
            "# clear PASSED == EXPECTED without a clean exit.",
            'if [ "${BINARY}" = "1.0" ]; then',
            '    if [ "${STATUS}" -ne 0 ] || [ "${FAILURES}" -ne 0 ] '
            '|| [ "${ERRORS}" -ne 0 ]; then',
            '        BINARY=0.0',
            '    fi',
            'fi',
            '',
            '# Extend reward.json with sub-counts so the shape matches the',
            '# reference grader (verify.py reads the 5 canonical keys and',
            '# ignores extras).',
            'printf \'{"reward": %s, "tests_passed": %s, "tests_total": %s, '
            '"binary": %s, "compiled": %s, '
            '"suites": %s, "tests_declared": %s, "tests_failed": %s, '
            '"tests_errored": %s, "tests_skipped": %s, "gradle_exit": %s}\\n\' \\',
            '    "${REWARD}" "${PASSED}" "${EXPECTED}" "${BINARY}" "${COMPILED}" \\',
            '    "${SUITES}" "${TESTS}" "${FAILURES}" '
            '"${ERRORS}" "${SKIPPED}" "${STATUS}" \\',
            '    > "${VERIFIER_DIR}/reward.json"',
            'echo "reward.json (extended) = $(cat "${VERIFIER_DIR}/reward.json")"',
            '',
            'exit 0',
            '',
        ])

    def measure_test_sh(self, *, graded: GradedSet | None = None, **kwargs) -> str:
        """Phase 1: run the WHOLE :tamboui-widgets:test against the intact tree.

        Floor-FREE by construction. `measure` writes tests_total = the summed
        `<testsuite tests=>` across every produced XML; a build that broke
        halfway registers as 0 (or a shrunken count) and
        `parse_measure_json` rejects zero.

        Runs OFFLINE because the measure Dockerfile has already warmed the
        cache at build time (see `render_measure_dockerfile`). --no-daemon
        matches the shipped test.sh.
        """
        del graded, kwargs
        results_dir = f'{WIDGETS_MODULE}/build/test-results/test'
        xml_parser_lines = [
            "import glob, os, sys, xml.etree.ElementTree as ET",
            "results = sys.argv[1]",
            "suites = tests = 0",
            "for path in sorted(glob.glob(os.path.join(results, 'TEST-*.xml'))):",
            "    try:",
            "        root = ET.parse(path).getroot()",
            "    except ET.ParseError:",
            "        continue",
            "    suites += 1",
            "    tests += int(root.get('tests', 0))",
            "print(f'{suites} {tests}')",
        ]
        return '\n'.join([
            '#!/usr/bin/env bash',
            '# Harbor MEASURE (phase 1) -- java. Floor-FREE by construction; the',
            '# pinned denominator is what THIS run measures against the intact',
            '# tree with the pre-warmed cache from the measure Dockerfile.',
            '',
            'set -uo pipefail',
            '',
            f'REPO=${{REPO:-{B.WORKDIR}}}',
            f'RESULTS={results_dir}',
            B.measure_emitter_block(_MEASURE_LOGS_DEFAULT),
            'MEASURE_LOG="${MEASURE_DIR}/gradle-test.log"',
            '',
            'cd "${REPO}" || { echo "no ${REPO}" >&2; measure 0 \'\'; exit 0; }',
            '',
            'rm -rf "${RESULTS}"',
            '',
            '# --rerun defeats gradle build-cache stale-green here too; the',
            "# measure image's Dockerfile already ran the suite once at build",
            '# time (that is how the cache got warm), so without --rerun this',
            '# would hit the cache and report the same numbers without running.',
            f'{TEST_COMMAND} > "${{MEASURE_LOG}}" 2>&1 || true',
            'tail -40 "${MEASURE_LOG}"',
            '',
            "TALLY=$(python3 - \"${RESULTS}\" <<'PY_EOF'",
            *xml_parser_lines,
            'PY_EOF',
            ')',
            'read -r SUITES TESTS <<EOF',
            '${TALLY:-0 0}',
            'EOF',
            'SUITES=${SUITES:-0}',
            'TESTS=${TESTS:-0}',
            'echo "MEASURE: suites=${SUITES} tests=${TESTS}"',
            '',
            'measure "${TESTS}" \'\'',
            'exit 0',
            '',
        ])

    # --- axis 7 -----------------------------------------------------------

    def post_restore_block(self) -> str:
        """Invalidate stale gradle build output so GREEN rebuilds honestly.

        The `--rerun` in test.sh handles task-level caching, but leftover
        artifacts under `build/` from a previous partial run would let a
        JUnit XML from before the oracle restore linger on disk. The scrub
        keeps the post-restore state predictable.
        """
        return (
            'rm -rf "${REPO}/tamboui-widgets/build" '
            '"${REPO}/tamboui-widgets/.gradle" '
            '"${REPO}/.gradle"'
        )

    # --- axis 8 + the image ----------------------------------------------

    def render_measure_dockerfile(self, env: EnvSpec) -> str:
        """The stripped Dockerfile for the never-ship measure image (phase 1).

        Same toolchain (JDK25 install byte-identical to the graded one for
        BuildKit layer reuse) plus a build-time warm-and-run of
        :tamboui-widgets:test against the INTACT tree. That single run does
        double duty: it downloads every dependency the graded --offline run
        will need (network is allowed at BUILD time), and it produces the
        JUnit XML the measure script can re-count later at container time
        with `--offline --rerun`.

        NO leak gate, NO tripwire scan, NO carve-metadata assert (intact-tree
        exemption -- all three would fire by construction on the intact tree).
        `measure_image_tag` marks the image as never-ship and `measure.py`
        deletes it in a finally block.
        """
        base_image = self.toolchain_spec().base_image
        toolchain = self.toolchain()
        return '\n'.join([
            '# syntax=docker/dockerfile:1.7',
            f'# Harbor MEASURE image -- {env.repo_name} ({self.name}). NEVER SHIP.',
            '#',
            '# Built by measure.py phase 1 to count the intact JUnit suite.',
            '# Contains the intact tree by construction -- an escaped measure',
            '# image is not a partial leak, it is the whole answer.',
            "# `measure_image_tag` marks it as never-ship and `measure.py`",
            '# deletes it in a finally block.',
            '',
            f'FROM {base_image} AS measure',
            '',
            toolchain,
            '',
            '# no separate warm stage (measure warms its own cache below)',
            '',
            f'# The measure phase points {env.repo_context} at the INTACT repo',
            '# directly, not a staging tree, so there is no repo/ prefix to copy',
            '# from. That is the only structural difference from the shipped',
            '# Dockerfile.',
            f'COPY --from={env.repo_context} . {env.workdir}/',
            f'RUN mkdir -p {env.logs_dir}',
            '',
            '# --- warm-and-run against the intact tree ---------------------------',
            '# Network is allowed at BUILD time here (measure.py builds the',
            '# measure image with the host default networking), so this single',
            "# gradle test run downloads every dependency the graded --offline",
            '# run will need and produces the JUnit XML the measure script',
            '# re-counts later at container time with --offline --rerun.',
            'RUN set -eux; \\',
            f'    cd {env.workdir}; \\',
            f'    gradle --no-daemon --console=plain :{WIDGETS_MODULE}:test; \\',
            '    gradle --no-daemon --stop || true',
            '',
            '# --- measure script (COPYed, not carved-tree, so lives in a layer) ---',
            f'COPY measure.sh {env.tests_dir}/measure.sh',
            f'RUN chmod 0555 {env.tests_dir}/measure.sh',
            '',
        ])


B.register(JavaPlugin())
