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

DEP-WARM: separate `FROM harbor-base-java AS warm` stage that COPYs the
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

GRADER FINGERPRINT. `grader_fingerprint_globs_for` narrows to the CARVED
module's own test root (`tamboui-widgets/src/test/**`), derived from the carve
rather than declared, so a monorepo's other modules do not get locked with
files no graded task can reach. It expands host-side at generate time to every
file under that root (49 .java + resources / metadata) and every path gets an
sha256 baked inline via `fingerprint_gate_block`. The reference asset instead
ships a `test-tree.sha256` file into `/opt/harbor-tooling` at image build time;
the inline per-file lock is strictly stronger -- a solver cannot swap one test
file for another of matching aggregate hash, and the fingerprint document
never lands on disk as a rewritable file inside the container.

NO REPO CONSTANTS IN THE RENDERER. 823 (JUnit test methods) threads through
`graded.expected` -- measured host-side in phase 1 by `measure.py` running the
intact tree's graded task once and summing `<testsuite tests=>` across every
XML. Everything else the two scripts and the warm stage used to hardcode about
java-tamboui -- the module, the graded gradle task, the JUnit results
directory, the buildSrc convention build, the stale-output wipe, and the 69
suite count -- is now `DepPlan.harness`, and `JAVA_TAMBOUI_HARNESS` is
java-tamboui's own answer to it (the fallback is not a guess; it is the same
bytes under a name).

THE SUITE COUNT IS MEASURED, NOT DECLARED. `SUITES == 69` was the structural
net that caught a graded run in which JUnit failed to fan out every declared
class; a shrunken denominator is SELF-CONSISTENT at the wrong number, so
measure, the equality floor and RED/GREEN would all pass while part of the suite
went ungraded. The literal is replaced by an INDEPENDENT host-side count:
`emit._java_grader_metadata` counts JUnit test CLASSES in the carved module's
own test sources, brace-depth-aware so a `@Nested` class counts as the separate
`TEST-*.xml` JUnit writes for it, and refuses to answer rather than answer low
(see `_junit_suite_classes`). That number -- not a plugin constant and not a
resolver's guess -- is what the shipped `EXPECTED_SUITES` gate asserts.

It is NOT a plan slot, and that is a leak-boundary consequence rather than a
preference: the count depends on class structure INSIDE the test files, the
resolver is shown test PATHS only, and a slot whose correct value is
underivable from a resolver's inputs is a slot that can only be guessed. The
scope cross-check that IS answerable from paths -- which module the graded task
and its report directory belong to -- stays plan-fed and is enforced against the
carve by `_assert_scope_agrees`.

THE LEAK SCAN'S SUBJECT COMES FROM THE CARVE. The warm stage asserts that
nothing under `$GRADLE_USER_HOME` names the carved package namespace. That
namespace is derived from the carved relpaths host-side (`CarveFacts`), never
from the plan: a namespace a model supplied would be a fossil scan whose
subject was guessed, and a wrong guess passes vacuously on every repo. A carve
whose package root cannot be determined is REFUSED rather than scanned for
nothing.

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
from .base import CarveFacts, DepWarmSpec, EnvSpec, GradedSet, ToolchainSpec

__all__ = [
    'BAKED_CAPABILITIES',
    'BASE_IMAGE',
    'GRADER_FINGERPRINT_GLOBS',
    'Harness',
    'JAVA_HOME',
    'JAVA_MEASURE_DEP_PLAN',
    'JAVA_TAMBOUI_CARVE',
    'JAVA_TAMBOUI_HARNESS',
    'JAVA_VERSION',
    'JavaPlugin',
    'MEASURE_NO_WARM_COMMENT',
    'REQUIRED_BUILD_FLAGS',
    'REQUIRED_PLAN_SLOTS',
    'REQUIRED_TEST_INVOCATION_KEYS',
    'TEST_COMMAND',
    'TEST_SOURCE_ROOT',
    'module_dir_of_carve',
    'package_roots_of_carve',
    'read_harness',
]

#: The JDK the base bakes and this task selects. settings.gradle.kts hard-fails
#: on any launcher JVM older than 25, so the pin cannot go below it.
JAVA_VERSION = 'temurin-25.0.4+7.0.LTS'

#: Where mise unpacks every JDK it installs. Named separately from `JAVA_HOME`
#: because `render_gap` derives the home of the JDK a PLAN selected, which is
#: not necessarily the one this module's default pins.
JAVA_INSTALL_ROOT = '/opt/mise/installs/java'

#: Derived from the mise install rather than a distro path. The old literal
#: `/usr/lib/jvm/java-25-openjdk-arm64` does not exist on amd64, so it pinned the
#: whole pipeline to one architecture.
JAVA_HOME = f'{JAVA_INSTALL_ROOT}/{JAVA_VERSION}'

#: Per-language base carrying temurin 17/21/25 and gradle, so no task build has
#: to apt-install a ~700MB JDK.
BASE_IMAGE = '426628337772.dkr.ecr.ap-south-2.amazonaws.com/triton/base-java@sha256:41679c33cc35b89771757cbfac358c0e678f3da165505fed52880ffaf7d8d1d1'

#: Where harbor-base already exports the Gradle home. Written into the
#: toolchain env dict for redundancy (base already exports it) so the plugin
#: source is self-describing.
GRADLE_USER_HOME = '/opt/gradle-home'

#: java-tamboui's own harness as a `DepPlan.harness` section -- the fixed point
#: of the refactor: rendered, it reproduces the scripts and the warm stage this
#: plugin used to hardcode byte for byte. Every value is a fact about the
#: java-tamboui REPOSITORY, which is exactly why none of it belongs here.
#:
#: The SUITE COUNT is deliberately absent: how many `TEST-*.xml` files JUnit
#: writes depends on how many test CLASSES the sources declare (a `@Nested`
#: class gets its own report, which is why java-tamboui's 49 test files produce
#: 69 reports). That is a fact about test BODIES, and the resolver is shown test
#: PATHS only -- so asking it would be asking for something its inputs cannot
#: answer. It is counted HOST-SIDE instead, by `emit._java_grader_metadata`.
JAVA_TAMBOUI_HARNESS: tuple[tuple[str, HarnessValue], ...] = (
    ('buildsrc_path', ':buildSrc'),
    ('project_label', 'java-tamboui'),
    ('results_dir', 'tamboui-widgets/build/test-results/test'),
    ('stale_paths', (
        'tamboui-widgets/build', 'tamboui-widgets/.gradle', '.gradle',
    )),
    ('test_task', ':tamboui-widgets:test'),
)

#: java-tamboui's own CARVE, for the callers that hold no carve of their own
#: (`render_dockerfile(EnvSpec(repo_name=...))`, every test written before the
#: warm stage read one). Not a default a real generate can reach:
#: `emit._java_carve_facts` derives these from the carved relpaths and REFUSES
#: when it cannot, so the fossil scan a shipped image runs always names the
#: namespace that was actually removed from THAT repo.
JAVA_TAMBOUI_CARVE = CarveFacts(
    root='tamboui-widgets/src/main/java',
    package_roots=('dev/tamboui/widgets',),
    file_count=85,
    grader_source_count=49,
)

#: The graded test command as argv. Rendered into instruction.md so the model
#: sees exactly what test.sh runs. `--rerun` defeats gradle's build-cache
#: stale-green (spike caught this returning a cached PASS with zero test runs).
#: `--offline` refuses the network; `--no-daemon` matches the reference and
#: keeps each container invocation self-contained.
_TAMBOUI_TEST_ARGV: tuple[str, ...] = (
    'gradle', '--offline', '--no-daemon', '--console=plain',
    ':tamboui-widgets:test', '--rerun',
)

#: What the plugin locks against tampering, when it has no carve to narrow to:
#: every file under the graded module's `src/test/` (49 .java files plus any
#: resources / metadata). The whole test tree IS the grader; a solver who edited
#: one to make the suite trivially pass would trip the per-file sha256 check
#: inline in `render_test_sh` and score zero. Strictly stronger than the
#: reference asset's aggregate-hash file at /opt/harbor-tooling/test-tree.sha256.
#:
#: `grader_fingerprint_globs_for` narrows this to the CARVED module, because a
#: monorepo's other modules have test trees no graded task can reach and locking
#: those pins files the verifier never runs.
GRADER_FINGERPRINT_GLOBS: tuple[str, ...] = ('*/src/test/**', 'src/test/**')

_LOGS_DEFAULT = '${VERIFIER_DIR:-' + B.LOGS_DIR + '}'
_MEASURE_LOGS_DEFAULT = '${MEASURE_DIR:-' + B.LOGS_DIR + '}'

#: The apt+gradle-properties toolchain install block. RAW string, byte-identical
#: across warm / graded / measure so BuildKit dedupes the ~700 MB JDK layer.
#: Uses LITERAL paths (not $JAVA_HOME) because ToolchainSpec.render emits the
#: install_block BEFORE the ENV lines, so shell expansion of $JAVA_HOME here
#: would resolve to empty.
_INSTALL_BLOCK = (
    '# The base bakes temurin 17/21/25 and gradle; select one rather than\n'
    '# apt-installing a JDK on every task build.\n'
    f'RUN set -eux; mise use -g java@{JAVA_VERSION}\n'
    '# profile.d is sourced in sorted order and the base re-exports its own PATH\n'
    '# ahead of the mise shims, so only a zz- file keeps the pin in front for the\n'
    "# login shells harbor's test.sh and solve.sh actually run as.\n"
    'RUN set -eux; \\\n'
    '    printf \'export PATH="/opt/mise/shims:$PATH"\\n\' > /etc/profile.d/zz-harbor-toolchain-pin.sh; \\\n'
    '    chmod 0644 /etc/profile.d/zz-harbor-toolchain-pin.sh\n'
    'RUN set -eux; \\\n'
    '    v="$(bash -lc \'java -version 2>&1 | head -1\')"; \\\n'
    '    case "$v" in \\\n'
    '        *\\"25.*) echo "TOOLCHAIN PIN OK (login shell): $v" ;; \\\n'
    '        *) echo "TOOLCHAIN PIN FAILED (login shell): got $v want 25.x" >&2; exit 42 ;; \\\n'
    '    esac\n'
    "# org.gradle.java.home pins the COMPILE JVM; the launcher JVM comes from PATH.\n"
    'RUN set -eux; \\\n'
    f'    mkdir -p {GRADLE_USER_HOME}; \\\n'
    f"    printf 'org.gradle.java.home=%s\\n' {JAVA_HOME} \\\n"
    f'        >> {GRADLE_USER_HOME}/gradle.properties'
)

#: The measure image's dep-warm slot. java warms its own gradle cache inside the
#: measure image (see `render_measure_dockerfile`), so there is no separate warm
#: stage to COPY from and this comment IS the slot's content.
MEASURE_NO_WARM_COMMENT = '# no separate warm stage (measure warms its own cache below)'

#: The JDK majors the base image bakes, as the gap states them. A base that
#: stopped shipping one of these would still render, which is why the pin assert
#: below is on the SELECTED version rather than on this list.
BAKED_JDKS = '17/21/25'

#: What the base image already provides, as `--resolve-env` states it to a
#: model. Sorted and version-exact, so the same prompt is built on every run.
BAKED_CAPABILITIES: tuple[str, ...] = (
    'apt-get (build time only; the graded run has no network)',
    f'gradle, with its cache home already exported as {GRADLE_USER_HOME}',
    'mise, which selects among the baked JDKs',
    f'temurin JDK {BAKED_JDKS}',
)

#: Every `build_flags` key `render_gap` interpolates. Enumerated once, read by
#: BOTH the renderer and `JavaPlugin.validate_dep_plan`, so a flag added to the
#: rendered text without being added here is the only way the two can disagree.
REQUIRED_BUILD_FLAGS: tuple[str, ...] = ('baked_jdks',)

#: Every `test_invocation` key the java measure path needs. `render_gap` itself
#: does not interpolate one -- the measure script runs the graded task -- but a
#: plan that does not state the command it was resolved FOR is a plan nobody can
#: check against the image it produced, so it is required at the same gate.
REQUIRED_TEST_INVOCATION_KEYS: tuple[str, ...] = ('command',)

#: The same requirements as the resolver is told them, before it answers. Kept
#: adjacent to the checks above so the ask and the rejection cannot drift apart.
#: Leak-safe: slot names, shapes and toolchain versions only -- never a repo
#: path, a module name or a source body.
REQUIRED_PLAN_SLOTS: tuple[str, ...] = (
    'build_flags["baked_jdks"] must be a non-empty string: the JDK majors the '
    'base image already bakes, e.g. "17/21/25". The gap states them in the '
    'comment that explains why nothing apt-installs a JDK',
    'install_commands must NOT compile the graded sources -- test_invocation '
    'builds and runs the suite itself -- so for a gradle-driven repo this list '
    'is USUALLY EMPTY. Add a step only to PREPARE the environment (fetch or '
    'vendor a dependency); never a bare "gradle build". A build step that '
    'cannot succeed makes the whole plan unbuildable, not merely suboptimal',
    'apt_packages lists ONLY system libraries the sources need that the base '
    'image lacks; the JDK and gradle are already baked in, so naming them here '
    'buys nothing and costs a network fetch',
    'package_manager must be gradle: the gap writes gradle.properties and pins '
    'the compile JVM through it',
    'test_invocation["command"] must be a non-empty list of argv tokens: the '
    'command that runs the whole graded suite, and it must NAME '
    'harness["test_task"], e.g. ["gradle", "--offline", "--no-daemon", '
    '"--console=plain", ":my-module:test", "--rerun"]. A command that runs a '
    'different task than the one the results directory belongs to grades one '
    'module and measures another',
    'toolchain_version must be the mise JDK identifier, '
    'distribution-then-version, e.g. "temurin-25.0.4+7.0.LTS": the gap selects '
    'it by that exact string and derives the runtime pin from its major',
    'SCOPE, and it overrides the general "enumerate every corpus" rule for '
    'java: this task grades exactly ONE gradle module -- the one whose main '
    'sources are removed -- and NOT the whole build. A multi-module repository '
    'is the NORMAL case and is never a reason to REFUSE: do not refuse because '
    'a root `test` task fans out, because several modules own separate '
    '`:module:test` tasks, or because each writes its own test-results '
    'directory. Name the single module under test. If you cannot tell which '
    'one it is, name your best candidate and answer anyway -- the harness knows '
    'which module was carved, checks your answer against it before anything is '
    'built, and names the right module in a repair message you can act on. An '
    'unnecessary REFUSE produces no task at all, which is strictly worse than '
    'one repairable guess',
    'harness is REQUIRED for java and describes THIS repository\'s own test '
    'harness; it is what the graded verifier script and the dependency-warming '
    'stage are rendered from. State it from the build manifests and the TEST '
    'FILE PATHS you were given, and from nothing else. '
    'harness["test_task"] is the ABSOLUTE gradle task path of the graded test '
    'task for that ONE module, leading colon included, e.g. ":my-module:test" '
    '(":test" for a single-project build). It must be a concrete module task, '
    'never an aggregate lifecycle task. The verifier derives the '
    '`:dependencies` task it proves the offline cache against from the same '
    'project path',
    'harness["results_dir"] is the repo-relative directory gradle writes the '
    'JUnit `TEST-*.xml` reports into for that ONE task -- the gradle default '
    'is "<module>/build/test-results/test". One directory, matching the one '
    'module test_task names; the other modules keep their own and are not '
    'graded by this task',
    'harness["stale_paths"] lists the build-output directories the oracle wipes '
    'after it restores, e.g. ["my-module/build", "my-module/.gradle", '
    '".gradle"]. Gradle reports a cached PASS from a previous run with zero '
    'recompiled code, so name EVERY directory the graded task writes into; '
    'this list may not be empty',
    'harness["buildsrc_path"] is the gradle project path of a convention build '
    'that must be assembled before dependencies resolve, e.g. ":buildSrc". '
    'Answer "" when the repository has no buildSrc directory -- a repo without '
    'one must not be asserted to have one. harness["project_label"] is the '
    'project\'s own name and is prose only; it may be ""',
)

#: The one package manager java's gap has prose (and a properties file) for.
#: `maven` is schema-legal for java but this gap writes gradle.properties and
#: exports a gradle cache home; rendering it for maven would emit a pin nothing
#: reads.
_SUPPORTED_MANAGERS: frozenset[str] = frozenset({'gradle'})


def _flag_str(build_flags: tuple[tuple[str, FlagValue], ...], key: str) -> str:
    """A build flag the gap's rendered text needs, or a refusal to render at all."""
    for name, value in build_flags:
        if name == key:
            if isinstance(value, str) and value:
                return value
            break
    raise B.LangError(
        f'the java gap needs build_flags[{key!r}]: it must be a non-empty '
        'string. A plan without it would render a Dockerfile comment with a '
        'hole in it, and a hole in the toolchain description is how a base swap '
        'goes unnoticed'
    )


def _test_tokens(
    test_invocation: tuple[tuple[str, TestValue], ...], key: str,
) -> tuple[str, ...]:
    """A test_invocation entry the java measure path needs, as argv tokens."""
    for name, value in test_invocation:
        if name == key:
            tokens = (value,) if isinstance(value, str) else tuple(value)
            if tokens and all(token.strip() for token in tokens):
                return tokens
            break
    raise B.LangError(
        f'the java plan needs test_invocation[{key!r}]: it must be a non-empty '
        'list of argv tokens naming the command that runs the graded suite. A '
        'plan that does not state what it was resolved to RUN cannot be checked '
        'against the image it produced'
    )


def _slot_error(key: str, want: str) -> B.LangError:
    return B.LangError(
        f'the java harness needs harness[{key!r}]: {want}. Without it the '
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


def _require_relative(path: str, key: str) -> str:
    """A repo-relative path, or a refusal. `..` escapes the tree it describes."""
    if path.startswith('/') or path.split('/')[0] == '..' or '/../' in path:
        raise _slot_error(
            key,
            f'a repo-relative path, but {path!r} is absolute or climbs out of '
            'the checkout; the verifier resolves it against ${REPO} and a path '
            'that escapes there names something no carve controls',
        )
    return path


@dataclass(frozen=True)
class Harness:
    """A java test harness as data: run it, find its reports, size its suite.

    Everything the two rendered scripts and the warm stage used to hardcode
    about the java-tamboui REPO, read off `DepPlan.harness`. `read_harness` is
    the only constructor, so a slot that is missing, mistyped or inconsistent
    with its siblings becomes a `LangError` before any container exists rather
    than a shell script that counts an empty directory and reports a floor of
    zero.
    """

    test_task: str
    results_dir: str
    stale_paths: tuple[str, ...]
    buildsrc_path: str
    project_label: str

    @property
    def project_path(self) -> str:
        """`:tamboui-widgets` -- the gradle project the graded task belongs to.

        `''` for a single-project build, whose task path is `:test`; that is
        also what makes `dependencies_task` come out as `:dependencies` there.
        """
        return self.test_task.rsplit(':', 1)[0]

    @property
    def dependencies_task(self) -> str:
        return f'{self.project_path}:dependencies'

    @property
    def buildsrc_task(self) -> str:
        return f'{self.buildsrc_path}:assemble'

    @property
    def buildsrc_dir(self) -> str:
        """`:buildSrc` as the directory the carve assert stats: `buildSrc`."""
        return self.buildsrc_path.strip(':').replace(':', '/')


def read_harness(plan: DepPlan) -> Harness:
    """`plan.harness` as the record the renderers read, or a `LangError`.

    Every check here is reachable from the refine loop, so every message names
    the slot and says what a correct value looks like: a resolver that gets one
    wrong must be able to repair it without being shown the repository.
    """
    values = {key: value for key, value in canonicalize(plan).harness}
    harness = Harness(
        test_task=_str_slot(
            values, 'test_task',
            'the absolute gradle task path of the graded test task, leading '
            'colon included, e.g. ":my-module:test"',
            required=True,
        ),
        results_dir=_require_relative(
            _str_slot(
                values, 'results_dir',
                'the repo-relative directory gradle writes the JUnit '
                '`TEST-*.xml` reports into for the graded task',
                required=True,
            ),
            'results_dir',
        ),
        stale_paths=_tuple_slot(
            values, 'stale_paths',
            'the build-output directories the oracle wipes after it restores, '
            'so no cached gradle output can report a pass with zero recompiled '
            'code',
            required=True,
        ),
        buildsrc_path=_str_slot(
            values, 'buildsrc_path',
            'the gradle project path of a convention build that must be '
            'assembled before dependencies resolve, or "" for a repo with no '
            'buildSrc',
        ),
        project_label=_str_slot(
            values, 'project_label', "the project's own name",
        ),
    )
    for key, path in (('test_task', harness.test_task),
                      ('buildsrc_path', harness.buildsrc_path)):
        if path and not path.startswith(':'):
            raise _slot_error(
                key,
                f'an ABSOLUTE gradle project path starting with ":", but '
                f'{path!r} does not. A relative task path resolves against '
                "gradle's current project, which is not a property this "
                'renderer can see or pin',
            )
    if not harness.test_task.rpartition(':')[2]:
        raise _slot_error(
            'test_task',
            f'a task path whose last segment NAMES a task, but {harness.test_task!r} '
            'ends at a project path. e.g. ":my-module:test"',
        )
    for path in harness.stale_paths:
        _require_relative(path, 'stale_paths')
    return harness


#: How a java source root separates the module from the package namespace.
#: `src/<sourceSet>/java` is the maven-and-gradle layout every JVM build tool
#: reimplements, so the segment AFTER `java` is the package root and the segment
#: BEFORE `src` is the module -- the only two facts the carve has to yield.
_JAVA_SOURCE_MARKER = 'java'
_SRC_SEGMENT = 'src'

#: Gradle's `test` task compiles the `test` source set, whose conventional root
#: this is. It is what the grader calls the oracle tree, and what
#: `emit._java_grader_metadata` counts JUnit classes under.
TEST_SOURCE_ROOT = 'src/test'


def module_dir_of_carve(carved_relpaths) -> str:
    """The repo-relative directory of the module the carve removed code from.

    `tamboui-widgets/src/main/java/dev/tamboui/widgets/X.java` -> `tamboui-widgets`;
    `src/main/java/...` -> `''`, the root project. Derived from the CARVE rather
    than from the plan because which module is graded is decided by the operator
    who chose the include globs, not by whoever resolved the environment -- and
    a plan that grades a module nobody carved is the mis-scope this feeds the
    cross-check to catch.
    """
    modules = set()
    for rel in carved_relpaths:
        parts = str(rel).split('/')
        if _SRC_SEGMENT not in parts:
            raise B.LangError(
                f'the java carve holds {rel!r}, which sits under no `src/` '
                'source root, so the module it belongs to cannot be '
                'determined. The graded gradle task, the report directory and '
                'the fingerprint are all scoped to one module; refusing rather '
                'than guessing which'
            )
        modules.add('/'.join(parts[:parts.index(_SRC_SEGMENT)]))
    if len(modules) != 1:
        raise B.LangError(
            f'the java carve spans {len(modules)} modules '
            f'({", ".join(sorted(repr(m) for m in modules))}), but the graded '
            'surface is ONE gradle task with ONE report directory. Carving '
            'across modules would leave every module but the graded one '
            'stubbed and ungraded, so it is refused rather than partly measured'
        )
    return modules.pop()


def package_roots_of_carve(carved_relpaths) -> tuple[str, ...]:
    """The package namespace(s) the carve removed, relative to the source root.

    THE LEAK SCAN'S SUBJECT. Compiled java carries its package as a UTF8
    constant in every `.class` and as a path in every jar entry, so the warm
    stage's fossil scan looks for exactly this string. It is derived from the
    carved paths and never from a plan: a namespace a model supplied would be a
    scan whose subject was guessed, and on a repo whose guess was wrong the scan
    passes vacuously while the cache holds the answer.

    Returns the COMMON prefix per source root rather than one entry per file,
    because `dev/tamboui/widgets` subsumes `dev/tamboui/widgets/tabs` and a scan
    for the parent finds every child. An empty prefix -- carved classes in the
    default package, or sitting directly under the source root -- is REFUSED:
    a `-path '**'` scan matches the whole cache and proves nothing.
    """
    roots: list[tuple[str, ...]] = []
    for rel in carved_relpaths:
        parts = str(rel).split('/')
        if _JAVA_SOURCE_MARKER not in parts[:-1]:
            raise B.LangError(
                f'the java carve holds {rel!r}, which sits under no `java` '
                'source root, so the package namespace it removed cannot be '
                'read off its path. The warm stage asserts that namespace is '
                "absent from the gradle cache, and a scan whose subject is "
                'unknown is a scan that passes vacuously -- refusing instead'
            )
        index = len(parts) - 1 - parts[::-1].index(_JAVA_SOURCE_MARKER)
        roots.append(tuple(parts[index + 1:-1]))
    if not roots:
        raise B.LangError(
            'the java carve removed no files, so there is no package namespace '
            'for the warm stage to prove absent from the gradle cache'
        )
    common = roots[0]
    for parts in roots[1:]:
        keep = 0
        while keep < min(len(common), len(parts)) and common[keep] == parts[keep]:
            keep += 1
        common = common[:keep]
    if not common:
        raise B.LangError(
            'the carved java sources share no package namespace (they sit in '
            'the default package or span unrelated packages), so the warm '
            "stage's fossil scan would have to look for the empty string -- "
            'which matches the whole dependency cache and proves nothing. '
            'Refusing rather than shipping a leak assert that cannot fail'
        )
    return ('/'.join(common),)


_ENGLISH_TITLE: tuple[str, ...] = (
    'Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight',
)


def _carved_nouns(package_roots: tuple[str, ...]) -> tuple[str, str]:
    """`('widget', 'widgets')` -- how the warm stage's COMMENTS name the carve.

    The leaf of the carved package namespace, which is the word a java project
    already chose for the thing it removed. Reaches comments only: no rendered
    INSTRUCTION is derived from it, so a naive de-pluralisation costs at worst a
    slightly odd sentence and can never change what the image does.
    """
    plural = package_roots[0].rsplit('/', 1)[-1] if package_roots else 'carved'
    singular = plural[:-1] if plural.endswith('s') and not plural.endswith('ss') else plural
    return singular, plural


def _quoted_roots(package_roots: tuple[str, ...]) -> str:
    return ', '.join(f"'{root}'" for root in package_roots)


def _dotted_roots(package_roots: tuple[str, ...]) -> str:
    """The namespaces as java spells them in a class file's constant pool."""
    return ', '.join(root.replace('/', '.') for root in package_roots)


def _find_path_expr(package_roots: tuple[str, ...]) -> str:
    """`find`'s primary for "names any carved namespace", parenthesised only if needed.

    A single root renders as the bare `-path ... -print` a one-package carve has
    always produced; several are OR-ed inside `\\( \\)`, because `find`'s implicit
    AND binds tighter than `-o` and `-path A -o -path B -print` would print only
    the B matches -- reporting a leak as clean.
    """
    tests = ' -o '.join(f"-path '*{root}*'" for root in package_roots)
    return f'{tests} -print' if len(package_roots) == 1 else f'\\( {tests} \\) -print'


def _jdk_parts(toolchain_version: str) -> tuple[str, str]:
    """`(distribution, major)` of a mise JDK identifier, or a refusal to render.

    `mise use -g java@X` takes the whole identifier, but the login-shell pin
    asserts the MAJOR only (`java -version` prints `"25.0.4"`, never the mise
    spelling), and the comment above it names the distribution. A version with
    neither part is not renderable: an assert derived from a non-numeric major
    would either match nothing or -- worse -- match everything.
    """
    distribution, sep, version = toolchain_version.partition('-')
    major = version.partition('.')[0]
    if not sep or not distribution or not major.isdigit():
        raise B.LangError(
            f'toolchain_version {toolchain_version!r} is not a mise JDK '
            'identifier of the form distribution-major.minor.patch, e.g. '
            '"temurin-25.0.4+7.0.LTS"; the java gap selects the JDK by that '
            'exact string and derives its runtime pin from the major'
        )
    return distribution, major


def _java_measure_dep_plan() -> DepPlan:
    """java's own environment as the record a resolver would have to produce.

    The fixed point of the exercise: fed to `JavaPlugin.render_gap`, this plan
    reproduces the gap bytes `render_measure_dockerfile` hardcodes today.

    `apt_packages` and `install_commands` are EMPTY and must stay empty. The JDK
    and gradle are BAKED into the per-language base -- that is precisely why the
    install block selects one with mise instead of apt-installing ~700 MB -- and
    these two fields mean "what the gap INSTALLS". Listing a baked component
    would make the gap emit an `apt-get` line today's bytes do not contain.

    The warm stage and the pre-leakgate blocks are NOT described here: they are
    fixed scaffolding (a resolve-only gradle pass plus its leak assert), not
    provisioning a resolver may vary, and they stay hardcoded.
    """
    plan = DepPlan(
        lang='java',
        toolchain_version=JAVA_VERSION,
        package_manager='gradle',
        manifest_files=('build.gradle.kts', 'settings.gradle.kts'),
        apt_packages=(),
        install_commands=(),
        build_flags=(('baked_jdks', BAKED_JDKS),),
        test_invocation=(('command', _TAMBOUI_TEST_ARGV),),
        harness=JAVA_TAMBOUI_HARNESS,
        needs_git_metadata=False,
    )
    validate(plan)
    return canonicalize(plan)


#: java's canonical, validated environment plan. Module-level so a test can
#: assert the rendered gap against it without re-deriving the facts it states.
JAVA_MEASURE_DEP_PLAN: DepPlan = _java_measure_dep_plan()


def _test_command(plan: DepPlan) -> str:
    """The graded command as the one prose line instruction.md shows.

    Derived from the plan rather than spelled again: the argv is already the
    authority the two scripts render from, and a second spelling of it is how
    the instruction the model reads starts describing a command the grader does
    not run.
    """
    return ' '.join(_test_tokens(plan.test_invocation, 'command'))


#: The graded command as prose, for instruction.md and the graded set.
#: java-tamboui's, because it is derived from java-tamboui's plan -- a resolved
#: plan replaces it through `test_command_from_plan` before an entry is written.
TEST_COMMAND = _test_command(JAVA_MEASURE_DEP_PLAN)


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
    #: the three spellings of a test dir for cpp; the graded module's own
    #: `src/test/**` for java, narrowed to the carve by
    #: `grader_fingerprint_globs_for`.
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
    java_home: ClassVar[str] = JAVA_HOME
    gradle_user_home: ClassVar[str] = GRADLE_USER_HOME

    def _harness(self, dep_plan: DepPlan | None) -> tuple[DepPlan, Harness]:
        """The plan the harness comes from, and the harness itself.

        `dep_plan=None` is the pre-resolution path (`--no-resolve-env`, and
        every caller written before the harness was plan-driven). It falls back
        to java-tamboui's own canonical plan, which is exactly the environment
        those callers were hardcoded against -- so the fallback is not a guess,
        it is the same bytes under a name.
        """
        plan = canonicalize(JAVA_MEASURE_DEP_PLAN if dep_plan is None else dep_plan)
        return plan, read_harness(plan)

    @staticmethod
    def _carve(env: EnvSpec | None) -> CarveFacts:
        """What the host carved, or java-tamboui's own carve for a caller with none.

        A caller holding a `root` but no `package_roots` is REFUSED rather than
        silently given the fallback: that shape means a carve WAS derived and
        its namespace could not be, and quietly substituting another
        repository's namespace is precisely the vacuous fossil scan this whole
        derivation exists to make impossible.
        """
        carve = env.carve if env is not None else CarveFacts()
        if not carve.root and not carve.package_roots:
            return JAVA_TAMBOUI_CARVE
        if not carve.package_roots:
            raise B.LangError(
                f'the carve at {carve.root!r} yielded no java package root, so '
                "the warm stage's fossil scan has no namespace to prove absent "
                'from the gradle cache. A scan whose subject is unknown passes '
                'vacuously on every repository; refusing to render one'
            )
        return carve

    @staticmethod
    def _assert_scope_agrees(harness: Harness, graded_module: str) -> None:
        """THE MIS-SCOPE GUARD: the plan grades the module the carve stubbed.

        The other half of what replaced `EXPECTED_SUITES` (the first half being
        the host-side suite count `render_test_sh` bakes in). `graded_module`
        comes from the CARVE, so a plan naming a DIFFERENT module -- one whose
        sources are all still present -- is refused before anything is built.
        That is the worst failure this language can have: such a task grades a
        module the carve never touched, so the stub scores a perfect suite and
        RED and GREEN are indistinguishable.

        Both messages name the module the carve actually stubbed, so a resolver
        that guessed wrong repairs in one attempt instead of REFUSING -- which
        matters, because the resolver is shown test PATHS only and several
        modules' paths look alike.
        """
        scope = f'{graded_module}/' if graded_module else ''
        if not harness.results_dir.startswith(scope):
            raise B.LangError(
                f'harness["results_dir"] is {harness.results_dir!r}, which is '
                f'not inside the module the carve stubbed ({graded_module!r}). '
                'The verifier would count reports produced by a module whose '
                'sources are all still present, score a perfect suite against '
                'them, and never run the graded code at all'
            )
        if graded_module and f':{graded_module}:' not in harness.test_task:
            raise B.LangError(
                f'harness["test_task"] is {harness.test_task!r}, which does not '
                f'name the module the carve stubbed ({graded_module!r}). The '
                'graded task and the carve must describe one module, or the '
                'task under test is not the code that was removed'
            )

    def assert_repo_agrees(self, plan: DepPlan, *, graded_module: str) -> None:
        """`validate_dep_plan` plus what only the CARVE can answer."""
        self._assert_scope_agrees(read_harness(plan), str(graded_module))

    def test_command_from_plan(self, dep_plan: DepPlan | None) -> str:
        """What the rendered scripts ACTUALLY run, for instruction.md to quote."""
        plan, _harness = self._harness(dep_plan)
        return _test_command(plan)

    def grader_fingerprint_globs_for(self, carved_relpaths) -> tuple[str, ...]:
        """The carved module's own test tree, and no other module's.

        `('*/src/test/**', 'src/test/**')` matches every module in a monorepo,
        and locking java-tamboui's other four modules would pin ~200 files the
        graded task never reads -- while moving the shipped test.sh for a repo
        whose layout did not change. Narrowing to the module the carve stubbed
        is both the correct scope and the stable one.
        """
        module = module_dir_of_carve(carved_relpaths)
        return (f'{module}/src/test/**' if module else 'src/test/**',)

    # --- axis 1 -----------------------------------------------------------

    def toolchain_spec(self) -> ToolchainSpec:
        """JDK 25 apt install + gradle.properties, byte-identical across stages."""
        return ToolchainSpec(
            base_image=BASE_IMAGE,
            install_block=_INSTALL_BLOCK,
            env={
                'DEBIAN_FRONTEND': 'noninteractive',
                'GRADLE_USER_HOME': GRADLE_USER_HOME,
                'JAVA_HOME': JAVA_HOME,
                # ENV lines are sorted by ToolchainSpec.render, so PATH lands
                # after JAVA_HOME and can reference it. Bake JAVA_HOME's literal
                # anyway rather than $JAVA_HOME so a base image that unsets it
                # cannot silently break the prepend.
                'PATH': '/opt/mise/shims:${PATH}',
            },
            workdir=B.WORKDIR,
        )

    # --- axis 2 -----------------------------------------------------------

    def dep_warm_spec(self) -> DepWarmSpec:
        """The warm stage for a caller holding neither a plan nor a carve."""
        return self.dep_warm_spec_for(None, None)

    def dep_warm_spec_for(
        self, env: EnvSpec | None, dep_plan: DepPlan | None,
    ) -> DepWarmSpec:
        """Resolve-only warm stage: no carved bytecode possible, gradle-home clean.

        `files_needed=()` because the framework's single-line COPY concat
        mangles directory layout for the ~30 build files gradle needs across
        modules and a convention build. The stage_block below COPYs the whole
        carved repoctx instead -- which by construction has NO sources under the
        carved source root -- and drives Gradle through a resolve-only pass that
        never produces bytecode for the carved package. The scrub + namespace-
        absence assert at the end is the load-bearing leak proof, and its
        subject comes from the CARVE (`env.carve`), never from the plan.

        `copy_paths=(('/opt/gradle-home', '/opt/gradle-home'),)` makes the
        framework emit `COPY --from=warm /opt/gradle-home /opt/gradle-home`
        into the graded stage (invariant 7: COPY --from=warm only, never
        FROM warm).
        """
        _plan, harness = self._harness(dep_plan)
        carve = self._carve(env)
        noun, plural = _carved_nouns(carve.package_roots)
        buildsrc = harness.buildsrc_dir
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

        carve_assert = [
            f'    test "$(find {workdir}/{carve.root} '
            "-name '*.java' 2>/dev/null | wc -l)\" -eq 0",
        ]
        if buildsrc:
            carve_assert = [
                carve_assert[0] + '; \\',
                f'    test "$(find {workdir}/{buildsrc}/src/main '
                "-type f 2>/dev/null | wc -l)\" -gt 0",
            ]
            carve_note = [
                f'# Assert the carve landed as expected: no {noun} main sources, {buildsrc}',
                f'# convention plugins present, {plural} test tree intact '
                f'({carve.grader_source_count} files, per',
                '# the fingerprint the graded stage will lock).',
            ]
        else:
            carve_note = [
                f'# Assert the carve landed as expected: no {noun} main sources, and the',
                f'# {plural} test tree intact ({carve.grader_source_count} files, per the',
                '# fingerprint the graded stage will lock).',
            ]

        buildsrc_pass = [
            f"#   1) {harness.buildsrc_task} compiles {buildsrc}'s Kotlin convention",
            "#      plugins into $GRADLE_USER_HOME's instrumented-jar cache. NOT",
            f"#      {harness.buildsrc_path}:build (which triggers "
            f"{harness.buildsrc_path}:compileTestJava,",
            "#      whose testCompileClasspath resolves junit-jupiter WITHOUT a",
            f"#      version -- {harness.project_label}'s {buildsrc} references "
            'it as a plain',
            "#      dependency instead of via a BOM, and gradle fails the whole",
            "#      task rather than skip the version-less coord).",
        ] if buildsrc else []
        first = len(buildsrc_pass) and 1
        warm_passes = [
            *buildsrc_pass,
            f"#   {first + 1}) harborResolveAll .resolve()s every configuration in every",
            "#      subproject (compileClasspath, testCompileClasspath,",
            "#      testRuntimeClasspath, annotationProcessor, ...) so no config",
            "#      the graded run touches is unresolved.",
            f"#   {first + 2}) --stop kills the daemon before scrub so no daemon-owned lock",
            "#      files under $GRADLE_USER_HOME/daemon would survive.",
        ]

        stage_block = '\n'.join([
            f'# The carved repoctx into the warm stage. {carve.root}',
            f'# is empty by construction ({carve.file_count} files carved), '
            f'so no {noun} code can be',
            '# compiled here. Every other module is intact -- resolve-only needs the',
            '# whole project tree to CONFIGURE the build graph.',
            f'COPY --from={B.REPO_CONTEXT} repo/ {workdir}/',
            '',
            *carve_note,
            'RUN set -eux; \\',
            *carve_assert,
            '',
            '# Resolve-only init script. `harborResolveAll` iterates every project',
            (f'# (including {buildsrc}) and calls .resolve() on every resolvable'
             if buildsrc else '# and calls .resolve() on every resolvable'),
            '# configuration -- downloads every jar the later --offline run will',
            f"# ask for, without compiling anything derived from {noun} sources.",
            "# `<<'HARBOR_INIT_EOF'` disables shell expansion so Kotlin's ${...}",
            '# survives into the file verbatim.',
            f'RUN cat > /tmp/harbor-resolve.init.gradle.kts '
            "<<'HARBOR_INIT_EOF'",
            init_body,
            'HARBOR_INIT_EOF',
            '',
            f'# Warm the cache. {_ENGLISH_TITLE[first + 2]} passes '
            'matching the Oracle design:',
            *warm_passes,
            'RUN set -eux; \\',
            f'    cd {workdir}; \\',
            '    gradle --version; \\',
            *([f'    gradle --no-daemon --console=plain {harness.buildsrc_task}; \\']
              if buildsrc else []),
            '    gradle --no-daemon --console=plain '
            '-I /tmp/harbor-resolve.init.gradle.kts harborResolveAll; \\',
            '    gradle --no-daemon --stop || true; \\',
            '    rm -f /tmp/harbor-resolve.init.gradle.kts',
            '',
            '# Scrub the cache. build-cache-1 holds task OUTPUTS keyed by input',
            f'# hash -- for {harness.project_path}:compileJava that would be the answer',
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
            f"# may name {_quoted_roots(carve.package_roots)}: bytecode holds FQNs as UTF8",
            '# constants, jars hold entry names in the central directory, and',
            '# either would be a decompile-back-to-source leak of the worst kind.',
            '# The resolve-only path above should never produce such artifacts',
            f'# (no {noun} sources to compile), but this assert stops a future',
            '# stage-graph change from silently smuggling them through.',
            'RUN set -eu; \\',
            f'    hits=$(find {GRADLE_USER_HOME} '
            f'{_find_path_expr(carve.package_roots)} 2>/dev/null || true); \\',
            '    if [ -n "$hits" ]; then \\',
            f'        echo "LEAK: {GRADLE_USER_HOME} holds '
            f'{_dotted_roots(carve.package_roots)} artifacts:" >&2; \\',
            '        echo "$hits" >&2; \\',
            '        exit 1; \\',
            '    fi; \\',
            f'    echo "ASSERT: {GRADLE_USER_HOME} carries no '
            f'{_dotted_roots(carve.package_roots)} symbols or paths"',
        ]) + '\n'

        return DepWarmSpec(
            stage_block=stage_block,
            files_needed=(),
            copy_paths=((f'{GRADLE_USER_HOME}', f'{GRADLE_USER_HOME}'),),
        )

    # --- axes 3-6 ---------------------------------------------------------

    def pre_leakgate_blocks(self, env: EnvSpec) -> tuple[str, ...]:
        """The plan-free proof, for a caller with no resolved environment."""
        return self.pre_leakgate_blocks_for(env, None)

    def pre_leakgate_blocks_for(
        self, env: EnvSpec, dep_plan: DepPlan | None,
    ) -> tuple[str, ...]:
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
        _plan, harness = self._harness(dep_plan)
        workdir = env.workdir
        proof = '\n'.join([
            '# Offline-sufficiency proof: prove the warm cache covers every',
            f'# configuration the graded {harness.test_task} task needs, at',
            '# BUILD time (network refused via --offline), so a genuine cache',
            '# gap fails the image build rather than surfacing later as a',
            '# black-box compile error inside the graded container.',
            'RUN set -eux; \\',
            f'    cd {workdir}; \\',
            '    for cfg in testRuntimeClasspath testCompileClasspath '
            'compileClasspath annotationProcessor testAnnotationProcessor; do \\',
            '        echo "=== --offline :dependencies --configuration $cfg ==="; \\',
            '        gradle --offline --no-daemon --console=plain '
            f'{harness.dependencies_task} --configuration "$cfg" \\',
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
        test_suites: int | None = None,
        graded_module: str | None = None,
        dep_plan: DepPlan | None = None,
    ) -> str:
        expected = graded.expected if expected is None else int(expected)
        fingerprint = graded.fingerprint_sha256 if fingerprint is None else fingerprint
        if test_suites is None or graded_module is None:
            raise B.LangError(
                'java plugin needs test_suites and graded_module threaded from '
                'the intact tree at generate time; emit._render_test_sh '
                'supplies them (no repo constants in the plugin source)'
            )
        plan, harness = self._harness(dep_plan)
        self._assert_scope_agrees(harness, str(graded_module))
        test_cmd = _test_command(plan)
        results_dir = harness.results_dir
        grader_tree = (
            f'{graded_module}/{TEST_SOURCE_ROOT}' if graded_module
            else TEST_SOURCE_ROOT
        )

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
            f'across {test_suites} suites).',
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
            f'EXPECTED_SUITES={test_suites}',
            '',
            '# --- integrity guards -----------------------------------------------',
            f'# {grader_tree} IS the grader. The per-file lock below',
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

    def measure_test_sh(
        self,
        *,
        graded: GradedSet | None = None,
        dep_plan: DepPlan | None = None,
        **kwargs,
    ) -> str:
        """Phase 1: run the WHOLE graded gradle task against the intact tree.

        Floor-FREE by construction. `measure` writes tests_total = the summed
        `<testsuite tests=>` across every produced XML; a build that broke
        halfway registers as 0 (or a shrunken count) and
        `parse_measure_json` rejects zero.

        Runs OFFLINE because the measure Dockerfile has already warmed the
        cache at build time (see `render_measure_dockerfile`). --no-daemon
        matches the shipped test.sh.
        """
        del graded, kwargs
        plan, harness = self._harness(dep_plan)
        results_dir = harness.results_dir
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
            f'{_test_command(plan)} > "${{MEASURE_LOG}}" 2>&1 || true',
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
        return self.post_restore_for(None)

    def post_restore_for(self, dep_plan: DepPlan | None) -> str:
        """Invalidate stale gradle build output so GREEN rebuilds honestly.

        The `--rerun` in test.sh handles task-level caching, but leftover
        artifacts under `build/` from a previous partial run would let a
        JUnit XML from before the oracle restore linger on disk. The scrub
        keeps the post-restore state predictable. WHICH directories is the
        plan's answer, not this method's -- a module's own `build/` is that
        repository's name for one, not gradle's.
        """
        _plan, harness = self._harness(dep_plan)
        return 'rm -rf ' + ' '.join(
            f'"${{REPO}}/{path}"' for path in harness.stale_paths
        )

    # --- axis 8 + the image ----------------------------------------------

    def validate_dep_plan(self, plan: DepPlan) -> None:
        """Every precondition `render_gap` has, checked before a container exists.

        The same set of checks `render_gap` makes, hoisted to where they cost
        nothing and can still be repaired: same helpers, same messages, so a
        plan accepted here cannot then fail to render. Enumerated from the
        renderer below -- the JDK identifier's distribution and major, the
        manager whose properties file it writes, the baked-JDK list and the
        graded command -- because a slot the renderer reads and this gate does
        not is exactly the crash this exists to prevent.
        """
        validate(plan)
        plan = canonicalize(plan)
        if plan.lang != self.name:
            raise B.LangError(
                f'the java gap cannot render a {plan.lang!r} plan; a gap is the '
                'one part of a Dockerfile that is language-specific by definition'
            )
        _jdk_parts(plan.toolchain_version)
        if plan.package_manager not in _SUPPORTED_MANAGERS:
            raise B.LangError(
                f'the java gap has no prose for package_manager '
                f'{plan.package_manager!r}; expected one of '
                f'{", ".join(sorted(_SUPPORTED_MANAGERS))}'
            )
        for key in REQUIRED_BUILD_FLAGS:
            _flag_str(plan.build_flags, key)
        for key in REQUIRED_TEST_INVOCATION_KEYS:
            _test_tokens(plan.test_invocation, key)
        harness = read_harness(plan)
        if harness.test_task not in _test_tokens(plan.test_invocation, 'command'):
            raise B.LangError(
                f'test_invocation["command"] does not name '
                f'harness["test_task"] ({harness.test_task!r}): the command '
                'the verifier runs and the task whose reports it counts must '
                'be the same task, or the suite is measured in one module and '
                'graded in another'
            )

    def _gap_body(self, plan: DepPlan) -> str:
        """java's toolchain bytes, rendered from a plan instead of a literal.

        The part BOTH the measure and the shipped render take; the measure-only
        no-warm note is `render_gap`'s (see `LangPlugin._gap_body`).

        The SCAFFOLDING stays fixed and stays hardcoded: the zz- profile.d file
        that keeps the mise shims ahead of the base's own PATH, the login-shell
        pin assert, the gradle cache home, and -- crucially -- the whole
        `dep_warm_spec` warm stage and its leak assert, which are not
        provisioning a resolver may vary but the proof that no carved bytecode
        survives. What comes from the PLAN is the JDK the image selects, the
        baked-JDK list the comment states, the manager whose properties file is
        written, and the apt/install lines a repo needing system libraries adds.

        The apt and install blocks are unreachable for `JAVA_MEASURE_DEP_PLAN`
        (both fields are empty by construction, because the base BAKES the JDK)
        and are rendered from the plan for the case where a resolved plan does
        declare them. They are the only lines here that are not in today's image.

        `JAVA_HOME` is re-derived from the plan's JDK rather than taken from
        `toolchain_spec()`: it is the very path the gradle.properties line
        writes, so a plan that moved the JDK and left that ENV behind would ship
        an image whose compile JVM and whose environment disagree.
        """
        validate(plan)
        plan = canonicalize(plan)
        if plan.lang != self.name:
            raise B.LangError(
                f'the java gap cannot render a {plan.lang!r} plan; a gap is the '
                'one part of a Dockerfile that is language-specific by definition'
            )
        if plan.package_manager not in _SUPPORTED_MANAGERS:
            raise B.LangError(
                f'the java gap has no prose for package_manager '
                f'{plan.package_manager!r}; expected one of '
                f'{", ".join(sorted(_SUPPORTED_MANAGERS))}'
            )

        distribution, major = _jdk_parts(plan.toolchain_version)
        java_home = f'{JAVA_INSTALL_ROOT}/{plan.toolchain_version}'
        lines = [
            f'# The base bakes {distribution} '
            f'{_flag_str(plan.build_flags, "baked_jdks")} and '
            f'{plan.package_manager}; select one rather than',
            '# apt-installing a JDK on every task build.',
            f'RUN set -eux; mise use -g java@{plan.toolchain_version}',
            '# profile.d is sourced in sorted order and the base re-exports its own PATH',
            '# ahead of the mise shims, so only a zz- file keeps the pin in front for the',
            "# login shells harbor's test.sh and solve.sh actually run as.",
            'RUN set -eux; \\',
            '    printf \'export PATH="/opt/mise/shims:$PATH"\\n\' > /etc/profile.d/zz-harbor-toolchain-pin.sh; \\',
            '    chmod 0644 /etc/profile.d/zz-harbor-toolchain-pin.sh',
            'RUN set -eux; \\',
            '    v="$(bash -lc \'java -version 2>&1 | head -1\')"; \\',
            '    case "$v" in \\',
            f'        *\\"{major}.*) echo "TOOLCHAIN PIN OK (login shell): $v" ;; \\',
            f'        *) echo "TOOLCHAIN PIN FAILED (login shell): got $v want '
            f'{major}.x" >&2; exit 42 ;; \\',
            '    esac',
            '# org.gradle.java.home pins the COMPILE JVM; the launcher JVM comes from PATH.',
            'RUN set -eux; \\',
            f'    mkdir -p {GRADLE_USER_HOME}; \\',
            f"    printf 'org.gradle.java.home=%s\\n' {java_home} \\",
            f'        >> {GRADLE_USER_HOME}/gradle.properties',
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
            env={**spec.env, 'JAVA_HOME': java_home},
        ).render()

    def render_gap(self, plan: DepPlan) -> str:
        """The measure image's gap: the shared body plus the no-warm note.

        The note is the one line the shipped image must NOT carry: java's
        shipped render warms the gradle cache in a separate stage and COPYs it.
        """
        return '\n'.join([self._gap_body(plan), '', MEASURE_NO_WARM_COMMENT])

    def render_measure_dockerfile(
        self, env: EnvSpec, *, dep_plan: DepPlan | None = None,
    ) -> str:
        """The stripped Dockerfile for the never-ship measure image (phase 1).

        Same toolchain (JDK25 install byte-identical to the graded one for
        BuildKit layer reuse) plus a build-time warm-and-run of the plan's
        graded task against the INTACT tree. That single run does
        double duty: it downloads every dependency the graded --offline run
        will need (network is allowed at BUILD time), and it produces the
        JUnit XML the measure script can re-count later at container time
        with `--offline --rerun`.

        NO leak gate, NO tripwire scan, NO carve-metadata assert (intact-tree
        exemption -- all three would fire by construction on the intact tree).
        `measure_image_tag` marks the image as never-ship and `measure.py`
        deletes it in a finally block.

        `dep_plan` swaps the hardcoded gap for `render_gap(dep_plan)` in the
        SAME slot and touches nothing else -- notably NOT the warm-and-run block
        below, which is scaffolding. `dep_plan=None` is what emit.py passes and
        renders the bytes it always did.
        """
        base_image = self.toolchain_spec().base_image
        _plan, harness = self._harness(dep_plan)
        gap = (
            '\n'.join([self.toolchain(), '', MEASURE_NO_WARM_COMMENT])
            if dep_plan is None
            else self.render_gap(dep_plan)
        )
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
            gap,
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
            f'    gradle --no-daemon --console=plain {harness.test_task}; \\',
            '    gradle --no-daemon --stop || true',
            '',
            '# --- measure script (COPYed, not carved-tree, so lives in a layer) ---',
            f'COPY measure.sh {env.tests_dir}/measure.sh',
            f'RUN chmod 0555 {env.tests_dir}/measure.sh',
            '',
        ])


B.register(JavaPlugin())
