"""The java plugin: whole-suite grading, measured floor, JDK25+gradle+JUnit5.

Docker is MOCKED here -- nothing below builds an image. What is asserted is
the TEXT the plugin renders, plus a docker-less exercise of the fingerprint
plumbing. The java-tamboui task's full docker matrix runs separately (see the
plan's ACCEPTANCE GATE in Section 7 of the delivery brief).

Key java-tamboui properties this file locks:

JAVA1  whole-suite, no parser. `SELECTOR_KIND['java']='whole-suite'` and the
       plugin declares `parser_backed=False`. `emit.plan_carve` takes the
       no-parser branch and produces a graded set built by
       `whole_suite_selection`, not by `derive_graded_set`.
JAVA2  equality floor. JUnit reports per-test tallies in one XML per suite,
       and a compile failure or crashed JVM produces zero JUnit XML (caught by
       SUITES > 0 ahead of the floor). observed==EXPECTED is a real assertion.
JAVA3  fingerprint plumbing. The plugin declares
       `grader_fingerprint_globs=('tamboui-widgets/src/test/**',)`;
       emit.plan_carve expands those against the intact repo, `_graded_spec`
       sha256s the resulting paths, and `fingerprint_gate_block` bakes the
       per-file checks inline.
JAVA4  no magic repo-numbers in the plugin source (mostly). 823 (JUnit test
       methods) comes from `graded.expected`; EXPECTED_SUITES=69 is the ONE
       exception, threaded as a plugin ClassVar with a docstring rather than
       sprinkled through the rendered bash (see plugin docstring's rationale).
JAVA5  JDK25 install + gradle.properties. Byte-identical toolchain block
       across warm/graded/measure so BuildKit reuses the ~700 MB JDK layer.
JAVA6  DEP-WARM stage is Option A (Oracle's decision): FROM harbor-base AS
       warm, COPY the carved repoctx (no widget src/main/java by construction),
       :buildSrc:build + harborResolveAll init-script resolve-only pass,
       scrub cache, assert no dev/tamboui/widgets anywhere under gradle-home.
       Only /opt/gradle-home is COPY --from=warm'd forward.
JAVA7  DOCKERFILE INVARIANTS. `_assert_dockerfile_invariants` runs inside
       render_dockerfile and rejects `repos-src`, `repo-src`, `FROM warm`,
       `git init` (java does NOT synthesize git), and any second reference to
       the solution_mount.
JAVA8  OFFLINE-SUFFICIENCY PROOF at build time. `pre_leakgate_blocks` runs
       `gradle --offline :dependencies` against every config the graded run
       will touch and fails the IMAGE BUILD if any jar is missing. This
       replaces the reference Dockerfile's build-time :dependencies check.
"""

from __future__ import annotations

import pytest

from taskgen.langs import base as B
from taskgen.langs import java as JAVA


@pytest.fixture
def plugin():
    return JAVA.JavaPlugin()


@pytest.fixture
def graded():
    """A graded set mirroring what emit._measure_and_pin would produce for java-tamboui.

    expected=823 is the pinned intact denominator (JUnit test methods). The
    fingerprint dict here is a stand-in for the ~49-entry lock emit builds
    host-side; it is enough to exercise `fingerprint_gate_block` without
    needing 49 real files.
    """
    return B.GradedSet(
        expected=823,
        floor_mode='equality',
        kind='whole-suite',
        test_command=JAVA.TEST_COMMAND,
        fingerprint_sha256={
            'tamboui-widgets/src/test/java/dev/tamboui/widgets/'
            'ButtonWidgetTest.java': 'a' * 64,
            'tamboui-widgets/src/test/java/dev/tamboui/widgets/'
            'BarChartWidgetTest.java': 'b' * 64,
            'tamboui-widgets/src/test/java/dev/tamboui/widgets/'
            'LabelWidgetTest.java': 'c' * 64,
        },
    )

#: What `emit._render_test_sh` threads in for java: the INDEPENDENT host-side
#: signals the plugin refuses to render without. 69 and 'tamboui-widgets' are
#: java-tamboui's, measured by `emit._java_grader_metadata` from the intact tree.
HOST_SIDE = {'test_suites': 69, 'graded_module': 'tamboui-widgets'}


def render_test_sh(plugin, graded, **kwargs):
    return plugin.render_test_sh(graded, **{**HOST_SIDE, **kwargs})


def test_plugin_declares_its_five_axes(plugin):
    assert plugin.name == 'java'
    assert plugin.toml_family == 'java-hybrid'
    assert plugin.floor_mode == 'equality'
    assert plugin.parser_backed is False
    assert plugin.synthesizes_git is False


def test_java_uses_equality_floor_not_pinned_denominator(plugin):
    """JUnit tests are separate method invocations; equality is real (not G1)."""
    assert JAVA.JavaPlugin.floor_mode == 'equality'


def test_plugin_is_registered_under_its_name():
    from taskgen import langs

    assert langs.get('java') is not None
    assert langs.get('java').name == 'java'


def test_java_is_no_longer_in_the_unlanded_tuple():
    """test_langs_base.py used to assert `B.get('java')` raised Wave 2; this
    plugin's landing must remove that assertion (else the base test regresses)."""
    plugin = B.get('java')
    assert plugin is not None
    assert plugin.name == 'java'


def test_test_command_uses_offline_no_daemon_rerun_console_plain(plugin):
    """The reference test.sh's exact incantation for the graded run."""
    assert '--offline' in plugin.test_command
    assert '--no-daemon' in plugin.test_command
    assert '--rerun' in plugin.test_command
    assert '--console=plain' in plugin.test_command
    assert ':tamboui-widgets:test' in plugin.test_command


def test_selector_kind_registers_java_as_whole_suite():
    from taskgen import gradedset

    assert gradedset.SELECTOR_KIND['java'] == 'whole-suite'


def test_whole_suite_selection_accepts_fingerprint_relpaths():
    from taskgen.gradedset import whole_suite_selection

    sel = whole_suite_selection(
        'java', expected=823, test_command=JAVA.TEST_COMMAND,
        fingerprint_relpaths=(
            'tamboui-widgets/src/test/java/dev/tamboui/widgets/ButtonWidgetTest.java',
            'tamboui-widgets/src/test/java/dev/tamboui/widgets/BarChartWidgetTest.java',
        ),
    )
    assert sel.kind == 'whole-suite'
    assert sel.expected == 823
    assert sel.fingerprint_relpaths == (
        'tamboui-widgets/src/test/java/dev/tamboui/widgets/BarChartWidgetTest.java',
        'tamboui-widgets/src/test/java/dev/tamboui/widgets/ButtonWidgetTest.java',
    )


def test_default_test_globs_covers_java_src_test():
    """java-tamboui tests live under */src/test/**; the guard refuses to carve them."""
    from taskgen.scope import DEFAULT_TEST_GLOBS

    assert DEFAULT_TEST_GLOBS['java'] == ('**/src/test/**',)


def test_stub_phrase_for_java(plugin):
    """java carves --delete-whole-file; the prose stub is the java-body variant."""
    from taskgen.contexts import fence, stub_phrase

    assert stub_phrase('java') == 'throw new UnsupportedOperationException()'
    assert fence('java') == 'java'


def test_default_stub_body_for_java_is_a_java_body():
    """java carve is --delete-whole-file always, but default_stub_body('java')
    must resolve rather than raise (mirrors rust/c/cpp)."""
    from taskgen.carve_file import default_stub_body

    body = default_stub_body('java')
    assert body.startswith('{')
    assert body.endswith('}')
    assert 'UnsupportedOperationException' in body


def test_toolchain_uses_harbor_base_and_declares_java_home(plugin):
    tc = plugin.toolchain_spec()
    assert tc.base_image.startswith('426628337772.dkr.ecr.ap-south-2.amazonaws.com/triton/base-java@sha256:')
    assert tc.workdir == B.WORKDIR
    assert tc.env['JAVA_HOME'] == f'/opt/mise/installs/java/{JAVA.JAVA_VERSION}'
    assert '/opt/mise/shims' in tc.env['PATH']
    assert tc.env['GRADLE_USER_HOME'] == '/opt/gradle-home'


def test_toolchain_selects_jdk25_the_base_already_bakes(plugin):
    """harbor-base ships JDK 21; settings.gradle.kts hard-fails on <25.

    JDK 25 has to be installed here at build time (grading is --network=none).
    """
    block = plugin.toolchain_spec().install_block
    assert 'mise use -g java@temurin-25' in block
    assert 'TOOLCHAIN PIN FAILED' in block


def test_toolchain_writes_gradle_properties_pointing_at_jdk25(plugin):
    """org.gradle.java.home pins the COMPILE JVM for gradle's daemon-less runs."""
    block = plugin.toolchain_spec().install_block
    assert 'gradle.properties' in block
    assert 'org.gradle.java.home' in block
    assert '/opt/mise/installs/java/temurin-25' in block


def test_toolchain_install_block_is_byte_identical_across_calls(plugin):
    """BuildKit dedupes layers by RUN content; a non-deterministic install
    would rebuild the ~700 MB JDK layer in warm, graded and measure separately."""
    a = plugin.toolchain_spec().install_block
    b = plugin.toolchain_spec().install_block
    c = JAVA.JavaPlugin().toolchain_spec().install_block
    assert a == b == c


def test_dep_warm_declares_copy_paths_for_gradle_home(plugin):
    """The graded stage receives /opt/gradle-home via COPY --from=warm."""
    dw = plugin.dep_warm_spec()
    assert dw.copy_paths == (('/opt/gradle-home', '/opt/gradle-home'),)


def test_dep_warm_files_needed_is_empty(plugin):
    """The framework's single-line files_needed COPY mangles directory layout
    for gradle's ~30 build files; the stage_block does all COPYs itself."""
    dw = plugin.dep_warm_spec()
    assert dw.files_needed == ()


def test_dep_warm_stage_copies_carved_repoctx_not_intact_tree(plugin):
    """Warm stage sees ONLY the carved repoctx: widget src/main/java is empty.
    Nothing in the stage can compile widget bytecode."""
    dw = plugin.dep_warm_spec()
    assert 'COPY --from=repoctx repo/' in dw.stage_block


def test_dep_warm_stage_asserts_widget_main_is_empty(plugin):
    """Sanity check that the carve landed: 0 widget main .java files present."""
    dw = plugin.dep_warm_spec()
    assert 'tamboui-widgets/src/main/java' in dw.stage_block
    assert '-eq 0' in dw.stage_block


def test_dep_warm_stage_runs_buildsrc_assemble_and_resolve_all(plugin):
    """The Oracle's warm recipe: :buildSrc:assemble (NOT :build -- see plugin
    docstring for the junit-jupiter versionless-coord reason) + harborResolveAll
    init script."""
    dw = plugin.dep_warm_spec()
    assert 'gradle --no-daemon --console=plain :buildSrc:assemble' in dw.stage_block
    assert 'harborResolveAll' in dw.stage_block
    assert 'harbor-resolve.init.gradle.kts' in dw.stage_block


def test_dep_warm_stage_scrubs_build_cache_and_scratch_dirs(plugin):
    """Local build-cache holds task OUTPUTS -- widget compiled classes would
    be the answer outright. Scrub wholesale."""
    dw = plugin.dep_warm_spec()
    assert 'caches/build-cache-1' in dw.stage_block
    assert '/daemon' in dw.stage_block
    assert '/workers' in dw.stage_block


def test_dep_warm_stage_asserts_no_widget_paths_in_gradle_home(plugin):
    """The load-bearing leak assert: gradle-home must not name dev/tamboui/widgets."""
    dw = plugin.dep_warm_spec()
    assert 'dev/tamboui/widgets' in dw.stage_block
    assert 'LEAK' in dw.stage_block
    assert 'exit 1' in dw.stage_block


def test_pre_leakgate_offline_sufficiency_proof_is_present(plugin):
    """Build-time proof that the warm cache covers every graded configuration."""
    env = B.EnvSpec(repo_name='java-tamboui')
    blocks = plugin.pre_leakgate_blocks(env)
    joined = '\n'.join(blocks)
    assert 'gradle --offline --no-daemon' in joined
    assert ':tamboui-widgets:dependencies' in joined
    assert 'testRuntimeClasspath' in joined
    assert 'testCompileClasspath' in joined
    assert 'compileClasspath' in joined
    assert 'annotationProcessor' in joined
    assert 'Could not resolve' in joined
    assert 'OFFLINE PROOF' in joined


def test_pre_leakgate_resets_project_cache(plugin):
    """After the offline proof, .gradle and per-project build/ must be wiped
    so the graded run is genuinely from-scratch offline."""
    env = B.EnvSpec(repo_name='java-tamboui')
    blocks = plugin.pre_leakgate_blocks(env)
    joined = '\n'.join(blocks)
    assert 'rm -rf .gradle' in joined
    assert 'name build -prune' in joined


def test_no_extra_ctx_assets(plugin):
    """Unlike rust (wabt tarball), java ships nothing extra."""
    assert plugin.extra_ctx_assets() == ()


def test_render_test_sh_bakes_expected_from_graded(plugin, graded):
    """823 is threaded from graded.expected -- NOT a literal in the plugin source."""
    script = render_test_sh(plugin, graded)
    assert 'EXPECTED=823' in script


def test_render_test_sh_bakes_fingerprint_checks(plugin, graded):
    script = render_test_sh(plugin, graded)
    assert 'check_sha256' in script
    a_line = (
        "check_sha256 'tamboui-widgets/src/test/java/dev/tamboui/widgets/"
        "ButtonWidgetTest.java' '" + ('a' * 64) + "'"
    )
    assert a_line in script


def test_render_test_sh_runs_offline_no_daemon_rerun(plugin, graded):
    """The graded run must use exactly the flags the reference test.sh uses."""
    script = render_test_sh(plugin, graded)
    assert 'gradle --offline --no-daemon --console=plain :tamboui-widgets:test --rerun' in script


def test_render_test_sh_removes_stale_junit_xml_before_running(plugin, graded):
    """A stale XML from a prior run would let compilation-skipped tests
    report green via the cached counts."""
    script = render_test_sh(plugin, graded)
    assert 'rm -rf "${RESULTS}"' in script
    assert 'tamboui-widgets/build/test-results/test' in script


def test_render_test_sh_parses_junit_xml_via_python3(plugin, graded):
    """The JUnit XML is authoritative, not Gradle's console summary."""
    script = render_test_sh(plugin, graded)
    assert 'python3 - ' in script
    assert 'xml.etree.ElementTree' in script
    assert "root.get('tests'" in script
    assert "root.get('failures'" in script
    assert "root.get('errors'" in script
    assert "root.get('skipped'" in script


def test_render_test_sh_gates_suites_at_expected_suites(plugin, graded):
    """The structural anti-gaming gate: JUnit fanned out every declared suite."""
    script = render_test_sh(plugin, graded)
    assert '"${SUITES}" -eq "${EXPECTED_SUITES}"' in script
    # 69 is java-tamboui's, and it now reaches the script from emit's own
    # host-side scan of the intact tree rather than from a plugin literal.
    assert 'EXPECTED_SUITES=69' in script
    assert 'EXPECTED_SUITES=61' in render_test_sh(plugin, graded, test_suites=61)


def test_render_test_sh_gates_suites_gt_zero_before_the_floor(plugin, graded):
    """No JUnit XML at all means compile failed; caught ahead of the floor
    so a build failure reports compiled=0.0 rather than tripping the equality gate."""
    script = render_test_sh(plugin, graded)
    assert '"${SUITES}" -eq 0' in script
    assert 'compilation failed' in script


def test_render_test_sh_computes_passed_without_subtracting_skipped_from_denom(
    plugin, graded,
):
    """A skipped test is not a passed test; but the denominator stays at
    EXPECTED so a solver cannot @Disabled their way to a perfect score."""
    script = render_test_sh(plugin, graded)
    assert 'PASSED=$((TESTS - FAILURES - ERRORS - SKIPPED))' in script


def test_render_test_sh_gates_binary_on_exit_zero_and_no_failures(plugin, graded):
    """BINARY==1 requires gradle exit==0 AND zero failures AND zero errors."""
    script = render_test_sh(plugin, graded)
    assert '"${STATUS}" -ne 0' in script
    assert '"${FAILURES}" -ne 0' in script
    assert '"${ERRORS}" -ne 0' in script


def test_render_test_sh_does_not_hardcode_repo_magic_numbers(plugin, graded):
    """823 (test count) must NOT appear as a bash-substantive literal in the source.

    The plugin's own source is inspected here: 823 may appear ONLY inside
    docstrings/comments, never as a bare shell/python literal. It must always
    thread through graded.expected. The suite count is no longer an exception:
    it lives in `JAVA_TAMBOUI_HARNESS` -- java-tamboui's own plan -- and is
    cross-checked against an independent host-side scan.
    """
    import inspect

    src = inspect.getsource(JAVA)
    # Strip module docstring + comments to check only "code" characters.
    lines_no_comments = []
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        lines_no_comments.append(line)
    joined = '\n'.join(lines_no_comments)
    # Strip triple-quoted docstrings (a coarse pass that survives the two-per-
    # file convention rust/c/cpp use).
    import re

    joined = re.sub(r'"""[\s\S]*?"""', '', joined)
    joined = re.sub(r"'''[\s\S]*?'''", '', joined)
    assert '823' not in joined, (
        '823 must not appear as a bare literal in java.py source; it must '
        'thread through graded.expected'
    )


def test_render_measure_dockerfile_is_stripped(plugin):
    """Measure image has NO leak gate, NO tripwire, NO carve-metadata assert."""
    env = B.EnvSpec(repo_name='java-tamboui')
    df = plugin.render_measure_dockerfile(env)
    assert f'FROM {plugin.toolchain_spec().base_image} AS measure' in df
    assert 'NEVER SHIP' in df
    assert 'mise use -g java@temurin-25' in df
    assert 'leakscan.sh' not in df
    assert 'TRIPWIRE_FILE' not in df
    assert 'carve_receipt.json' not in df


def test_render_measure_dockerfile_warms_intact_tree_at_build_time(plugin):
    """Network is allowed at build time; the measure image runs the full
    test suite at build time so measure.sh can re-count offline at runtime."""
    env = B.EnvSpec(repo_name='java-tamboui')
    df = plugin.render_measure_dockerfile(env)
    assert 'gradle --no-daemon --console=plain :tamboui-widgets:test' in df


def test_render_measure_dockerfile_copies_intact_tree_without_repo_prefix(plugin):
    """The measure phase points repoctx at --repo directly; no repo/ prefix
    unlike the shipped Dockerfile (see rust/c/cpp render_measure_dockerfile)."""
    env = B.EnvSpec(repo_name='java-tamboui')
    df = plugin.render_measure_dockerfile(env)
    assert 'COPY --from=repoctx . /opt/harbor/repo/' in df


def test_measure_test_sh_reruns_offline_and_counts_from_xml(plugin):
    """Measure re-runs the test task offline (cache warm from build time)
    and re-counts JUnit XML for the pinned denominator."""
    script = plugin.measure_test_sh()
    assert '--offline' in script
    assert '--rerun' in script
    assert 'xml.etree.ElementTree' in script
    assert 'measure "${TESTS}"' in script


def test_render_dockerfile_ships_no_solution_no_warm_from_no_git_init(plugin):
    """The composed Dockerfile must satisfy _assert_dockerfile_invariants:
    no repos-src / repo-src / FROM warm / git init tokens (all banned)."""
    env = B.EnvSpec(repo_name='java-tamboui')
    df = plugin.render_dockerfile(env)
    # Only comments may name these tokens; the invariant checker inspects
    # INSTRUCTIONS. Verify it did not raise.
    assert f'FROM {plugin.toolchain_spec().base_image} AS graded' in df
    assert 'COPY --from=warm /opt/gradle-home /opt/gradle-home' in df
    # FROM warm as bare instruction (not COPY --from=warm) is banned. Grep it.
    for line in df.splitlines():
        s = line.lstrip()
        if s.startswith('#'):
            continue
        # The base check bans 'FROM warm' in instructions; sanity-check here.
        if s.startswith('FROM ') and 'warm' in s:
            assert 'AS warm' in s, f'illegal FROM-warm reference: {line!r}'


def test_render_dockerfile_includes_the_dep_warm_stage_header(plugin):
    """The warm stage must be declared BEFORE the graded stage (COPY --from=warm
    needs a stage named 'warm' to exist)."""
    env = B.EnvSpec(repo_name='java-tamboui')
    df = plugin.render_dockerfile(env)
    assert f'FROM {plugin.toolchain_spec().base_image} AS warm' in df
    warm_at = df.index(f'FROM {plugin.toolchain_spec().base_image} AS warm')
    graded_at = df.index(f'FROM {plugin.toolchain_spec().base_image} AS graded')
    assert warm_at < graded_at, 'warm stage must precede graded'


def test_render_dockerfile_ships_leak_gate_and_carve_receipt_absence(plugin):
    """Every shipped image runs harbor's leakscan.sh + asserts carve receipt
    is absent. The base class emits both; a plugin override could not remove
    them because render_dockerfile is inherited."""
    env = B.EnvSpec(repo_name='java-tamboui')
    df = plugin.render_dockerfile(env)
    assert 'leakscan.sh' in df
    assert 'carve_receipt.json' in df


# ------------------------------------------ wave 1-2: versioned per-lang base --


def test_java_builds_on_the_versioned_per_language_base(plugin):
    assert plugin.toolchain_spec().base_image.startswith('426628337772.dkr.ecr.ap-south-2.amazonaws.com/triton/base-java@sha256:')


def test_java_home_is_not_architecture_hardcoded(plugin):
    """`/opt/mise/installs/java/temurin-25` does not exist on amd64.

    The base README documents this exact literal as a multi-arch defect it
    fixed; taskgen still carried it, so a task image could only ever build on
    arm64. Deriving JAVA_HOME from the mise install keeps it arch-neutral.
    """
    env = plugin.toolchain_spec().env
    assert 'arm64' not in env['JAVA_HOME']
    assert 'amd64' not in env['JAVA_HOME']
    assert 'arm64' not in plugin.toolchain()


def test_java_does_not_apt_install_a_jdk_the_base_already_carries(plugin):
    assert 'apt-get install' not in plugin.toolchain()


def test_java_pins_the_runtime_and_proves_it_under_a_login_shell(plugin):
    tc = plugin.toolchain()
    assert 'mise use -g java@' in tc
    assert '/etc/profile.d/zz-harbor-toolchain-pin.sh' in tc
    assert 'TOOLCHAIN PIN FAILED' in tc
    assert 'bash -lc' in tc
