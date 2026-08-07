r"""The cpp harness seam: a plan renders the scripts the plugin used to hardcode.

The same discipline as `test_c_harness_golden`, one language over. `render_gap`
proved a DepPlan could reproduce the measure Dockerfile's toolchain bytes; this
module proves a DepPlan can reproduce the two GRADING scripts -- `tests/test.sh`
and the phase-1 measure script -- which is where every remaining cpp-Rux-shaped
constant lived (the seven named `Tests/Unit/*` oracle files, the
`Bin/Tests/Unit/rux-tests` binary path, `-DRUX_WERROR=ON -DRUX_BUILD_TESTS=ON`,
`Build`/`Bin`, and the `CTEST_RAN -eq 1` registry assert).

The digests below were measured against the commit BEFORE the harness became
plan-driven. They are the safety net for an already-proven task: as long as they
hold, cpp-Rux's shipped bytes did not move, and c/java/rust were not in this
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
from taskgen.langs import cpp as CPP

#: sha256 of `render_test_sh` / `measure_test_sh` / `render_solve_sh` for
#: cpp-Rux as rendered by the commit before `DepPlan.harness` reached cpp.
#: `(digest, length)`; the length is carried too so a collision claim has to
#: explain two numbers, not one.
PRE_HARNESS_TEST_SH = (
    '565d1a9616b2b57ffe15d44b3895deb55a60dfa38750f90c4c7e0a14d241d377', 11293,
)
PRE_HARNESS_MEASURE_SH = (
    '82cf5987dc4df7bb40987553bdac46c6b7db05230c62c102e3cccb98cbd596c5', 1658,
)
PRE_HARNESS_SOLVE_SH = (
    '6ef47ac4bd4aaf8bd1bdbb657ced659f200f570241736615a94acdbfc1694864', 1329,
)

#: The real cpp-Rux shape: `Tests/**` fingerprinted, 170 doctest TEST_CASEs,
#: ONE registered ctest, `Compiler/` carved.
FINGERPRINT = {
    'Tests/Unit/CMakeLists.txt': 'a' * 64,
    'Tests/Unit/Main.cpp': 'b' * 64,
    'Tests/Unit/SemanticTests.cpp': 'c' * 64,
}
REGISTERED_TESTS = 1
CARVE_ROOT = 'Compiler'
CARVED = ('Compiler/Semantic/A.cpp',)


@pytest.fixture
def plugin():
    return CPP.CppPlugin()


@pytest.fixture
def graded():
    return B.GradedSet(
        expected=170, floor_mode='equality', kind='whole-suite',
        test_command=CPP.TEST_COMMAND, fingerprint_sha256=FINGERPRINT,
    )


def render_test_sh(plugin, graded, plan=None, **kwargs):
    kwargs.setdefault('registered_tests', REGISTERED_TESTS)
    kwargs.setdefault('carve_root', CARVE_ROOT)
    return plugin.render_test_sh(graded, dep_plan=plan, **kwargs)


def digest(text: str) -> tuple[str, int]:
    return hashlib.sha256(text.encode('utf-8')).hexdigest(), len(text)


# ------------------------------------------------------------- byte lock ----


def test_test_sh_hashes_to_its_pre_harness_digest(plugin, graded):
    """cpp-Rux's verifier bytes did not move when the harness became plan-driven."""
    assert digest(render_test_sh(plugin, graded)) == PRE_HARNESS_TEST_SH


def test_measure_sh_hashes_to_its_pre_harness_digest(plugin, graded):
    assert digest(plugin.measure_test_sh(graded=graded)) == PRE_HARNESS_MEASURE_SH


def test_solve_sh_hashes_to_its_pre_harness_digest(plugin):
    """The oracle's post-restore wipe moved into the plan without moving a byte."""
    assert digest(plugin.render_solve_sh(CARVED)) == PRE_HARNESS_SOLVE_SH


def test_the_canonical_plan_renders_the_same_bytes_as_no_plan(plugin, graded):
    """The load-bearing one: passing cpp-Rux's plan MOVED nothing.

    `dep_plan=None` is the pre-resolution path every older caller takes;
    `CPP_MEASURE_DEP_PLAN` is that same environment written down. If the two
    ever diverged, the fallback would be a second, unreviewed definition of the
    cpp-Rux harness rather than the same one under a name.
    """
    assert render_test_sh(plugin, graded, CPP.CPP_MEASURE_DEP_PLAN) == (
        render_test_sh(plugin, graded)
    )
    assert plugin.measure_test_sh(graded=graded, dep_plan=CPP.CPP_MEASURE_DEP_PLAN) == (
        plugin.measure_test_sh(graded=graded)
    )
    assert plugin.render_solve_sh(CARVED, dep_plan=CPP.CPP_MEASURE_DEP_PLAN) == (
        plugin.render_solve_sh(CARVED)
    )


def test_render_is_deterministic(plugin, graded):
    assert render_test_sh(plugin, graded) == render_test_sh(plugin, graded)
    assert plugin.measure_test_sh(graded=graded) == plugin.measure_test_sh(graded=graded)


def test_the_test_command_is_derived_from_the_plan_not_respelled(plugin):
    """instruction.md's prose and the rendered argv are one statement, not two."""
    assert CPP.TEST_COMMAND == (
        'cmake -S . -B Build -G Ninja -DCMAKE_BUILD_TYPE=Release '
        '-DRUX_WERROR=ON -DRUX_BUILD_TESTS=ON '
        '&& cmake --build Build --config Release --parallel 4 '
        '&& ctest --test-dir Build --output-on-failure -C Release'
    )
    assert plugin.test_command_from_plan(None) == CPP.TEST_COMMAND
    assert plugin.test_command_from_plan(CPP.CPP_MEASURE_DEP_PLAN) == CPP.TEST_COMMAND


def test_the_renderer_carries_no_cpp_rux_constant():
    """The point of the slice: cpp-the-language, not cpp-Rux-the-repository.

    Scoped to the PLUGIN CLASS, which is everything that renders. The cpp-Rux
    facts are still allowed to appear in `CPP_RUX_HARNESS` (that repository's
    own plan) and as `e.g.` illustrations inside the slot descriptions the
    resolver prompt is built from -- an example is what makes a model fill a
    slot right on attempt 1, and it reaches no rendered byte.
    """
    import inspect

    renderer = inspect.getsource(CPP.CppPlugin)
    for token in ('rux-tests', 'Bin/Tests', 'Tests/Unit', 'RUX_WERROR',
                  'RUX_BUILD_TESTS', 'UnitTests', 'doctest.h', 'Compiler/'):
        assert token not in renderer, (
            f'{token!r} is still hardcoded in the cpp renderer'
        )


# ------------------------------------------------------------- the seam -----


def _swapped_plan() -> depplan.DepPlan:
    """A DIFFERENT repo's harness: lowercase build dirs, no -Werror, no notes."""
    plan = dataclasses.replace(
        CPP.CPP_MEASURE_DEP_PLAN,
        test_invocation=(
            ('build', ('cmake', '--build', 'build')),
            ('configure', ('cmake', '-S', '.', '-B', 'build',
                           '-DDEMO_BUILD_TESTS=ON')),
            ('test', ('ctest', '--test-dir', 'build')),
        ),
        harness=(
            ('build_artifacts', ('build/test/demo_tests',)),
            ('harness_files', ('test/main.cpp',)),
            ('project_label', 'demo'),
            ('registered_test_names', ('demo-unit',)),
            ('registered_tests', 1),
            ('stale_paths', ('build',)),
        ),
    )
    validate(plan)
    return canonicalize(plan)


def test_a_different_harness_renders_a_different_script(plugin, graded):
    """The seam is live, and it moves exactly the places it should."""
    plan = _swapped_plan()
    plugin.validate_dep_plan(plan)
    swapped = dataclasses.replace(
        graded, fingerprint_sha256={'test/main.cpp': 'd' * 64},
    )
    script = render_test_sh(plugin, swapped, plan, carve_root='src')

    assert '# test/ tree is the doctest oracle.' in script

    assert 'TEST_BIN=build/test/demo_tests' in script
    assert 'rm -rf build\n' in script
    assert 'cmake -S . -B build \\' in script
    assert '      -DDEMO_BUILD_TESTS=ON \\' in script
    assert 'cmake --build build --parallel 4 \\' in script
    assert 'ctest --test-dir build \\' in script
    assert 'demo registers `demo-unit`, the single demo_tests binary' in script
    assert '[ -f "$f" ] || fail "graded harness file missing: $f"' in script
    assert '(expected while src/* is carved)' in script
    assert 'macOS.yml' not in script, 'the -Werror rationale is not this repo\'s'

    for gone in ('rux-tests', 'Bin/Tests', 'Tests/Unit', 'RUX_WERROR',
                 'RUX_BUILD_TESTS', 'UnitTests', 'Compiler/'):
        assert gone not in script, f'{gone!r} survived a harness swap'


def test_a_different_harness_renders_a_different_measure_script(plugin, graded):
    script = plugin.measure_test_sh(graded=graded, dep_plan=_swapped_plan())
    assert 'TEST_BIN=build/test/demo_tests' in script
    assert 'measure: no ${TEST_BIN} produced by intact build' in script
    assert 'rux-tests' not in script
    assert 'RUX_WERROR' not in script
    assert 'measure "${CASES}"' in script


def test_a_different_harness_renders_a_different_oracle(plugin):
    script = plugin.render_solve_sh(CARVED, dep_plan=_swapped_plan())
    assert 'rm -rf "${REPO}/build"' in script
    assert '${REPO}/Bin' not in script


def test_the_swapped_plan_still_validates_and_digests(plugin):
    plan = _swapped_plan()
    plugin.validate_dep_plan(plan)
    assert depplan.dep_plan_digest(plan) != (
        depplan.dep_plan_digest(CPP.CPP_MEASURE_DEP_PLAN)
    )


def test_a_harness_with_no_optional_prose_still_renders(plugin, graded):
    """"" for an inapplicable optional slot must render, not explode.

    The prompt tells a resolver to answer "" for a slot its repo has no answer
    for, so rejecting it would punish a model for following the ask.
    """
    plan = dataclasses.replace(
        CPP.CPP_MEASURE_DEP_PLAN,
        harness=tuple(
            (k, '' if k in ('project_label', 'harness_files_note') else v)
            for k, v in CPP.CPP_MEASURE_DEP_PLAN.harness
        ),
    )
    validate(plan)
    script = render_test_sh(plugin, graded, canonicalize(plan))
    assert 'the build registers `UnitTests`' in script
    assert 'must all still be present.' in script


# ------------------------------------------- the under-enumeration guard ----


def test_a_plan_that_under_enumerates_the_registry_is_refused(plugin, graded):
    """THE GUARD. A shrunken denominator is self-consistent at the wrong number.

    `CTEST_RAN -eq 1` used to be the structural net that caught this. Now that
    the count is plan-fed, the net is an INDEPENDENT host-side scan of the
    intact tree's own `add_test` registrations, and the two must agree.
    """
    with pytest.raises(B.LangError, match='under-enumerates'):
        render_test_sh(plugin, graded, registered_tests=3)


def test_a_plan_that_over_enumerates_the_registry_is_refused_too(plugin, graded):
    """Symmetric: a count nobody can satisfy fails the runtime assert anyway."""
    plan = dataclasses.replace(
        CPP.CPP_MEASURE_DEP_PLAN,
        harness=tuple(
            (k, ('a', 'b') if k == 'build_artifacts' else
                (2 if k == 'registered_tests' else
                 ('x', 'y') if k == 'registered_test_names' else v))
            for k, v in CPP.CPP_MEASURE_DEP_PLAN.harness
        ),
    )
    with pytest.raises(B.LangError, match='grades ONE doctest binary'):
        plugin.validate_dep_plan(plan)


def test_registered_tests_must_match_the_binaries_that_serve_them(plugin):
    """A registration the plan does not name a binary for is graded by nobody."""
    plan = dataclasses.replace(
        CPP.CPP_MEASURE_DEP_PLAN,
        harness=tuple(
            (k, 4 if k == 'registered_tests' else v)
            for k, v in CPP.CPP_MEASURE_DEP_PLAN.harness
            if k != 'registered_test_names'
        ),
    )
    with pytest.raises(B.LangError, match='denominator silently shrinks'):
        plugin.validate_dep_plan(plan)


def test_a_short_registered_names_column_is_refused(plugin):
    plan = dataclasses.replace(
        CPP.CPP_MEASURE_DEP_PLAN,
        harness=tuple(
            (k, () if k == 'registered_test_names' else
                (2 if k == 'registered_tests' else v))
            for k, v in CPP.CPP_MEASURE_DEP_PLAN.harness
        ),
    )
    with pytest.raises(B.LangError, match='registered_tests'):
        plugin.validate_dep_plan(plan)


def test_the_host_side_scan_counts_add_test_in_the_intact_tree(tmp_path):
    """The independent signal, and its determinism: generated cmake is skipped."""
    from taskgen.emit import _cpp_grader_metadata, plan_carve

    repo = tmp_path / 'repo'
    (repo / 'Compiler').mkdir(parents=True)
    (repo / 'Tests' / 'Unit').mkdir(parents=True)
    (repo / 'Build').mkdir()
    (repo / 'CMakeLists.txt').write_text(
        'project(fake)\nadd_subdirectory(Tests/Unit)\n'
    )
    (repo / 'Compiler' / 'A.cpp').write_text('int a;\n')
    (repo / 'Tests' / 'Unit' / 'CMakeLists.txt').write_text(
        '# add_test in a comment does not count\n'
        'add_test(NAME One COMMAND t)\n'
        'ADD_TEST(NAME Two COMMAND t)\n'
    )
    (repo / 'Tests' / 'Unit' / 'Main.cpp').write_text('int main(){}\n')
    (repo / 'Build' / 'CTestTestfile.cmake').write_text(
        'add_test(One "t")\nadd_test(Two "t")\n'
    )

    plan = plan_carve(
        repo, tmp_path / 'out', lang='cpp', carve_scope='folder',
        include=('Compiler/**',), delete_whole_file=True,
    )
    assert _cpp_grader_metadata(plan) == {'registered_tests': 2}


def test_the_registry_cross_check_reaches_the_refine_loop(tmp_path):
    """The guard has to fire where a plan can still be REPAIRED.

    `_candidate_validator` is what `_measure_and_pin` hands the refine loop, so
    a plan whose registry disagrees with the tree comes back as one more
    resolver attempt with a precise message -- not as an abort halfway through
    writing eleven entries.
    """
    from taskgen.emit import _candidate_validator, plan_carve

    repo = tmp_path / 'repo'
    (repo / 'Compiler').mkdir(parents=True)
    (repo / 'Tests').mkdir()
    (repo / 'CMakeLists.txt').write_text('project(fake)\n')
    (repo / 'Compiler' / 'A.cpp').write_text('int a;\n')
    (repo / 'Tests' / 'CMakeLists.txt').write_text(
        'add_test(NAME One COMMAND t)\nadd_test(NAME Two COMMAND t)\n'
    )
    (repo / 'Tests' / 'Main.cpp').write_text('int main(){}\n')

    plan = plan_carve(
        repo, tmp_path / 'out', lang='cpp', carve_scope='folder',
        include=('Compiler/**',), delete_whole_file=True,
    )
    validate_candidate = _candidate_validator(CPP.CppPlugin(), plan)
    with pytest.raises(B.LangError, match='under-enumerates'):
        validate_candidate(CPP.CPP_MEASURE_DEP_PLAN)


def test_a_registration_inside_a_repeated_cmake_block_refuses_to_be_counted(tmp_path):
    """The scan must never answer LOW. A helper called ten times registers ten.

    Observed on a real candidate repo: one textual `add_test` inside a cmake
    `function(make_test ...)` invoked once per test target and per C++ standard.
    A regex count would have said 1, the plan would have agreed with it, and the
    task would have shipped a floor over one binary out of ten -- self-consistent
    at the wrong number, which is exactly the failure this guard exists for.
    """
    from taskgen.emit import _cpp_grader_metadata, plan_carve

    repo = tmp_path / 'repo'
    (repo / 'Compiler').mkdir(parents=True)
    (repo / 'Tests').mkdir()
    (repo / 'CMakeLists.txt').write_text('project(fake)\n')
    (repo / 'Compiler' / 'A.cpp').write_text('int a;\n')
    (repo / 'Tests' / 'CMakeLists.txt').write_text(
        'function(make_test target)\n'
        '  add_test(NAME ${target} COMMAND ${target})\n'
        'endfunction()\n'
        'make_test(one)\n'
        'make_test(two)\n'
    )
    (repo / 'Tests' / 'Main.cpp').write_text('int main(){}\n')

    plan = plan_carve(
        repo, tmp_path / 'out', lang='cpp', carve_scope='folder',
        include=('Compiler/**',), delete_whole_file=True,
    )
    with pytest.raises(B.LangError, match='repeated cmake block'):
        _cpp_grader_metadata(plan)


# ----------------------------------------------------------- refusals -------


@pytest.mark.parametrize(
    ('drop', 'message'),
    [
        ('build_artifacts', 'build_artifacts'),
        ('registered_tests', 'registered_tests'),
    ],
)
def test_every_slot_the_renderer_reads_is_enforced(plugin, drop, message):
    """A bad plan must be a REPAIRABLE refine signal, never a crash mid-render."""
    plan = dataclasses.replace(
        CPP.CPP_MEASURE_DEP_PLAN,
        harness=tuple(
            (k, v) for k, v in CPP.CPP_MEASURE_DEP_PLAN.harness if k != drop
        ),
    )
    with pytest.raises(B.LangError, match=message):
        plugin.validate_dep_plan(plan)


def test_a_zero_registry_is_refused(plugin):
    plan = dataclasses.replace(
        CPP.CPP_MEASURE_DEP_PLAN,
        harness=tuple(
            (k, 0 if k == 'registered_tests' else v)
            for k, v in CPP.CPP_MEASURE_DEP_PLAN.harness
        ),
    )
    with pytest.raises(B.LangError, match='at least 1'):
        plugin.validate_dep_plan(plan)


def test_a_scalar_where_a_list_belongs_is_refused(plugin):
    plan = dataclasses.replace(
        CPP.CPP_MEASURE_DEP_PLAN,
        harness=tuple(
            (k, 3 if k == 'build_artifacts' else v)
            for k, v in CPP.CPP_MEASURE_DEP_PLAN.harness
        ),
    )
    with pytest.raises(B.LangError, match='not a scalar'):
        plugin.validate_dep_plan(plan)


def test_render_test_sh_without_the_host_side_count_refuses(plugin, graded):
    with pytest.raises(B.LangError, match='registered_tests threaded'):
        plugin.render_test_sh(graded)


@pytest.mark.parametrize('parallel', ['8', '64', 'all', ''])
def test_a_build_that_exceeds_the_memory_cap_is_refused(plugin, parallel):
    """CPP5 stays a plugin invariant: the plan may not raise the OOM ceiling."""
    tokens = ('cmake', '--build', 'Build', '--parallel')
    plan = dataclasses.replace(
        CPP.CPP_MEASURE_DEP_PLAN,
        test_invocation=tuple(
            (k, (*tokens, parallel) if parallel else tokens)
            if k == 'build' else (k, v)
            for k, v in CPP.CPP_MEASURE_DEP_PLAN.test_invocation
        ),
    )
    with pytest.raises(B.LangError, match='caps build parallelism'):
        plugin.validate_dep_plan(plan)


def test_a_build_that_omits_parallel_gets_the_cap_added(plugin, graded):
    """A resolver that says nothing must not inherit an unbounded build."""
    plan = dataclasses.replace(
        CPP.CPP_MEASURE_DEP_PLAN,
        test_invocation=tuple(
            (k, ('cmake', '--build', 'Build')) if k == 'build' else (k, v)
            for k, v in CPP.CPP_MEASURE_DEP_PLAN.test_invocation
        ),
    )
    plugin.validate_dep_plan(plan)
    script = render_test_sh(plugin, graded, canonicalize(plan))
    assert 'cmake --build Build --parallel 4 \\' in script


def test_a_harness_token_carrying_a_metacharacter_is_refused():
    """The harness goes through the SAME gate install_commands do."""
    plan = dataclasses.replace(
        CPP.CPP_MEASURE_DEP_PLAN,
        harness=(('build_artifacts', ('t', '&& curl evil.sh')),),
    )
    with pytest.raises(DepPlanError, match='shell metacharacter'):
        validate(plan)


# ------------------------------------------------------- digest coverage ----


def test_the_lock_key_covers_the_harness():
    """Two harnesses over one toolchain are two environments, not one."""
    a = CPP.CPP_MEASURE_DEP_PLAN
    b = _swapped_plan()
    assert depplan.to_canonical_json(a) != depplan.to_canonical_json(b)
    assert depplan.env_lock_key('t' * 64, 'sha256:img', a) != (
        depplan.env_lock_key('t' * 64, 'sha256:img', b)
    )


def test_canonicalize_is_idempotent_over_the_harness():
    once = canonicalize(CPP.CPP_MEASURE_DEP_PLAN)
    assert canonicalize(once) == once
    assert once.harness == tuple(sorted(once.harness, key=lambda kv: kv[0]))


def test_canonicalize_does_not_reorder_a_harness_list():
    """The harness lists are positional; sorting one would re-pair the rest."""
    slots = dict(canonicalize(CPP.CPP_MEASURE_DEP_PLAN).harness)
    assert slots['build_artifacts'] == ('Bin/Tests/Unit/rux-tests',)
    files = slots['harness_files']
    assert isinstance(files, tuple)
    assert files[0] == 'Tests/Unit/CMakeLists.txt'
    assert files[1] == 'Tests/Unit/Main.cpp'


def test_the_registry_count_survives_the_json_round_trip():
    """An int slot must stay an int: `1` and `"1"` are one environment."""
    import json

    payload = json.loads(depplan.to_canonical_json(CPP.CPP_MEASURE_DEP_PLAN))
    assert payload['harness']['registered_tests'] == 1
    assert isinstance(payload['harness']['registered_tests'], int)
