"""The go plugin: pinned-denominator floor, -json parsing, offline toolchain.

Docker is MOCKED here -- nothing below builds an image. What is asserted is the
TEXT the plugin renders, because that text is the whole of the contract the
Wave 4 dry-run will execute, and every property it must have is a property of
the string.

The three go-specific hazards the S-GO spike measured, and where each is tested:

  G1  a panic ABORTS the whole test binary, so only the first selected test ever
      emits an event. `TOTAL == EXPECTED` therefore mis-fires as "refusing
      partial credit" on precisely the runs that should score 0. The floor is
      pinned-denominator: reward = TOP_PASS/EXPECTED, with a scope-GROWTH guard
      (observed <= expected) and sha256 + file-count locks as the anti-SHRINK
      half.
  G2  a zero-value stub scores 0.25 by accident, so the carve stub is
      `panic("not implemented")`. That is the carver's job, but the reward
      arithmetic here is what makes the distinction visible.
  G8  the graded run is fully offline: GOPROXY=off, GOTOOLCHAIN=local, and a
      module cache warmed in a SEPARATE image.
"""

from __future__ import annotations

import subprocess

import pytest

from taskgen.langs import base as B
from taskgen.langs import go as GO

PKG = 'github.com/multigres/multigres/go/common/pgprotocol/server'
TESTS = (
    'TestAssignConnectionID_EncodesGatewayPrefix',
    'TestAssignConnectionID_FailsWhenAllIDsUsed',
    'TestAssignConnectionID_SkipsInUsePIDs',
    'TestAssignConnectionID_WrapsAndFindsSlot',
)
TEST_SHA = '9369f5d1393d4a0349f4f6e3bd34d5e51cbb7cec54f02cd3a61e9664678e64af'
TEST_REL = 'go/common/pgprotocol/server/listener_test.go'


@pytest.fixture
def plugin():
    return GO.GoPlugin()


@pytest.fixture
def graded():
    """The S-GO frozen graded set."""
    return B.GradedSet(
        expected=4,
        floor_mode='pinned-denominator',
        kind='go-run',
        selectors=TESTS,
        packages=(PKG,),
        fingerprint_sha256={TEST_REL: TEST_SHA},
        test_command=f"go test -run '^({'|'.join(TESTS)})$' {PKG}",
    )


# ------------------------------------------------------------------ axes ----


def test_plugin_declares_its_three_axes(plugin):
    assert plugin.name == 'go'
    assert plugin.toml_family == 'A'
    assert plugin.floor_mode == 'pinned-denominator'
    assert plugin.parser_backed is True


def test_go_is_the_one_language_that_is_not_equality_floored():
    """Stated as a test so nobody 'fixes' it back to harbor's default (G1)."""
    assert GO.GoPlugin.floor_mode == 'pinned-denominator'


def test_plugin_is_registered_under_its_name():
    from taskgen import langs

    assert isinstance(langs.get('go'), GO.GoPlugin)
    assert 'go' in langs.available()


# ------------------------------------------------------- axis 1: toolchain --


def test_toolchain_installs_the_pinned_go_tarball(plugin):
    block = plugin.toolchain()
    assert 'go1.26.5' in block
    assert '/opt/go1.26' in block
    assert '.tar.gz' in block


def test_toolchain_puts_go_on_path_and_pins_the_toolchain_to_local(plugin):
    """GOTOOLCHAIN=local stops a `go 1.2x` directive downloading a compiler --
    which offline is a build failure, and online is an unpinned toolchain."""
    env = plugin.toolchain_spec().env
    assert env['GOTOOLCHAIN'] == 'local'
    assert '/opt/go1.26/bin' in env['PATH']
    assert env['GOROOT'] == '/opt/go1.26'


def test_the_graded_image_is_offline_by_environment_not_by_convention(plugin):
    assert plugin.toolchain_spec().env['GOPROXY'] == 'off'


# -------------------------------------------------------- axis 2: dep warm --


def test_dep_warm_downloads_modules_in_a_scratch_dir_never_the_repo(plugin):
    spec = plugin.dep_warm_spec()
    assert 'go mod download all' in spec.stage_block
    assert spec.files_needed == ('go.mod', 'go.sum')
    assert '/opt/modwarm' in spec.stage_block


def test_dep_warm_is_the_only_step_allowed_a_proxy(plugin):
    spec = plugin.dep_warm_spec()
    assert 'GOPROXY=' in spec.stage_block
    assert 'GOPROXY=off' not in spec.stage_block


def test_dep_warm_reaches_the_graded_image_by_copy_not_by_from(plugin):
    """invariant 7: `FROM warm` inherits every layer the warm build touched."""
    block = plugin.dep_warm()
    assert 'COPY --from=warm /opt/go/pkg/mod /opt/go/pkg/mod' in block
    assert 'FROM warm' not in block


# ------------------------------------------------- axis 3: the test command --


def test_test_sh_runs_the_anchored_run_regex_over_the_owning_packages(plugin, graded):
    sh = plugin.render_test_sh(graded)
    assert f"-run '^({'|'.join(TESTS)})$'" in sh
    assert PKG in sh


def test_test_sh_defeats_the_test_cache_and_bounds_the_run(plugin, graded):
    sh = plugin.render_test_sh(graded)
    assert '-count=1' in sh, 'a cached PASS would grade the previous build'
    assert '-short' in sh
    assert '-timeout=' in sh


def test_test_sh_asks_for_json_because_the_plain_output_has_no_per_test_verdict(
    plugin, graded
):
    assert '-json' in plugin.render_test_sh(graded)


# ---------------------------------------------- axis 4: the -json reward -----


def test_test_sh_counts_only_TOP_LEVEL_tests(plugin, graded):
    """`"Test":"TestX/sub"` is a subtest; counting it would inflate the
    numerator past the pinned denominator and score >1."""
    sh = plugin.render_test_sh(graded)
    assert '[^"/]' in sh, 'the top-level filter must exclude Test names with a /'
    assert '"Action":"pass"' in sh


def test_test_sh_emits_every_common_reward_key(plugin, graded):
    sh = plugin.render_test_sh(graded)
    for key in B.REWARD_KEYS:
        assert f'"{key}"' in sh, f'reward.json is missing {key}'
    assert 'reward.txt' in sh


def test_test_sh_is_fail_closed_with_a_zero_on_disk_before_go_runs(plugin, graded):
    sh = plugin.render_test_sh(graded)
    assert sh.index('emit 0.0') < sh.index('go test')


def test_reward_divides_by_the_PINNED_expected_never_by_the_observed_total(
    plugin, graded
):
    """G1: a panic aborts the binary, so the observed total is not a denominator."""
    sh = plugin.render_test_sh(graded)
    assert 'EXPECTED=4' in sh
    assert '-v e="${EXPECTED}"' in sh
    assert 'p / e' in sh


def test_scope_growth_is_guarded_but_a_short_run_is_not_punished(plugin, graded):
    sh = plugin.render_test_sh(graded)
    assert 'scope-growth' in sh
    assert '-gt "${EXPECTED}"' in sh
    assert '-ne "${EXPECTED}"' not in sh, 'equality would mis-fire on a panic (G1)'


def test_compiled_is_zero_when_the_package_did_not_build(plugin, graded):
    sh = plugin.render_test_sh(graded)
    assert 'build failed' in sh or 'build-fail' in sh
    assert 'COMPILED=0.0' in sh
    assert 'COMPILED=1.0' in sh


# --------------------------------------------- axis 5: the anti-shrink lock --


def test_test_sh_pins_the_sha256_of_every_graded_test_file(plugin, graded):
    sh = plugin.render_test_sh(graded)
    assert TEST_REL in sh
    assert TEST_SHA in sh


def test_test_sh_locks_the_number_of_test_files_in_the_graded_package(plugin, graded):
    """The sha lock pins the graded FILE; the count lock pins the graded
    PACKAGE, so a helper `_test.go` cannot be deleted to change what compiles."""
    sh = plugin.render_test_sh(
        graded, test_file_counts={'go/common/pgprotocol/server': 25},
    )
    assert '_test.go' in sh
    assert '25' in sh


def test_the_count_lock_is_visibly_absent_rather_than_silently_skipped(plugin, graded):
    sh = plugin.render_test_sh(graded)
    assert 'no _test.go count lock' in sh


# ------------------------------------------------ the floor-FREE measure sh --


def test_measure_lists_the_selected_tests_instead_of_running_them(plugin, graded):
    sh = plugin.measure_test_sh(graded=graded)
    assert '-list' in sh
    assert PKG in sh


def test_measure_asserts_no_floor_at_all(plugin, graded):
    """A floor cannot be enforced by the run that is supposed to discover it."""
    sh = plugin.measure_test_sh(graded=graded)
    assert 'EXPECTED' not in sh
    assert 'measure.json' in sh
    for key in B.MEASURE_KEYS:
        assert key in sh


# ------------------------------------------------------- axis 7: the oracle --


def test_solve_sh_restores_from_the_run_time_mount(plugin):
    sh = plugin.render_solve_sh(('go/common/pgprotocol/server/listener.go',))
    assert B.SOLUTION_MOUNT in sh
    assert 'COPY' not in sh


def test_post_restore_invalidates_the_build_cache_of_the_restored_code(plugin):
    """Go caches compiled packages by content; a stale entry is not a risk, but
    a stale TEST RESULT is -- `-count=1` and this hook are the two defences."""
    assert 'go clean' in plugin.post_restore_block()


# ----------------------------------------------------- the image invariants --


def _df(plugin) -> str:
    return plugin.render_dockerfile(B.EnvSpec(repo_name='go-multigres'))


def test_dockerfile_ships_only_the_host_carved_tree(plugin):
    df = _df(plugin)
    assert f'COPY --from={B.REPO_CONTEXT} repo/ /opt/harbor/repo' in df
    assert 'repos-src' not in df
    assert 'repo-src' not in df


def test_dockerfile_bakes_no_oracle(plugin):
    df = _df(plugin)
    assert 'COPY --from=entry solution' not in df
    assert df.count(B.SOLUTION_MOUNT) == 1, 'only the absence assertion may name it'


def test_dockerfile_runs_the_leak_scan_and_the_no_metadata_assertions(plugin):
    df = _df(plugin)
    assert 'leakscan.sh' in df
    assert 'carve_receipt.json' in df


def test_dockerfile_asserts_the_pinned_toolchain_and_a_warm_module_cache(plugin):
    df = _df(plugin)
    assert 'go version' in df
    assert 'GOTOOLCHAIN' in df


def test_dockerfile_never_compiles_the_carved_repo_at_build_time(plugin):
    """A compiled artifact of the carved package carries its export data
    (invariant 5). The go image warms modules only; the honest rebuild happens
    at grade time."""
    df = _df(plugin)
    assert 'go build ./...' not in df
    assert 'go test' not in df


# -------------------------------------------------------------- determinism --


@pytest.mark.parametrize('render', [
    lambda p, g: p.render_test_sh(g),
    lambda p, g: p.render_test_sh(g, test_file_counts={'a': 2, 'b': 3}),
    lambda p, g: p.measure_test_sh(graded=g),
    lambda p, g: p.render_solve_sh(('a.go', 'b/c.go')),
    lambda p, g: p.render_dockerfile(B.EnvSpec(repo_name='go-multigres')),
])
def test_every_render_is_byte_stable(render, plugin, graded):
    assert render(plugin, graded) == render(GO.GoPlugin(), graded)


def test_multi_package_selection_is_sorted(plugin):
    g = B.GradedSet(expected=2, floor_mode='pinned-denominator',
                    selectors=('TestB', 'TestA'), packages=('z/pkg', 'a/pkg'))
    sh = plugin.render_test_sh(g)
    assert sh.index('a/pkg') < sh.index('z/pkg')
    assert '^(TestA|TestB)$' in sh


@pytest.mark.parametrize('render', [
    lambda p, g: p.render_test_sh(g, test_file_counts={'go/x': 25}),
    lambda p, g: p.measure_test_sh(graded=g),
    lambda p, g: p.render_solve_sh(('a.go',)),
])
def test_every_rendered_script_is_valid_bash(render, plugin, graded, tmp_path):
    script = tmp_path / 's.sh'
    script.write_text(render(plugin, graded))
    proc = subprocess.run(['bash', '-n', str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# ------------------------------------- the reward arithmetic, actually run ---


def _run(sh: str, tmp_path, json_lines: str):
    """Execute the rendered test.sh with `go` and `sha256sum` MOCKED.

    Asserting on the text proves the script SAYS the right thing; running it
    against a captured -json transcript proves it COMPUTES the right thing,
    which is the half that a typo in an awk expression would otherwise hide
    until a 15-minute docker leg in Wave 4.
    """
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    (bin_dir / 'go').write_text(
        '#!/usr/bin/env bash\n'
        f'cat <<\'EOF\'\n{json_lines}\nEOF\n'
        'exit 0\n'
    )
    (bin_dir / 'sha256sum').write_text(
        '#!/usr/bin/env bash\n'
        f'echo "{TEST_SHA}  $1"\n'
    )
    for name in ('go', 'sha256sum'):
        (bin_dir / name).chmod(0o755)

    repo = tmp_path / 'repo'
    (repo / TEST_REL).parent.mkdir(parents=True)
    (repo / TEST_REL).write_text('package server\n')
    logs = tmp_path / 'logs'

    script = tmp_path / 'test.sh'
    script.write_text(sh)
    env = {
        'PATH': f'{bin_dir}:/usr/bin:/bin',
        'REPO': str(repo),
        'VERIFIER_DIR': str(logs),
        'HOME': str(tmp_path),
    }
    subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True,
                   check=False)
    import json

    return json.loads((logs / 'reward.json').read_text())


def _event(action, test=None):
    t = f',"Test":"{test}"' if test else ''
    return f'{{"Time":"2026-01-01T00:00:00Z","Action":"{action}","Package":"{PKG}"{t}}}'


def test_all_four_passing_scores_one(plugin, graded, tmp_path):
    lines = '\n'.join(
        [_event('run', t) for t in TESTS] + [_event('pass', t) for t in TESTS]
    )
    got = _run(plugin.render_test_sh(graded), tmp_path, lines)
    assert got == {'reward': 1.0, 'tests_passed': 4, 'tests_total': 4,
                   'binary': 1.0, 'compiled': 1.0}


def test_a_panic_that_aborts_the_binary_scores_zero_not_a_floor_violation(
    plugin, graded, tmp_path
):
    """G1 in one test: the panic kills the binary after ONE event, so the
    observed total is 1. An equality floor would call that a broken denominator
    and refuse to grade; the pinned floor scores it 0/4, which is the truth."""
    lines = '\n'.join([_event('run', TESTS[0]), _event('fail', TESTS[0])])
    got = _run(plugin.render_test_sh(graded), tmp_path, lines)
    assert got['reward'] == 0.0
    assert got['binary'] == 0.0
    assert got['tests_passed'] == 0


def test_partial_credit_is_over_the_pinned_denominator(plugin, graded, tmp_path):
    lines = '\n'.join(
        [_event('run', t) for t in TESTS[:2]] + [_event('pass', TESTS[0]),
                                                 _event('fail', TESTS[1])]
    )
    got = _run(plugin.render_test_sh(graded), tmp_path, lines)
    assert got['reward'] == 0.25, 'one of the PINNED four, not one of the two seen'
    assert got['tests_passed'] == 1
    assert got['binary'] == 0.0


def test_subtests_do_not_inflate_the_numerator(plugin, graded, tmp_path):
    lines = '\n'.join(
        [_event('run', t) for t in TESTS]
        + [_event('pass', t) for t in TESTS]
        + [_event('pass', f'{TESTS[0]}/case_one'), _event('pass', f'{TESTS[0]}/case_two')]
    )
    got = _run(plugin.render_test_sh(graded), tmp_path, lines)
    assert got['tests_passed'] == 4
    assert got['reward'] == 1.0


def test_scope_growth_fails_closed(plugin, graded, tmp_path):
    """More top-level tests than were pinned means the graded set moved."""
    extra = TESTS + ('TestSomethingNew',)
    lines = '\n'.join([_event('run', t) for t in extra] + [_event('pass', t) for t in extra])
    got = _run(plugin.render_test_sh(graded), tmp_path, lines)
    assert got['reward'] == 0.0


def test_a_build_failure_reports_compiled_zero(plugin, graded, tmp_path):
    lines = 'FAIL\t' + PKG + ' [build failed]'
    got = _run(plugin.render_test_sh(graded), tmp_path, lines)
    assert got['compiled'] == 0.0
    assert got['reward'] == 0.0


def test_a_tampered_test_file_fails_the_fingerprint_gate(plugin, graded, tmp_path):
    sh = plugin.render_test_sh(
        graded, fingerprint={TEST_REL: 'ff' * 32},
    )
    lines = '\n'.join([_event('run', t) for t in TESTS] + [_event('pass', t) for t in TESTS])
    got = _run(sh, tmp_path, lines)
    assert got['reward'] == 0.0, 'a rewritten graded test must not score'


# ------------------------------- every COPY --from must resolve (real docker) --


def _copy_from_names(text: str) -> set[str]:
    return {
        line.split('--from=', 1)[1].split()[0]
        for line in text.splitlines()
        if line.lstrip().startswith('COPY') and '--from=' in line
    }


def _declared_stages(text: str) -> set[str]:
    return {
        line.rsplit(' AS ', 1)[1].strip()
        for line in text.splitlines()
        if line.startswith('FROM ') and ' AS ' in line
    }


def test_the_dockerfile_declares_the_warm_stage_it_copies_from(plugin):
    """A `COPY --from=warm` with no `FROM ... AS warm` is not a stage reference.

    Docker resolves the unknown name as an IMAGE, so the build tried to pull
    `docker.io/library/warm:latest` and died on authorization. Nothing in the
    mocked suite could see it: `dep_warm()` was only ever asserted in
    isolation, never against the assembled file.
    """
    env = B.EnvSpec(repo_name='go-multigres')
    text = plugin.render_dockerfile(env)
    named_contexts = {
        env.repo_context, env.entry_context, env.trip_context, env.tooling_context,
    }
    unresolved = _copy_from_names(text) - _declared_stages(text) - named_contexts
    assert not unresolved, f'COPY --from={unresolved} resolves to no stage or context'


def test_the_warm_stage_never_sees_the_repo(plugin):
    """invariant 7/2: a warm stage with the sources has the answer in a layer."""
    env = B.EnvSpec(repo_name='go-multigres')
    text = plugin.render_dockerfile(env)
    warm, _, graded = text.partition(f'FROM harbor-base:local AS {env.stage}')
    assert 'AS warm' in warm
    assert f'COPY --from={env.repo_context} repo/ ' not in warm, 'warm stage got the tree'
    for manifest in plugin.dep_warm_spec().files_needed:
        assert manifest in warm, f'{manifest} never reaches the warm stage'
    directives = [ln for ln in graded.splitlines() if ln.startswith('FROM ')]
    assert not any(ln.startswith('FROM warm') for ln in directives)
