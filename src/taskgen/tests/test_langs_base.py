"""Contract tests for the language-plugin framework.

No plugin exists yet -- python and go land in Wave 2 -- so the contract is
driven by a FAKE plugin defined here. That is deliberate: if the only thing
that can satisfy `LangPlugin` is the plugin the interface was reverse-
engineered from, the interface is not an interface. Everything asserted below
is something a Wave 2 plugin author must be able to rely on.

Four properties matter more than the shape of the dataclasses:

  * the reward schema is COMMON -- every language writes the same five keys, so
    verify.py can grade any language without a per-language parser;
  * the floor mode is a FIELD, not a convention, because go's floor is
    pinned-denominator (a panic aborts the whole test binary, so the observed
    total is not the denominator) while python's is equality;
  * the oracle is never baked -- solve.sh restores from a RUN-TIME MOUNT;
  * the rendered Dockerfile ships the staged carved tree and nothing else: no
    intact repo, no solution, no carve receipt, no carve tooling.
"""

from __future__ import annotations

import pytest

from taskgen import langs
from taskgen.langs import base as B


# ------------------------------------------------------------------ fake ---


class FakePlugin(B.LangPlugin):
    """The minimum a Wave 2 plugin must write. Everything else is inherited."""

    name = 'fake'
    toml_family = 'A'
    floor_mode = 'equality'
    parser_backed = True

    def toolchain_spec(self) -> B.ToolchainSpec:
        return B.ToolchainSpec(
            base_image='harbor-base:local',
            install_block='RUN install-fake-toolchain',
            env={'FAKE_OFFLINE': '1', 'FAKE_HOME': '/opt/fake'},
            workdir='/opt/harbor/repo',
        )

    def dep_warm_spec(self) -> B.DepWarmSpec:
        return B.DepWarmSpec(
            stage_block='RUN fake-fetch-deps',
            files_needed=('fake.lock',),
            copy_paths=(('/opt/fake-home', '/opt/fake-home'),),
        )

    def render_test_sh(self, graded, *, expected=None, fingerprint=None) -> str:
        expected = graded.expected if expected is None else expected
        return '\n'.join([
            '#!/usr/bin/env bash',
            'set -uo pipefail',
            B.reward_emitter_block(),
            B.fail_closed_preamble(expected),
            B.fingerprint_gate_block(
                graded.fingerprint_sha256 if fingerprint is None else fingerprint
            ),
            'PASSED=$(fake-test --count-passed)',
            'TOTAL=$(fake-test --count-total)',
            B.floor_gate_block(graded.floor_mode, expected),
            '',
        ])

    def measure_test_sh(self, *, test_command='fake-test --all') -> str:
        return '\n'.join([
            '#!/usr/bin/env bash',
            'set -uo pipefail',
            f'{test_command} > /tmp/out.txt',
            'TOTAL=$(grep -c . /tmp/out.txt)',
            B.measure_emitter_block(),
            'measure "${TOTAL}" "$(cat /tmp/out.txt)"',
            '',
        ])


class HalfPlugin(B.LangPlugin):
    """Missing every abstract method: must not be instantiable."""

    name = 'half'
    toml_family = 'A'
    floor_mode = 'equality'


# ------------------------------------------------------------- dataclasses --


def test_graded_set_carries_expected_floor_mode_and_fingerprints():
    g = B.GradedSet(expected=5, floor_mode='equality',
                    fingerprint_sha256={'tests/test_task.py': 'ab' * 32})
    assert g.expected == 5
    assert g.floor_mode == 'equality'
    assert g.fingerprint_sha256['tests/test_task.py'] == 'ab' * 32
    assert g.fingerprint_relpaths == ('tests/test_task.py',)


def test_floor_mode_is_a_field_with_exactly_two_legal_values():
    assert set(B.FLOOR_MODES) == {'equality', 'pinned-denominator'}
    assert B.GradedSet(expected=4, floor_mode='pinned-denominator').floor_mode == \
        'pinned-denominator'
    with pytest.raises(B.LangError, match='floor_mode'):
        B.GradedSet(expected=4, floor_mode='ge')


def test_graded_set_refuses_an_empty_denominator():
    """expected=0 makes reward = passed/0 -- a floor of nothing is not a floor."""
    with pytest.raises(B.LangError, match='expected'):
        B.GradedSet(expected=0)


def test_graded_set_fingerprints_are_sorted_so_renders_are_byte_stable():
    a = B.GradedSet(expected=1, fingerprint_sha256={'b.py': '1' * 64, 'a.py': '0' * 64})
    b = B.GradedSet(expected=1, fingerprint_sha256={'a.py': '0' * 64, 'b.py': '1' * 64})
    assert a.fingerprint_relpaths == ('a.py', 'b.py') == b.fingerprint_relpaths


def test_graded_set_is_frozen():
    g = B.GradedSet(expected=1)
    with pytest.raises(Exception):
        g.expected = 9  # type: ignore[misc]


def test_reward_counts_emit_the_common_schema():
    counts = B.RewardCounts(reward=0.5, tests_passed=2, tests_total=4,
                            binary=0.0, compiled=1.0)
    assert set(counts.to_dict()) == set(B.REWARD_KEYS)
    assert counts.to_dict()['reward'] == 0.5


# ---------------------------------------------------------------- registry --


def test_langs_registry_is_a_dict_and_starts_without_wave2_plugins():
    assert isinstance(B.LANGS, dict)
    assert 'fake' not in B.LANGS


def test_register_then_get_roundtrips():
    plugin = FakePlugin()
    B.register(plugin)
    try:
        assert B.get('fake') is plugin
        assert 'fake' in B.available()
    finally:
        B.LANGS.pop('fake', None)


def test_register_refuses_to_silently_replace_a_plugin():
    plugin = FakePlugin()
    B.register(plugin)
    try:
        with pytest.raises(B.LangError, match='already registered'):
            B.register(FakePlugin())
        B.register(FakePlugin(), overwrite=True)
    finally:
        B.LANGS.pop('fake', None)


def test_register_refuses_something_that_is_not_a_plugin():
    with pytest.raises(B.LangError, match='LangPlugin'):
        B.register(object())


def test_get_of_an_unknown_language_names_what_is_available():
    with pytest.raises(B.LangError, match='unknown language'):
        B.get('cobol')


def test_get_of_an_unlanded_language_says_so_instead_of_raising_importerror():
    """java/rust/c/cpp are later increments; the message must not be a bare
    ImportError. python and go HAVE landed, so they are asserted the other way
    round -- see test_lang_python.py / test_lang_go.py."""
    for planned in ('java', 'rust', 'c', 'cpp', 'csharp'):
        with pytest.raises(B.LangError, match='Wave 2'):
            B.get(planned)


def test_plugin_validates_its_own_axes_at_construction():
    class BadFamily(FakePlugin):
        name = 'bad-family'
        toml_family = 'Z'

    class BadFloor(FakePlugin):
        name = 'bad-floor'
        floor_mode = 'ge'

    with pytest.raises(B.LangError, match='toml_family'):
        BadFamily()
    with pytest.raises(B.LangError, match='floor_mode'):
        BadFloor()


def test_a_plugin_missing_an_axis_cannot_be_instantiated():
    with pytest.raises(TypeError):
        HalfPlugin()


def test_the_package_reexports_the_registry():
    assert langs.get is B.get
    assert langs.LANGS is B.LANGS
    assert langs.LangPlugin is B.LangPlugin


# ------------------------------------------------------- axis 1-2: toolchain --


def test_toolchain_renders_a_dockerfile_snippet():
    snippet = FakePlugin().toolchain()
    assert 'RUN install-fake-toolchain' in snippet
    assert 'ENV FAKE_OFFLINE=1' in snippet
    assert 'WORKDIR /opt/harbor/repo' in snippet


def test_toolchain_env_is_sorted_so_the_snippet_is_byte_stable():
    snippet = FakePlugin().toolchain()
    assert snippet.index('FAKE_HOME') < snippet.index('FAKE_OFFLINE')


def test_dep_warm_pulls_from_the_warm_stage_and_never_inherits_it():
    """`FROM warm` would inherit every layer the warm build touched (invariant 7)."""
    snippet = FakePlugin().dep_warm()
    assert 'COPY --from=warm /opt/fake-home /opt/fake-home' in snippet
    assert 'FROM warm' not in snippet


def test_dep_warm_spec_warms_from_manifest_files_only():
    spec = FakePlugin().dep_warm_spec()
    assert spec.files_needed == ('fake.lock',)


# --------------------------------------------- axis 3-4: test.sh + reward ----


def _test_sh(**kw) -> str:
    graded = kw.pop('graded', None) or B.GradedSet(
        expected=5, fingerprint_sha256={'tests/test_task.py': 'ab' * 32}
    )
    return FakePlugin().render_test_sh(graded, **kw)


def test_rendered_test_sh_emits_every_common_reward_key():
    sh = _test_sh()
    for key in B.REWARD_KEYS:
        assert f'"{key}"' in sh, f'reward.json is missing {key}'


def test_reward_json_is_authoritative_and_reward_txt_holds_the_binary():
    """harbor convention: reward.txt is the binary bar, reward.json the detail."""
    sh = B.reward_emitter_block()
    assert 'reward.json' in sh
    assert 'reward.txt' in sh
    # emit <reward> <passed> <total> <binary> <compiled>; $4 is binary.
    assert '"$4"' in sh.split('reward.txt')[0].split('reward.json')[1] or \
        'BINARY' in sh


def test_test_sh_is_fail_closed_and_writes_a_zero_before_anything_runs():
    sh = _test_sh()
    first_emit = sh.index('emit 0.0')
    assert first_emit < sh.index('fake-test'), 'a zero must land before the suite runs'


def test_test_sh_pins_the_fingerprint_of_every_graded_test_file():
    sh = _test_sh()
    assert 'tests/test_task.py' in sh
    assert 'ab' * 32 in sh


def test_equality_floor_asserts_the_observed_total_equals_expected():
    sh = _test_sh()
    assert '-eq "${EXPECTED}"' in sh.replace('"$EXPECTED"', '"${EXPECTED}"')


def test_pinned_denominator_floor_divides_by_expected_not_by_the_observed_total():
    """G1: a go panic aborts the binary, so the observed total is not a denominator."""
    graded = B.GradedSet(expected=4, floor_mode='pinned-denominator')
    sh = FakePlugin().render_test_sh(graded)
    assert 'scope-growth' in sh.lower() or 'gt "${EXPECTED}"' in sh
    assert '${EXPECTED}' in sh
    assert 'TOTAL' in sh


def test_test_sh_render_is_deterministic():
    assert _test_sh() == _test_sh()


# ------------------------------------------------ measure: the floor-FREE sh --


def test_measure_test_sh_only_counts_and_asserts_no_floor():
    sh = FakePlugin().measure_test_sh()
    assert 'EXPECTED' not in sh, 'the measure phase must not know a floor -- it makes one'
    assert 'measure.json' in sh
    for key in B.MEASURE_KEYS:
        assert key in sh


def test_measure_test_sh_render_is_deterministic():
    p = FakePlugin()
    assert p.measure_test_sh() == p.measure_test_sh()


# ------------------------------------------ axis 7: run-time oracle mount ----


def test_solve_sh_restores_from_the_run_time_mount():
    sh = FakePlugin().render_solve_sh(('src/a2a/utils/task.py', 'src/a2a/utils/other.py'))
    assert B.SOLUTION_MOUNT in sh
    assert 'HARBOR_SOLUTION' in sh
    assert 'src/a2a/utils/task.py' in sh


def test_solve_sh_asserts_it_restored_everything_it_promised():
    sh = FakePlugin().render_solve_sh(('a.py', 'b.py'))
    assert '2' in sh
    assert 'SOLVE OK' in sh


def test_solve_sh_never_bakes_the_oracle_into_a_layer():
    sh = FakePlugin().render_solve_sh(('a.py',))
    assert 'COPY' not in sh


def test_solve_sh_render_is_deterministic_and_order_independent():
    p = FakePlugin()
    assert p.render_solve_sh(('b.py', 'a.py')) == p.render_solve_sh(('a.py', 'b.py'))


# ----------------------------------------- axis 6+8: dockerfile invariants ---


def _dockerfile() -> str:
    return FakePlugin().render_dockerfile(B.EnvSpec(repo_name='python-a2a-python'))


def test_dockerfile_ships_only_the_staged_carved_tree():
    df = _dockerfile()
    assert f'COPY --from={B.REPO_CONTEXT} repo/ /opt/harbor/repo' in df


def test_dockerfile_never_copies_the_intact_repo():
    """A1: an intact COPY leaves the answer in a layer that `docker save` reads."""
    df = _dockerfile()
    assert 'repos-src' not in df
    assert 'repo-src' not in df


def test_dockerfile_never_copies_the_solution():
    df = _dockerfile()
    assert B.SOLUTION_MOUNT not in df.replace(f'test ! -e {B.SOLUTION_MOUNT}', '')
    assert 'COPY --from=entry solution' not in df


def test_dockerfile_runs_the_harbor_leak_scan_as_a_build_gate():
    df = _dockerfile()
    assert 'leakscan.sh' in df
    assert f'from={B.TRIP_CONTEXT}' in df
    assert 'type=bind' in df, 'tripwires must arrive on a bind mount, leaving no layer'


def test_dockerfile_asserts_no_carve_metadata_and_no_carve_tooling():
    df = _dockerfile()
    assert 'carve_receipt.json' in df
    assert 'carve.py' in df


def test_dockerfile_makes_the_verifier_log_dir():
    assert 'mkdir -p /logs/verifier' in _dockerfile()


def test_dockerfile_render_is_deterministic():
    assert _dockerfile() == _dockerfile()


# ------------------------------------------------------ the scripts run ------


@pytest.mark.parametrize('render', [
    lambda p: p.render_test_sh(B.GradedSet(expected=4, floor_mode='pinned-denominator',
                                           fingerprint_sha256={'x_test.go': 'ab' * 32})),
    lambda p: p.render_test_sh(B.GradedSet(expected=5)),
    lambda p: p.measure_test_sh(),
    lambda p: p.render_solve_sh(('a.py', 'b/c.py')),
])
def test_every_rendered_script_is_valid_bash(render, tmp_path):
    """`bash -n` the renders: a template that does not parse fails inside docker,
    minutes into a build, as an unattributable non-zero exit."""
    import subprocess
    script = tmp_path / 's.sh'
    script.write_text(render(FakePlugin()))
    proc = subprocess.run(['bash', '-n', str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------- WAVE 2 gaps: two ordering hooks ----
# Both were found by building the python plugin against this contract. Neither
# is a python detail: any language whose dependency resolution must run against
# the CARVED tree hits the first, and any language whose build system demands
# VCS metadata hits the second.


class GitPlugin(FakePlugin):
    """A plugin that must synthesise git metadata over the carved tree."""

    name = 'gitty'
    synthesizes_git = True

    def pre_leakgate_blocks(self, env):
        return (
            'RUN git init -q -b main && git add -A && git commit -m x',
            'RUN set -eux; \\',
            "    test \"$(git rev-list --objects --all --reflog | wc -l)\" -gt 0; \\",
            '    test "$(git rev-list --count --all)" = "1"; \\',
            '    test -z "$(git fsck --unreachable --dangling --no-progress)"',
        )


def test_pre_leakgate_blocks_land_after_the_repo_and_before_the_scan():
    """Dependency resolution has to see the carved tree, and the leak scan has
    to see whatever that resolution left behind -- so there must be a slot
    BETWEEN them. `extra_dockerfile_blocks` runs after the gate, which is too
    late for anything the gate should inspect."""
    df = GitPlugin().render_dockerfile(B.EnvSpec(repo_name='r'))
    assert df.index(f'COPY --from={B.REPO_CONTEXT}') < df.index('git init')
    assert df.index('git init') < df.index('leakscan.sh')


def test_git_init_stays_banned_for_a_plugin_that_did_not_declare_it():
    """Leak route C: `git add -A` over an intact tree puts the answer in
    .git/objects, recoverable with one `git checkout --`."""
    class Sneaky(FakePlugin):
        name = 'sneaky'

        def pre_leakgate_blocks(self, env):
            return ('RUN git init -q && git add -A',)

    with pytest.raises(B.LangError, match='git init'):
        Sneaky().render_dockerfile(B.EnvSpec(repo_name='r'))


def test_declaring_synthesizes_git_is_not_enough_without_the_audit():
    """The declaration buys nothing on its own: the tree must be PROVEN to
    carry version metadata and nothing else."""
    class Unaudited(FakePlugin):
        name = 'unaudited'
        synthesizes_git = True

        def pre_leakgate_blocks(self, env):
            return ('RUN git init -q && git add -A && git commit -m x',)

    with pytest.raises(B.LangError, match='audit'):
        Unaudited().render_dockerfile(B.EnvSpec(repo_name='r'))


def test_a_plugin_synthesises_no_git_by_default():
    assert FakePlugin().synthesizes_git is False
    assert FakePlugin().pre_leakgate_blocks(B.EnvSpec(repo_name='r')) == ()
