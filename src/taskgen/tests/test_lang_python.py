"""The python plugin: today's PROVEN entry, re-expressed through the plugin base.

Python is the only language with a shipped, docker-proven entry already (WAVE 0:
GREEN-baseline 5/5, RED-stub 0/5, GREEN-oracle 5/5, three stable repeats). So
this plugin is not a new design, it is a MIGRATION, and the tests are shaped
accordingly: one half asserts PARITY -- every flag, path and gate that the
proven run depended on must survive, enumerated as literals below so the list
itself is the frozen reference -- and the other half asserts the things that
deliberately CHANGED.

What changed, and why it is not a regression:

  * the intact tree no longer enters any build context. The old Dockerfile did
    `COPY --from=repo-src . ${HARBOR_REPO}` and then overwrote one file, leaving
    the answer in an earlier layer that `docker save` reads back (plan A1). The
    carve now happens on the HOST and only the carved tree is shipped.
  * `git init` survives, but only over the ALREADY-CARVED tree. uv-dynamic-
    versioning demands git metadata; doing it before the carve is leak route C
    (the answer in `.git/objects`). It is gated on a compensating audit.
  * reward.json gains `binary` and `compiled` (the COMMON schema). For a
    function-scope task the numbers still move 0 -> 1 exactly as before.

Docker is MOCKED. `uv` is a shell stub, so the reward arithmetic is executed
rather than merely read.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from taskgen.langs import base as B
from taskgen.langs import python as PY

NODEIDS = (
    'tests/utils/test_task.py::TestApplyHistoryLength::test_large_history_length_returns_full_history',
    'tests/utils/test_task.py::TestApplyHistoryLength::test_none_config_returns_full_history',
    'tests/utils/test_task.py::TestApplyHistoryLength::test_positive_history_length_truncates',
    'tests/utils/test_task.py::TestApplyHistoryLength::test_unset_history_length_returns_full_history',
    'tests/utils/test_task.py::TestApplyHistoryLength::test_zero_history_length_returns_empty_history',
)
TEST_REL = 'tests/utils/test_task.py'
TEST_SHA = 'ab' * 32


@pytest.fixture
def plugin():
    return PY.PythonPlugin()


@pytest.fixture
def graded():
    return B.GradedSet(
        expected=5,
        floor_mode='equality',
        kind='pytest-allowlist',
        selectors=NODEIDS,
        fingerprint_sha256={TEST_REL: TEST_SHA},
    )


# ------------------------------------------------------------------ axes ----


def test_plugin_declares_its_three_axes(plugin):
    assert plugin.name == 'python'
    assert plugin.toml_family == 'A'
    assert plugin.floor_mode == 'equality'
    assert plugin.parser_backed is True


def test_plugin_is_registered_under_its_name():
    from taskgen import langs

    assert isinstance(langs.get('python'), PY.PythonPlugin)
    assert 'python' in langs.available()


# ---------------------------------------------- axis 1-2: parity of the env --


def test_toolchain_installs_the_same_pinned_uv_as_the_proven_entry(plugin):
    from taskgen import emit

    assert f'uv=={PY.UV_VERSION}' in plugin.toolchain()
    assert PY.UV_VERSION == emit.UV_VERSION, \
        'the pin must match the entry this plugin replaces'


def test_toolchain_keeps_the_uv_environment_of_the_proven_entry(plugin):
    env = plugin.toolchain_spec().env
    assert env['UV_PROJECT_ENVIRONMENT'] == '/opt/harbor/repo/.venv'
    assert env['UV_LINK_MODE'] == 'copy'
    assert env['HARBOR_REPO'] == '/opt/harbor/repo'


def test_dep_warm_syncs_the_lockfile_and_proves_the_install_is_editable(plugin):
    """The benchmark's premise: a regenerated function must take effect with no
    reinstall, which is false if uv baked a non-editable copy into site-packages."""
    block = plugin.dep_warm_spec().stage_block
    assert 'uv sync --locked --all-extras' in block
    assert 'NOT EDITABLE' in block
    assert plugin.dep_warm_spec().files_needed == ('pyproject.toml', 'uv.lock')


def test_dep_warm_synthesises_the_git_metadata_uv_dynamic_versioning_needs(plugin):
    block = plugin.dep_warm_spec().stage_block
    assert 'git init' in block
    assert 'uv-dynamic-versioning' in block or 'describe' in block


def test_python_declares_that_it_synthesises_git_so_the_base_can_gate_it(plugin):
    assert plugin.synthesizes_git is True


# --------------------------------------------- axis 3-4: parity of test.sh ---


@pytest.mark.parametrize('flag', [
    "-o addopts=''",
    '-p no:xdist',
    '-p no:cacheprovider',
    '-p harbor_filter',
    'uv run --no-sync --offline pytest',
    'HARBOR_ALLOWLIST',
    'HARBOR_SELECTED_OUT',
])
def test_test_sh_keeps_every_load_bearing_flag_of_the_proven_run(plugin, graded, flag):
    """These are not style. `--dist loadgroup` in the project's own addopts makes
    pytest refuse to start once xdist is off; selection by DESELECTION preserves
    the order the allowlist was measured under; and the explicit
    HARBOR_SELECTED_OUT exists because the plugin swallows OSError, so a path
    typo would read as `selected=0` rather than as an error."""
    sh = plugin.render_test_sh(graded)
    assert flag in sh


def test_test_sh_grades_through_the_allowlist_not_the_command_line(plugin, graded):
    sh = plugin.render_test_sh(graded)
    assert 'allowlist.txt' in sh
    for nodeid in NODEIDS:
        assert nodeid not in sh, 'node ids on argv reorder an order-dependent suite'


def test_test_sh_emits_every_common_reward_key(plugin, graded):
    sh = plugin.render_test_sh(graded)
    for key in B.REWARD_KEYS:
        assert f'"{key}"' in sh
    assert 'reward.txt' in sh


def test_test_sh_is_fail_closed_with_a_zero_before_pytest_runs(plugin, graded):
    sh = plugin.render_test_sh(graded)
    assert sh.index('emit 0.0') < sh.index('uv run --no-sync --offline pytest')


def test_python_uses_the_equality_floor(plugin, graded):
    """Unlike go: nothing in pytest aborts the process on one failure, so the
    observed total IS a denominator and `>=` would be a free reward."""
    sh = plugin.render_test_sh(graded)
    assert '-ne "${EXPECTED}"' in sh
    assert 'scope-growth' not in sh


def test_test_sh_reports_python_as_always_compiled(plugin, graded):
    assert 'COMPILED=1.0' in plugin.render_test_sh(graded)


def test_test_sh_pins_the_fingerprint_of_the_graded_test_file(plugin, graded):
    sh = plugin.render_test_sh(graded)
    assert TEST_REL in sh and TEST_SHA in sh


def test_measure_test_sh_collects_the_allowlist_and_asserts_no_floor(plugin, graded):
    sh = plugin.measure_test_sh(graded=graded)
    assert '--collect-only' in sh
    assert 'EXPECTED' not in sh
    assert 'measure.json' in sh


# ------------------------------------------ the Dockerfile: what CHANGED -----


def _df(plugin) -> str:
    return plugin.render_dockerfile(B.EnvSpec(repo_name='python-a2a-python'))


def test_dockerfile_ships_the_host_carved_tree_and_never_the_intact_one(plugin):
    """A1: the old entry's `COPY --from=repo-src .` put the answer in a layer."""
    df = _df(plugin)
    assert f'COPY --from={B.REPO_CONTEXT} repo/ /opt/harbor/repo' in df
    assert 'repo-src' not in df
    assert 'repos-src' not in df
    assert 'COPY --from=repo' not in df.replace('COPY --from=repoctx', '')


def test_dockerfile_has_no_intact_stage_at_all(plugin):
    df = _df(plugin)
    assert 'AS intact' not in df
    assert 'FROM intact' not in df


def test_dockerfile_bakes_no_oracle(plugin):
    df = _df(plugin)
    assert 'COPY --from=entry solution' not in df
    assert df.count(B.SOLUTION_MOUNT) == 1


def test_git_is_synthesised_only_over_the_already_carved_tree(plugin):
    """Leak route C: `git add -A` on the INTACT tree puts the answer in
    .git/objects, recoverable with one `git checkout --` even without
    `docker save`."""
    df = _df(plugin)
    assert 'git init' in df
    assert df.index(f'COPY --from={B.REPO_CONTEXT}') < df.index('git init')


def test_the_git_synthesis_carries_its_compensating_audit(plugin):
    """The base refuses `git init` unless the tree is proven to be metadata
    only: one commit, no stash, no unreachable or dangling object."""
    df = _df(plugin)
    assert 'git rev-list --objects --all --reflog' in df
    assert 'git fsck' in df
    assert "git rev-list --count --all" in df


def test_the_leak_scan_runs_after_the_deps_and_the_git_synthesis(plugin):
    """Otherwise the scan never sees .git/objects or the built venv -- the two
    places a python answer would actually end up."""
    df = _df(plugin)
    assert df.index('git init') < df.index('leakscan.sh')
    assert df.index('uv sync') < df.index('leakscan.sh')


def test_dockerfile_still_asserts_the_editable_install(plugin):
    df = _df(plugin)
    assert 'NOT EDITABLE' in df
    assert 'uv sync --locked --all-extras' in df


def test_dockerfile_asserts_no_carve_metadata(plugin):
    df = _df(plugin)
    assert 'carve_receipt.json' in df
    assert 'carve.py' in df


# ---------------------------------------------------------------- oracle ----


def test_solve_sh_restores_from_the_run_time_mount(plugin):
    sh = plugin.render_solve_sh(('src/a2a/utils/task.py',))
    assert B.SOLUTION_MOUNT in sh
    assert 'COPY' not in sh


def test_post_restore_clears_stale_bytecode(plugin):
    """A cached .pyc would import the carved module after the restore."""
    assert '__pycache__' in plugin.post_restore_block()


# ----------------------------------------------------------- determinism ----


@pytest.mark.parametrize('render', [
    lambda p, g: p.render_test_sh(g),
    lambda p, g: p.measure_test_sh(graded=g),
    lambda p, g: p.render_solve_sh(('b.py', 'a.py')),
    lambda p, g: p.render_dockerfile(B.EnvSpec(repo_name='python-a2a-python')),
])
def test_every_render_is_byte_stable(render, plugin, graded):
    assert render(plugin, graded) == render(PY.PythonPlugin(), graded)


@pytest.mark.parametrize('render', [
    lambda p, g: p.render_test_sh(g),
    lambda p, g: p.measure_test_sh(graded=g),
    lambda p, g: p.render_solve_sh(('a.py',)),
])
def test_every_rendered_script_is_valid_bash(render, plugin, graded, tmp_path):
    script = tmp_path / 's.sh'
    script.write_text(render(plugin, graded))
    proc = subprocess.run(['bash', '-n', str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# --------------------------------------- the reward arithmetic, actually run -


def _run(sh, tmp_path, *, selected, tests, failures, status=0, sha=TEST_SHA):
    """Execute the rendered test.sh with `uv` and `sha256sum` MOCKED."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    (bin_dir / 'uv').write_text(
        '#!/usr/bin/env bash\n'
        'J=""\n'
        'for a in "$@"; do case "$a" in --junitxml=*) J="${a#--junitxml=}";; esac; done\n'
        '[ -n "$J" ] && mkdir -p "$(dirname "$J")" && printf \'%s\\n\' \\\n'
        f'  \'<testsuites><testsuite tests="{tests}" failures="{failures}" '
        'errors="0" skipped="0"/></testsuites>\' > "$J"\n'
        'mkdir -p "$(dirname "${HARBOR_SELECTED_OUT}")"\n'
        f'printf \'%s\\n\' {selected} > "${{HARBOR_SELECTED_OUT}}"\n'
        f'exit {status}\n'
    )
    (bin_dir / 'sha256sum').write_text(f'#!/usr/bin/env bash\necho "{sha}  $1"\n')
    for name in ('uv', 'sha256sum'):
        (bin_dir / name).chmod(0o755)

    repo = tmp_path / 'repo'
    (repo / TEST_REL).parent.mkdir(parents=True)
    (repo / TEST_REL).write_text('# test\n')
    allowlist = tmp_path / 'allowlist.txt'
    allowlist.write_text('\n'.join(NODEIDS) + '\n')
    logs = tmp_path / 'logs'

    script = tmp_path / 'test.sh'
    script.write_text(sh)
    env = {
        'PATH': f'{bin_dir}:/usr/bin:/bin',
        'REPO': str(repo),
        'VERIFIER_DIR': str(logs),
        'ALLOWLIST': str(allowlist),
        'HOME': str(tmp_path),
    }
    subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True,
                   check=False)
    return json.loads((logs / 'reward.json').read_text())


def test_all_five_passing_scores_one(plugin, graded, tmp_path):
    got = _run(plugin.render_test_sh(graded), tmp_path,
               selected=5, tests=5, failures=0)
    assert got == {'reward': 1.0, 'tests_passed': 5, 'tests_total': 5,
                   'binary': 1.0, 'compiled': 1.0}


def test_the_red_stub_leg_scores_zero(plugin, graded, tmp_path):
    """The WAVE 0 RED proof: all five fail, and only those five ran."""
    got = _run(plugin.render_test_sh(graded), tmp_path,
               selected=5, tests=5, failures=5, status=1)
    assert got['reward'] == 0.0
    assert got['binary'] == 0.0
    assert got['tests_passed'] == 0


def test_partial_credit_is_fractional(plugin, graded, tmp_path):
    got = _run(plugin.render_test_sh(graded), tmp_path,
               selected=5, tests=5, failures=2, status=1)
    assert got['reward'] == 0.6
    assert got['tests_passed'] == 3
    assert got['binary'] == 0.0


def test_a_shrunken_denominator_fails_closed(plugin, graded, tmp_path):
    """The equality gate: a run trimmed to a passing subset still exits 0."""
    got = _run(plugin.render_test_sh(graded), tmp_path,
               selected=2, tests=2, failures=0, status=0)
    assert got['reward'] == 0.0


def test_a_tampered_test_file_fails_the_fingerprint_gate(plugin, graded, tmp_path):
    got = _run(plugin.render_test_sh(graded), tmp_path,
               selected=5, tests=5, failures=0, sha='ff' * 32)
    assert got['reward'] == 0.0
