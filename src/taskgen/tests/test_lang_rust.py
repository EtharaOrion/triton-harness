"""The rust plugin: whole-suite grading, measured floor, no-parser carve path.

Docker is MOCKED here -- nothing below builds an image. What is asserted is
the TEXT the plugin renders, plus a docker-less exercise of the reward
arithmetic against a captured `cargo test` transcript. The rust-spacewasm
task's docker matrix runs separately (see the plan's ACCEPTANCE GATE).

Key rust properties this file locks:

  R1  whole-suite, no parser. `SELECTOR_KIND['rust']='whole-suite'` and the
      plugin declares `parser_backed=False`. `emit.plan_carve` takes the
      no-parser branch and produces a graded set built by
      `whole_suite_selection`, not by `derive_graded_set`.
  R2  equality floor. rust libtest does NOT abort a whole test binary on a
      panic (that's a Go hazard, spike G1); observed==EXPECTED is a real
      assertion.
  R3  measured denominator. The `expected` on the graded set is a placeholder
      out of `plan_carve` and is replaced by `emit_all._measure_and_pin`
      against the intact tree before test.sh is rendered. Nothing hardcodes
      92 in shipped code -- but this file DOES assert measured==92 for the
      spacewasm-shaped fixture because that is the empirical spike's number.
  R4  DOCKERFILE INVARIANTS. `_assert_dockerfile_invariants` runs inside
      render_dockerfile and rejects `repos-src`, `repo-src`, `FROM warm`,
      `git init` (rust does NOT synthesize git), and any second reference
      to the solution_mount. Rendering the composed dockerfile is the test.
  R5  4-harness / 92-test / WAST-corpus structural gates. These are
      anti-gaming: a solver who deleted a harness gets 0, not a fraction.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from taskgen.langs import base as B
from taskgen.langs import rust as R


@pytest.fixture
def plugin():
    return R.RustPlugin()


@pytest.fixture
def graded():
    """A graded set that mirrors what emit._measure_and_pin would produce.

    expected=92 is the empirically-measured intact denominator; it must NOT be
    hardcoded in shipped code, but tests are the place to assert the number the
    substrate actually produces.
    """
    return B.GradedSet(
        expected=92,
        floor_mode='equality',
        kind='whole-suite',
        test_command=R.TEST_COMMAND,
    )


# ------------------------------------------------------------------ axes ----


def test_plugin_declares_its_three_axes(plugin):
    assert plugin.name == 'rust'
    assert plugin.toml_family == 'A'
    assert plugin.floor_mode == 'equality'
    assert plugin.parser_backed is False
    assert plugin.synthesizes_git is False


def test_rust_uses_equality_floor_not_pinned_denominator(plugin):
    """rust libtest does not abort a binary on a panic, so observed==EXPECTED
    is a real assertion. Stated as a test so nobody 'fixes' it to match go."""
    assert R.RustPlugin.floor_mode == 'equality'


def test_plugin_is_registered_under_its_name():
    from taskgen import langs

    assert isinstance(langs.get('rust'), R.RustPlugin)
    assert 'rust' in langs.available()


def test_selector_kind_registered_for_rust():
    from taskgen.gradedset import SELECTOR_KIND

    assert SELECTOR_KIND['rust'] == 'whole-suite'


# ------------------------------------------------------- axis 1: toolchain --


def test_toolchain_selects_pinned_rustc_from_the_base(plugin):
    block = plugin.toolchain()
    assert '1.87.0' in block
    assert 'TOOLCHAIN PIN FAILED' in block
    assert '1.87.0' in block
    assert 'rustup' not in block, 'the base carries rustc; do not reinstall it'


def test_toolchain_does_not_set_cargo_net_offline_true_in_env(plugin):
    """harbor-base already exports it; baking it here would make the vendor
    RUN un-overridable and break dependency resolution."""
    env = plugin.toolchain_spec().env
    assert env.get('CARGO_NET_OFFLINE') is None


def test_toolchain_workdir_matches_base_workdir(plugin):
    assert plugin.toolchain_spec().workdir == B.WORKDIR


# --------------------------------------------------- axis 2: no dep-warm ----


def test_dep_warm_is_empty_because_rust_vendors_in_the_graded_image(plugin):
    spec = plugin.dep_warm_spec()
    assert spec.stage_block == ''
    assert spec.copy_paths == ()
    assert spec.files_needed == ()


def test_dep_warm_renders_no_warm_stage(plugin):
    """A DepWarmSpec with no copy_paths produces the '# no warmed dependencies'
    comment, not a FROM warm stage."""
    assert 'no warmed dependencies' in plugin.dep_warm()
    assert 'FROM' not in plugin.dep_warm()


# -------------------------------------- axis 3-6: the graded verifier -----


def test_test_sh_runs_the_four_harnesses_with_skip_memory_max(plugin, graded):
    sh = plugin.render_test_sh(graded)
    for harness in R.HARNESSES:
        assert f'--test {harness}' in sh
    assert '-- --skip memory_max' in sh


def test_test_sh_emits_every_common_reward_key(plugin, graded):
    sh = plugin.render_test_sh(graded)
    for key in B.REWARD_KEYS:
        assert f'"{key}"' in sh, f'reward.json is missing {key}'
    assert 'reward.txt' in sh


def test_test_sh_is_fail_closed_before_cargo_runs(plugin, graded):
    sh = plugin.render_test_sh(graded)
    assert sh.index('emit 0.0') < sh.index('cargo test')


def test_test_sh_pins_expected_from_graded_not_hardcoded(plugin):
    sh_92 = plugin.render_test_sh(
        B.GradedSet(expected=92, floor_mode='equality', kind='whole-suite',
                    test_command=R.TEST_COMMAND)
    )
    sh_50 = plugin.render_test_sh(
        B.GradedSet(expected=50, floor_mode='equality', kind='whole-suite',
                    test_command=R.TEST_COMMAND)
    )
    assert 'EXPECTED=92' in sh_92
    assert 'EXPECTED=50' in sh_50
    assert 'EXPECTED=92' not in sh_50


def test_test_sh_uses_equality_floor_guard(plugin, graded):
    """observed != EXPECTED must fail. `-ge` or `>=` would let a solver skip
    tests and clear the floor."""
    sh = plugin.render_test_sh(graded)
    assert '-ne "${EXPECTED}"' in sh
    assert '-ge "${EXPECTED}"' not in sh


def test_test_sh_asserts_all_four_harnesses_produced_a_summary(plugin, graded):
    sh = plugin.render_test_sh(graded)
    assert 'SUMMARIES' in sh
    assert '${SUMMARIES}" -eq 4' in sh
    assert 'refusing partial credit' in sh


def test_test_sh_asserts_no_ignored_tests(plugin, graded):
    """A solver who marks tests #[ignore] must not clear the floor."""
    sh = plugin.render_test_sh(graded)
    assert 'IGNORED' in sh
    assert '${IGNORED}" -eq 0' in sh


def test_test_sh_gates_on_wast_corpus_floor(plugin, graded):
    """The .wast corpus IS the oracle; a truncated corpus makes the reward
    a lie."""
    sh = plugin.render_test_sh(graded)
    assert 'WAST_COUNT' in sh
    assert f'-ge {R.MIN_WAST}' in sh


def test_test_sh_gates_on_every_integrity_harness_file(plugin, graded):
    sh = plugin.render_test_sh(graded)
    for rel in R.INTEGRITY_HARNESS_FILES:
        assert rel in sh


def test_test_sh_gates_on_wast2json_presence(plugin, graded):
    sh = plugin.render_test_sh(graded)
    assert 'command -v wast2json' in sh


def test_compiled_is_zero_when_no_test_result_line_seen(plugin, graded):
    """A crate that failed to build produces no `test result:` line. The
    reward schema distinguishes 'did not build' (compiled=0) from 'built but
    failed' (compiled=1, reward<1)."""
    sh = plugin.render_test_sh(graded)
    assert 'test result:' in sh
    assert 'COMPILED=1.0' in sh
    assert 'emit 0.0 0 "${EXPECTED}" 0.0 0.0' in sh


def test_test_sh_runs_the_graded_test_command_from_the_graded_set(plugin, graded):
    sh = plugin.render_test_sh(graded)
    assert graded.test_command in sh


# ------------------------------------------------ the floor-FREE measure sh --


def test_measure_runs_the_same_command_as_the_graded_verifier(plugin, graded):
    sh = plugin.measure_test_sh(graded=graded)
    assert graded.test_command in sh


def test_measure_asserts_no_floor(plugin, graded):
    """A floor cannot be enforced by the run that is supposed to discover it."""
    sh = plugin.measure_test_sh(graded=graded)
    assert 'EXPECTED' not in sh
    for key in B.MEASURE_KEYS:
        assert key in sh
    assert 'measure.json' in sh


def test_measure_uses_measure_dir_not_verifier_dir(plugin, graded):
    """The two live in the same path in the image but come from DIFFERENT env
    vars; the measure phase must not depend on VERIFIER_DIR being set."""
    sh = plugin.measure_test_sh(graded=graded)
    assert 'MEASURE_DIR' in sh
    assert 'VERIFIER_DIR' not in sh


# ------------------------------------------------------- axis 7: the oracle --


def test_solve_sh_restores_carved_set_from_run_time_mount(plugin):
    sh = plugin.render_solve_sh(('src/lib.rs', 'src/foo/bar.rs'))
    assert B.SOLUTION_MOUNT in sh
    assert 'COPY' not in sh
    assert "restore 'src/lib.rs'" in sh
    assert "restore 'src/foo/bar.rs'" in sh


def test_solve_sh_promises_the_correct_file_count(plugin):
    sh = plugin.render_solve_sh(('src/a.rs', 'src/b.rs', 'src/c.rs'))
    assert 'WANT=3' in sh


# ----------------------------------------------------- the image invariants --


def _df(plugin) -> str:
    return plugin.render_dockerfile(B.EnvSpec(repo_name='rust-spacewasm'))


def test_dockerfile_ships_only_the_host_carved_tree(plugin):
    df = _df(plugin)
    assert f'COPY --from={B.REPO_CONTEXT} repo/ /opt/harbor/repo' in df
    assert 'repos-src' not in df
    assert 'repo-src' not in df


def test_dockerfile_bakes_no_oracle(plugin):
    """`_assert_dockerfile_invariants` refuses solution_mount anywhere except
    the absence assertion. Composing the file is the test."""
    df = _df(plugin)
    absence = f'test ! -e {B.SOLUTION_MOUNT}'
    assert absence in df
    assert df.replace(absence, '').count(B.SOLUTION_MOUNT) == 0


def test_dockerfile_runs_the_leak_scan_and_receipt_asserts(plugin):
    df = _df(plugin)
    assert 'leakscan.sh' in df
    assert 'carve_receipt.json' in df


def test_dockerfile_does_not_synthesize_git(plugin):
    """python needs `git init` for uv-dynamic-versioning; rust does not, and
    doing it anyway would trip _assert_dockerfile_invariants (leak route C)."""
    df = _df(plugin)
    assert 'git init' not in df


def test_dockerfile_includes_wabt_install_and_prune_and_vendor(plugin):
    df = _df(plugin)
    assert 'wabt' in df.lower()
    assert 'wast2json --version' in df
    assert 'rm -rf crates fuzz' in df
    assert 'cargo vendor' in df


def test_dockerfile_includes_strings_target_leak_assert(plugin):
    """target/*.o carries mangled symbol names and debug-info paths; strings|
    grep is what makes a solver's `docker save` -> `strings` attack fail."""
    df = _df(plugin)
    assert 'strings' in df
    assert '/opt/harbor/repo/src' in df
    assert 'LEAK' in df


def test_dockerfile_still_binds_leakscan_from_the_tooling_context(plugin):
    """Dropping the wabt asset must not disturb the OTHER tooling-context user.

    leakscan.sh reaches the image by bind mount so it leaves no layer of its
    own; removing the wabt tarball from the same context is exactly the kind of
    change that could take leakscan with it.
    """
    df = _df(plugin)
    assert f'--mount=type=bind,from={B.TOOLING_CONTEXT}' in df
    assert 'leakscan.sh' in df


def test_dockerfile_declares_the_graded_stage(plugin):
    df = _df(plugin)
    assert f'FROM {plugin.toolchain_spec().base_image} AS graded' in df


def test_measure_dockerfile_has_no_leak_gate_or_strings_assert(plugin):
    """The measure image is built on the INTACT tree; the strings-assert and
    the leakscan would fire by construction and are correctly absent."""
    df = plugin.render_measure_dockerfile(B.EnvSpec(repo_name='rust-spacewasm'))
    # Ignore comment-only mentions (a comment cannot fire an assert).
    instructions = '\n'.join(
        ln for ln in df.splitlines() if not ln.lstrip().startswith('#')
    )
    assert 'leakscan.sh' not in instructions
    assert 'xargs -0 -r strings' not in instructions, 'strings-assert must be absent'
    assert 'LEAK' not in instructions
    assert 'tripwire' not in instructions.lower()
    assert 'carve_receipt' not in instructions


def test_measure_dockerfile_includes_toolchain_and_wabt_and_vendor(plugin):
    df = plugin.render_measure_dockerfile(B.EnvSpec(repo_name='rust-spacewasm'))
    assert 'rustc' in df and '1.87.0' in df
    assert 'wast2json --version' in df
    assert 'cargo vendor' in df
    assert 'measure.sh' in df


def test_measure_dockerfile_tags_the_stage_as_never_ship(plugin):
    df = plugin.render_measure_dockerfile(B.EnvSpec(repo_name='rust-spacewasm'))
    assert 'AS measure' in df
    assert 'NEVER SHIP' in df.upper() or 'never-ship' in df.lower() or 'never ship' in df.lower()


# ---------------------------------------------- plugin static assets --------


def test_extra_ctx_assets_is_empty_now_the_base_carries_wabt(plugin):
    """rust used to stage a per-architecture tarball here; the base ships the
    wabt suite, so rust stages nothing and matches python and go."""
    assert plugin.extra_ctx_assets() == ()


def test_extra_ctx_assets_base_default_is_empty():
    """python and go must return () -- adding the rust hook must not change
    their staging."""
    from taskgen.langs.python import PythonPlugin
    from taskgen.langs.go import GoPlugin

    assert PythonPlugin().extra_ctx_assets() == ()
    assert GoPlugin().extra_ctx_assets() == ()


# ----------------------------------------------- render_measure_dockerfile --


def test_base_default_render_measure_dockerfile_raises_for_parser_backed():
    """Only whole-suite plugins need one; parser-backed languages derive the
    denominator without an intact-tree image."""
    from taskgen.langs.python import PythonPlugin

    with pytest.raises(B.LangError, match='does not render a measure dockerfile'):
        PythonPlugin().render_measure_dockerfile(B.EnvSpec(repo_name='x'))


# -------------------------------------------------------------- determinism --


@pytest.mark.parametrize('render', [
    lambda p, g: p.render_test_sh(g),
    lambda p, g: p.measure_test_sh(graded=g),
    lambda p, g: p.render_solve_sh(('src/lib.rs', 'src/foo.rs')),
    lambda p, g: p.render_dockerfile(B.EnvSpec(repo_name='rust-spacewasm')),
    lambda p, g: p.render_measure_dockerfile(B.EnvSpec(repo_name='rust-spacewasm')),
])
def test_every_render_is_byte_stable(render, plugin, graded):
    assert render(plugin, graded) == render(R.RustPlugin(), graded)


@pytest.mark.parametrize('render', [
    lambda p, g: p.render_test_sh(g),
    lambda p, g: p.measure_test_sh(graded=g),
    lambda p, g: p.render_solve_sh(('src/lib.rs',)),
])
def test_every_rendered_script_is_valid_bash(render, plugin, graded, tmp_path):
    script = tmp_path / 's.sh'
    script.write_text(render(plugin, graded))
    proc = subprocess.run(['bash', '-n', str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# ---------------------------- the reward arithmetic, actually run ----------


def _run(sh: str, tmp_path: Path, cargo_transcript: str,
         extra_repo_paths: dict[str, str] | None = None):
    """Execute the rendered test.sh with `cargo`, `wast2json`, `rustc` mocked.

    Asserting on the text proves the script SAYS the right thing; running it
    against a captured cargo transcript proves it COMPUTES the right thing --
    the awk sum_field is exactly the kind of thing a typo would silently
    invert.
    """
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    (bin_dir / 'cargo').write_text(
        '#!/usr/bin/env bash\n'
        f'cat <<\'EOF\'\n{cargo_transcript}\nEOF\n'
        'exit 0\n'
    )
    (bin_dir / 'wast2json').write_text(
        '#!/usr/bin/env bash\necho "wast2json 1.0.41"\n'
    )
    (bin_dir / 'rustc').write_text(
        '#!/usr/bin/env bash\necho "rustc 1.87.0"\n'
    )
    for name in ('cargo', 'wast2json', 'rustc'):
        (bin_dir / name).chmod(0o755)

    repo = tmp_path / 'repo'
    repo.mkdir()
    for rel in R.INTEGRITY_HARNESS_FILES:
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('// test harness\n')
    tests_dir = repo / 'tests'
    for i in range(R.MIN_WAST + 2):
        (tests_dir / f'case_{i:03d}.wast').write_text(';; wast\n')
    for rel, content in (extra_repo_paths or {}).items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    logs = tmp_path / 'logs'
    script = tmp_path / 'test.sh'
    script.write_text(sh)
    env = {
        'PATH': f'{bin_dir}:/usr/bin:/bin',
        'REPO': str(repo),
        'VERIFIER_DIR': str(logs),
        'HOME': str(tmp_path),
    }
    subprocess.run(
        ['bash', str(script)], env=env, capture_output=True, text=True, check=False,
    )
    import json

    return json.loads((logs / 'reward.json').read_text())


def _summary(passed: int, failed: int = 0, ignored: int = 0) -> str:
    return (
        f'test result: ok. {passed} passed; {failed} failed; '
        f'{ignored} ignored; 0 measured; 0 filtered out\n'
    )


def _four_summaries(counts: list[tuple[int, int, int]]) -> str:
    """Four `test result:` lines, one per harness."""
    return ''.join(_summary(p, f, i) for p, f, i in counts)


def test_all_92_pass_scores_one(plugin, graded, tmp_path):
    got = _run(plugin.render_test_sh(graded), tmp_path,
               _four_summaries([(23, 0, 0)] * 4))
    assert got == {'reward': 1.0, 'tests_passed': 92, 'tests_total': 92,
                   'binary': 1.0, 'compiled': 1.0}


def test_partial_credit_is_over_the_pinned_denominator(plugin, graded, tmp_path):
    got = _run(plugin.render_test_sh(graded), tmp_path,
               _four_summaries([(23, 0, 0), (20, 3, 0), (23, 0, 0), (23, 0, 0)]))
    assert got['tests_passed'] == 89
    assert got['tests_total'] == 92
    assert got['reward'] == pytest.approx(89 / 92, rel=1e-3)
    assert got['binary'] == 0.0


def test_a_crate_that_did_not_build_scores_compiled_zero(plugin, graded, tmp_path):
    got = _run(plugin.render_test_sh(graded), tmp_path,
               'error[E0432]: unresolved import\n')
    assert got == {'reward': 0.0, 'tests_passed': 0, 'tests_total': 92,
                   'binary': 0.0, 'compiled': 0.0}


def test_missing_harness_summary_fails_closed(plugin, graded, tmp_path):
    """Only 3 of 4 harnesses reported. Equality floor rejects and refuses
    partial credit."""
    got = _run(plugin.render_test_sh(graded), tmp_path,
               _four_summaries([(30, 0, 0), (30, 0, 0), (32, 0, 0)]))
    assert got['reward'] == 0.0
    assert got['binary'] == 0.0


def test_ignored_tests_fail_closed(plugin, graded, tmp_path):
    """A solver who marks tests #[ignore] and passes the rest scores 0."""
    got = _run(plugin.render_test_sh(graded), tmp_path,
               _four_summaries([(20, 0, 3), (23, 0, 0), (23, 0, 0), (23, 0, 0)]))
    assert got['reward'] == 0.0


def test_shrunk_denominator_fails_closed(plugin, graded, tmp_path):
    """4 summaries, 0 ignored, but only 90 tests -- someone deleted 2 cases."""
    got = _run(plugin.render_test_sh(graded), tmp_path,
               _four_summaries([(22, 0, 0), (23, 0, 0), (22, 0, 0), (23, 0, 0)]))
    assert got['reward'] == 0.0
    assert got['binary'] == 0.0


def test_missing_wast2json_fails_closed(plugin, graded, tmp_path):
    """A solver who deleted wast2json to skirt the harness must score 0."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    (bin_dir / 'cargo').write_text('#!/usr/bin/env bash\nexit 0\n')
    (bin_dir / 'rustc').write_text('#!/usr/bin/env bash\necho "rustc 1.87.0"\n')
    for name in ('cargo', 'rustc'):
        (bin_dir / name).chmod(0o755)
    # deliberately do NOT create wast2json
    repo = tmp_path / 'repo'
    for rel in R.INTEGRITY_HARNESS_FILES:
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('// test harness\n')
    tests_dir = repo / 'tests'
    for i in range(R.MIN_WAST + 2):
        (tests_dir / f'case_{i:03d}.wast').write_text(';; wast\n')

    logs = tmp_path / 'logs'
    script = tmp_path / 'test.sh'
    script.write_text(plugin.render_test_sh(graded))
    env = {
        'PATH': f'{bin_dir}:/usr/bin:/bin',
        'REPO': str(repo),
        'VERIFIER_DIR': str(logs),
        'HOME': str(tmp_path),
    }
    subprocess.run(
        ['bash', str(script)], env=env, capture_output=True, text=True, check=False,
    )
    import json

    got = json.loads((logs / 'reward.json').read_text())
    assert got['reward'] == 0.0
    assert got['compiled'] == 0.0


def test_missing_harness_file_fails_closed(plugin, graded, tmp_path):
    """A solver who deleted `tests/util/spectest.rs` must score 0."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    (bin_dir / 'cargo').write_text('#!/usr/bin/env bash\nexit 0\n')
    (bin_dir / 'wast2json').write_text('#!/usr/bin/env bash\necho "wast2json 1.0.41"\n')
    (bin_dir / 'rustc').write_text('#!/usr/bin/env bash\necho "rustc 1.87.0"\n')
    for name in ('cargo', 'wast2json', 'rustc'):
        (bin_dir / name).chmod(0o755)
    repo = tmp_path / 'repo'
    # deliberately skip tests/util/spectest.rs
    for rel in R.INTEGRITY_HARNESS_FILES:
        if 'spectest' in rel:
            continue
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('// test harness\n')
    tests_dir = repo / 'tests'
    for i in range(R.MIN_WAST + 2):
        (tests_dir / f'case_{i:03d}.wast').write_text(';; wast\n')

    logs = tmp_path / 'logs'
    script = tmp_path / 'test.sh'
    script.write_text(plugin.render_test_sh(graded))
    env = {
        'PATH': f'{bin_dir}:/usr/bin:/bin',
        'REPO': str(repo),
        'VERIFIER_DIR': str(logs),
        'HOME': str(tmp_path),
    }
    subprocess.run(
        ['bash', str(script)], env=env, capture_output=True, text=True, check=False,
    )
    import json

    got = json.loads((logs / 'reward.json').read_text())
    assert got['reward'] == 0.0


def test_truncated_wast_corpus_fails_closed(plugin, graded, tmp_path):
    """A solver who deleted .wast files to hit "trivially passing" scores 0."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    (bin_dir / 'cargo').write_text('#!/usr/bin/env bash\nexit 0\n')
    (bin_dir / 'wast2json').write_text('#!/usr/bin/env bash\necho "wast2json 1.0.41"\n')
    (bin_dir / 'rustc').write_text('#!/usr/bin/env bash\necho "rustc 1.87.0"\n')
    for name in ('cargo', 'wast2json', 'rustc'):
        (bin_dir / name).chmod(0o755)
    repo = tmp_path / 'repo'
    for rel in R.INTEGRITY_HARNESS_FILES:
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('// test harness\n')
    tests_dir = repo / 'tests'
    # only 10 .wast files -- well under MIN_WAST (90)
    for i in range(10):
        (tests_dir / f'case_{i:03d}.wast').write_text(';; wast\n')

    logs = tmp_path / 'logs'
    script = tmp_path / 'test.sh'
    script.write_text(plugin.render_test_sh(graded))
    env = {
        'PATH': f'{bin_dir}:/usr/bin:/bin',
        'REPO': str(repo),
        'VERIFIER_DIR': str(logs),
        'HOME': str(tmp_path),
    }
    subprocess.run(
        ['bash', str(script)], env=env, capture_output=True, text=True, check=False,
    )
    import json

    got = json.loads((logs / 'reward.json').read_text())
    assert got['reward'] == 0.0


# -------------------------------- COPY --from must resolve (docker parity) --


def _copy_from_names(text: str) -> set[str]:
    return {
        line.split('--from=', 1)[1].split()[0].split(',')[0]
        for line in text.splitlines()
        if line.lstrip().startswith(('COPY', 'RUN'))
        and '--from=' in line
    }


def _declared_stages(text: str) -> set[str]:
    return {
        line.rsplit(' AS ', 1)[1].strip()
        for line in text.splitlines()
        if line.startswith('FROM ') and ' AS ' in line
    }


def test_every_copy_from_in_the_dockerfile_resolves_to_a_stage_or_context(plugin):
    env = B.EnvSpec(repo_name='rust-spacewasm')
    text = plugin.render_dockerfile(env)
    named_contexts = {
        env.repo_context, env.entry_context, env.trip_context, env.tooling_context,
    }
    unresolved = _copy_from_names(text) - _declared_stages(text) - named_contexts
    assert not unresolved, f'COPY --from={unresolved} resolves to no stage/context'


# ------------------------------------------ plan_carve (no-parser branch) --


def test_plan_carve_takes_the_no_parser_branch_for_rust(tmp_path):
    """`emit.plan_carve` must skip parse_linked/derive_graded_set for rust
    (which has no tree-sitter parser here). This test drives it through with
    a minimal repo layout and confirms the whole-suite graded set comes out."""
    from taskgen import emit

    repo = tmp_path / 'rust-spacewasm'
    (repo / 'src').mkdir(parents=True)
    (repo / 'src' / 'lib.rs').write_text(
        'pub fn foo() {}\npub const CARVED_CRATE_MARKER_VALUE: u32 = 41;\n'
    )
    (repo / 'src' / 'util.rs').write_text('pub fn bar() {}\n')
    (repo / 'tests' / 'util').mkdir(parents=True)
    for rel in R.INTEGRITY_HARNESS_FILES:
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('#[test] fn t() {}\n')
    (repo / 'Cargo.toml').write_text(
        '[package]\nname = "spacewasm"\nversion = "0.1.0"\nedition = "2024"\n'
    )
    (repo / 'Cargo.lock').write_text('# lockfile\n')

    plan = emit.plan_carve(
        repo=repo, out=tmp_path / 'out', lang='rust', carve_scope='folder',
        include=('src/**',), delete_whole_file=True,
    )
    assert plan.lang == 'rust'
    assert plan.graded.kind == 'whole-suite'
    assert plan.graded.test_command == R.TEST_COMMAND
    assert plan.graded.expected == 1  # placeholder before measure
    assert 'src/lib.rs' in plan.carve.carved_relpaths
    assert 'src/util.rs' in plan.carve.carved_relpaths


# ------------------------------------------ wave 1-2: versioned per-lang base --


def test_rust_builds_on_the_versioned_per_language_base(plugin):
    assert plugin.toolchain_spec().base_image.startswith('426628337772.dkr.ecr.ap-south-2.amazonaws.com/triton/base-rust@sha256:')


def test_rust_does_not_rustup_install_a_toolchain_the_base_carries(plugin):
    """The base ships rustc/cargo 1.87.0, so `rustup toolchain install` is a
    network reach and a layer for a compiler already present."""
    assert 'rustup' not in plugin.toolchain()


def test_rust_wabt_is_not_architecture_hardcoded(plugin):
    """`wabt-1.0.41-linux-arm64.tar.gz` names one architecture, so a task image
    could only ever build on arm64. The base ships wasm2wat 1.0.41 already, so
    the vendored tarball -- and its arch literal -- can go entirely."""
    assert 'linux-arm64' not in plugin.toolchain()
    assert 'arm64' not in plugin.toolchain()


def test_rust_pins_the_runtime_and_proves_it_under_a_login_shell(plugin):
    tc = plugin.toolchain()
    assert "bash -lc" in tc, 'the pin must be proven as a login shell'
    assert '/etc/profile.d/zz-harbor-toolchain-pin.sh' in tc
    assert 'TOOLCHAIN PIN FAILED' in tc
    assert 'bash -lc' in tc


def test_rust_dockerfile_carries_no_architecture_locked_wabt_tarball(plugin):
    """`wabt-1.0.41-linux-arm64.tar.gz` names one architecture in the emitted
    Dockerfile, so the image could only ever build on arm64. The base ships the
    whole wabt suite -- wast2json included -- at the pinned 1.0.41, so both the
    vendored tarball and its arch literal are unnecessary. The dependency must
    still be ASSERTED, just not installed.
    """
    env = B.EnvSpec(repo_name='rust-spacewasm')
    df = plugin.render_dockerfile(env)
    assert 'linux-arm64' not in df
    assert 'wabt.tar.gz' not in df
    assert 'wast2json' in df


def test_rust_stages_no_wabt_asset_into_the_tooling_context(plugin):
    """The tooling context still carries leakscan.sh; it must no longer carry a
    per-architecture tarball the base already provides."""
    assets = plugin.extra_ctx_assets()
    assert not any('wabt' in name for _, name in assets)
