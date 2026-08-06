"""The taskgen <-> verifier glue: inputs, the soundness executor, and the driver.

Offline and deterministic by construction. Every LLM step runs through
`verifier_fixtures`' scripted MockClient, and the one thing that is genuinely
executed -- pytest against a golden and a stub tree -- runs against a synthetic
stdlib-only repo, so nothing here needs the proxy, litellm, docker or network.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from taskgen import emit as emit_mod
from taskgen import staging as staging_mod
from verifier import bundle as bundle_mod
from verifier import exec_env as exec_env_mod
from verifier import inputs as inputs_mod
from verifier.generators import layout, manifest, verifier_loop

from . import verifier_fixtures as fx


# --------------------------------------------------------------- fixtures ---


@dataclass
class FakeCarve:
    """The `CarveLike` slice inputs.py and exec_env.py read."""

    carved_relpaths: tuple[str, ...]
    originals: dict[str, str]
    overlay: dict[str, str]
    deleted_relpaths: tuple[str, ...] = ()
    docstring: str = ''


GOLDEN_CALC = '''"""Arithmetic helpers."""


def add(a, b):
    """Return the sum of two numbers."""
    return a + b


def scale(xs, factor):
    """Return every element of xs multiplied by factor."""
    return [x * factor for x in xs]
'''

STUB_CALC = '''"""Arithmetic helpers."""


def add(a, b):
    raise NotImplementedError


def scale(xs, factor):
    raise NotImplementedError
'''

CARVED_REL = 'mypkg/calc.py'


@pytest.fixture
def carve() -> FakeCarve:
    return FakeCarve(
        carved_relpaths=(CARVED_REL,),
        originals={CARVED_REL: GOLDEN_CALC},
        overlay={CARVED_REL: STUB_CALC},
    )


@pytest.fixture
def synthetic_repo(tmp_path: Path) -> Path:
    """A stdlib-only checkout whose carved module is really importable."""
    repo = tmp_path / 'src-repo'
    (repo / 'mypkg').mkdir(parents=True)
    (repo / 'mypkg' / '__init__.py').write_text('', encoding='utf-8')
    (repo / 'mypkg' / 'calc.py').write_text(GOLDEN_CALC, encoding='utf-8')
    (repo / 'README.md').write_text('# synthetic\n', encoding='utf-8')
    return repo


@dataclass
class FakeTarget:
    docstring: str = 'Return the sum of two numbers.'
    tests: list = field(default_factory=list)
    repo: Path | None = None


# ------------------------------------------------------------- inputs.py ---


def test_golden_diff_runs_stub_to_oracle(carve):
    diff = inputs_mod.golden_diff(carve)
    assert f'--- a/{CARVED_REL}' in diff
    assert f'+++ b/{CARVED_REL}' in diff
    # The ORACLE body is what a correct solve ADDS; the stub marker is removed.
    assert '+    return a + b' in diff
    assert '-    raise NotImplementedError' in diff


def test_golden_diff_is_the_shape_the_manifest_parses(carve):
    diff = inputs_mod.golden_diff(carve)
    assert manifest.target_files_from_diff(diff) == [CARVED_REL]
    assert manifest.required_target_files(diff) == [CARVED_REL]


def test_golden_diff_is_deterministic(carve):
    assert inputs_mod.golden_diff(carve) == inputs_mod.golden_diff(carve)


def test_golden_diff_treats_a_deleted_file_as_wholly_added():
    carve = FakeCarve(
        carved_relpaths=(CARVED_REL,),
        originals={CARVED_REL: GOLDEN_CALC},
        overlay={},
        deleted_relpaths=(CARVED_REL,),
    )
    diff = inputs_mod.golden_diff(carve)
    assert '+def add(a, b):' in diff
    removed = [ln for ln in diff.splitlines()
               if ln.startswith('-') and not ln.startswith('---')]
    assert removed == []


def test_build_truth_inputs_uses_taskgen_artifacts_only(carve):
    inp = inputs_mod.build_truth_inputs(
        carve, FakeTarget(), 'python',
        repo_name='synthetic', fail_to_pass=['tests/test_calc.py::test_add'],
    )
    assert inp.repo == 'synthetic'
    assert inp.language == 'python'
    assert inp.stub_files == [CARVED_REL]
    assert inp.fail_to_pass == ['tests/test_calc.py::test_add']
    assert inp.spec_text == 'Return the sum of two numbers.'
    assert '+    return a + b' in inp.golden_diff


def test_build_truth_inputs_derives_node_ids_from_the_target(carve):
    from taskgen.select import Linked

    target = FakeTarget(tests=[
        Linked('tests/test_calc.py', 'TestAdd', 'test_two'),
        Linked('tests/test_calc.py', '', 'test_one'),
    ])
    inp = inputs_mod.build_truth_inputs(carve, target, 'python')
    assert inp.fail_to_pass == [
        'tests/test_calc.py::TestAdd::test_two',
        'tests/test_calc.py::test_one',
    ]


def test_solution_code_carries_both_bare_sides(carve):
    code = inputs_mod.solution_code(carve)
    assert 'return a + b' in code.golden
    assert 'return a + b' not in code.stub
    assert 'raise NotImplementedError' in code.stub


# ----------------------------------------------------------- exec_env.py ---


def test_golden_tree_is_the_oracle_and_stub_tree_is_the_carve(
        synthetic_repo, carve, tmp_path):
    env = exec_env_mod.make_exec_env(
        'python', repo=synthetic_repo, carve=carve, work_dir=tmp_path / 'trees')
    golden = (env.tree('golden') / CARVED_REL).read_text()
    stub = (env.tree('stub') / CARVED_REL).read_text()
    assert golden == GOLDEN_CALC
    assert stub == STUB_CALC
    # The source checkout is never touched.
    assert (synthetic_repo / CARVED_REL).read_text() == GOLDEN_CALC


def test_the_trees_are_materialised_once_and_reused(synthetic_repo, carve, tmp_path):
    env = exec_env_mod.make_exec_env(
        'python', repo=synthetic_repo, carve=carve, work_dir=tmp_path / 'trees')
    assert env.tree('golden') is env.tree('golden')


def test_an_unknown_tree_kind_is_refused(synthetic_repo, carve, tmp_path):
    env = exec_env_mod.make_exec_env(
        'python', repo=synthetic_repo, carve=carve, work_dir=tmp_path / 'trees')
    with pytest.raises(exec_env_mod.ExecEnvError):
        env.tree('oracle')


def test_materialising_inside_the_source_checkout_is_refused(synthetic_repo, carve):
    with pytest.raises(exec_env_mod.ExecEnvError):
        exec_env_mod.make_exec_env(
            'python', repo=synthetic_repo, carve=carve,
            work_dir=synthetic_repo / 'trees')


@pytest.mark.parametrize('language', ['go', 'rust', 'c', 'cpp', 'java'])
def test_no_soundness_executor_for_the_whole_suite_languages(
        language, synthetic_repo, carve, tmp_path):
    """Never silently pass: an unexecuted check is exactly what v2 forbids."""
    with pytest.raises(NotImplementedError, match='only python is implemented'):
        exec_env_mod.make_exec_env(
            language, repo=synthetic_repo, carve=carve, work_dir=tmp_path / 't')


DISCRIMINATING_TEST = '''from mypkg.calc import add, scale


def test_add_returns_the_sum():
    assert add(2, 3) == 5


def test_scale_multiplies_every_element():
    assert scale([1, 2], 3) == [3, 6]
'''

NON_DISCRIMINATING_TEST = '''def test_arithmetic_still_works():
    assert 1 + 1 == 2
'''


def test_a_discriminating_test_passes_on_golden_and_fails_on_stub(
        synthetic_repo, carve, tmp_path):
    env = exec_env_mod.make_exec_env(
        'python', repo=synthetic_repo, carve=carve, work_dir=tmp_path / 'trees')
    golden = env.run(DISCRIMINATING_TEST, 'golden')
    stub = env.run(DISCRIMINATING_TEST, 'stub')
    assert golden['test_add_returns_the_sum'] == 'pass'
    assert stub['test_add_returns_the_sum'] in ('fail', 'error')


def test_a_non_discriminating_test_passes_on_both_trees(
        synthetic_repo, carve, tmp_path):
    env = exec_env_mod.make_exec_env(
        'python', repo=synthetic_repo, carve=carve, work_dir=tmp_path / 'trees')
    assert env.run(NON_DISCRIMINATING_TEST, 'golden')['test_arithmetic_still_works'] == 'pass'
    assert env.run(NON_DISCRIMINATING_TEST, 'stub')['test_arithmetic_still_works'] == 'pass'


MIXED_TEST_MODULE = DISCRIMINATING_TEST + '''

def test_add_is_commutative():
    assert add(1, 9) == add(9, 1)


def test_scale_by_zero_zeroes_everything():
    assert scale([4, 5], 0) == [0, 0]


def test_arithmetic_still_works():
    assert 1 + 1 == 2


def test_strings_still_concatenate():
    assert "a" + "b" == "ab"
'''


def test_only_pass_golden_and_fail_stub_checks_survive_the_loop(
        synthetic_repo, carve, tmp_path):
    """THE soundness gate, executed for real: kept vs dropped.

    Four tests exercise the carved behaviour (pass on golden, fail on stub) and
    two are true on any tree at all. Only the four discriminating ones may reach
    a bundle; the two that pass on BOTH prove nothing and are pruned out of the
    module entirely.
    """
    from verifier.generators.model_client import MockClient

    env = exec_env_mod.make_exec_env(
        'python', repo=synthetic_repo, carve=carve, work_dir=tmp_path / 'trees')
    client = MockClient(responder=lambda system, user: f'```python\n{MIXED_TEST_MODULE}```')

    code, result = verifier_loop.sound_pytest_loop(
        fx.TRUTH_MD, [CARVED_REL], GOLDEN_CALC, env.as_run_pytest(), client)

    assert result.ok
    assert result.sound == [
        'test_add_is_commutative',
        'test_add_returns_the_sum',
        'test_scale_by_zero_zeroes_everything',
        'test_scale_multiplies_every_element',
    ]
    assert result.dropped_non_discriminating == [
        'test_arithmetic_still_works', 'test_strings_still_concatenate',
    ]
    assert 'def test_add_returns_the_sum' in code
    assert 'def test_arithmetic_still_works' not in code


# ------------------------------------------------------------- bundle.py ---


def _build(base_dir, synthetic_repo, carve, tmp_path, *, client=None, **kwargs):
    env = exec_env_mod.make_exec_env(
        'python', repo=synthetic_repo, carve=carve, work_dir=tmp_path / 'trees')
    inp = inputs_mod.build_truth_inputs(carve, FakeTarget(), 'python',
                                        repo_name='synthetic')
    code = inputs_mod.solution_code(carve)
    return bundle_mod.build_verifier_bundle(
        inp, client or fx.mock_client(), base_dir,
        exec_runner=env.as_run_pytest(),
        golden_code=code.golden, stub_code=code.stub,
        echo=lambda _m: None, **kwargs)


def test_a_sound_bundle_freezes_all_six_named_artifacts(
        synthetic_repo, carve, tmp_path):
    base = tmp_path / 'bundle'
    assert _build(base, synthetic_repo, carve, tmp_path) is True
    for name, path in layout.bundle_paths(base).items():
        assert path.is_file(), name
    rubric = json.loads(layout.rubric_path(base).read_text())
    assert len([c for c in rubric['criteria'] if c['anchorable']]) >= 6
    assert json.loads(layout.manifest_path(base).read_text())['stage1'] == [CARVED_REL]


def test_the_frozen_pytest_module_holds_only_sound_tests(
        synthetic_repo, carve, tmp_path):
    base = tmp_path / 'bundle'
    _build(base, synthetic_repo, carve, tmp_path)
    code = layout.pytest_code_path(base).read_text()
    assert 'def test_no_unimplemented_body_marker' in code


def test_below_the_test_floor_nothing_at_all_is_written(
        synthetic_repo, carve, tmp_path):
    """A suite that passes on both trees is unsound: no bundle, no partial write."""
    base = tmp_path / 'bundle'
    shipped = _build(base, synthetic_repo, carve, tmp_path,
                     client=fx.non_discriminating_mock_client())
    assert shipped is False
    assert not base.exists()


def test_raising_the_criteria_floor_above_what_survives_aborts(
        synthetic_repo, carve, tmp_path):
    base = tmp_path / 'bundle'
    assert _build(base, synthetic_repo, carve, tmp_path, min_criteria=99) is False
    assert not base.exists()


def test_the_abort_reason_is_logged_not_swallowed(synthetic_repo, carve, tmp_path):
    logged: list[str] = []
    env = exec_env_mod.make_exec_env(
        'python', repo=synthetic_repo, carve=carve, work_dir=tmp_path / 'trees')
    inp = inputs_mod.build_truth_inputs(carve, FakeTarget(), 'python', repo_name='s')
    code = inputs_mod.solution_code(carve)
    bundle_mod.build_verifier_bundle(
        inp, fx.non_discriminating_mock_client(), tmp_path / 'bundle',
        exec_runner=env.as_run_pytest(), golden_code=code.golden,
        stub_code=code.stub, echo=logged.append)
    assert any('no bundle' in line and 'sound test' in line for line in logged)


def test_a_truth_doc_that_leaks_the_golden_aborts_the_bundle(
        synthetic_repo, carve, tmp_path):
    """The verbatim-golden leak-guard stays load-bearing, not advisory."""
    leaky = fx.TRUTH_MD + '\n'.join(
        line.strip() for line in GOLDEN_CALC.splitlines() if line.strip())
    base = tmp_path / 'bundle'
    assert _build(base, synthetic_repo, carve, tmp_path,
                  client=fx.mock_client(truth_md=leaky)) is False
    assert not base.exists()


def test_a_model_failure_propagates_instead_of_aborting_quietly(
        synthetic_repo, carve, tmp_path):
    """PLAN V4: an unreachable model is NOT the non-blocking soundness case."""
    class _Down:
        def complete(self, system, user, *, max_tokens=8192, reasoning_effort=None):
            raise ConnectionError('proxy refused the connection')

    with pytest.raises(ConnectionError):
        _build(tmp_path / 'bundle', synthetic_repo, carve, tmp_path, client=_Down())
    assert not (tmp_path / 'bundle').exists()


def test_the_judge_treats_an_empty_response_as_inconclusive():
    """A silent judge must not read as 'every criterion failed'."""
    from verifier.generators.model_client import MockClient
    from verifier.generators.rubric import Rubric, backbone_rubric

    judge = bundle_mod.judge_code_factory('contract', MockClient(responses=['']))
    result = judge(Rubric(criteria=backbone_rubric()), 'code')
    assert result.ok is False


# --------------------------------------------------------------- emit.py ---


def test_the_min_criteria_default_never_drifts_from_the_ported_floor():
    assert emit_mod.DEFAULT_VERIFIER_MIN_CRITERIA == verifier_loop.MIN_SOUND_CRITERIA


def test_a_missing_llm_config_is_a_loud_failure_not_a_silent_skip(tmp_path, monkeypatch):
    monkeypatch.delenv(emit_mod.VERIFIER_FAKE_CLIENT_ENV, raising=False)
    with pytest.raises(SystemExit) as exc:
        emit_mod.resolve_verifier_client(tmp_path / 'nope.json', echo=lambda _m: None)
    assert 'cannot read the LLM config' in str(exc.value)


def test_a_malformed_fake_client_hook_is_refused(monkeypatch):
    monkeypatch.setenv(emit_mod.VERIFIER_FAKE_CLIENT_ENV, 'not-a-callable-spec')
    with pytest.raises(SystemExit, match='module:callable'):
        emit_mod.resolve_verifier_client(None, echo=lambda _m: None)


def test_the_fake_client_hook_announces_itself(monkeypatch):
    monkeypatch.setenv(emit_mod.VERIFIER_FAKE_CLIENT_ENV,
                       'taskgen.tests.verifier_fixtures:sound_mock_client')
    logged: list[str] = []
    client = emit_mod.resolve_verifier_client(None, echo=logged.append)
    assert client is not None
    assert any('FAKE client hook' in line for line in logged)


def test_copying_a_bundle_normalises_line_endings(tmp_path):
    src = tmp_path / 'src'
    (src / 'pytest').mkdir(parents=True)
    (src / 'TRUTH.md').write_bytes(b'a\r\nb\r\n')
    (src / 'pytest' / 'predicates.json').write_bytes(b'[]\n')
    written = emit_mod._copy_bundle(src, tmp_path / 'dest')
    assert written == ['TRUTH.md', 'pytest/predicates.json']
    assert (tmp_path / 'dest' / 'TRUTH.md').read_bytes() == b'a\nb\n'


def test_the_verifier_flag_defaults_to_off_in_the_cli():
    from taskgen.cli import build_parser

    args = build_parser().parse_args(['generate', '--repo', '.', '--out', 'o'])
    assert args.verifier is False
    assert args.llm_config is None
    assert args.verifier_min_criteria == emit_mod.DEFAULT_VERIFIER_MIN_CRITERIA


def test_the_cli_threads_every_verifier_flag():
    from taskgen.cli import build_parser

    args = build_parser().parse_args([
        'generate', '--repo', '.', '--out', 'o', '--verifier',
        '--llm-config', '/tmp/cfg.json', '--verifier-min-criteria', '9',
    ])
    assert (args.verifier, args.llm_config, args.verifier_min_criteria) == (
        True, '/tmp/cfg.json', 9)


# ------------------------------------------- leak hardening (D6) ----------


def _leak_scene(tmp_path):
    repo_dir = tmp_path / 'ctx' / 'repo'
    (repo_dir / 'mypkg').mkdir(parents=True)
    (repo_dir / 'mypkg' / 'calc.py').write_text(STUB_CALC, encoding='utf-8')
    oracle_dir = tmp_path / 'oracle'
    (oracle_dir / 'mypkg').mkdir(parents=True)
    (oracle_dir / 'mypkg' / 'calc.py').write_text(
        GOLDEN_CALC + '    verified_total = compute_running_total(rows, rate=1)\n',
        encoding='utf-8')
    bundle_dir = tmp_path / 'bundle'
    bundle_dir.mkdir()
    return repo_dir, oracle_dir, bundle_dir


def test_an_answer_fragment_quoted_by_the_bundle_becomes_a_tripwire(tmp_path):
    repo_dir, oracle_dir, bundle_dir = _leak_scene(tmp_path)
    (bundle_dir / 'test_truth_generated.py').write_text(
        'def test_quotes_the_answer():\n'
        '    expected = "verified_total = compute_running_total(rows, rate=1)"\n'
        '    assert expected\n', encoding='utf-8')
    lines = staging_mod.bundle_tripwire_lines(bundle_dir, repo_dir, oracle_dir)
    assert lines == ('verified_total = compute_running_total(rows, rate=1)',)


def test_a_generic_bundle_line_is_never_promoted_to_a_tripwire(tmp_path):
    """Regression: this exact line failed leakscan against coverage/regions.py."""
    repo_dir, oracle_dir, bundle_dir = _leak_scene(tmp_path)
    (bundle_dir / 'test_truth_generated.py').write_text(
        'import ast\n\n\n'
        'def test_walk():\n'
        '    for node in ast.walk(tree):\n'
        '        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):\n'
        '            pass\n', encoding='utf-8')
    assert staging_mod.bundle_tripwire_lines(bundle_dir, repo_dir, oracle_dir) == ()


def test_a_line_that_survives_in_the_staged_tree_is_never_a_tripwire(tmp_path):
    repo_dir, oracle_dir, bundle_dir = _leak_scene(tmp_path)
    surviving = '"""Arithmetic helpers that live on in the carved tree."""'
    (repo_dir / 'mypkg' / 'other.py').write_text(surviving + '\n', encoding='utf-8')
    (oracle_dir / 'mypkg' / 'other.py').write_text(surviving + '\n', encoding='utf-8')
    (bundle_dir / 'TRUTH.md').write_text(surviving + '\n', encoding='utf-8')
    assert surviving not in staging_mod.bundle_tripwire_lines(
        bundle_dir, repo_dir, oracle_dir)


def test_merging_rewrites_the_one_file_leakscan_and_archaeology_both_read(tmp_path):
    repo_dir, oracle_dir, bundle_dir = _leak_scene(tmp_path)
    trip = tmp_path / 'trip' / 'tripwires.txt'
    trip.parent.mkdir()
    trip.write_text('an existing carved tripwire line of ample length\n', encoding='utf-8')
    (bundle_dir / 'x.py').write_text(
        'q = "verified_total = compute_running_total(rows, rate=1)"\n', encoding='utf-8')

    @dataclass
    class _Staged:
        tripwire_path: Path
        repo_dir: Path
        oracle_dir: Path

    added = staging_mod.merge_bundle_tripwires(
        _Staged(trip, repo_dir, oracle_dir), bundle_dir)
    assert len(added) == 1
    assert trip.read_text().splitlines() == [
        'an existing carved tripwire line of ample length',
        'verified_total = compute_running_total(rows, rate=1)',
    ]


def test_merging_nothing_leaves_the_tripwire_file_untouched(tmp_path):
    repo_dir, oracle_dir, bundle_dir = _leak_scene(tmp_path)
    trip = tmp_path / 'trip' / 'tripwires.txt'
    trip.parent.mkdir()
    trip.write_text('an existing carved tripwire line of ample length\n', encoding='utf-8')
    (bundle_dir / 'TRUTH.md').write_text('nothing here quotes the answer at all\n',
                                         encoding='utf-8')

    @dataclass
    class _Staged:
        tripwire_path: Path
        repo_dir: Path
        oracle_dir: Path

    before = trip.read_bytes()
    assert staging_mod.merge_bundle_tripwires(
        _Staged(trip, repo_dir, oracle_dir), bundle_dir) == ()
    assert trip.read_bytes() == before


# ---------------------------------------------- test.sh results.xml (8b) ---


def test_results_xml_is_written_at_run_time_by_the_rendered_script(tmp_path):
    """The JUnit report is persisted as results.xml, in the container, at run time."""
    from taskgen.langs import base as B
    from taskgen.langs.python import PythonPlugin

    graded = B.GradedSet(expected=1, floor_mode='equality', kind='pytest-allowlist',
                         selectors=('t.py::test_x',), fingerprint_sha256={})
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    (bin_dir / 'uv').write_text(
        '#!/usr/bin/env bash\n'
        'J=""\n'
        'for a in "$@"; do case "$a" in --junitxml=*) J="${a#--junitxml=}";; esac; done\n'
        '[ -n "$J" ] && mkdir -p "$(dirname "$J")" && printf \'%s\\n\' \\\n'
        '  \'<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/>'
        '</testsuites>\' > "$J"\n'
        'mkdir -p "$(dirname "${HARBOR_SELECTED_OUT}")"\n'
        'printf \'%s\\n\' 1 > "${HARBOR_SELECTED_OUT}"\n'
        'exit 0\n')
    (bin_dir / 'uv').chmod(0o755)

    repo = tmp_path / 'repo'
    repo.mkdir()
    allowlist = tmp_path / 'allowlist.txt'
    allowlist.write_text('t.py::test_x\n')
    logs = tmp_path / 'logs'
    script = tmp_path / 'test.sh'
    script.write_text(PythonPlugin().render_test_sh(graded))

    subprocess.run(['bash', str(script)], check=False, capture_output=True, env={
        'PATH': f'{bin_dir}:/usr/bin:/bin', 'REPO': str(repo),
        'VERIFIER_DIR': str(logs), 'ALLOWLIST': str(allowlist), 'HOME': str(tmp_path),
    })

    results = logs / 'results' / 'results.xml'
    assert results.is_file()
    assert results.read_text() == (logs / 'junit.xml').read_text()
    reward = json.loads((logs / 'reward.json').read_text())
    assert reward['reward'] == 1.0 and reward['binary'] == 1.0


def test_results_xml_is_never_an_emitted_or_image_bound_asset():
    """It is a run-time log, so it must not appear in the Dockerfile or solve.sh."""
    from taskgen.langs import base as B
    from taskgen.langs.python import PythonPlugin

    plugin = PythonPlugin()
    dockerfile = plugin.render_dockerfile(B.EnvSpec(repo_name='r'))
    assert 'results.xml' not in dockerfile
    assert 'results' not in plugin.render_solve_sh(('a.py',))


def test_the_results_dir_lives_under_the_run_time_logs_dir():
    from taskgen.langs import base as B
    from taskgen.langs.python import PythonPlugin

    graded = B.GradedSet(expected=1, floor_mode='equality', kind='pytest-allowlist',
                         selectors=('t.py::test_x',), fingerprint_sha256={})
    sh = PythonPlugin().render_test_sh(graded)
    assert 'RESULTS_DIR=${VERIFIER_DIR}/results' in sh
    assert 'RESULTS_XML=${RESULTS_DIR}/results.xml' in sh


def test_the_follow_up_matrix_names_all_five_unimplemented_languages():
    """PLAN 8b: documented, never faked -- and no other plugin secretly emits one."""
    from taskgen.langs import python as py_mod

    langs_dir = Path(py_mod.__file__).parent
    python_src = (langs_dir / 'python.py').read_text()
    for tool in ('gotestsum', 'surefire', 'nextest', '--output-junit', 'doctest'):
        assert tool in python_src, tool
    for other in ('go', 'rust', 'c', 'cpp', 'java'):
        assert 'results.xml' not in (langs_dir / f'{other}.py').read_text(), other
