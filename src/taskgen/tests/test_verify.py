"""Unit tests for the docker oracle verifier's pure logic.

Docker itself is NOT exercised here -- these tests must stay runnable on a
machine with no daemon, in a fraction of a second. Every subprocess boundary is
crossed through the injectable `Runner` protocol, so what is under test is the
part that decides PASS/FAIL, not the part that shells out.

The real docker build/run is the manual integration proof
(`python -m taskgen.cli verify --all .taskgen_out`), deliberately not gated
behind this suite.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from taskgen import verify as V


# --------------------------------------------------------------- fixtures ---

RED_JSON = '{"reward": 0.0, "tests_passed": 0, "tests_total": 5, "binary": 0.0, "compiled": 0.0}'
GREEN_JSON = '{"reward": 1.0, "tests_passed": 5, "tests_total": 5, "binary": 1.0, "compiled": 1.0}'
CLEAN_OUT = 'NO_SOLUTION\n'


class FakeRunner:
    """Records every script it is asked to run and replays canned stdout."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.scripts: list[str] = []
        self.builds: list[tuple] = []
        self.mounts: list[tuple] = []

    def build(self, plan):
        self.builds.append(plan)
        return 'BUILD OK'

    def run(self, image, script, quiet=False, mounts=()):
        self.scripts.append(script)
        self.mounts.append(tuple(mounts))
        # Diagnostic re-runs happen after the canned graded runs and their
        # output is only ever echoed, never parsed.
        return self._replies.pop(0) if self._replies else ''

    def remove_image(self, image):
        self.scripts.append(f'<rm {image}>')
        return ''


def _write_entry(root: Path, entry_id: str, condition: str, *, carve_body: str = 'STUB\n') -> Path:
    d = root / entry_id
    (d / 'tests').mkdir(parents=True)
    (d / 'solution' / 'carved' / 'src').mkdir(parents=True)
    (d / 'environment' / 'carve' / 'src').mkdir(parents=True)
    (d / 'task.toml').write_text(
        textwrap.dedent(f"""
        schema_version = "1.4"
        [task]
        name = "triton/python-a2a-python__apply_history_length__{condition}"
        [metadata]
        id = "{entry_id}"
        name = "python-a2a-python__apply_history_length__{condition}"
        condition = "{condition}"
        target_file = "src/a2a/utils/task.py"
        target_func = "apply_history_length"
        graded_tests = ["t.py::a", "t.py::b"]
        [environment]
        docker_image = "taskgen-python-a2a-python:local"
        """).strip()
        + '\n'
    )
    (d / 'tests' / 'allowlist.txt').write_text('t.py::a\nt.py::b\n')
    (d / 'tests' / 'graded.json').write_text(
        '{"expected": 2, "kind": "pytest-allowlist", "selectors": ["t.py::a", "t.py::b"]}\n'
    )
    (d / 'tests' / 'harbor_filter.py').write_text('# filter\n')
    (d / 'tests' / 'test.sh').write_text(f'#!/bin/sh\n# verifier -- {condition}\nEXPECTED=2\n')
    (d / 'solution' / 'solve.sh').write_text('#!/bin/sh\necho SOLVE OK\n')
    (d / 'solution' / 'carved' / 'src' / 'task.py').write_text('INTACT\n')
    (d / 'environment' / 'Dockerfile').write_text(f'# env -- {condition}\nFROM harbor-base:local\n')
    (d / 'environment' / 'carve' / 'src' / 'task.py').write_text(carve_body)
    return d


# ------------------------------------------------------- reward.json parse ---


def test_parse_plain_reward_json():
    r = V.parse_reward_json(RED_JSON)
    assert (r.reward, r.tests_passed, r.tests_total) == (0.0, 0, 5)


def test_parse_ignores_leading_solve_output():
    """The GREEN command prints solve.sh's chatter before the json."""
    payload = 'restored src/a2a/utils/task.py (sha256 96e2)\nimport a2a OK\nSOLVE OK\n' + GREEN_JSON
    r = V.parse_reward_json(payload)
    assert r.reward == 1.0
    assert r.tests_passed == 5


def test_parse_takes_the_last_json_object():
    r = V.parse_reward_json(RED_JSON + '\n' + GREEN_JSON + '\n')
    assert r.reward == 1.0


def test_parse_empty_output_is_an_error():
    """A missing reward.json means `cat` failed -- never a silent zero."""
    with pytest.raises(V.VerifyError, match='no reward json'):
        V.parse_reward_json('   \n')


def test_parse_garbage_is_an_error():
    with pytest.raises(V.VerifyError, match='no reward json'):
        V.parse_reward_json('bash: line 1: uv: command not found\n')


def test_parse_rejects_json_without_reward_key():
    with pytest.raises(V.VerifyError, match='no reward json'):
        V.parse_reward_json('{"tests_passed": 5}')


def test_parse_rejects_non_numeric_reward():
    with pytest.raises(V.VerifyError, match='not a number'):
        V.parse_reward_json('{"reward": "1.0", "tests_passed": 5, "tests_total": 5}')


# ------------------------------------------------------------- verdicts -----


def test_red_verdict_accepts_zero():
    assert V.check_red(V.parse_reward_json(RED_JSON)) == []


def test_red_verdict_rejects_one():
    """A task that is green before any work is worthless -- this must fail."""
    reasons = V.check_red(V.parse_reward_json(GREEN_JSON))
    assert reasons and 'reward' in reasons[0]


def test_green_verdict_accepts_full_marks():
    assert V.check_green(V.parse_reward_json(GREEN_JSON), expected=5) == []


def test_green_verdict_rejects_zero_reward():
    assert V.check_green(V.parse_reward_json(RED_JSON), expected=5)


def test_green_verdict_rejects_partial_pass():
    partial = '{"reward": 1.0, "tests_passed": 4, "tests_total": 5}'
    reasons = V.check_green(V.parse_reward_json(partial), expected=5)
    assert any('tests_passed' in r for r in reasons)


def test_green_verdict_rejects_shrunken_denominator():
    """reward=1 on 2/2 when 5 were promised is a shrunken suite, not a pass."""
    shrunk = '{"reward": 1.0, "tests_passed": 2, "tests_total": 2}'
    reasons = V.check_green(V.parse_reward_json(shrunk), expected=5)
    assert any('tests_total' in r for r in reasons)


# ------------------------------------------------------ state orchestration --


def test_run_states_issues_the_three_canonical_commands():
    runner = FakeRunner([CLEAN_OUT, RED_JSON, GREEN_JSON])
    result = V.run_states('taskgen-x:local', runner, expected=5, solution_dir='/entry/solution')

    # `runtime_ok`, not `passed`: run_states grades what three containers can
    # show. `passed` additionally requires the `docker save` layer audit, which
    # verify_entry owns -- an image no one has unpacked is not a clean image.
    assert result.runtime_ok
    assert len(runner.scripts) == 3
    probe, red, green = runner.scripts
    assert f'test ! -e {V.SOLUTION_MOUNTPOINT}' in probe
    assert 'solve.sh' not in red
    assert '/opt/harbor/tests/test.sh' in red
    assert 'reward.json' in red
    # GREEN must run the oracle first and only grade if it succeeded.
    assert green.index('solve.sh') < green.index('test.sh')
    assert '&&' in green


def test_only_the_green_run_mounts_the_oracle():
    """The probe and RED must see the image exactly as the agent will."""
    runner = FakeRunner([CLEAN_OUT, RED_JSON, GREEN_JSON])
    V.run_states('taskgen-x:local', runner, expected=5, solution_dir='/entry/solution')

    probe_mounts, red_mounts, green_mounts = runner.mounts
    assert probe_mounts == ()
    assert red_mounts == ()
    assert green_mounts == (('/entry/solution', V.SOLUTION_MOUNTPOINT),)


def test_run_states_fails_when_the_stub_is_already_green():
    runner = FakeRunner([CLEAN_OUT, GREEN_JSON, GREEN_JSON])
    result = V.run_states('taskgen-x:local', runner, expected=5)
    assert not result.passed
    assert result.red_reasons
    assert len(runner.scripts) == 4, 'the failing state must be re-run for diagnostics'


def test_run_states_fails_when_the_oracle_does_not_score_one():
    runner = FakeRunner([CLEAN_OUT, RED_JSON, RED_JSON])
    result = V.run_states('taskgen-x:local', runner, expected=5)
    assert not result.passed
    assert result.green_reasons
    assert result.red_reasons == []


# ------------------------------------------------------------ image hygiene --


def test_image_without_the_oracle_passes():
    assert V.check_image_has_no_solution('NO_SOLUTION\n') == []


def test_image_carrying_the_oracle_fails():
    leak = 'SOLUTION_LEAK\n/opt/harbor/solution:\ncarved\nsolve.sh\n'
    reasons = V.check_image_has_no_solution(leak)
    assert reasons and 'shell' in reasons[0]


def test_a_clean_marker_inside_a_leaked_listing_does_not_excuse_the_leak():
    """A file named NO_SOLUTION in the leaked tree must not flip the verdict."""
    leak = 'SOLUTION_LEAK\n/opt/harbor/solution:\nNO_SOLUTION\nsolve.sh\n'
    assert V.check_image_has_no_solution(leak)


def test_a_silent_probe_is_a_failure_not_a_pass():
    """No marker means the probe never ran; absence must be proven, not assumed."""
    assert V.check_image_has_no_solution('')
    assert V.check_image_has_no_solution('bash: docker: command not found\n')


def test_a_leaking_image_fails_the_whole_verdict_even_when_red_and_green_are_right():
    runner = FakeRunner(['SOLUTION_LEAK\nsolve.sh\n', RED_JSON, GREEN_JSON])
    result = V.run_states('taskgen-x:local', runner, expected=5, solution_dir='/entry/solution')
    assert result.red.ok and result.green.ok
    assert not result.passed, 'a baked-in oracle must sink the verdict on its own'


# ------------------------------------------------------------ entry specs ---


def test_load_entry_reads_image_and_expected(tmp_path):
    d = _write_entry(tmp_path, '11111111-1111-5111-8111-111111111111', 'no_context')
    spec = V.load_entry(d)
    assert spec.image == 'taskgen-python-a2a-python:local'
    assert spec.expected == 2
    assert spec.condition == 'no_context'
    assert spec.repo_dirname == 'python-a2a-python'


def test_load_entry_rejects_allowlist_that_disagrees_with_task_toml(tmp_path):
    d = _write_entry(tmp_path, '11111111-1111-5111-8111-111111111111', 'no_context')
    (d / 'tests' / 'allowlist.txt').write_text('t.py::a\n')
    with pytest.raises(V.VerifyError, match='allowlist'):
        V.load_entry(d)


def test_load_entry_rejects_a_directory_that_is_not_an_entry(tmp_path):
    with pytest.raises(V.VerifyError, match='task.toml'):
        V.load_entry(tmp_path)


# ---------------------------------------------- shared-image equivalence ----


def test_entries_sharing_one_carve_are_reported_identical(tmp_path):
    a = _write_entry(tmp_path, 'aaaaaaaa-1111-5111-8111-111111111111', 'no_context')
    b = _write_entry(tmp_path, 'bbbbbbbb-1111-5111-8111-111111111111', 'bm25')
    shared = V.compare_entries([a, b])
    assert shared.equivalent
    assert 'tests/allowlist.txt' in shared.identical
    assert 'solution/solve.sh' in shared.identical
    assert 'environment/carve/src/task.py' in shared.identical
    # The slug comment in the header is the only permitted drift.
    assert 'tests/test.sh' in shared.comment_only
    assert 'environment/Dockerfile' in shared.comment_only
    assert shared.mismatched == ()


def test_a_differing_carve_breaks_the_shared_image_claim(tmp_path):
    a = _write_entry(tmp_path, 'aaaaaaaa-1111-5111-8111-111111111111', 'no_context')
    b = _write_entry(tmp_path, 'bbbbbbbb-1111-5111-8111-111111111111', 'bm25',
                     carve_body='DIFFERENT STUB\n')
    shared = V.compare_entries([a, b])
    assert not shared.equivalent
    assert 'environment/carve/src/task.py' in shared.mismatched


def test_a_differing_graded_command_breaks_the_shared_image_claim(tmp_path):
    a = _write_entry(tmp_path, 'aaaaaaaa-1111-5111-8111-111111111111', 'no_context')
    b = _write_entry(tmp_path, 'bbbbbbbb-1111-5111-8111-111111111111', 'bm25')
    (b / 'tests' / 'test.sh').write_text('#!/bin/sh\n# verifier -- bm25\nEXPECTED=1\n')
    shared = V.compare_entries([a, b])
    assert not shared.equivalent
    assert 'tests/test.sh' in shared.mismatched


def test_entries_with_different_images_are_not_covered_by_one_build(tmp_path):
    a = _write_entry(tmp_path, 'aaaaaaaa-1111-5111-8111-111111111111', 'no_context')
    b = _write_entry(tmp_path, 'bbbbbbbb-1111-5111-8111-111111111111', 'bm25')
    (b / 'task.toml').write_text(
        (b / 'task.toml').read_text().replace('taskgen-python-a2a-python:local', 'other:local')
    )
    with pytest.raises(V.VerifyError, match='docker_image'):
        V.discover_entries(tmp_path)


def test_discover_entries_is_sorted_and_skips_non_entries(tmp_path):
    _write_entry(tmp_path, 'bbbbbbbb-1111-5111-8111-111111111111', 'bm25')
    _write_entry(tmp_path, 'aaaaaaaa-1111-5111-8111-111111111111', 'no_context')
    (tmp_path / 'not-an-entry').mkdir()
    specs = V.discover_entries(tmp_path)
    assert [s.entry_id for s in specs] == [
        'aaaaaaaa-1111-5111-8111-111111111111',
        'bbbbbbbb-1111-5111-8111-111111111111',
    ]


def test_discover_entries_refuses_an_empty_directory(tmp_path):
    with pytest.raises(V.VerifyError, match='no entries'):
        V.discover_entries(tmp_path)


# ------------------------------------------------------------- cleanup ------


def test_only_taskgen_images_may_be_removed():
    assert V.is_removable_image('taskgen-python-a2a-python:local')
    assert not V.is_removable_image('harbor-base:local')
    assert not V.is_removable_image('harbor-py-a2a-intact:local')


def test_cleanup_refuses_to_delete_a_harbor_image():
    runner = FakeRunner([])
    with pytest.raises(V.VerifyError, match='refusing'):
        V.cleanup_image('harbor-base:local', runner)


def test_cleanup_removes_the_built_taskgen_image():
    runner = FakeRunner([])
    V.cleanup_image('taskgen-x:local', runner)
    assert runner.scripts == ['<rm taskgen-x:local>']
