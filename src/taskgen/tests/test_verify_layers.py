"""T5: fractional reward, floor modes, layer archaeology, harbor gate wrappers.

Docker is MOCKED throughout -- these must stay runnable without a daemon. The
real build/run proof is Wave 4.

The layer audit is the reason this file exists. The shipped verifier only ever
asked the FINAL filesystem whether the oracle was present, and that check passes
on an image whose first layer contains the intact answer: `docker save` unpacks
every layer, and a file deleted or overwritten in layer N is still readable in
layer N-1. Every test below that plants a tripwire line inside an inner layer is
a regression test for exactly that miss.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from taskgen import cli
from taskgen import verify as V


# --------------------------------------------------------------- fixtures ---

BODY_LINE = 'del task_copy.history[:-history_length]'
INTACT = f'def apply_history_length(task, config):\n    {BODY_LINE}\n    return task\n'
STUB = 'def apply_history_length(task, config):\n    raise NotImplementedError\n'

FRACTIONAL = ('{"reward": 0.5, "tests_passed": 2, "tests_total": 4, '
              '"binary": 0.0, "compiled": 1.0}')
GREEN = ('{"reward": 1.0, "tests_passed": 4, "tests_total": 4, '
         '"binary": 1.0, "compiled": 1.0}')
RED = ('{"reward": 0.0, "tests_passed": 0, "tests_total": 4, '
       '"binary": 0.0, "compiled": 1.0}')


def _layer_tar(dest: Path, members: dict[str, str]) -> None:
    """One `docker save` layer blob: a tar of files inside the image."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, 'w') as tf:
        for name, text in members.items():
            data = text.encode('utf-8')
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _saved_image(root: Path, *, leaking: bool) -> Path:
    """An unpacked `docker save` tree: OCI blobs plus a manifest.

    The final layer always holds the STUB -- that is what makes the leak
    invisible to a runtime-only check. `leaking` decides whether the EARLIER
    layer holds the intact original, which is the shape the shipped python
    Dockerfile actually produced (COPY intact, then overwrite one file).
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / 'manifest.json').write_text(json.dumps([{'Layers': ['blobs/sha256/a']}]))
    _layer_tar(
        root / 'blobs' / 'sha256' / 'a',
        {'opt/harbor/repo/src/a2a/utils/task.py': INTACT if leaking else STUB},
    )
    _layer_tar(root / 'blobs' / 'sha256' / 'b', {'opt/harbor/repo/README.md': '# repo\n'})
    _layer_tar(
        root / 'blobs' / 'sha256' / 'c',
        {'opt/harbor/repo/src/a2a/utils/task.py': STUB},
    )
    return root


class FakeRunner:
    """Records every docker interaction and replays canned results."""

    def __init__(self, replies=(), *, save_leaking=False, script_results=()):
        self._replies = list(replies)
        self._script_results = list(script_results)
        self.scripts: list[str] = []
        self.mounts: list[tuple] = []
        self.saves: list[tuple[str, str]] = []
        self.argvs: list[list[str]] = []
        self._save_leaking = save_leaking

    def build(self, image, dockerfile, context, repo, entry, target):
        return 'BUILD OK'

    def run(self, image, script, quiet=False, mounts=()):
        self.scripts.append(script)
        self.mounts.append(tuple(mounts))
        return self._replies.pop(0) if self._replies else ''

    def save(self, image, dest):
        self.saves.append((image, str(dest)))
        _saved_image(Path(dest), leaking=self._save_leaking)
        return 'SAVE OK'

    def script(self, argv, env=None):
        self.argvs.append(list(argv))
        return self._script_results.pop(0) if self._script_results else (0, '')

    def remove_image(self, image):
        self.scripts.append(f'<rm {image}>')
        return ''


# ------------------------------------------------------ fractional reward ---


def test_reward_json_carries_binary_and_compiled():
    r = V.parse_reward_json(FRACTIONAL)
    assert (r.reward, r.tests_passed, r.tests_total) == (0.5, 2, 4)
    assert r.binary == 0.0
    assert r.compiled == 1.0


def test_a_reward_json_without_binary_is_not_a_pass():
    """Fail closed: an unspoken binary bar is an unproven one, not a satisfied one."""
    legacy = '{"reward": 1.0, "tests_passed": 4, "tests_total": 4}'
    reasons = V.check_green(V.parse_reward_json(legacy), expected=4)
    assert any('binary' in r for r in reasons)


def test_green_requires_both_binary_one_and_reward_one():
    assert V.check_green(V.parse_reward_json(GREEN), expected=4) == []
    almost = ('{"reward": 1.0, "tests_passed": 4, "tests_total": 4, '
              '"binary": 0.0, "compiled": 1.0}')
    reasons = V.check_green(V.parse_reward_json(almost), expected=4)
    assert any('binary' in r for r in reasons)


def test_green_rejects_a_fractional_oracle():
    """The oracle restores the intact file: anything below 1.0 is unsolvable."""
    reasons = V.check_green(V.parse_reward_json(FRACTIONAL), expected=4)
    assert reasons


def test_red_accepts_zero_reward():
    assert V.check_red(V.parse_reward_json(RED)) == []


def test_red_rejects_a_zero_reward_that_claims_a_binary_pass():
    """Contradictory keys mean the parser is wrong; an inconsistent RED is not a RED."""
    weird = ('{"reward": 0.0, "tests_passed": 0, "tests_total": 4, '
             '"binary": 1.0, "compiled": 1.0}')
    assert V.check_red(V.parse_reward_json(weird))


# ------------------------------------------------------------- floor modes --


def test_equality_floor_rejects_a_shrunken_denominator():
    shrunk = ('{"reward": 1.0, "tests_passed": 2, "tests_total": 2, '
              '"binary": 1.0, "compiled": 1.0}')
    reasons = V.check_green(V.parse_reward_json(shrunk), expected=4,
                            floor_mode='equality')
    assert any('tests_total' in r for r in reasons)


def test_pinned_denominator_tolerates_an_observed_total_below_expected():
    """Spike G1: a go panic aborts the binary, so only some tests emit events.

    The oracle leg still passes all four, so the observed total may legitimately
    be smaller than the pinned denominator without the run being partial.
    """
    partial_report = ('{"reward": 1.0, "tests_passed": 4, "tests_total": 1, '
                      '"binary": 1.0, "compiled": 1.0}')
    reasons = V.check_green(V.parse_reward_json(partial_report), expected=4,
                            floor_mode='pinned-denominator')
    assert reasons == []


def test_pinned_denominator_still_rejects_scope_growth():
    grown = ('{"reward": 1.0, "tests_passed": 4, "tests_total": 9, '
             '"binary": 1.0, "compiled": 1.0}')
    reasons = V.check_green(V.parse_reward_json(grown), expected=4,
                            floor_mode='pinned-denominator')
    assert any('scope' in r.lower() or 'grew' in r.lower() for r in reasons)


def test_pinned_denominator_still_requires_every_expected_test_to_pass():
    short = ('{"reward": 0.75, "tests_passed": 3, "tests_total": 3, '
             '"binary": 0.0, "compiled": 1.0}')
    assert V.check_green(V.parse_reward_json(short), expected=4,
                         floor_mode='pinned-denominator')


def test_an_unknown_floor_mode_is_refused_rather_than_defaulted():
    with pytest.raises(V.VerifyError, match='floor_mode'):
        V.check_green(V.parse_reward_json(GREEN), expected=4, floor_mode='ge')


# ------------------------------------------------------- layer archaeology --


def test_a_tripwire_in_an_inner_layer_is_found(tmp_path):
    root = _saved_image(tmp_path / 'saved', leaking=True)
    hits = V.scan_extracted_image(root, patterns=[BODY_LINE])
    assert hits
    assert any('task.py' in h.member for h in hits)


def test_a_clean_save_yields_no_hits(tmp_path):
    root = _saved_image(tmp_path / 'saved', leaking=False)
    assert V.scan_extracted_image(root, patterns=[BODY_LINE]) == ()


def test_the_answer_file_is_found_by_digest_even_when_renamed(tmp_path):
    """A verbatim copy under another name defeats a path check and a line grep
    only if the line was boilerplate -- the content digest catches it anyway."""
    root = tmp_path / 'saved'
    _layer_tar(root / 'blobs' / 'sha256' / 'a', {'var/backup/whatever.bin': INTACT})
    import hashlib
    digest = hashlib.sha256(INTACT.encode()).hexdigest()
    hits = V.scan_extracted_image(root, patterns=[], digests=[digest])
    assert hits and hits[0].kind == 'digest'


def test_loose_files_outside_a_layer_tar_are_scanned_too(tmp_path):
    """`docker save` also writes config json blobs; a leak there is still a leak."""
    root = tmp_path / 'saved'
    root.mkdir()
    (root / 'config.json').write_text(json.dumps({'history': [BODY_LINE]}))
    assert V.scan_extracted_image(root, patterns=[BODY_LINE])


def test_layer_scan_verdict_is_clean_only_with_zero_hits():
    assert V.check_layers(()) == []
    hit = V.LayerHit(layer='blobs/sha256/a', member='opt/harbor/repo/x.py',
                     kind='tripwire', detail=BODY_LINE)
    assert V.check_layers((hit,))


def test_layer_audit_saves_the_image_scans_it_and_deletes_the_temp_dir(tmp_path):
    runner = FakeRunner(save_leaking=False)
    result = V.audit_image_layers('taskgen-x:local', runner,
                                  patterns=[BODY_LINE], tmp_root=tmp_path)
    assert result.ok
    assert runner.saves and runner.saves[0][0] == 'taskgen-x:local'
    workdir = Path(runner.saves[0][1])
    assert not workdir.exists(), 'an unpacked image is gigabytes -- it must not survive'


def test_layer_audit_fails_on_a_leaking_image_and_still_cleans_up(tmp_path):
    runner = FakeRunner(save_leaking=True)
    result = V.audit_image_layers('taskgen-x:local', runner,
                                  patterns=[BODY_LINE], tmp_root=tmp_path)
    assert not result.ok
    assert any('layer' in r.lower() for r in result.reasons)
    assert not Path(runner.saves[0][1]).exists()


def test_the_layer_audit_refuses_to_save_a_measure_image(tmp_path):
    """A measure image is built from the INTACT tree: saving it IS the leak."""
    runner = FakeRunner()
    with pytest.raises(Exception, match='never'):
        V.audit_image_layers('harbor-x-measure:local', runner, patterns=[BODY_LINE],
                             tmp_root=tmp_path)
    assert runner.saves == []


def test_layer_audit_with_no_tripwires_fails_closed(tmp_path):
    """No patterns means nothing was checked; an unproven image is not a clean one."""
    runner = FakeRunner()
    result = V.audit_image_layers('taskgen-x:local', runner, patterns=[],
                                  tmp_root=tmp_path)
    assert not result.ok


def test_layer_audit_cleans_up_even_when_docker_save_explodes(tmp_path):
    class Exploding(FakeRunner):
        def save(self, image, dest):
            self.saves.append((image, str(dest)))
            Path(dest).mkdir(parents=True, exist_ok=True)
            raise V.VerifyError('docker save failed (exit 1)')

    runner = Exploding()
    with pytest.raises(V.VerifyError):
        V.audit_image_layers('taskgen-x:local', runner, patterns=[BODY_LINE],
                             tmp_root=tmp_path)
    assert not Path(runner.saves[0][1]).exists()


# ------------------------------------------------- harbor script wrappers ---


def test_the_wrapped_harbor_scripts_exist_where_we_say_they_do():
    assert V.harbor_script('leakscan.sh').is_file()
    assert V.harbor_script('three_state_gate.sh').is_file()


def test_an_absent_harbor_script_is_an_error_not_a_reimplementation():
    with pytest.raises(V.VerifyError, match='harbor'):
        V.harbor_script('does_not_exist.sh')


def test_leakscan_exit_codes_map_to_verdicts():
    assert V.leakscan_verdict(0, 'LEAKSCAN PASS [image]: 0 hits') == []
    assert V.leakscan_verdict(1, 'LEAKSCAN FAIL [image]: ...')


def test_leakscan_could_not_run_fails_closed():
    """Exit 2 is `leakscan.sh`'s own could-not-run state (:23, :35)."""
    reasons = V.leakscan_verdict(2, 'LEAKSCAN FATAL: no tripwires at /tmp/x')
    assert reasons and 'could not run' in reasons[0].lower()


def test_an_unexpected_leakscan_exit_code_fails_closed():
    assert V.leakscan_verdict(127, 'bash: leakscan.sh: No such file')


def test_a_zero_exit_without_the_pass_line_fails_closed():
    """Silence proves nothing: the scanner must say it scanned."""
    assert V.leakscan_verdict(0, '')


def test_three_state_gate_verdict_needs_both_exit_zero_and_the_pass_line():
    assert V.three_state_verdict(0, 'PASS: RED -> 0\nPASS: GREEN -> 1\nGATE PASS: img') == []
    assert V.three_state_verdict(1, 'FAIL: GREEN -> 0, wanted 1\nGATE FAIL: img')
    assert V.three_state_verdict(0, 'some output with no verdict line')


def test_three_state_gate_reports_the_leg_that_failed():
    reasons = V.three_state_verdict(1, "FAIL: GREEN -> '0', wanted 1\nGATE FAIL: img")
    assert any('GREEN' in r for r in reasons)


def test_run_leakscan_invokes_the_harbor_script_with_the_tripwire_file(tmp_path):
    trip = tmp_path / 'tripwires.txt'
    trip.write_text(BODY_LINE + '\n')
    runner = FakeRunner(script_results=[(0, 'LEAKSCAN PASS [x]: 0 hits')])
    result = V.run_leakscan('taskgen-x:local', trip, runner)
    assert result.ok
    argv = runner.argvs[0]
    assert 'leakscan.sh' in ' '.join(argv)
    assert str(trip) in ' '.join(argv)


def test_run_three_state_gate_passes_harbor_the_five_documented_arguments(tmp_path):
    runner = FakeRunner(script_results=[(0, 'GATE PASS: img')])
    result = V.run_three_state_gate(
        'taskgen-x:local', '/opt/harbor/repo', tmp_path / 'oracle',
        'bash /opt/harbor/tests/test.sh', runner,
    )
    assert result.ok
    argv = runner.argvs[0]
    assert argv[0].endswith('three_state_gate.sh')
    assert argv[1:5] == ['taskgen-x:local', '/opt/harbor/repo',
                         str(tmp_path / 'oracle'), 'bash /opt/harbor/tests/test.sh']


# ------------------------------------------------------------- the verdict --


def _result(*, image_clean=True, layer_clean=True, red_ok=True, green_ok=True,
            integrity=True):
    red = V.StateResult('RED', V.parse_reward_json(RED))
    green = V.StateResult('GREEN', V.parse_reward_json(GREEN))
    if not red_ok:
        red.reasons = ['nope']
    if not green_ok:
        green.reasons = ['nope']
    return V.VerifyResult(
        image='taskgen-x:local', expected=4, red=red, green=green,
        image_clean=V.CheckResult('IMAGE', [] if image_clean else ['leak']),
        layer_clean=V.CheckResult('LAYER', [] if layer_clean else ['leak']),
        integrity=V.CheckResult('INTEGRITY', [] if integrity else ['tampered']),
    )


def test_the_verdict_needs_every_gate():
    assert _result().passed
    assert not _result(image_clean=False).passed
    assert not _result(layer_clean=False).passed
    assert not _result(red_ok=False).passed
    assert not _result(green_ok=False).passed
    assert not _result(integrity=False).passed


def test_an_unaudited_image_does_not_pass():
    """layer_clean=None means the audit never ran; that is not a clean bill."""
    result = _result()
    result.layer_clean = None
    assert not result.passed


# ------------------------------------------------------------------- cli ----


def test_verify_accepts_lang_and_carve_scope():
    args = cli.build_parser().parse_args(
        ['verify', '--all', 'out', '--lang', 'go', '--carve-scope', 'file']
    )
    assert args.lang == 'go'
    assert args.carve_scope == 'file'


def test_verify_defaults_to_python_function_scope():
    args = cli.build_parser().parse_args(['verify', '--entry', 'e'])
    assert args.lang == 'python'
    assert args.carve_scope == 'function'


def test_cli_passes_lang_and_scope_through_to_verify(monkeypatch):
    seen = {}
    monkeypatch.setattr(V, 'main', lambda **kw: seen.update(kw) or 0)
    args = cli.build_parser().parse_args(
        ['verify', '--all', 'out', '--lang', 'go', '--carve-scope', 'folder']
    )
    assert args.func_impl(args) == 0
    assert seen['lang'] == 'go'
    assert seen['carve_scope'] == 'folder'


def test_an_unknown_carve_scope_is_rejected_at_the_cli():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(['verify', '--entry', 'e', '--carve-scope', 'module'])
