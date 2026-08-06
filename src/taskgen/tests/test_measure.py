"""T13: the two-phase measure/lock that breaks the floor circularity (plan A2).

A pinned denominator can only be measured against the INTACT tree, but the
thing that enforces the floor is the same test.sh that would measure it, and the
only image that can measure it contains the answer by construction. Phase 1
resolves that: a throwaway MEASURE image, built from an intact `repoctx`, running
a floor-FREE test.sh that counts and asserts nothing, writing `graded.lock.json`.

Two properties are non-negotiable and are what most of this file tests:

  * the lock is DETERMINISTIC -- no timestamp, no ordering luck -- because
    `generate` consumes it offline and regeneration must stay byte-identical;
  * the measure image is NEVER-SHIP: never `docker save`d, never entered into the
    dry-run matrix, and deleted in a `finally` even when the run explodes. It
    holds the intact tree by construction, so an escaped measure image is a
    total leak.

Docker is MOCKED. Real measurement runs in Wave 2, once a language plugin can
supply a toolchain.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taskgen import measure as M
from taskgen.langs import base as B


# --------------------------------------------------------------- fixtures ---


class FakePlugin(B.LangPlugin):
    name = 'fake'
    toml_family = 'A'
    floor_mode = 'equality'

    def toolchain_spec(self):
        return B.ToolchainSpec(install_block='RUN true')

    def dep_warm_spec(self):
        return B.DepWarmSpec()

    def render_test_sh(self, graded, *, expected=None, fingerprint=None):
        return '#!/usr/bin/env bash\nEXPECTED=%d\n' % (expected or graded.expected)

    def measure_test_sh(self, **kwargs):
        return '#!/usr/bin/env bash\n# floor-free: counts only\nmeasure 5 ""\n'


MEASURE_OUT = ('{"tests_total": 5, "graded": ["t.py::b", "t.py::a", "t.py::e", '
               '"t.py::d", "t.py::c"]}')
GRADED_SORTED = ['t.py::a', 't.py::b', 't.py::c', 't.py::d', 't.py::e']


class FakeRunner:
    def __init__(self, replies=(), digests=None, *, run_raises=False):
        self._replies = list(replies)
        self._digests = digests or {}
        self._run_raises = run_raises
        self.builds: list[dict] = []
        self.runs: list[tuple[str, str]] = []
        self.removed: list[str] = []
        self.saved: list[str] = []

    def build_with_contexts(self, image, dockerfile, context, contexts, target=None):
        self.builds.append({
            'image': image, 'dockerfile': str(dockerfile), 'context': str(context),
            'contexts': {k: str(v) for k, v in contexts.items()}, 'target': target,
        })
        return 'BUILD OK'

    def run(self, image, script, quiet=False, mounts=()):
        self.runs.append((image, script))
        if self._run_raises:
            raise M.MeasureError('container exploded')
        return self._replies.pop(0) if self._replies else ''

    def image_digest(self, image):
        return self._digests.get(image, 'sha256:' + 'de' * 32)

    def remove_image(self, image):
        self.removed.append(image)
        return ''

    def save(self, image, dest):
        self.saved.append(image)


def _request(tmp_path: Path, **kw) -> M.MeasureRequest:
    repo = tmp_path / 'repo'
    (repo / 'src').mkdir(parents=True, exist_ok=True)
    (repo / 'src' / 'mod.py').write_text('def f():\n    return 1\n')
    defaults = dict(
        repo=repo,
        repo_name='python-a2a-python',
        lang='fake',
        scope='function',
        intact_image='harbor-py-a2a-intact:local',
        dockerfile=tmp_path / 'Dockerfile.measure',
        context=tmp_path / 'ctx',
        fingerprint_sha256={'tests/test_task.py': 'ab' * 32},
        floor_mode='equality',
    )
    defaults.update(kw)
    (tmp_path / 'Dockerfile.measure').write_text('FROM harbor-base:local\n')
    (tmp_path / 'ctx').mkdir(exist_ok=True)
    return M.MeasureRequest(**defaults)


# ------------------------------------------------- the never-ship contract ---


def test_the_measure_image_tag_is_recognisable_as_never_ship():
    tag = M.measure_image_tag('python-a2a-python')
    assert tag == 'harbor-python-a2a-python-measure:local'
    assert M.is_measure_image(tag)
    assert not M.is_measure_image('taskgen-python-a2a-python:local')
    assert not M.is_measure_image('harbor-py-a2a-intact:local')


def test_saving_a_measure_image_is_refused():
    """It contains the intact tree by construction: `docker save` would be the leak."""
    with pytest.raises(M.MeasureError, match='never'):
        M.assert_never_shipped('harbor-x-measure:local')
    M.assert_never_shipped('taskgen-x:local')


def test_measure_deletes_the_measure_image(tmp_path):
    runner = FakeRunner([MEASURE_OUT])
    M.measure(_request(tmp_path), FakePlugin(), runner, echo=lambda *_: None)
    assert runner.removed == [M.measure_image_tag('python-a2a-python')]


def test_measure_deletes_the_measure_image_even_when_the_run_fails(tmp_path):
    runner = FakeRunner(run_raises=True)
    with pytest.raises(M.MeasureError):
        M.measure(_request(tmp_path), FakePlugin(), runner, echo=lambda *_: None)
    assert runner.removed == [M.measure_image_tag('python-a2a-python')], \
        'an escaped measure image holds the intact tree -- deletion cannot be skipped'


def test_measure_deletes_the_measure_image_even_when_the_output_is_garbage(tmp_path):
    runner = FakeRunner(['bash: no such file'])
    with pytest.raises(M.MeasureError):
        M.measure(_request(tmp_path), FakePlugin(), runner, echo=lambda *_: None)
    assert runner.removed


def test_measure_never_saves_the_image(tmp_path):
    runner = FakeRunner([MEASURE_OUT])
    M.measure(_request(tmp_path), FakePlugin(), runner, echo=lambda *_: None)
    assert runner.saved == []


# ------------------------------------------------------------ phase 1 build --


def test_the_measure_image_is_built_from_the_INTACT_repo(tmp_path):
    """No carve: phase 1 measures what the suite is before anything is removed."""
    request = _request(tmp_path)
    runner = FakeRunner([MEASURE_OUT])
    M.measure(request, FakePlugin(), runner, echo=lambda *_: None)
    build = runner.builds[0]
    assert build['contexts'][B.REPO_CONTEXT] == str(request.repo)
    assert build['image'] == M.measure_image_tag('python-a2a-python')


def test_the_measure_run_uses_the_plugins_floor_free_script(tmp_path):
    runner = FakeRunner([MEASURE_OUT])
    M.measure(_request(tmp_path), FakePlugin(), runner, echo=lambda *_: None)
    _, script = runner.runs[0]
    assert 'measure.json' in script
    assert 'EXPECTED' not in script


# ------------------------------------------------------ the measured output --


def test_parse_measure_json_reads_the_count_and_the_selectors():
    total, graded = M.parse_measure_json(MEASURE_OUT)
    assert total == 5
    assert list(graded) == GRADED_SORTED, 'selectors must be sorted, not run-ordered'


def test_parse_measure_json_ignores_chatter_before_the_object():
    total, _ = M.parse_measure_json('compiling...\nok 5 tests\n' + MEASURE_OUT)
    assert total == 5


def test_a_missing_measure_json_is_an_error_not_a_zero():
    with pytest.raises(M.MeasureError, match='measure json'):
        M.parse_measure_json('')


def test_a_measure_of_zero_tests_is_refused():
    """A floor of zero is not a floor -- every carve would grade green for free."""
    with pytest.raises(M.MeasureError, match='zero'):
        M.parse_measure_json('{"tests_total": 0, "graded": []}')


def test_selectors_that_disagree_with_the_count_are_refused():
    """If the counter and the enumerator disagree, neither can be trusted."""
    with pytest.raises(M.MeasureError, match='disagree'):
        M.parse_measure_json('{"tests_total": 5, "graded": ["a", "b"]}')


# ------------------------------------------------------------------- lock ----


def _lock(tmp_path, **kw):
    runner = FakeRunner([MEASURE_OUT])
    return M.measure(_request(tmp_path, **kw), FakePlugin(), runner, echo=lambda *_: None)


def test_the_lock_pins_the_measured_denominator(tmp_path):
    lock = _lock(tmp_path)
    assert lock['expected'] == 5
    assert lock['graded'] == GRADED_SORTED
    assert lock['floor_mode'] == 'equality'
    assert lock['lang'] == 'fake'
    assert lock['scope'] == 'function'


def test_the_lock_carries_its_provenance(tmp_path):
    lock = _lock(tmp_path)
    prov = lock['provenance']
    assert prov['repo_sha256']
    assert prov['intact_image'] == 'harbor-py-a2a-intact:local'
    assert prov['intact_image_digest'].startswith('sha256:')


def test_the_lock_has_no_timestamp_and_no_host_paths(tmp_path):
    """`generate` consumes the lock offline; regeneration must stay byte-identical."""
    text = M.render_lock(_lock(tmp_path))
    assert 'generated_at' not in text
    assert 'timestamp' not in text
    assert str(tmp_path) not in text


def test_the_lock_is_byte_identical_across_runs(tmp_path):
    first = M.render_lock(_lock(tmp_path))
    second = M.render_lock(_lock(tmp_path))
    assert first == second
    assert first.endswith('\n')


def test_the_lock_round_trips_through_disk(tmp_path):
    lock = _lock(tmp_path)
    path = tmp_path / 'graded.lock.json'
    M.write_lock(path, lock)
    assert M.load_lock(path) == lock
    assert path.name == M.LOCK_FILENAME


def test_the_lock_fingerprints_the_graded_test_files(tmp_path):
    lock = _lock(tmp_path)
    assert lock['fingerprint_sha256'] == {'tests/test_task.py': 'ab' * 32}


def test_the_lock_yields_a_graded_set_the_plugins_can_consume(tmp_path):
    graded = M.graded_set_from_lock(_lock(tmp_path))
    assert isinstance(graded, B.GradedSet)
    assert graded.expected == 5
    assert graded.floor_mode == 'equality'
    assert graded.fingerprint_relpaths == ('tests/test_task.py',)


# ------------------------------------------------------------ provenance -----


def test_provenance_accepts_the_repo_it_was_measured_from(tmp_path):
    request = _request(tmp_path)
    lock = _lock(tmp_path)
    assert M.check_provenance(lock, M.repo_tree_sha256(request.repo),
                              lock['provenance']['intact_image_digest']) == []


def test_provenance_rejects_a_different_repo(tmp_path):
    lock = _lock(tmp_path)
    reasons = M.check_provenance(lock, 'f' * 64,
                                 lock['provenance']['intact_image_digest'])
    assert any('repo' in r for r in reasons)


def test_provenance_rejects_a_different_intact_image(tmp_path):
    request = _request(tmp_path)
    lock = _lock(tmp_path)
    reasons = M.check_provenance(lock, M.repo_tree_sha256(request.repo), 'sha256:beef')
    assert any('image' in r for r in reasons)


def test_repo_tree_sha256_is_deterministic_and_content_sensitive(tmp_path):
    repo = tmp_path / 'r'
    (repo / 'a').mkdir(parents=True)
    (repo / 'a' / 'x.py').write_text('one\n')
    (repo / 'b.py').write_text('two\n')
    first = M.repo_tree_sha256(repo)
    assert first == M.repo_tree_sha256(repo)
    (repo / 'b.py').write_text('three\n')
    assert M.repo_tree_sha256(repo) != first


def test_repo_tree_sha256_ignores_vcs_and_build_detritus(tmp_path):
    """.git carries the carved files' full history: it must not move the provenance."""
    repo = tmp_path / 'r'
    repo.mkdir()
    (repo / 'x.py').write_text('one\n')
    before = M.repo_tree_sha256(repo)
    (repo / '.git').mkdir()
    (repo / '.git' / 'HEAD').write_text('ref: refs/heads/main\n')
    (repo / '__pycache__').mkdir()
    (repo / '__pycache__' / 'x.pyc').write_bytes(b'\x00\x01')
    assert M.repo_tree_sha256(repo) == before


# ----------------------------------------------------------- floor sanity ----


def test_a_plugin_whose_floor_mode_disagrees_with_the_request_is_refused(tmp_path):
    """The lock records ONE floor mode; two sources for it is one too many."""
    runner = FakeRunner([MEASURE_OUT])
    with pytest.raises(M.MeasureError, match='floor_mode'):
        M.measure(_request(tmp_path, floor_mode='pinned-denominator'), FakePlugin(),
                  runner, echo=lambda *_: None)


def test_the_lock_json_is_sorted_so_two_hosts_agree(tmp_path):
    text = M.render_lock(_lock(tmp_path))
    keys = list(json.loads(text))
    assert keys == sorted(keys)


# ------------------------------------------- the success path is UNCHANGED ---

#: Captured by running the pre-stderr-capture measure.py at f046b4f. If surfacing
#: a diagnostic on FAILURE ever moves a byte of the SUCCESS path, this literal is
#: what catches it: `generate` consumes the lock offline and regeneration is a
#: `diff -r` proof.
GOLDEN_MEASURE_JSON = '{"tests_total": 3, "graded": ["t.py::c", "t.py::a", "t.py::b"]}'
GOLDEN_LOCK_TEXT = (
    '{\n  "carve_set_sha256": "",\n  "expected": 3,\n  "fingerprint_sha256": {\n'
    '    "tests/test_x.py": "' + 'ab' * 32 + '"\n  },\n  "floor_mode": "equality",\n'
    '  "graded": [\n    "t.py::a",\n    "t.py::b",\n    "t.py::c"\n  ],\n'
    '  "lang": "fake",\n  "provenance": {\n'
    '    "intact_image": "harbor-x-intact:local",\n'
    '    "intact_image_digest": "sha256:' + 'ef' * 32 + '",\n'
    '    "repo_sha256": "' + 'cd' * 32 + '"\n  },\n  "schema": 1,\n'
    '  "scope": "function",\n  "tool_version": "taskgen/1"\n}\n'
)


def test_the_lock_for_a_fixed_measure_json_is_byte_identical_to_the_golden():
    total, graded = M.parse_measure_json(GOLDEN_MEASURE_JSON)
    lock = M.build_lock(
        lang='fake', scope='function', expected=total, graded=graded,
        floor_mode='equality', fingerprint_sha256={'tests/test_x.py': 'ab' * 32},
        repo_sha256='cd' * 32, intact_image='harbor-x-intact:local',
        intact_image_digest='sha256:' + 'ef' * 32,
    )
    assert M.render_lock(lock) == GOLDEN_LOCK_TEXT


def test_capturing_stderr_does_not_change_what_the_parser_reads():
    """The marker splits the run output; the json half parses as it always did."""
    run_output = (
        f'{GOLDEN_MEASURE_JSON}\n{M.MEASURE_STDERR_MARKER}\n'
        'cc1: error: no such file or directory\n'
    )
    payload, stderr = M.split_measure_output(run_output)
    assert payload == GOLDEN_MEASURE_JSON
    assert M.parse_measure_json(payload) == M.parse_measure_json(GOLDEN_MEASURE_JSON)
    assert stderr.strip() == 'cc1: error: no such file or directory'


def test_a_stderr_tail_that_looks_like_measure_json_cannot_move_the_floor():
    """Reading the raw output would take the LAST json line -- the stderr one."""
    run_output = (
        f'{GOLDEN_MEASURE_JSON}\n{M.MEASURE_STDERR_MARKER}\n'
        '{"tests_total": 1, "graded": []}\n'
    )
    payload, _ = M.split_measure_output(run_output)
    assert M.parse_measure_json(payload)[0] == 3


def test_output_without_the_marker_is_all_measure_json():
    assert M.split_measure_output(MEASURE_OUT) == (MEASURE_OUT, '')


def test_the_measure_run_script_captures_stderr_but_still_cats_the_json():
    script = M.MEASURE_RUN_SCRIPT
    assert f'cat {M.MEASURE_JSON_PATH}' in script, 'the json must come from cat, as before'
    assert '2>&1' not in script, 'stderr must not be merged into the parsed output'
    assert f'2>{M.MEASURE_STDERR_PATH}' in script
    assert M.MEASURE_STDERR_MARKER in script
    assert 'EXPECTED' not in script, 'phase 1 stays floor-FREE'


# ------------------------------------------------------- scrubbed stderr -----


def test_scrub_stderr_keeps_only_the_last_cap_bytes():
    text = '\n'.join(f'line {i}' for i in range(500))
    out = M.scrub_stderr(text, cap_bytes=64)
    assert len(out.encode('utf-8')) <= 64
    assert 'line 499' in out
    assert 'line 0\n' not in out


def test_scrub_stderr_drops_lines_naming_the_intact_tree():
    """A compiler error quoting a carved path is a leak, not a diagnostic."""
    text = (
        'gcc: warning: something bland\n'
        '/opt/harbor/repo/src/secret.c:41: error: expected declaration\n'
        'ld: fatal: link step aborted\n'
    )
    out = M.scrub_stderr(text)
    assert '/opt/harbor/repo' not in out
    assert 'secret.c' not in out
    assert 'ld: fatal: link step aborted' in out


def test_scrub_stderr_drops_lines_naming_an_extra_relpath():
    text = 'error: in src/carved/target.py\nerror: unrelated failure\n'
    out = M.scrub_stderr(text, extra_relpaths=('src/carved/target.py',))
    assert 'src/carved/target.py' not in out
    assert 'error: unrelated failure' in out


def test_scrub_stderr_prefers_signal_lines_and_dedupes_them():
    text = (
        'building 1/9\n'
        'undefined reference to `mg_start\'\n'
        'building 2/9\n'
        'undefined reference to `mg_start\'\n'
        'building 3/9\n'
        'FATAL: collect2 returned 1\n'
    )
    out = M.scrub_stderr(text).splitlines()
    assert out == ['undefined reference to `mg_start\'', 'FATAL: collect2 returned 1']


def test_scrub_stderr_falls_back_to_the_capped_tail_when_nothing_signals():
    text = 'compiling a\ncompiling b\ncompiling c\n'
    assert M.scrub_stderr(text).splitlines() == ['compiling a', 'compiling b', 'compiling c']


def test_scrub_stderr_is_deterministic_and_introduces_no_host_path(tmp_path):
    text = f'error: build failed under {tmp_path}\nerror: build failed\n'
    first = M.scrub_stderr(text, extra_relpaths=(str(tmp_path),))
    assert first == M.scrub_stderr(text, extra_relpaths=(str(tmp_path),))
    assert str(tmp_path) not in first


def test_scrub_stderr_never_emits_a_half_line_from_a_truncating_cut():
    """The cut can land mid-path; half of `/opt/harbor/repo/...` evades redaction."""
    text = 'padding\n/opt/harbor/repo/src/secret.c:1: error: boom\n'
    out = M.scrub_stderr(text, cap_bytes=30)
    assert 'secret.c' not in out


# ---------------------------------------- the failure path carries a reason --


class FailingRunner(FakeRunner):
    def run(self, image, script, quiet=False, mounts=()):
        self.runs.append((image, script))
        return (
            f'{M.MEASURE_STDERR_MARKER}\n'
            'building...\n'
            '/opt/harbor/repo/src/secret.c:9: error: leaked path\n'
            'cc1: fatal error: stdio.h: No such file or directory\n'
        )


def test_a_failed_measure_surfaces_the_scrubbed_stderr(tmp_path):
    runner = FailingRunner()
    with pytest.raises(M.MeasureError) as caught:
        M.measure(_request(tmp_path), FakePlugin(), runner, echo=lambda *_: None)
    assert 'stdio.h: No such file or directory' in caught.value.stderr
    assert 'secret.c' not in caught.value.stderr
    assert 'stdio.h' in str(caught.value), 'the message must carry it too'
    assert runner.removed, 'the NEVER-SHIP image still goes'


def test_a_build_failure_surfaces_its_reason_even_with_no_captured_stderr(tmp_path):
    class BrokenBuild(FakeRunner):
        def build_with_contexts(self, image, dockerfile, context, contexts, target=None):
            raise RuntimeError('docker build failed (exit 1): cannot find base image')

    runner = BrokenBuild()
    with pytest.raises(M.MeasureError) as caught:
        M.measure(_request(tmp_path), FakePlugin(), runner, echo=lambda *_: None)
    assert 'cannot find base image' in caught.value.stderr
    assert runner.removed


def test_a_successful_measure_carries_no_stderr_and_the_same_lock(tmp_path):
    """SUCCESS is untouched: same lock bytes whether or not stderr was captured."""
    plain = FakeRunner([MEASURE_OUT])
    with_tail = FakeRunner([
        f'{MEASURE_OUT}\n{M.MEASURE_STDERR_MARKER}\nwarning: noisy but harmless\n'
    ])
    first = M.measure(_request(tmp_path), FakePlugin(), plain, echo=lambda *_: None)
    second = M.measure(_request(tmp_path), FakePlugin(), with_tail, echo=lambda *_: None)
    assert M.render_lock(first) == M.render_lock(second)
    assert first['expected'] == 5
