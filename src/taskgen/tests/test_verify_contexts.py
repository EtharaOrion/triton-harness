"""The verifier must build the entry the way the entry says it is built.

Two defects motivate this file, both of which let a green-looking verifier
prove nothing about the artefact that actually ships.

BUILD CONTEXTS (FIX-1). `emit` moved to a host-side carve: the image is
assembled from four BuildKit named contexts -- `repoctx` (the staged CARVED
tree), `trip` (tripwire lines for the leak gate), `tooling` (harbor's
leakscan.sh) and `entry` -- and the stage is named in the entry too. The
verifier used to hard-code the previous shape (`repo-src=<repo>`, `entry=`,
`--target carved`). That is not "slightly stale": `repo-src` would hand the
build the INTACT tree, which is the very leak the host-side carve exists to
close, and a build that fails outright is the good outcome. So the build
command is assembled from `environment/contexts.json` and from nowhere else.

ORACLE INTEGRITY (FIX-2). solve.sh no longer pins the intact sha256, because
solve.sh is readable from inside the container and that digest IS the answer.
Removing it left the oracle gate with nothing checking that
`solution/carved/<relpath>` really is the upstream file: a tampered oracle
that restores a hand-written "solution" would still score reward 1.0 and the
GREEN leg would applaud. The check therefore moves to the HOST, where it can
compare against `repos-src` without the container ever seeing the digest.
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path

import pytest

from taskgen import verify as V

INTACT = 'def apply_history_length(task, config):\n    return task\n'


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _write_repo(root: Path, *, body: str = INTACT) -> Path:
    repo = root / 'repos-src' / 'python-a2a-python'
    (repo / 'src' / 'a2a' / 'utils').mkdir(parents=True)
    (repo / 'src' / 'a2a' / 'utils' / 'task.py').write_text(body)
    return repo


def _write_entry(root: Path, entry_id: str = 'aaaaaaaa-1111-5111-8111-111111111111',
                 *, contexts: dict | None = None, oracle: str = INTACT) -> Path:
    """An entry in the post-W3 shape: staged contexts beside the entry dirs."""
    d = root / 'out' / entry_id
    (d / 'tests').mkdir(parents=True)
    (d / 'environment').mkdir(parents=True)
    (d / 'task.toml').write_text(textwrap.dedent(f"""
        schema_version = "1.4"
        [task]
        name = "mrgctx/python-a2a-python__apply_history_length__no_context"
        [metadata]
        id = "{entry_id}"
        name = "python-a2a-python__apply_history_length__no_context"
        condition = "no_context"
        language = "python"
        carve_scope = "function"
        target_file = "src/a2a/utils/task.py"
        target_func = "apply_history_length"
        graded_tests = ["t.py::a", "t.py::b"]
        [environment]
        docker_image = "taskgen-python-a2a-python:local"
        """).strip() + '\n')
    (d / 'tests' / 'allowlist.txt').write_text('t.py::a\nt.py::b\n')
    (d / 'tests' / 'test.sh').write_text('#!/bin/sh\nEXPECTED=2\n')
    (d / 'environment' / 'Dockerfile').write_text('FROM harbor-base:local AS graded\n')

    oracle_file = d / 'solution' / 'carved' / 'src' / 'a2a' / 'utils' / 'task.py'
    oracle_file.parent.mkdir(parents=True)
    oracle_file.write_text(oracle)
    (d / 'solution' / 'solve.sh').write_text('#!/bin/sh\necho SOLVE OK\n')

    staging = root / 'out' / '_staging' / 'python-a2a-python__function__deadbeef'
    for leaf in ('ctx', 'trip', 'tooling'):
        (staging / leaf).mkdir(parents=True, exist_ok=True)
    (staging / 'trip' / 'tripwires.txt').write_text('a distinctive carved line\n')
    (staging / 'tooling' / 'leakscan.sh').write_text('#!/bin/sh\nexit 0\n')

    payload = contexts if contexts is not None else {
        'image': 'taskgen-python-a2a-python:local',
        'stage': 'graded',
        'build_contexts': {
            'repoctx': '../../_staging/python-a2a-python__function__deadbeef/ctx',
            'trip': '../../_staging/python-a2a-python__function__deadbeef/trip',
            'tooling': '../../_staging/python-a2a-python__function__deadbeef/tooling',
            'entry': '..',
        },
    }
    (d / 'environment' / 'contexts.json').write_text(json.dumps(payload, indent=2) + '\n')
    return d


# ------------------------------------------------------------------ FIX-1 ---


def test_build_plan_comes_from_the_entrys_own_contexts_json(tmp_path):
    entry = _write_entry(tmp_path)
    plan = V.load_build_plan(entry)

    assert plan.image == 'taskgen-python-a2a-python:local'
    assert plan.stage == 'graded'
    assert plan.dockerfile == entry / 'environment' / 'Dockerfile'
    names = dict(plan.contexts)
    assert set(names) == {'repoctx', 'trip', 'tooling', 'entry'}
    # Resolved against environment/, which is what the emitted relpaths mean.
    assert names['entry'] == entry
    assert names['repoctx'].name == 'ctx'
    assert (names['repoctx'] / '..' / 'trip').resolve() == names['trip']


def test_build_argv_is_assembled_from_contexts_json(tmp_path):
    entry = _write_entry(tmp_path)
    argv = V.build_argv('docker', V.load_build_plan(entry))

    pairs = [argv[i + 1] for i, tok in enumerate(argv) if tok == '--build-context']
    assert sorted(p.split('=', 1)[0] for p in pairs) == [
        'entry', 'repoctx', 'tooling', 'trip',
    ]
    for pair in pairs:
        _, path = pair.split('=', 1)
        assert Path(path).is_absolute(), f'{pair} is not an absolute host path'
        assert Path(path).is_dir(), f'{pair} does not exist'

    # The pre-W3 shape handed the build the INTACT tree. It must be gone.
    assert not any(p.startswith('repo-src=') for p in pairs)
    assert argv[argv.index('--target') + 1] == 'graded'
    assert argv[argv.index('-t') + 1] == 'taskgen-python-a2a-python:local'
    assert argv[argv.index('-f') + 1] == str(entry / 'environment' / 'Dockerfile')


def test_a_missing_contexts_json_is_an_error_not_a_guessed_default(tmp_path):
    entry = _write_entry(tmp_path)
    (entry / 'environment' / 'contexts.json').unlink()
    with pytest.raises(V.VerifyError, match='contexts.json'):
        V.load_build_plan(entry)


def test_a_context_directory_that_does_not_exist_fails_closed(tmp_path):
    entry = _write_entry(tmp_path)
    doc = json.loads((entry / 'environment' / 'contexts.json').read_text())
    doc['build_contexts']['repoctx'] = '../../_staging/gone/ctx'
    (entry / 'environment' / 'contexts.json').write_text(json.dumps(doc))
    with pytest.raises(V.VerifyError, match='repoctx'):
        V.load_build_plan(entry)


def test_contexts_json_without_a_repoctx_is_rejected(tmp_path):
    """No repoctx means no carved tree, so whatever built is not the task."""
    entry = _write_entry(tmp_path, contexts={
        'image': 'taskgen-x:local', 'stage': 'graded',
        'build_contexts': {'entry': '..'},
    })
    with pytest.raises(V.VerifyError, match='repoctx'):
        V.load_build_plan(entry)


def test_the_layer_audit_finds_the_staged_tripwires_through_contexts_json(tmp_path):
    """The tripwire file lives in the `trip` context now, not in the entry."""
    entry = _write_entry(tmp_path)
    patterns, digests = V.oracle_signals(entry)
    assert 'a distinctive carved line' in patterns
    assert _sha(INTACT) in digests


# ------------------------------------------------------------------ FIX-2 ---


def test_oracle_integrity_passes_when_the_oracle_is_the_upstream_file(tmp_path):
    entry = _write_entry(tmp_path)
    repo = _write_repo(tmp_path)
    assert V.check_oracle_integrity(entry, repo).ok


def test_a_tampered_oracle_fails_integrity(tmp_path):
    """The oracle payload must be the ANSWER, not a hand-written stand-in."""
    entry = _write_entry(tmp_path, oracle=INTACT + '# forged\n')
    repo = _write_repo(tmp_path)
    result = V.check_oracle_integrity(entry, repo)
    assert not result.ok
    assert any('src/a2a/utils/task.py' in r for r in result.reasons)
    assert any(_sha(INTACT)[:12] in r for r in result.reasons)


def test_an_oracle_for_a_file_the_repo_does_not_have_fails_integrity(tmp_path):
    entry = _write_entry(tmp_path)
    repo = _write_repo(tmp_path)
    (repo / 'src' / 'a2a' / 'utils' / 'task.py').unlink()
    result = V.check_oracle_integrity(entry, repo)
    assert not result.ok
    assert any('missing' in r for r in result.reasons)


def test_an_empty_oracle_fails_integrity_rather_than_passing_vacuously(tmp_path):
    entry = _write_entry(tmp_path)
    for path in (entry / 'solution' / 'carved').rglob('*'):
        if path.is_file():
            path.unlink()
    repo = _write_repo(tmp_path)
    result = V.check_oracle_integrity(entry, repo)
    assert not result.ok
    assert any('no oracle' in r or 'empty' in r for r in result.reasons)


def test_integrity_is_computed_on_the_host_against_repos_src(tmp_path):
    """Nothing may read the digest from inside the image: it is the answer."""
    entry = _write_entry(tmp_path)
    repo = _write_repo(tmp_path)

    class ExplodingRunner:
        def run(self, *a, **k):
            raise AssertionError('the integrity check must not touch a container')

    V.check_oracle_integrity(entry, repo, runner=ExplodingRunner())


def test_a_result_without_the_integrity_check_never_passes(tmp_path):
    """An unproven oracle is not a proven one -- same rule as the layer audit."""
    red = V.StateResult('RED', V.parse_reward_json(
        '{"reward": 0.0, "tests_passed": 0, "tests_total": 2, "binary": 0.0, "compiled": 1.0}'))
    green = V.StateResult('GREEN', V.parse_reward_json(
        '{"reward": 1.0, "tests_passed": 2, "tests_total": 2, "binary": 1.0, "compiled": 1.0}'))
    result = V.VerifyResult(
        image='taskgen-x:local', expected=2, red=red, green=green,
        image_clean=V.CheckResult('IMAGE'), layer_clean=V.CheckResult('LAYER'),
    )
    assert not result.passed, 'integrity unproven must not read as proven'
    result.integrity = V.CheckResult('INTEGRITY')
    assert result.passed
    result.integrity = V.CheckResult('INTEGRITY', ['oracle tampered'])
    assert not result.passed


# ------------------------------------------- language-agnostic entry loading --


def _write_go_entry(root: Path, entry_id: str, condition: str = 'no_context') -> Path:
    """A go entry: `go test -run` selectors, so there is no pytest allowlist."""
    d = root / 'out' / entry_id
    (d / 'tests').mkdir(parents=True)
    (d / 'environment').mkdir(parents=True)
    (d / 'task.toml').write_text(textwrap.dedent(f"""
        schema_version = "1.4"
        [task]
        name = "mrgctx/go-multigres__assignConnectionID__{condition}"
        [metadata]
        id = "{entry_id}"
        name = "go-multigres__assignConnectionID__{condition}"
        language = "go"
        condition = "{condition}"
        carve_scope = "function"
        target_file = "go/common/pgprotocol/server/listener.go"
        target_func = "(*Listener).assignConnectionID"
        graded_tests = ["TestA", "TestB", "TestC", "TestD"]
        [environment]
        docker_image = "taskgen-go-multigres:local"
        """).strip() + '\n')
    (d / 'tests' / 'graded.json').write_text(json.dumps({
        'kind': 'go-run', 'expected': 4,
        'selectors': ['TestA', 'TestB', 'TestC', 'TestD'],
        'packages': ['github.com/multigres/multigres/go/common/pgprotocol/server'],
        'fingerprint_relpaths': ['go/common/pgprotocol/server/listener_test.go'],
    }, indent=2, sort_keys=True) + '\n')
    (d / 'tests' / 'test.sh').write_text(f'#!/bin/sh\n# verifier -- {condition}\nEXPECTED=4\n')
    (d / 'environment' / 'Dockerfile').write_text(f'# env -- {condition}\nFROM harbor-base:local\n')
    carved = d / 'solution' / 'carved' / 'go' / 'common' / 'pgprotocol' / 'server'
    carved.mkdir(parents=True)
    (carved / 'listener.go').write_text('package server\n')
    (d / 'solution' / 'solve.sh').write_text('#!/bin/sh\necho SOLVE OK\n')
    return d


def test_a_go_entry_loads_without_a_pytest_allowlist(tmp_path):
    """`tests/allowlist.txt` is the pytest selector format, not the contract.

    Requiring it made every non-python entry unloadable, so the go legs of the
    matrix could not even be attempted.
    """
    d = _write_go_entry(tmp_path, 'cccccccc-1111-5111-8111-111111111111')
    spec = V.load_entry(d)
    assert spec.lang == 'go'
    assert spec.expected == 4
    assert spec.repo_dirname == 'go-multigres'


def test_the_graded_count_comes_from_the_pinned_expected(tmp_path):
    """`expected` is the PINNED denominator; go's selectors may under-report it."""
    d = _write_go_entry(tmp_path, 'cccccccc-1111-5111-8111-111111111111')
    doc = json.loads((d / 'tests' / 'graded.json').read_text())
    doc['expected'] = 7
    (d / 'tests' / 'graded.json').write_text(json.dumps(doc))
    assert V.load_entry(d).expected == 7


def test_graded_json_disagreeing_with_task_toml_is_refused(tmp_path):
    d = _write_go_entry(tmp_path, 'cccccccc-1111-5111-8111-111111111111')
    doc = json.loads((d / 'tests' / 'graded.json').read_text())
    doc['selectors'] = ['TestA']
    (d / 'tests' / 'graded.json').write_text(json.dumps(doc))
    with pytest.raises(V.VerifyError, match='graded'):
        V.load_entry(d)


def test_a_missing_graded_json_is_refused(tmp_path):
    d = _write_go_entry(tmp_path, 'cccccccc-1111-5111-8111-111111111111')
    (d / 'tests' / 'graded.json').unlink()
    with pytest.raises(V.VerifyError, match='graded.json'):
        V.load_entry(d)


def test_go_entries_are_compared_without_python_only_assets(tmp_path):
    """compare_entries used to read tests/allowlist.txt unconditionally."""
    a = _write_go_entry(tmp_path, 'aaaaaaaa-1111-5111-8111-111111111111', 'no_context')
    b = _write_go_entry(tmp_path, 'bbbbbbbb-1111-5111-8111-111111111111', 'bm25')
    shared = V.compare_entries([a, b])
    assert shared.equivalent
    assert 'tests/graded.json' in shared.identical
    assert 'solution/carved/go/common/pgprotocol/server/listener.go' in shared.identical


def test_a_drifting_graded_set_still_breaks_the_shared_image_claim(tmp_path):
    a = _write_go_entry(tmp_path, 'aaaaaaaa-1111-5111-8111-111111111111', 'no_context')
    b = _write_go_entry(tmp_path, 'bbbbbbbb-1111-5111-8111-111111111111', 'bm25')
    doc = json.loads((b / 'tests' / 'graded.json').read_text())
    doc['expected'] = 3
    (b / 'tests' / 'graded.json').write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    shared = V.compare_entries([a, b])
    assert not shared.equivalent
    assert 'tests/graded.json' in shared.mismatched
