"""Full harbor entry emission: nine dirs, valid task.toml, byte-identical reruns."""

from __future__ import annotations

import filecmp
import hashlib
import os
import tomllib

import pytest

from taskgen.contexts import CONTEXT_TYPES
from taskgen.emit import emit_all

from taskgen import templates

from .conftest import (
    FROZEN_FILE,
    FROZEN_FUNC,
    FROZEN_NODEIDS,
    FROZEN_SHA256,
    FROZEN_STUB_SHA256,
)

FRAMING = 'These conditions vary context PROVISIONING'


@pytest.fixture(scope='module')
def emitted(repo, tmp_path_factory):
    """The frozen dry-run target, emitted once and inspected by most tests."""
    out = tmp_path_factory.mktemp('emit')
    entries = emit_all(
        repo=repo, out=out, package_base='src/', file=FROZEN_FILE, func=FROZEN_FUNC
    )
    return out, entries


def test_emits_nine_entries(emitted):
    out, entries = emitted
    assert len(entries) == 9
    assert {e.context_type for e in entries} == set(CONTEXT_TYPES)
    # `_staging/` sits alongside the entries and is deliberately not one: it is
    # the shared host-side carve, keyed by task.toml's absence.
    assert len([p for p in out.iterdir() if (p / 'task.toml').is_file()]) == 9


def test_entry_dirs_are_the_uuid5_ids(emitted):
    _out, entries = emitted
    for e in entries:
        assert e.path.name == e.entry_id


def test_every_entry_has_the_full_harbor_layout(emitted):
    _out, entries = emitted
    for e in entries:
        for rel in (
            'task.toml',
            'instruction.md',
            'environment/Dockerfile',
            'environment/Dockerfile.dockerignore',
            f'environment/carve/{e.target_relpath}',
            'tests/test.sh',
            'tests/allowlist.txt',
            'tests/harbor_filter.py',
            'solution/solve.sh',
            f'solution/carved/{e.target_relpath}',
        ):
            assert (e.path / rel).is_file(), f'{e.context_type}: missing {rel}'


def test_task_toml_parses_with_the_family_a_schema(emitted):
    _out, entries = emitted
    for e in entries:
        data = tomllib.loads((e.path / 'task.toml').read_text())
        assert data['schema_version'] == '1.4'
        for block in ('task', 'metadata', 'verifier', 'agent', 'environment'):
            assert block in data, f'{e.context_type}: missing [{block}]'
        assert data['metadata']['condition'] == e.context_type
        assert data['metadata']['id'] == e.entry_id
        assert data['metadata']['context_budget_tokens'] == 100000
        assert data['metadata']['tokenizer'] == 'chars4'
        assert data['metadata']['seed'] == 0
        assert data['task']['name'] == f'mrgctx/{e.slug}'
        assert data['verifier']['network_mode'] == 'allowlist'
        assert data['agent']['allowed_hosts'] == ['172.17.0.1']
        assert data['environment']['network_mode'] == 'public'


def test_instruction_md_carries_the_framing_and_the_condition(emitted):
    _out, entries = emitted
    for e in entries:
        text = (e.path / 'instruction.md').read_text()
        assert FRAMING in text, e.context_type
        assert f'**`{e.context_type}`**' in text
        assert FROZEN_FUNC in text
        assert '## Generation metadata' in text


def test_allowlist_is_the_five_linked_nodeids(emitted):
    _out, entries = emitted
    for e in entries:
        lines = (e.path / 'tests/allowlist.txt').read_text().splitlines()
        assert lines == FROZEN_NODEIDS


def test_instruction_md_lists_the_graded_nodeids(emitted):
    _out, entries = emitted
    for e in entries:
        text = (e.path / 'instruction.md').read_text()
        for nid in FROZEN_NODEIDS:
            assert nid in text


def test_test_sh_grades_exactly_five(emitted):
    _out, entries = emitted
    for e in entries:
        sh = (e.path / 'tests/test.sh').read_text()
        assert 'EXPECTED=5' in sh
        assert "-o addopts=''" in sh
        assert '-p no:xdist' in sh
        assert '-p no:cacheprovider' in sh
        assert '-p harbor_filter' in sh
        assert 'HARBOR_SELECTED_OUT="${SELECTED}"' in sh
        assert 'HARBOR_ALLOWLIST="${ALLOWLIST}"' in sh
        assert 'ALLOWLIST=${ALLOWLIST:-/opt/harbor/tests/allowlist.txt}' in sh


def test_test_sh_emits_the_common_fractional_reward_schema(emitted):
    """Wave 2 moved every language onto one reward.json shape; python included."""
    _out, entries = emitted
    sh = (entries[0].path / 'tests/test.sh').read_text()
    for key in ('reward', 'tests_passed', 'tests_total', 'binary', 'compiled'):
        assert f'"{key}"' in sh
    assert 'emit 0.0 0 "${EXPECTED}" 0.0 1.0' in sh, 'a zero must land first'
    assert 'the denominator moved' in sh, 'python keeps the equality floor'


def test_solution_carved_is_the_intact_original(emitted):
    _out, entries = emitted
    for e in entries:
        raw = (e.path / f'solution/carved/{e.target_relpath}').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == FROZEN_SHA256


def test_environment_carve_is_the_stub(emitted):
    _out, entries = emitted
    for e in entries:
        text = (e.path / f'environment/carve/{e.target_relpath}').read_text()
        assert 'raise NotImplementedError' in text
        assert hashlib.sha256(text.encode()).hexdigest() != FROZEN_SHA256


def test_solve_sh_restores_from_the_runtime_mount_and_counts(emitted):
    """The oracle reaches the container by mount, and proves it restored the lot.

    It no longer pins the intact sha256. That digest IDENTIFIES THE ANSWER, and
    solve.sh is the one entry asset that is readable from inside a task
    container the moment the gate mounts it; the restore is a copy of the intact
    original, so byte equality is a property of the copy rather than something
    the script has to re-derive. Tamper detection moved to `verify.py`, which
    compares solution/carved against the pristine repo from the HOST.
    """
    _out, entries = emitted
    for e in entries:
        solve = (e.path / 'solution/solve.sh').read_text()
        assert FROZEN_SHA256 not in solve
        assert f"restore '{e.target_relpath}'" in solve
        assert 'WANT=1' in solve
        assert 'HARBOR_SOLUTION:-/opt/harbor/solution' in solve
        assert 'restored ${RESTORED} file(s), promised ${WANT}' in solve


def test_harbor_filter_is_the_reference_verbatim(emitted):
    from taskgen.templates import HARBOR_FILTER_PY

    _out, entries = emitted
    for e in entries:
        assert (e.path / 'tests/harbor_filter.py').read_text() == HARBOR_FILTER_PY


def test_only_the_context_block_and_condition_differ(emitted):
    _out, entries = emitted
    docs = {e.context_type: (e.path / 'instruction.md').read_text() for e in entries}
    heads = {ct: d.split('## Context condition')[0] for ct, d in docs.items()}
    assert len(set(heads.values())) == 1


def test_two_runs_are_byte_identical(repo, emitted, tmp_path):
    first, _entries = emitted
    second = tmp_path / 'rerun'
    emit_all(
        repo=repo, out=second, package_base='src/', file=FROZEN_FILE, func=FROZEN_FUNC
    )
    assert _collect_diffs(filecmp.dircmp(first, second)) == []


def test_default_selection_generalises_to_an_unnamed_target(repo, tmp_path):
    """No --file/--func: the first eligible function must still emit 9 valid entries.

    This is the guard against a frozen target leaking into library code -- it
    runs the whole pipeline on whatever function the deterministic order picks.
    """
    entries = emit_all(repo=repo, out=tmp_path / 'auto', package_base='src/')
    assert len(entries) == 9
    assert {e.context_type for e in entries} == set(CONTEXT_TYPES)
    picked = {(e.target_relpath, e.nodeids) for e in entries}
    assert len(picked) == 1, 'the nine entries must share one carve'
    for e in entries:
        data = tomllib.loads((e.path / 'task.toml').read_text())
        assert data['schema_version'] == '1.4'
        assert e.nodeids, 'a task with no graded tests must never be emitted'
        assert (e.path / 'instruction.md').read_text().count(FRAMING) == 1


def _collect_diffs(dc, prefix=''):
    out = [f'{prefix}{n}' for n in dc.left_only + dc.right_only + dc.funny_files]
    out += [f'{prefix}{n}' for n in dc.diff_files]
    for name, sub in dc.subdirs.items():
        out += _collect_diffs(sub, f'{prefix}{name}/')
    return out


def test_solve_sh_actually_restores_and_imports(emitted, repo, tmp_path):
    """Execute the oracle for real against a throwaway copy of the repo tree.

    The oracle is the correctness bar for the whole generator, so it is run,
    not just grepped: stage the STUBBED file over a real `src/` tree, run
    solve.sh, and require exit 0, the original digest back, and a live import.
    """
    import shutil
    import subprocess

    _out, entries = emitted
    e = entries[0]

    fake_repo = tmp_path / 'repo'
    shutil.copytree(repo / 'src', fake_repo / 'src')
    stub = (e.path / f'environment/carve/{e.target_relpath}').read_text()
    (fake_repo / e.target_relpath).write_text(stub)
    assert 'raise NotImplementedError' in (fake_repo / e.target_relpath).read_text()

    proc = subprocess.run(
        ['bash', str(e.path / 'solution/solve.sh')],
        env={
            'PATH': os.environ['PATH'],
            'HARBOR_REPO': str(fake_repo),
            'HARBOR_SOLUTION': str(e.path / 'solution'),
        },
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f'stdout={proc.stdout}\nstderr={proc.stderr}'
    assert 'SOLVE OK' in proc.stdout
    restored = (fake_repo / e.target_relpath).read_bytes()
    assert hashlib.sha256(restored).hexdigest() == FROZEN_SHA256


def test_solve_sh_refuses_a_missing_oracle(emitted, repo, tmp_path):
    """No oracle mounted must fail loudly, not silently "restore" nothing.

    This is the fail-closed leg that matters now the oracle is never baked in:
    if the gate forgets the bind mount, solve.sh must not exit 0 having done
    nothing, because the graded run would then read as an honest RED.
    """
    import shutil
    import subprocess

    _out, entries = emitted
    e = entries[0]

    fake_repo = tmp_path / 'repo'
    shutil.copytree(repo / 'src', fake_repo / 'src')
    solution = tmp_path / 'solution'
    shutil.copytree(e.path / 'solution', solution)
    shutil.rmtree(solution / 'carved')

    proc = subprocess.run(
        ['bash', str(solution / 'solve.sh')],
        env={
            'PATH': os.environ['PATH'],
            'HARBOR_REPO': str(fake_repo),
            'HARBOR_SOLUTION': str(solution),
        },
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert 'mounted at run time, never baked' in proc.stdout + proc.stderr


def test_dockerfile_builds_from_the_host_staged_carve(emitted):
    """The two-stage intact->carved build is GONE, and that is the point.

    It COPYed the pristine tree into a layer and then overwrote one file in a
    later one; `docker save` reads every layer, so the answer was recoverable
    (plan A1). The carve now happens on the host and only the carved tree ever
    enters a build context.
    """
    _out, entries = emitted
    for e in entries:
        df = (e.path / 'environment/Dockerfile').read_text()
        assert 'FROM harbor-base:local AS graded' in df
        assert 'COPY --from=repoctx repo/ /opt/harbor/repo' in df
        assert 'FROM intact' not in df
        assert 'repo-src' not in df and 'repos-src' not in df
        assert 'uv sync --locked --all-extras' in df
        assert 'NOT EDITABLE' in df
        assert 'leakscan.sh' in df, 'the content leak gate must run at build time'


def test_dockerfile_declares_its_named_build_contexts(emitted):
    """The entry has to say how to build it, without baking in a host path."""
    import json

    _out, entries = emitted
    for e in entries:
        ctx = json.loads((e.path / 'environment/contexts.json').read_text())
        assert set(ctx['build_contexts']) == {'repoctx', 'trip', 'tooling', 'entry'}
        for value in ctx['build_contexts'].values():
            assert not value.startswith('/'), value
        staged = (e.path / 'environment' / ctx['build_contexts']['repoctx']).resolve()
        assert (staged / 'repo' / e.target_relpath).is_file()


def test_dockerfile_bakes_in_no_host_path(emitted):
    _out, entries = emitted
    for e in entries:
        df = (e.path / 'environment/Dockerfile').read_text()
        assert '/Users/' not in df
        assert 'harbor-tasks/repos-src' not in df


def test_dockerfile_never_ships_the_oracle_solution(emitted):
    """Harbor gives the agent a shell in this image. The answer cannot be in it.

    solution/ stays on disk in the entry -- the framework's solvability gate
    needs it -- but the image must never see it, at any stage. `intact` is a
    parent layer of `carved`, so a COPY there ships just as surely as one here.
    """
    _out, entries = emitted
    for e in entries:
        df = (e.path / 'environment/Dockerfile').read_text()
        assert 'COPY --from=entry solution' not in df, \
            f'{e.context_type}: the oracle is COPYed into the agent image'
        # The ONE permitted mention is the build-time assertion that the path is
        # absent -- which is a stronger guarantee than never naming it.
        assert df.count('/opt/harbor/solution') == 1, e.context_type
        assert 'test ! -e /opt/harbor/solution' in df, e.context_type
        assert 'COPY solution' not in df, \
            f'{e.context_type}: the image build still copies the oracle tree'


def test_dockerfile_names_neither_the_answer_nor_the_carve_metadata(emitted):
    """No digest of the intact file, no receipt, no carve tooling in the image.

    The in-image carve assertion is gone with the carve itself: there is nothing
    left to assert, because the tree arrives already carved. What replaces it is
    stronger -- the staged tree is checked on the HOST, and the image asserts the
    carve METADATA is absent (plan invariant 6), since the receipt names every
    removed path and the manifest names the globs.
    """
    _out, entries = emitted
    for e in entries:
        df = (e.path / 'environment/Dockerfile').read_text()
        assert FROZEN_SHA256 not in df, \
            'the intact digest identifies the answer and has no business in the image'
        assert 'find / -name carve_receipt.json' in df
        assert '| wc -l)" = "0"' in df
        assert 'test ! -e /opt/harbor-tooling/carve.py' in df


def test_no_python_template_shadows_the_plugin():
    """templates.py must not keep a second, divergent copy of the entry assets."""
    for gone in ('DOCKERFILE', 'TEST_SH', 'SOLVE_SH'):
        assert not hasattr(templates, gone), (
            f'templates.{gone} still exists; langs/python.py owns it now and two '
            'definitions of one artifact will drift'
        )
