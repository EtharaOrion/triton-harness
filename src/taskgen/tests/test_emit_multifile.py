"""emit across --lang x --carve-scope: multi-file entries, and the old path intact.

Three claims are under test, and they pull against each other:

  MULTI-FILE   `file`/`folder` scope carve N files, stage them all, union their
               graded sets and ship an oracle that restores every one of them
  BACKWARD     the default invocation (`python`, `function`) still selects the
  COMPAT       same target, mints the SAME uuid5 entry ids, and grades the
               same five node ids -- the ids are pinned as literals below, so a
               change to the id key is a failing test rather than a silent
               re-shuffle of every downstream artifact
  DETERMINISM  two runs are byte-identical, including the staged carve tree
"""

from __future__ import annotations

import filecmp
import hashlib
import tomllib

import pytest

from taskgen.contexts import CONTEXT_TYPES
from taskgen.emit import emit_all
from taskgen.scope import CarveScopeError

from .conftest import FROZEN_FILE, FROZEN_FUNC, FROZEN_NODEIDS, FROZEN_SHA256

#: The ids the proven single-function entry mints under the `triton.taskgen`
#: uuid5 namespace. They are the backward-compatibility contract:
#: `--carve-scope`/`--lang` were added to the uuid5 key, and the key must
#: collapse to its old form at the defaults. The nine originals are UNCHANGED by
#: the caller_* conditions, which is the point: the key is per-context_type, so
#: adding conditions cannot renumber the rest. Only re-seeding the namespace may
#: move this whole table at once, since a uuid5 is a pure function of seed+key.
FROZEN_ENTRY_IDS = {
    'no_context': '69e6a008-ea77-5237-8079-f2d8d8693c5a',
    'callee_func': '9fdea6ea-4a99-521a-8d48-58543fc316a7',
    'callee_sig': 'd07a6bcc-a761-5481-a630-1c0986cad095',
    'caller_func': 'b706af8b-0fde-525b-9124-fe782efa3001',
    'caller_sig': 'dd68fe30-53e0-5630-abdf-d96eef44060a',
    'in_file': 'c0605da8-3ce0-5c6c-91ed-f37ac9ebf150',
    'project': 'dbd1f5fe-c0c2-59f5-a1cd-973e0494adc5',
    'bm25': '9a41fac2-ac19-5e81-880c-f0cb45963d29',
    'embedding': 'ad3307cf-93c6-58ee-8ca3-b80ca830482e',
    'mix': 'ded73010-93b8-567f-b96e-07b0046e6dbb',
    'repo_coder': '87fc4553-36ac-5030-87a7-7393535ed67c',
}

UTILS_GLOB = 'src/a2a/utils/**'
#: 12 files match the glob; two (`__init__.py`, `constants.py`) hold no function
#: body at all, so a skeleton carve of them would be a no-op.
UTILS_CARVED = 10
UTILS_GRADED = 14

FAST = ('no_context', 'in_file', 'bm25')


# --------------------------------------------------------------------------
# backward compatibility of the default (python, function) path
# --------------------------------------------------------------------------


@pytest.fixture(scope='module')
def fn_scope(repo, tmp_path_factory):
    out = tmp_path_factory.mktemp('fn')
    return out, emit_all(
        repo=repo, out=out, package_base='src/', file=FROZEN_FILE, func=FROZEN_FUNC,
    )


def test_defaults_are_python_function_scope(fn_scope):
    _out, entries = fn_scope
    assert {e.lang for e in entries} == {'python'}
    assert {e.carve_scope for e in entries} == {'function'}


def test_the_original_entry_ids_are_unchanged(fn_scope):
    _out, entries = fn_scope
    assert {e.context_type: e.entry_id for e in entries} == FROZEN_ENTRY_IDS


def test_the_graded_set_is_still_the_five_linked_nodeids(fn_scope):
    _out, entries = fn_scope
    for e in entries:
        assert list(e.nodeids) == FROZEN_NODEIDS
        assert (e.path / 'tests/allowlist.txt').read_text().splitlines() == FROZEN_NODEIDS


def test_function_scope_carves_exactly_one_file(fn_scope):
    _out, entries = fn_scope
    for e in entries:
        assert e.carved_relpaths == (FROZEN_FILE,)
        raw = (e.path / f'solution/carved/{FROZEN_FILE}').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == FROZEN_SHA256


def test_function_scope_stubs_only_the_target_body(fn_scope):
    """The rest of the file survives -- that is what makes it FUNCTION scope."""
    _out, entries = fn_scope
    stub = (entries[0].path / f'environment/carve/{FROZEN_FILE}').read_text()
    assert stub.count('raise NotImplementedError') == 1
    assert 'def validate_history_length' in stub
    assert 'def encode_page_token' in stub


# --------------------------------------------------------------------------
# file scope
# --------------------------------------------------------------------------


@pytest.fixture(scope='module')
def file_scope(repo, tmp_path_factory):
    out = tmp_path_factory.mktemp('file')
    return out, emit_all(
        repo=repo, out=out, package_base='src/',
        carve_scope='file', include=[FROZEN_FILE],
    )


def test_file_scope_emits_eleven_entries(file_scope):
    _out, entries = file_scope
    assert len(entries) == 11
    assert {e.context_type for e in entries} == set(CONTEXT_TYPES)


def test_file_scope_ids_differ_from_function_scope(file_scope):
    """Same repo, same file, different carve: the ids must not collide."""
    _out, entries = file_scope
    minted = {e.entry_id for e in entries}
    assert minted.isdisjoint(set(FROZEN_ENTRY_IDS.values()))


def test_file_scope_skeletonises_every_function_in_the_file(file_scope):
    _out, entries = file_scope
    stub = (entries[0].path / f'environment/carve/{FROZEN_FILE}').read_text()
    assert stub.count('raise NotImplementedError') == 6
    assert 'def validate_history_length' in stub, 'signatures must survive'
    assert 'from a2a.types import' in stub or 'import' in stub


def test_file_scope_graded_set_is_the_union_over_the_file(file_scope):
    _out, entries = file_scope
    for e in entries:
        assert list(e.nodeids) == FROZEN_NODEIDS


def test_file_scope_oracle_restores_the_intact_original(file_scope):
    _out, entries = file_scope
    for e in entries:
        raw = (e.path / f'solution/carved/{FROZEN_FILE}').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == FROZEN_SHA256


# --------------------------------------------------------------------------
# folder scope
# --------------------------------------------------------------------------


@pytest.fixture(scope='module')
def folder_scope(repo, tmp_path_factory):
    out = tmp_path_factory.mktemp('folder')
    return out, emit_all(
        repo=repo, out=out, package_base='src/',
        carve_scope='folder', include=[UTILS_GLOB], context_types=FAST,
    )


def test_folder_scope_carves_ten_files(folder_scope):
    _out, entries = folder_scope
    for e in entries:
        assert len(e.carved_relpaths) == UTILS_CARVED
        assert all(r.startswith('src/a2a/utils/') for r in e.carved_relpaths)
        assert 'src/a2a/utils/__init__.py' not in e.carved_relpaths


def test_folder_scope_ships_an_intact_original_per_carved_file(folder_scope):
    _out, entries = folder_scope
    for e in entries:
        for rel in e.carved_relpaths:
            assert (e.path / f'solution/carved/{rel}').is_file(), rel
            assert (e.path / f'environment/carve/{rel}').is_file(), rel


def test_folder_scope_oracle_restores_every_carved_file(folder_scope):
    _out, entries = folder_scope
    solve = (entries[0].path / 'solution/solve.sh').read_text()
    assert f'WANT={UTILS_CARVED}' in solve
    for rel in entries[0].carved_relpaths:
        assert f"restore '{rel}'" in solve


def test_folder_scope_graded_set_is_the_union_across_files(folder_scope):
    _out, entries = folder_scope
    ids = list(entries[0].nodeids)
    assert len(ids) == UTILS_GRADED
    assert set(FROZEN_NODEIDS) < set(ids)
    assert len({i.split('::')[0] for i in ids}) == 5
    assert all(not i.startswith('src/a2a/utils/') for i in ids), \
        'a test living in a carved file must not grade itself'


def test_folder_scope_test_sh_pins_the_union_denominator(folder_scope):
    _out, entries = folder_scope
    sh = (entries[0].path / 'tests/test.sh').read_text()
    assert f'EXPECTED={UTILS_GRADED}' in sh


def test_folder_scope_instruction_lists_the_carved_files(folder_scope):
    _out, entries = folder_scope
    for e in entries:
        doc = (e.path / 'instruction.md').read_text()
        assert 'carve_scope = folder' in doc
        assert f'carved_files = {UTILS_CARVED}' in doc
        for rel in e.carved_relpaths:
            assert rel in doc, rel


# --------------------------------------------------------------------------
# staging, task.toml, determinism
# --------------------------------------------------------------------------


def test_the_staged_repoctx_holds_the_carved_tree_not_the_intact_one(folder_scope):
    out, entries = folder_scope
    staged = out / '_staging'
    assert staged.is_dir(), 'the host-side carve must be staged for --build-context'
    ctx = next(staged.glob('*/ctx/repo'))
    for rel in entries[0].carved_relpaths:
        text = (ctx / rel).read_text()
        assert 'raise NotImplementedError' in text, rel
        assert text != (entries[0].path / f'solution/carved/{rel}').read_text(), rel


def test_the_receipt_never_enters_the_shipped_context(folder_scope):
    out, _entries = folder_scope
    staged = next((out / '_staging').iterdir())
    assert (staged / 'carve_receipt.json').is_file(), 'the host keeps the receipt'
    assert list((staged / 'ctx').rglob('carve_receipt.json')) == []
    assert list((staged / 'trip').rglob('carve_receipt.json')) == []


def test_the_tripwire_file_is_staged_for_the_leak_gate(folder_scope):
    out, entries = folder_scope
    staged = next((out / '_staging').iterdir())
    trip = (staged / 'trip' / 'tripwires.txt').read_text()
    assert trip.strip(), 'an empty tripwire file downgrades the leak gate'
    # Leak-gate coverage spans both rungs: a strong grep line here, or a content
    # digest in tripwire-digests.txt when the only line is sub-strong (staging.py).
    lines = [l for l in trip.strip().splitlines() if l.strip()]
    digests = (staged / 'trip' / 'tripwire-digests.txt').read_text()
    digest_lines = [l for l in digests.splitlines() if l.strip()]
    assert len(lines) + len(digest_lines) == len(entries[0].carved_relpaths)


def test_task_toml_records_the_scope_and_the_carve_set(folder_scope):
    _out, entries = folder_scope
    for e in entries:
        data = tomllib.loads((e.path / 'task.toml').read_text())
        assert data['schema_version'] == '1.4'
        assert data['metadata']['language'] == 'python'
        assert data['metadata']['carve_scope'] == 'folder'
        assert data['metadata']['carved_files'] == list(e.carved_relpaths)


def test_dockerfile_ships_the_staged_tree_and_no_intact_copy(folder_scope):
    _out, entries = folder_scope
    for e in entries:
        df = (e.path / 'environment/Dockerfile').read_text()
        assert 'COPY --from=repoctx repo/' in df
        assert 'repo-src' not in df, 'the intact tree must never enter a layer'
        assert 'repos-src' not in df
        instructions = [l for l in df.splitlines() if not l.lstrip().startswith('#')]
        assert not any('/opt/harbor/solution' in l and 'test ! -e' not in l
                       for l in instructions)


def test_two_runs_are_byte_identical(repo, folder_scope, tmp_path):
    first, _entries = folder_scope
    second = tmp_path / 'rerun'
    emit_all(
        repo=repo, out=second, package_base='src/',
        carve_scope='folder', include=[UTILS_GLOB], context_types=FAST,
    )
    assert _diffs(filecmp.dircmp(first, second)) == []


def _diffs(dc, prefix=''):
    out = [f'{prefix}{n}' for n in dc.left_only + dc.right_only + dc.funny_files]
    out += [f'{prefix}{n}' for n in dc.diff_files]
    for name, sub in dc.subdirs.items():
        out += _diffs(sub, f'{prefix}{name}/')
    return out


# --------------------------------------------------------------------------
# fail-closed
# --------------------------------------------------------------------------


def test_file_scope_without_include_is_refused(repo, tmp_path):
    with pytest.raises(CarveScopeError, match='include'):
        emit_all(repo=repo, out=tmp_path / 'x', carve_scope='file', include=[])


def test_a_glob_matching_nothing_is_refused(repo, tmp_path):
    with pytest.raises(CarveScopeError, match='zero files'):
        emit_all(
            repo=repo, out=tmp_path / 'x', carve_scope='file',
            include=['src/a2a/nope/**'],
        )


def test_carving_a_graded_test_file_is_refused(repo, tmp_path):
    """Carving the tests that grade you is a free reward. Never a warning."""
    with pytest.raises(CarveScopeError, match='graded/fingerprinted'):
        emit_all(
            repo=repo, out=tmp_path / 'x', carve_scope='folder',
            include=['tests/utils/**'],
        )


def test_delete_whole_file_removes_the_file_from_the_staged_tree(repo, tmp_path):
    out = tmp_path / 'del'
    entries = emit_all(
        repo=repo, out=out, package_base='src/', carve_scope='file',
        include=['src/a2a/utils/error_handlers.py'], delete_whole_file=True,
        context_types=('no_context',),
    )
    e = entries[0]
    assert e.carved_relpaths == ('src/a2a/utils/error_handlers.py',)
    ctx = next((out / '_staging').glob('*/ctx/repo'))
    assert not (ctx / 'src/a2a/utils/error_handlers.py').exists()
    assert (e.path / 'solution/carved/src/a2a/utils/error_handlers.py').is_file()
    assert not (e.path / 'environment/carve/src/a2a/utils/error_handlers.py').exists()
