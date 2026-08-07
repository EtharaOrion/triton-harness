"""Go function-scope emission: a full harbor entry from the go plugin.

The frozen S-GO target (go-multigres, `(*Listener).assignConnectionID`) is
proven end-to-end in `test_go_select.py` and, with real Docker, in Wave 4. What
is proven HERE is the emit wiring -- that `--lang go` reaches `langs.go`, that
the selector becomes an anchored `-run` regex rather than a pytest allowlist,
that the stub panics rather than raising `NotImplementedError`, and that the
python-only assets are simply absent. A synthetic module is used so the test
costs milliseconds instead of parsing 1231 files.
"""

from __future__ import annotations

import hashlib
import tomllib

import pytest

from taskgen.emit import emit_all

MODULE = 'github.com/acme/widget'
SRC = 'pkg/dual.go'
FUNC = 'Alpha'
TESTS = ('TestAlphaOne', 'TestAlphaTwo')


@pytest.fixture(scope='module')
def go_module(tmp_path_factory):
    repo = tmp_path_factory.mktemp('gomod') / 'widget'
    (repo / 'pkg').mkdir(parents=True)
    (repo / 'go.mod').write_text(f'module {MODULE}\n\ngo 1.26\n')
    (repo / 'go.sum').write_text('')
    (repo / SRC).write_text(
        'package pkg\n'
        '\n'
        'type Alpha struct{}\n'
        '\n'
        'func helper() int { return 41 }\n'
        '\n'
        '// Alpha returns the alpha number, which is helper plus one.\n'
        'func (a *Alpha) Alpha() int {\n'
        '\tbase := helper()\n'
        '\tadjusted := base + alphaWidgetOffset // widget-specific alpha shift\n'
        '\treturn adjusted\n'
        '}\n'
        '\n'
        'const alphaWidgetOffset = 1\n'
    )
    (repo / 'pkg' / 'dual_test.go').write_text(
        'package pkg\n'
        '\n'
        'import "testing"\n'
        '\n'
        '// TestAlphaOne exercises Alpha.\n'
        'func TestAlphaOne(t *testing.T) {\n'
        '\tif (&Alpha{}).Alpha() != 42 {\n\t\tt.Fatal("no")\n\t}\n'
        '}\n'
        '\n'
        '// TestAlphaTwo exercises Alpha again.\n'
        'func TestAlphaTwo(t *testing.T) {\n'
        '\tif (&Alpha{}).Alpha() < 0 {\n\t\tt.Fatal("no")\n\t}\n'
        '}\n'
    )
    return repo


@pytest.fixture(scope='module')
def go_entries(go_module, tmp_path_factory):
    out = tmp_path_factory.mktemp('goout')
    return out, emit_all(
        repo=go_module, out=out, lang='go', file=SRC, func=FUNC,
    )


def test_go_emits_the_eleven_entries(go_entries):
    _out, entries = go_entries
    assert len(entries) == 11
    assert {e.lang for e in entries} == {'go'}
    assert {e.carve_scope for e in entries} == {'function'}


def test_go_entry_layout(go_entries):
    _out, entries = go_entries
    for e in entries:
        for rel in (
            'task.toml', 'instruction.md',
            'environment/Dockerfile', f'environment/carve/{SRC}',
            'tests/test.sh', 'tests/graded.json',
            'solution/solve.sh', f'solution/carved/{SRC}',
        ):
            assert (e.path / rel).is_file(), f'{e.context_type}: missing {rel}'


def test_go_gets_caller_context_from_the_same_inversion(go_entries):
    """`caller_*` is not a python-only feature bolted onto `select.py`.

    `GoTarget` carries no callee/caller surface at all -- both are derived in
    `carve.py` from the parser's graph, forwards and then backwards -- so if go
    resolves callees it must resolve callers, off the same edges.
    """
    _out, entries = go_entries
    by_type = {e.context_type: e for e in entries}
    callee = (by_type['callee_func'].path / 'instruction.md').read_text()
    assert '1 callee(s) were resolved' in callee
    assert 'func helper() int' in callee

    for ct, expected in (('caller_func', 'func TestAlphaOne('),
                         ('caller_sig', 'func TestAlphaOne(t *testing.T) { ... }')):
        text = (by_type[ct].path / 'instruction.md').read_text()
        assert '2 caller(s) were resolved' in text, ct
        assert expected in text, ct
        assert 'No first-party callers were found' not in text, ct


def test_go_ships_no_pytest_only_assets(go_entries):
    """allowlist.txt and harbor_filter.py are a pytest mechanism, not a task one."""
    _out, entries = go_entries
    for e in entries:
        assert not (e.path / 'tests/allowlist.txt').exists()
        assert not (e.path / 'tests/harbor_filter.py').exists()


def test_go_graded_set_is_the_linked_test_names(go_entries):
    import json

    _out, entries = go_entries
    for e in entries:
        assert tuple(e.nodeids) == TESTS
        graded = json.loads((e.path / 'tests/graded.json').read_text())
        assert graded['kind'] == 'go-run'
        assert tuple(graded['selectors']) == TESTS
        assert graded['packages'] == [f'{MODULE}/pkg']
        assert graded['expected'] == 2


def test_go_test_sh_uses_an_anchored_run_regex_and_a_pinned_floor(go_entries):
    _out, entries = go_entries
    sh = (entries[0].path / 'tests/test.sh').read_text()
    assert 'EXPECTED=2' in sh
    assert f"-run '^({'|'.join(TESTS)})$'" in sh
    assert '-count=1' in sh
    assert f'{MODULE}/pkg' in sh
    assert 'scope-growth' in sh, 'go grades on a PINNED denominator (spike G1)'
    assert 'harbor_filter' not in sh


def test_go_stub_panics_rather_than_raising_notimplementederror(go_entries):
    _out, entries = go_entries
    stub = (entries[0].path / f'environment/carve/{SRC}').read_text()
    assert 'panic("not implemented")' in stub
    assert 'raise NotImplementedError' not in stub
    assert 'func (a *Alpha) Alpha() int' in stub, 'the signature must survive'
    assert 'func helper() int { return 41 }' in stub, 'only the target is carved'


def test_go_solution_is_the_intact_original(go_module, go_entries):
    _out, entries = go_entries
    want = hashlib.sha256((go_module / SRC).read_bytes()).hexdigest()
    for e in entries:
        raw = (e.path / f'solution/carved/{SRC}').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == want


def test_go_solve_sh_restores_from_the_runtime_mount(go_entries):
    _out, entries = go_entries
    solve = (entries[0].path / 'solution/solve.sh').read_text()
    assert f"restore '{SRC}'" in solve
    assert 'WANT=1' in solve
    assert '/opt/harbor/solution' in solve
    assert 'go clean -testcache' in solve


def test_go_dockerfile_is_the_go_toolchain_over_a_staged_tree(go_entries):
    _out, entries = go_entries
    df = (entries[0].path / 'environment/Dockerfile').read_text()
    assert 'COPY --from=repoctx repo/' in df
    assert 'GOTOOLCHAIN=local' in df
    assert 'GOPROXY=off' in df
    assert 'repo-src' not in df and 'repos-src' not in df
    assert 'leakscan.sh' in df


def test_go_task_toml_is_family_a_and_declares_go(go_entries):
    _out, entries = go_entries
    for e in entries:
        data = tomllib.loads((e.path / 'task.toml').read_text())
        assert data['schema_version'] == '1.4'
        assert data['metadata']['language'] == 'go'
        assert data['metadata']['carve_scope'] == 'function'
        assert data['metadata']['condition'] == e.context_type
        assert data['metadata']['graded_tests'] == list(TESTS)


def test_go_instruction_names_the_receiver_and_the_go_selector(go_entries):
    _out, entries = go_entries
    for e in entries:
        doc = (e.path / 'instruction.md').read_text()
        assert '(*Alpha).Alpha' in doc
        assert '```go' in doc
        assert 'go test' in doc
        for name in TESTS:
            assert name in doc


def test_go_instruction_never_says_raise_notimplementederror(go_entries):
    """The prompt must describe the stub the solver actually sees."""
    _out, entries = go_entries
    for e in entries:
        assert 'raise NotImplementedError' not in (e.path / 'instruction.md').read_text()


def test_go_staged_tree_carries_the_panic_stub(go_entries):
    out, entries = go_entries
    ctx = next((out / '_staging').glob('*/ctx/repo'))
    assert 'panic("not implemented")' in (ctx / SRC).read_text()
    assert (ctx / 'pkg/dual_test.go').is_file(), 'the graded test must survive'


def test_go_is_deterministic(go_module, go_entries, tmp_path):
    import filecmp

    first, _entries = go_entries
    second = tmp_path / 'rerun'
    emit_all(repo=go_module, out=second, lang='go', file=SRC, func=FUNC)
    dc = filecmp.dircmp(first, second)

    def diffs(d, prefix=''):
        out = [f'{prefix}{n}' for n in d.left_only + d.right_only + d.funny_files]
        out += [f'{prefix}{n}' for n in d.diff_files]
        for name, sub in d.subdirs.items():
            out += diffs(sub, f'{prefix}{name}/')
        return out

    assert diffs(dc) == []


def test_go_and_python_ids_never_collide(go_entries):
    """`lang` is part of the uuid5 key, so the same path in two languages differs."""
    from taskgen.ids import entry_id

    py = entry_id('r', 'a/b.go', '', 'f', 'no_context')
    go = entry_id('r', 'a/b.go', '', 'f', 'no_context', lang='go')
    assert py != go


def test_unknown_language_is_refused(go_module, tmp_path):
    from taskgen.langs.base import LangError

    with pytest.raises((LangError, SystemExit)):
        emit_all(repo=go_module, out=tmp_path / 'x', lang='cobol', file=SRC, func=FUNC)
