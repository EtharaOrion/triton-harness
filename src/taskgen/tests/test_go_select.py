"""Go function-scope target selection, validated against the REAL go-multigres.

Three things here are not obvious and are the reason this module exists:

  * `GOParser.get_class_name` is BROKEN (go_parser.py:65-77 closes over `node`
    instead of its `root` argument), so `class_name` is ALWAYS "" for a Go
    method. `FunctionData.__hash__` is `file_path + name`, so two methods named
    `Run` on different receivers in one file are the SAME object to every set
    the parser builds. taskgen derives the receiver itself and refuses an
    ambiguous target rather than silently picking one (spike G3).
  * ~5% of parser-linked `test_funcs` are not runnable Go tests -- the parser's
    `is_test` is `"test" in name.lower()`, which matches `testFormatQualified`
    and `BenchmarkFoo` alike. Only `TestXxx(*testing.T)` in a `_test.go` file
    can appear in a `-run` regex (spike G4).
  * the graded selector is a package + a `-run` anchor, not a node id, so the
    package import path has to be reconstructed from go.mod.

The frozen target is the S-GO spike's: `assignConnectionID`, a method on
`*Listener`, 4 linked tests, EXPECTED=4. No docker is needed to check any of it.
"""

from __future__ import annotations

import pytest

from taskgen.langs import go_select as GS

FROZEN_RELPATH = 'go/common/pgprotocol/server/listener.go'
FROZEN_FUNC = 'assignConnectionID'
FROZEN_RECEIVER = 'Listener'
FROZEN_PACKAGE = 'github.com/multigres/multigres/go/common/pgprotocol/server'
FROZEN_TESTS = (
    'TestAssignConnectionID_EncodesGatewayPrefix',
    'TestAssignConnectionID_FailsWhenAllIDsUsed',
    'TestAssignConnectionID_SkipsInUsePIDs',
    'TestAssignConnectionID_WrapsAndFindsSlot',
)


@pytest.fixture(scope='session')
def go_repo(request):
    repo = request.config.rootpath / 'repo' / 'go-multigres'
    if not repo.is_dir():
        pytest.skip(f'go-multigres checkout not present: {repo}')
    return repo


@pytest.fixture(scope='session')
def frozen(go_repo):
    """The S-GO frozen target, selected through the public API.

    Scoped to the one package: the whole repo is 1231 files / ~2 minutes, and
    nothing about this target's linkage crosses a package boundary.
    """
    return GS.select_go_target(
        go_repo,
        file=FROZEN_RELPATH,
        func=FROZEN_FUNC,
        scope_dir='go/common/pgprotocol/server',
    )


# ---------------------------------------------------------------- go.mod ----


def test_module_path_is_read_from_go_mod(go_repo):
    assert GS.module_path(go_repo) == 'github.com/multigres/multigres'


def test_project_name_is_the_module_segment_the_parser_matches_on(go_repo):
    """GOParser resolves imports by `path.split('/')[2] == project_name`."""
    assert GS.project_name(go_repo) == 'multigres'


def test_package_import_path_is_module_plus_directory(go_repo):
    module = GS.module_path(go_repo)
    assert GS.package_import_path(go_repo, module, FROZEN_RELPATH) == FROZEN_PACKAGE
    assert GS.package_import_path(go_repo, module, 'go/common') == \
        'github.com/multigres/multigres/go/common'
    assert GS.package_import_path(go_repo, module, '.') == module


# ------------------------------------------------- G3: receiver derivation --


def test_receiver_is_derived_because_the_parser_always_returns_empty(frozen):
    """The whole point: `fd.class_name` is "" here, and `receiver` is not."""
    assert frozen.fd.class_name == '', 'go_parser.py:65-77 is expected to be broken'
    assert frozen.receiver == FROZEN_RECEIVER
    assert frozen.qualname == '(*Listener).assignConnectionID'


def test_a_plain_function_has_no_receiver(go_repo):
    _, recv = GS.find_go_function(
        go_repo, file=FROZEN_RELPATH, func='NewListener',
        scope_dir='go/common/pgprotocol/server',
    )
    assert recv == ''


# ---------------------------------------------------- the frozen selector ---


def test_frozen_target_resolves_to_the_spike_s_file_and_package(frozen):
    assert frozen.relpath == FROZEN_RELPATH
    assert frozen.name == FROZEN_FUNC
    assert frozen.package == FROZEN_PACKAGE
    assert frozen.packages == (FROZEN_PACKAGE,)


def test_frozen_target_links_exactly_the_four_spike_tests(frozen):
    assert tuple(t.name for t in frozen.tests) == FROZEN_TESTS
    assert frozen.test_names == FROZEN_TESTS


def test_frozen_expected_is_four(frozen):
    """EXPECTED is the pinned denominator; the spike measured 4 in-container."""
    assert frozen.expected == 4 == len(frozen.tests)


def test_frozen_run_regex_is_the_spike_s_anchored_alternation(frozen):
    assert frozen.run_regex == '^(' + '|'.join(FROZEN_TESTS) + ')$'


def test_every_linked_test_carries_its_owning_package(frozen):
    assert {t.package for t in frozen.tests} == {FROZEN_PACKAGE}
    assert {t.relpath for t in frozen.tests} == \
        {'go/common/pgprotocol/server/listener_test.go'}


def test_selection_is_deterministic(go_repo, frozen):
    again = GS.select_go_target(
        go_repo, file=FROZEN_RELPATH, func=FROZEN_FUNC,
        scope_dir='go/common/pgprotocol/server',
    )
    assert again.run_regex == frozen.run_regex
    assert again.test_names == frozen.test_names


# ------------------------------------------- G4: runnable-test filtering ----


@pytest.mark.parametrize('relpath, name, params, ok', [
    ('x_test.go', 'TestFoo', '(t *testing.T)', True),
    ('x_test.go', 'Test', '(t *testing.T)', True),
    ('x_test.go', 'Test_foo', '(t *testing.T)', True),
    # lowercase helper: `is_test` matches it, `go test -run` never runs it
    ('x_test.go', 'testFormatQualifiedName', '(t *testing.T)', False),
    # right name, wrong file: a Test func outside _test.go is not compiled in
    ('x.go', 'TestFoo', '(t *testing.T)', False),
    # benchmarks and fuzz targets are not -run selectable tests
    ('x_test.go', 'BenchmarkFoo', '(b *testing.B)', False),
    ('x_test.go', 'TestMain', '(m *testing.M)', False),
    ('x_test.go', 'FuzzFoo', '(f *testing.F)', False),
    # TestingT-alike must not be mistaken for *testing.T
    ('x_test.go', 'TestFoo', '(t *testing.TB)', False),
    ('x_test.go', 'TestFoo', '()', False),
    # subtest-style names starting lowercase after Test
    ('x_test.go', 'Testify', '(t *testing.T)', False),
])
def test_runnable_go_test_filter(relpath, name, params, ok):
    assert GS.is_runnable_test_signature(relpath, name, params) is ok


def test_only_runnable_tests_reach_the_run_regex(tmp_path):
    """A parser-linked lowercase helper must not end up in `-run`."""
    repo = _write_module(tmp_path, extra_tests=True)
    t = GS.select_go_target(repo, file='pkg/dual.go', func='Alpha')
    assert 'testAlphaHelper' not in t.run_regex
    assert 'BenchmarkAlpha' not in t.run_regex
    assert t.test_names == ('TestAlphaOne', 'TestAlphaTwo')


# ---------------------------------- G3: same-name methods must not collide --


def _write_module(tmp_path, *, extra_tests: bool = False) -> object:
    repo = tmp_path / 'mod'
    (repo / 'pkg').mkdir(parents=True)
    (repo / 'go.mod').write_text('module github.com/acme/widget\n\ngo 1.26\n')
    (repo / 'pkg' / 'dual.go').write_text(
        'package pkg\n'
        '\n'
        'type Alpha struct{}\n'
        '\n'
        'type Beta struct{}\n'
        '\n'
        'func helper() int { return 1 }\n'
        '\n'
        '// Alpha returns the alpha number.\n'
        'func (a *Alpha) Alpha() int {\n'
        '\treturn helper()\n'
        '}\n'
        '\n'
        '// Run is the ALPHA implementation.\n'
        'func (a *Alpha) Run() int {\n'
        '\treturn helper() + 10\n'
        '}\n'
        '\n'
        '// Run is the BETA implementation.\n'
        'func (b Beta) Run() int {\n'
        '\treturn helper() + 20\n'
        '}\n'
    )
    body = (
        'package pkg\n'
        '\n'
        'import "testing"\n'
        '\n'
        '// TestAlphaOne exercises Alpha.\n'
        'func TestAlphaOne(t *testing.T) {\n'
        '\ta := &Alpha{}\n'
        '\tif a.Alpha() != 1 {\n\t\tt.Fatal("no")\n\t}\n'
        '}\n'
        '\n'
        '// TestAlphaTwo exercises Alpha again.\n'
        'func TestAlphaTwo(t *testing.T) {\n'
        '\ta := &Alpha{}\n'
        '\tif a.Alpha() != 1 {\n\t\tt.Fatal("no")\n\t}\n'
        '}\n'
    )
    if extra_tests:
        body += (
            '\n'
            '// testAlphaHelper is a helper, not a test.\n'
            'func testAlphaHelper(t *testing.T) {\n'
            '\t_ = (&Alpha{}).Alpha()\n'
            '}\n'
            '\n'
            '// BenchmarkAlpha benchmarks Alpha.\n'
            'func BenchmarkAlpha(b *testing.B) {\n'
            '\t_ = (&Alpha{}).Alpha()\n'
            '}\n'
        )
    (repo / 'pkg' / 'dual_test.go').write_text(body)
    return repo


def test_same_name_methods_on_different_receivers_are_distinguishable(tmp_path):
    """`FunctionData.__hash__` is file+name, so these two are ONE object to the
    parser. Identity must come from the receiver we derive ourselves."""
    repo = _write_module(tmp_path)
    alpha, ra = GS.find_go_function(repo, file='pkg/dual.go', func='Run', receiver='Alpha')
    beta, rb = GS.find_go_function(repo, file='pkg/dual.go', func='Run', receiver='Beta')
    assert (ra, rb) == ('Alpha', 'Beta')
    assert 'helper() + 10' in alpha.get_body()
    assert 'helper() + 20' in beta.get_body()
    assert alpha.func_node.start_byte != beta.func_node.start_byte
    assert alpha == beta, 'the parser really cannot tell them apart (G3)'


def test_a_value_receiver_is_reported_without_the_pointer(tmp_path):
    repo = _write_module(tmp_path)
    _, recv = GS.find_go_function(repo, file='pkg/dual.go', func='Run', receiver='Beta')
    assert recv == 'Beta'


def test_an_ambiguous_target_is_refused_rather_than_guessed(tmp_path):
    """Identity is (file, receiver, name). Without the receiver it is not a key."""
    repo = _write_module(tmp_path)
    with pytest.raises(GS.GoSelectError, match='Alpha.*Beta|ambiguous'):
        GS.find_go_function(repo, file='pkg/dual.go', func='Run')
    with pytest.raises(GS.GoSelectError, match='Alpha.*Beta|ambiguous'):
        GS.select_go_target(repo, file='pkg/dual.go', func='Run')


def test_an_unknown_receiver_names_the_ones_that_exist(tmp_path):
    repo = _write_module(tmp_path)
    with pytest.raises(GS.GoSelectError, match='Alpha'):
        GS.find_go_function(repo, file='pkg/dual.go', func='Run', receiver='Gamma')


def test_a_target_with_no_runnable_linked_test_fails_closed(tmp_path):
    """The parser's linkage is receiver-blind (G3), so of the two `Run` methods
    at most one is reachable from a test. An empty graded set is a free reward;
    it must raise, not ship."""
    repo = _write_module(tmp_path)
    stranded = []
    for recv in ('Alpha', 'Beta'):
        try:
            GS.select_go_target(repo, file='pkg/dual.go', func='Run', receiver=recv)
        except GS.GoSelectError as exc:
            assert 'no runnable' in str(exc)
            stranded.append(recv)
    assert stranded, 'expected the receiver-blind linkage to strand a method'


def test_an_unknown_target_names_what_was_available(tmp_path):
    repo = _write_module(tmp_path)
    with pytest.raises(GS.GoSelectError, match='no go function'):
        GS.select_go_target(repo, file='pkg/dual.go', func='Nope')


# ----------------------------------------------------------- run regex ------


def test_run_regex_is_anchored_sorted_and_deduplicated():
    assert GS.run_regex(['TestB', 'TestA', 'TestB']) == '^(TestA|TestB)$'


def test_run_regex_refuses_an_empty_selection():
    """An unanchored or empty `-run` selects the WHOLE package."""
    with pytest.raises(GS.GoSelectError, match='empty'):
        GS.run_regex([])
