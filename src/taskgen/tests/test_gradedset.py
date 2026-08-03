"""The UNION-minus-carved-tests graded-set rule (plan T13/§graded-set).

The rule has to hold three properties at once, and each one is a separate way
to ship a broken task:

  UNION          every carved non-test function contributes its linked tests,
                 or a multi-file carve grades only a slice of what it removed
  MINUS          a test living in a CARVED file is dropped, or the task grades
                 a test whose own body was just stubbed out (free reward)
  COLLAPSE       for a single carved function the union IS today's
                 `linked_nodeids`, or function scope silently changes meaning
"""

from __future__ import annotations

import pytest

from taskgen.gradedset import (
    CarvedFunction,
    GradedSetError,
    LinkedTest,
    derive_graded_set,
    parse_linked,
    select_carved,
)

from .conftest import FROZEN_FILE, FROZEN_FUNC, FROZEN_NODEIDS

UTILS_FOLDER = 'src/a2a/utils/'


@pytest.fixture(scope='module')
def parsed(repo):
    """One full parse of the repo; the rule under test is the cheap half."""
    return parse_linked('python', repo, package_base='src/')


def _t(relpath, name, cls=''):
    return LinkedTest(relpath=relpath, class_name=cls, name=name)


def _f(relpath, name, tests):
    return CarvedFunction(relpath=relpath, qualname=name, name=name, tests=tuple(tests))


# --------------------------------------------------------------------------
# the rule, on synthetic input (no parser, no repo)
# --------------------------------------------------------------------------


def test_union_takes_every_carved_functions_tests():
    funcs = [
        _f('a.py', 'alpha', [_t('t_a.py', 'test_alpha')]),
        _f('b.py', 'beta', [_t('t_b.py', 'test_beta')]),
    ]
    graded = derive_graded_set('python', funcs, ('a.py', 'b.py'))
    assert graded.selectors == ('t_a.py::test_alpha', 't_b.py::test_beta')
    assert graded.expected == 2


def test_union_deduplicates_a_test_that_reaches_two_carved_functions():
    shared = _t('t.py', 'test_both')
    funcs = [_f('a.py', 'alpha', [shared]), _f('b.py', 'beta', [shared])]
    graded = derive_graded_set('python', funcs, ('a.py', 'b.py'))
    assert graded.selectors == ('t.py::test_both',)
    assert graded.expected == 1


def test_minus_drops_tests_that_live_in_a_carved_file():
    """A carved test is a stubbed test: grading it is a free reward."""
    funcs = [
        _f('a.py', 'alpha', [_t('a.py', 'test_inline'), _t('t_a.py', 'test_alpha')]),
    ]
    graded = derive_graded_set('python', funcs, ('a.py',))
    assert graded.selectors == ('t_a.py::test_alpha',)


def test_fails_closed_when_the_union_is_empty():
    funcs = [_f('a.py', 'alpha', [_t('a.py', 'test_inline')])]
    with pytest.raises(GradedSetError, match='empty graded set'):
        derive_graded_set('python', funcs, ('a.py',))


def test_fails_closed_when_no_carved_function_was_found():
    with pytest.raises(GradedSetError, match='no carved'):
        derive_graded_set('python', [], ('a.py',))


def test_nodeid_omits_the_class_segment_for_module_level_tests():
    funcs = [_f('a.py', 'alpha', [_t('t.py', 'test_x'), _t('t.py', 'test_y', 'TestK')])]
    graded = derive_graded_set('python', funcs, ('a.py',))
    assert graded.selectors == ('t.py::TestK::test_y', 't.py::test_x')


def test_go_selectors_are_bare_test_names_with_their_packages():
    funcs = [
        CarvedFunction(
            relpath='go/x/x.go', qualname='(*L).f', name='f',
            tests=(
                LinkedTest('go/x/x_test.go', '', 'TestB', 'example.com/m/go/x'),
                LinkedTest('go/x/x_test.go', '', 'TestA', 'example.com/m/go/x'),
            ),
        ),
    ]
    graded = derive_graded_set('go', funcs, ('go/x/x.go',))
    assert graded.kind == 'go-run'
    assert graded.selectors == ('TestA', 'TestB')
    assert graded.packages == ('example.com/m/go/x',)
    assert graded.expected == 2


def test_fingerprint_relpaths_are_the_graded_test_files():
    funcs = [
        _f('a.py', 'alpha', [_t('t_a.py', 'test_alpha'), _t('t_b.py', 'test_beta')]),
    ]
    graded = derive_graded_set('python', funcs, ('a.py',))
    assert graded.fingerprint_relpaths == ('t_a.py', 't_b.py')


# --------------------------------------------------------------------------
# the rule, against the real repo
# --------------------------------------------------------------------------


def test_collapses_to_todays_linked_nodeids_for_one_function(repo, target, parsed):
    """Function scope carves ONE function, so the union is that function's tests."""
    from taskgen.nodeids import linked_nodeids

    funcs = select_carved(parsed, (FROZEN_FILE,), only={FROZEN_FUNC})
    assert [f.name for f in funcs] == [FROZEN_FUNC]
    graded = derive_graded_set('python', funcs, (FROZEN_FILE,))
    assert list(graded.selectors) == FROZEN_NODEIDS == linked_nodeids(repo, target)


def test_file_scope_unions_every_function_in_the_carved_file(parsed):
    """All six non-test functions of the frozen file, not just the frozen one."""
    funcs = select_carved(parsed, (FROZEN_FILE,))
    assert len(funcs) == 6
    assert FROZEN_FUNC in {f.name for f in funcs}
    graded = derive_graded_set('python', funcs, (FROZEN_FILE,))
    assert list(graded.selectors) == FROZEN_NODEIDS


def test_folder_scope_unions_across_files(parsed):
    carved = tuple(
        f'{UTILS_FOLDER}{n}.py'
        for n in (
            '_async_queue_compat', 'error_handlers', 'errors', 'grpc_status',
            'json_utils', 'proto_utils', 'signing', 'task', 'telemetry',
            'version_validator',
        )
    )
    funcs = select_carved(parsed, carved)
    assert len(funcs) == 57
    graded = derive_graded_set('python', funcs, carved)
    assert graded.expected == 14
    assert set(FROZEN_NODEIDS) < set(graded.selectors)
    assert len({s.split('::')[0] for s in graded.selectors}) == 5


def test_collect_skips_test_functions_living_in_the_carve_set(parsed):
    """Carving a test file must not make its tests contribute to their own grade."""
    carved = (FROZEN_FILE, 'tests/utils/test_task.py')
    funcs = select_carved(parsed, carved)
    assert all(not f.relpath.startswith('tests/') for f in funcs)
    with pytest.raises(GradedSetError, match='empty graded set'):
        derive_graded_set('python', funcs, carved)
