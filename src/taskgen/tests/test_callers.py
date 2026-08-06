"""Graph inversion in isolation: transpose, order, and the cases that bite.

`test_contexts.py` proves the caller conditions against the frozen repo, which
costs a full parse and can only exercise the shapes that repo happens to have.
This module drives the inversion directly, so the shapes that matter -- a
function nothing calls, a recursive self-edge, two callers in one file -- are
pinned without a parser, and the DETERMINISM claim is testable at all: the
parser hands us `callee` as a `set`, whose iteration order is not stable across
processes, so the ordering guarantee has to survive shuffled input by
construction rather than by luck.
"""

from __future__ import annotations

from pathlib import Path

from taskgen.select import _sort_key, callers_of, invert_callee_edges

REPO = Path('/repo')


class FakeFunc:
    """`parser.FunctionData`'s identity contract: `(file_path, name)`, nothing more."""

    def __init__(self, relpath: str, name: str, class_name: str = ''):
        self.file_path = str(REPO / relpath)
        self.name = name
        self.class_name = class_name
        self.callee: set = set()

    def calls(self, *others) -> 'FakeFunc':
        self.callee.update(others)
        return self

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, FakeFunc)
            and (self.file_path, self.name) == (other.file_path, other.name)
        )

    def __hash__(self) -> int:
        return hash(self.file_path + self.name)

    def __repr__(self) -> str:
        return f'FakeFunc({self.file_path}::{self.class_name}.{self.name})'


def graph():
    target = FakeFunc('src/pkg/core.py', 'target')
    helper = FakeFunc('src/pkg/core.py', 'helper')
    b = FakeFunc('src/pkg/b.py', 'beta', class_name='Beta')
    a = FakeFunc('src/pkg/a.py', 'alpha', class_name='Alpha')
    lonely = FakeFunc('src/pkg/z.py', 'lonely')
    a.calls(target, helper)
    b.calls(target)
    target.calls(helper)
    return target, helper, a, b, lonely


def sorted_funcs(funcs):
    return sorted(funcs, key=lambda f: _sort_key(REPO, f))


def test_inversion_is_the_transpose_of_the_forward_edges():
    target, helper, a, b, _lonely = graph()
    inverse = invert_callee_edges(sorted_funcs([target, helper, a, b]))
    assert set(inverse[target]) == {a, b}
    assert set(inverse[helper]) == {a, target}


def test_a_function_nobody_calls_has_no_callers():
    target, helper, a, b, lonely = graph()
    assert callers_of(REPO, lonely, sorted_funcs([target, helper, a, b, lonely])) == []
    assert invert_callee_edges([lonely]) == {}


def test_callers_are_sorted_by_the_same_key_as_callees():
    target, helper, a, b, lonely = graph()
    callers = callers_of(REPO, target, sorted_funcs([target, helper, a, b, lonely]))
    assert callers == [a, b]
    assert [_sort_key(REPO, c) for c in callers] == sorted(
        _sort_key(REPO, c) for c in callers
    )


def test_order_survives_shuffled_input():
    """The parser's `callee` is a set, so insertion order is never a given."""
    import itertools

    target, helper, a, b, lonely = graph()
    every = [target, helper, a, b, lonely]
    orders = {
        tuple(
            _sort_key(REPO, c)
            for c in callers_of(REPO, target, sorted_funcs(list(perm)))
        )
        for perm in itertools.permutations(every)
    }
    assert len(orders) == 1
    assert orders.pop() == (
        ('src/pkg/a.py', 'Alpha', 'alpha'), ('src/pkg/b.py', 'Beta', 'beta'),
    )


def test_a_recursive_function_is_its_own_caller():
    """Which is why `contexts` still runs the carved-body-owner check on callers."""
    loop = FakeFunc('src/pkg/core.py', 'loop')
    loop.calls(loop)
    assert callers_of(REPO, loop, [loop]) == [loop]


def test_a_caller_is_reported_once_however_many_times_it_calls():
    target = FakeFunc('src/pkg/core.py', 'target')
    twin = FakeFunc('src/pkg/core.py', 'target')
    caller = FakeFunc('src/pkg/a.py', 'alpha')
    caller.calls(target)
    assert callers_of(REPO, target, sorted_funcs([caller, caller, target])) == [caller]
    assert callers_of(REPO, twin, sorted_funcs([caller, target])) == [caller]
