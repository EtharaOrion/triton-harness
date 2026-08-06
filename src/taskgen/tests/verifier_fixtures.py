"""A scripted stand-in for the authoring model, for offline tests AND evidence runs.

The LLM proxy is not reachable from the test suite (and must never be: the suite
is offline and deterministic), so every `--verifier` path is driven by a
`MockClient` whose responder is dispatched on the generators' own system prompts.
That keeps the fixture honest -- if a generator's prompt changes, the dispatch
stops matching and the test fails loudly rather than silently scripting the wrong
step.

WHAT THE FAKE TESTS ASSERT, AND WHY THEY ARE NOT BEHAVIOURAL. A real generated
suite imports the solution and exercises it. This one cannot: the sample repos'
third-party dependencies are not installed in taskgen's own environment, and
installing them at generate time would put the network on the default path. So
the scripted module asserts the weakest property that still genuinely
discriminates golden from stub -- "the carved bodies are implemented at all" --
using two text checks and two `ast` checks. It is a FIXTURE, and
`emit.VERIFIER_FAKE_CLIENT_ENV` announces itself on every run that uses it.

What it does prove is exactly what it is used to prove: that `exec_env` really
materialises both trees, really runs pytest in them, and that only pass-golden
AND fail-stub checks survive into a frozen bundle.
"""

from __future__ import annotations

import json
import re

from verifier.generators.model_client import MockClient

__all__ = [
    'STUB_MARKER',
    'ScriptedAuthor',
    'TRUTH_MD',
    'always_pass_pytest_module',
    'mock_client',
    'sound_pytest_module',
]

#: Present in the python stub body, absent from every sample golden file. The
#: judge responder uses it to tell which side it is being asked to score.
STUB_MARKER = 'raise NotImplementedError'

TRUTH_MD = """## Problem

The carved function bodies were removed and must be written again so the linked
tests pass.

## Behavioral contract

A correct solution restores the documented behaviour of every carved symbol: the
same inputs produce the same observable outputs, and the module keeps importing
cleanly.

## Solution decomposition

Read the surviving signature and its documentation, work out the required
mapping from inputs to outputs, then implement it.

## Solution space

Any implementation that satisfies the contract is acceptable: an explicit loop, a
comprehension, or a delegation to an existing helper are all valid routes.

## Known pitfalls

Boundary values, the empty input, and configuration objects that leave the field
unset are the cases most often missed.

## Cheat surface

Special-casing the exact values the frozen tests assert, or editing the tests
themselves, is not a solution.

## Success criteria

Every carved body is implemented, the module still imports, and the linked tests
pass.
"""

_RUBRIC_JSON = json.dumps([
    {'id': f'ts.criterion_{i}',
     'text': f'Does the solution satisfy behavioral-contract item {i} '
             'for the carved symbol, rather than special-casing an input?',
     'contract_ref': 'Behavioral contract'}
    for i in range(1, 9)
])

_PREDICATES_JSON = json.dumps([
    {'id': 'pt.no_stub_marker', 'type': 'pattern_absent',
     'target': r'NotImplementedError', 'truth_ref': 'Success criteria',
     'description': 'the carved body must be implemented, not left unimplemented',
     'negative_fixture': '    raise NotImplementedError\n'},
    {'id': 'pt.no_bare_pass_body', 'type': 'pattern_absent',
     'target': r'def \w+\([^)]*\):\n\s+pass\n', 'truth_ref': 'Cheat surface',
     'description': 'an empty body is not an implementation',
     'negative_fixture': 'def f():\n    pass\n'},
])

_SYSTEM_MARKERS = {
    'truth': 'You are authoring TRUTH.md',
    'pytest': 'You write a REAL pytest test module',
    'predicates': 'You generate DECIDABLE task-specific checks',
    'rubric': 'You design a binary evaluation rubric',
    'classify': 'You audit VERIFIERS',
    'judge': 'code-review JUDGE',
}

_STUB_FILES_RE = re.compile(r'^The implemented files are: (.+)$', re.M)


def _carved_relpaths(user: str) -> list[str]:
    m = _STUB_FILES_RE.search(user)
    return [p.strip() for p in m.group(1).split(',') if p.strip()] if m else []


def sound_pytest_module(relpaths: list[str]) -> str:
    """Four checks that pass on an implemented tree and fail on a carved one."""
    listing = ', '.join(repr(r) for r in relpaths)
    return f'''import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
CARVED = [{listing}]


def _sources():
    return [(rel, (ROOT / rel).read_text(encoding="utf-8")) for rel in CARVED]


def _functions(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def test_no_unimplemented_body_marker():
    for rel, text in _sources():
        assert "raise NotImplementedError" not in text, rel


def test_no_unimplemented_exception_reference():
    for rel, text in _sources():
        assert "NotImplementedError" not in text, rel


def test_no_function_body_is_a_single_raise_or_pass():
    for rel, text in _sources():
        for fn in _functions(ast.parse(text)):
            body = [s for s in fn.body
                    if not (isinstance(s, ast.Expr)
                            and isinstance(s.value, ast.Constant)
                            and isinstance(s.value.value, str))]
            assert not (len(body) == 1 and isinstance(body[0], (ast.Raise, ast.Pass))), \\
                f"{{rel}}::{{fn.name}}"


def test_every_function_keeps_its_documentation():
    for rel, text in _sources():
        for fn in _functions(ast.parse(text)):
            assert ast.get_docstring(fn), f"{{rel}}::{{fn.name}}"
'''


def always_pass_pytest_module(relpaths: list[str]) -> str:
    """A suite that passes on BOTH trees, so every test is non-discriminating."""
    del relpaths
    return '''def test_arithmetic_still_works():
    assert 1 + 1 == 2


def test_strings_still_concatenate():
    assert "a" + "b" == "ab"


def test_lists_still_append():
    xs = []
    xs.append(1)
    assert xs == [1]


def test_dicts_still_store():
    assert {"k": "v"}["k"] == "v"
'''


class ScriptedAuthor:
    """Dispatches on each generator's system prompt; no network, no litellm."""

    def __init__(self, *, pytest_module=sound_pytest_module,
                 truth_md: str = TRUTH_MD) -> None:
        self._pytest_module = pytest_module
        self._truth_md = truth_md

    def __call__(self, system: str, user: str) -> str:
        if _SYSTEM_MARKERS['truth'] in system:
            return self._truth_md
        if _SYSTEM_MARKERS['pytest'] in system:
            return f'```python\n{self._pytest_module(_carved_relpaths(user))}```'
        if _SYSTEM_MARKERS['predicates'] in system:
            return _PREDICATES_JSON
        if _SYSTEM_MARKERS['rubric'] in system:
            return _RUBRIC_JSON
        if _SYSTEM_MARKERS['classify'] in system:
            return '[]'
        if _SYSTEM_MARKERS['judge'] in system:
            return self._verdicts(user)
        raise AssertionError(f'no scripted response for system prompt: {system[:80]!r}')

    @staticmethod
    def _verdicts(user: str) -> str:
        # The rubric loop judges golden and stub with the same criteria; the only
        # way to tell them apart is the code under review, and the carved stub is
        # the only side that still carries the unimplemented marker.
        passed = STUB_MARKER not in user
        ids = re.findall(r'^- ([\w.]+):', user, re.M)
        return json.dumps([
            {'criterion_id': cid,
             'verdict': 'pass' if passed else 'fail',
             'justification': 'scripted fixture verdict'}
            for cid in ids
        ])


def mock_client(**kwargs) -> MockClient:
    return MockClient(responder=ScriptedAuthor(**kwargs))


def sound_mock_client() -> MockClient:
    """Entry point for `TASKGEN_VERIFIER_FAKE_CLIENT`."""
    return mock_client()


def non_discriminating_mock_client() -> MockClient:
    """Entry point that forces the soundness floor to abort the bundle."""
    return mock_client(pytest_module=always_pass_pytest_module)
