"""All eleven context builders run, differ, and leak nothing."""

from __future__ import annotations

import pytest

from taskgen.carve_fn import carve_function
from taskgen.contexts import CONTEXT_TYPES, ContextInputs, build_body
from taskgen.nodeids import linked_nodeids


@pytest.fixture(scope='module')
def inputs(repo, target):
    carve = carve_function(repo, target)
    return ContextInputs.build(
        repo=repo,
        repo_name='python-a2a-python',
        target=target,
        carve=carve,
        nodeids=linked_nodeids(repo, target),
        budget=100000,
        seed=0,
    )


@pytest.fixture(scope='module')
def bodies(inputs):
    return {ct: build_body(ct, inputs) for ct in CONTEXT_TYPES}


def test_there_are_eleven_context_types():
    assert len(CONTEXT_TYPES) == 11
    assert CONTEXT_TYPES == (
        'no_context', 'callee_func', 'callee_sig', 'caller_func', 'caller_sig',
        'in_file', 'project', 'bm25', 'embedding', 'mix', 'repo_coder',
    )


def test_every_builder_runs(bodies):
    assert set(bodies) == set(CONTEXT_TYPES)


def test_no_context_inlines_nothing(bodies):
    intro, blocks, stats = bodies['no_context']
    assert blocks == []
    assert stats['context_tokens'] == 0
    assert intro.startswith('## Pre-loaded context: none')


def test_callee_func_inlines_the_callee_source(bodies):
    _intro, blocks, _stats = bodies['callee_func']
    assert blocks
    assert 'HasField' in ''.join(blocks)


def test_callee_sig_has_signatures_but_not_the_callee_body(bodies):
    _intro, blocks, _stats = bodies['callee_sig']
    joined = ''.join(blocks)
    assert blocks
    assert 'HasField' in joined
    assert ': ...' in joined


#: A frozen production caller of the target: it is not a callee and not in the
#: target's file, so it discriminates caller_* from every condition we had.
CALLER_QUALNAME = 'DefaultRequestHandlerV2.on_message_send'
CALLER_FILE = 'src/a2a/server/request_handlers/default_request_handler_v2.py'
CALLER_DECL = 'async def on_message_send('


def test_the_target_knows_its_callers(target):
    quals = target.caller_qualnames()
    assert CALLER_QUALNAME in quals
    assert quals, 'the frozen target has in-repo callers; an empty set is a bug'
    assert set(quals).isdisjoint(target.callee_qualnames())


def test_caller_func_inlines_a_real_callers_whole_body(bodies):
    _intro, blocks, stats = bodies['caller_func']
    joined = ''.join(blocks)
    assert blocks
    assert CALLER_FILE in joined
    assert CALLER_DECL in joined
    assert 'apply_history_length' in joined, 'the call site itself must be present'
    assert stats['callers_resolved'] == stats['callers_inlined'] == len(blocks)


def test_caller_sig_has_the_caller_signature_but_not_its_body(bodies):
    _intro, blocks, _stats = bodies['caller_sig']
    joined = ''.join(blocks)
    assert blocks
    assert CALLER_DECL in joined
    assert ': ...' in joined
    assert 'apply_history_length' not in joined, 'a signature must not carry a call site'


def test_only_the_caller_conditions_inline_the_caller(bodies):
    """The discriminator: caller_* is not a rename of a condition we already had."""
    for ct in ('no_context', 'callee_func', 'callee_sig', 'in_file'):
        _intro, blocks, _stats = bodies[ct]
        assert CALLER_DECL not in ''.join(blocks), ct


def test_caller_ordering_is_sorted_and_stable(target, repo):
    from taskgen.select import _sort_key

    keys = [_sort_key(repo, c) for c in target.callers]
    assert keys == sorted(keys)
    assert len(keys) == len(set(keys)), 'a caller must not be reported twice'


def test_caller_context_never_carries_the_carved_body(bodies, inputs):
    from taskgen.contexts import assert_no_leakage

    assert inputs.tripwires
    body = inputs.carve.original_text.split(inputs.carve.signature, 1)[1]
    for ct in ('caller_func', 'caller_sig'):
        intro, blocks, _stats = bodies[ct]
        document = intro + ''.join(blocks)
        assert_no_leakage(document, inputs.tripwires)
        for line in body.splitlines():
            stripped = line.strip()
            if len(stripped) >= 24:
                assert stripped not in document, f'{ct} leaked {stripped!r}'


def test_in_file_inlines_the_stubbed_target_file(bodies, inputs):
    _intro, blocks, _stats = bodies['in_file']
    joined = ''.join(blocks)
    assert 'raise NotImplementedError' in joined
    assert inputs.target_relpath in joined


def test_retrieval_builders_return_ranked_chunk_blocks(bodies):
    for ct in ('bm25', 'embedding', 'mix', 'repo_coder'):
        _intro, blocks, stats = bodies[ct]
        assert blocks, ct
        assert stats['chunks_retrieved'] > 0, ct
        assert blocks[0].startswith('### ['), ct


def test_project_inlines_whole_files(bodies):
    _intro, blocks, stats = bodies['project']
    assert stats['files_inlined_whole'] > 0
    assert blocks[0].startswith('### `')


def test_every_builder_respects_the_budget(bodies, inputs):
    for ct, (_intro, _blocks, stats) in bodies.items():
        assert stats['context_tokens'] <= inputs.context_budget, ct


def test_bodies_differ_across_context_types(bodies):
    rendered = {ct: ''.join(b) for ct, (_i, b, _s) in bodies.items()}
    assert len(set(rendered.values())) >= 10


def test_no_builder_leaks_the_carved_body(bodies, inputs):
    from taskgen.contexts import assert_no_leakage

    assert inputs.tripwires, 'no tripwires built -- the leakage check is vacuous'
    for ct, (intro, blocks, _stats) in bodies.items():
        assert_no_leakage(intro + ''.join(blocks), inputs.tripwires)


def test_builders_are_deterministic(inputs):
    for ct in CONTEXT_TYPES:
        assert build_body(ct, inputs) == build_body(ct, inputs), ct


def test_unknown_context_type_is_rejected(inputs):
    with pytest.raises(ValueError):
        build_body('not-a-context-type', inputs)
