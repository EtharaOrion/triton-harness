"""All nine context builders run, differ, and leak nothing."""

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


def test_there_are_nine_context_types():
    assert len(CONTEXT_TYPES) == 9
    assert set(CONTEXT_TYPES) == {
        'no_context', 'callee_func', 'callee_sig', 'in_file',
        'project', 'bm25', 'embedding', 'mix', 'repo_coder',
    }


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
    assert len(set(rendered.values())) >= 8


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
