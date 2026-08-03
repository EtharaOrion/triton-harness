"""Multi-file surviving corpus: no carved body from ANY carved file may leak.

Single-file carve had one place the answer could enter a context -- the target's
own file -- and `contexts.py` closed it by swapping that one file for its stub.
A multi-file carve has N such places, and N-1 of them are invisible to that
design: `bm25`, `embedding`, `mix`, `repo_coder` and `project` all index the
whole repository, so a second carved file whose body is still on disk goes
straight into the prompt with full marks from the retriever.

So the SURVIVING tree is the repo with EVERY carved file replaced by its
skeleton (or removed outright under `--delete-whole-file`), and the tripwire set
is the UNION of the bodies of all of them. The negative controls below are the
part that matters: a test that only checks "no leak was found" passes just as
well when the checker is broken.
"""

from __future__ import annotations

import pytest

from taskgen.carve import build_carve_set
from taskgen.contexts import CONTEXT_TYPES, ContextInputs, LeakageError, build_body
from taskgen.gradedset import collect_carved_functions, derive_graded_set
from taskgen.scope import CarveScope

from .conftest import FROZEN_FILE

SECOND_FILE = 'src/a2a/utils/error_handlers.py'
THIRD_FILE = 'src/a2a/utils/version_validator.py'
CARVED = (FROZEN_FILE, SECOND_FILE, THIRD_FILE)


@pytest.fixture(scope='module')
def multi(repo):
    """A three-file skeleton carve, wired exactly as emit wires it."""
    carve = build_carve_set(
        repo, CarveScope.FILE, carved_relpaths=CARVED, language='python',
    )
    funcs = collect_carved_functions('python', repo, carve.carved_relpaths,
                                     package_base='src/')
    graded = derive_graded_set('python', funcs, carve.carved_relpaths)
    inp = ContextInputs.build(
        repo=repo, repo_name='python-a2a-python', target=carve.target_view,
        carve=carve, nodeids=graded.selectors, budget=100000, seed=0,
    )
    return carve, inp


@pytest.fixture(scope='module')
def multi_bodies(multi):
    _carve, inp = multi
    return {ct: build_body(ct, inp) for ct in CONTEXT_TYPES}


# --------------------------------------------------------------------------
# the surviving corpus
# --------------------------------------------------------------------------


def test_carve_set_holds_every_carved_file(multi):
    carve, _inp = multi
    assert carve.carved_relpaths == tuple(sorted(CARVED))
    assert set(carve.overlay) == set(CARVED)
    assert set(carve.originals) == set(CARVED)
    assert carve.deleted_relpaths == ()


def test_every_carved_file_enters_the_corpus_as_its_skeleton(multi):
    carve, inp = multi
    surviving = dict(inp.surviving)
    for rel in CARVED:
        assert surviving[rel] == carve.overlay[rel], rel
        assert surviving[rel] != carve.originals[rel], rel
        assert 'raise NotImplementedError' in surviving[rel], rel


def test_skeletons_keep_imports_and_signatures(multi):
    carve, _inp = multi
    skeleton = carve.overlay[SECOND_FILE]
    assert 'import' in skeleton
    assert 'def ' in skeleton


def test_tripwires_are_the_union_over_every_carved_file(multi):
    _carve, inp = multi
    origins = {rel for rel, _lineno in inp.tripwires.values()}
    assert origins == set(CARVED), (
        f'tripwires cover {sorted(origins)}; a carved file with no tripwire is a '
        'carved file whose leak cannot be detected'
    )


# --------------------------------------------------------------------------
# no leak, and the checker that proves it is not vacuous
# --------------------------------------------------------------------------


def test_no_builder_leaks_a_body_from_any_carved_file(multi_bodies, multi):
    _carve, inp = multi
    assert inp.tripwires
    from taskgen.contexts import assert_no_leakage

    for ct, (intro, blocks, _stats) in multi_bodies.items():
        assert_no_leakage(intro + ''.join(blocks), inp.tripwires)


@pytest.mark.parametrize('rel', CARVED)
def test_negative_control_a_spliced_body_line_from_any_file_is_caught(multi, rel):
    """Splice a real carved line back in: the checker MUST refuse it."""
    from taskgen.contexts import assert_no_leakage

    _carve, inp = multi
    lines = [t for t, (origin, _l) in inp.tripwires.items() if origin == rel]
    assert lines, f'{rel} contributed no tripwire, so this control proves nothing'
    with pytest.raises(LeakageError):
        assert_no_leakage(f'## context\n{lines[0]}\n', inp.tripwires)


def test_no_carved_body_line_survives_anywhere_in_the_corpus(multi):
    """The corpus IS the prompt material, so it is leak-checked like a document.

    Checked with the production line semantics rather than a substring scan: a
    tripwire is a normalised LINE, so a surviving line that merely contains the
    text (a test asserting on the same message, say) is not a leak.
    """
    from taskgen.contexts import assert_no_leakage

    _carve, inp = multi
    for rel, text in inp.surviving:
        assert_no_leakage(text, inp.tripwires)
        del rel


# --------------------------------------------------------------------------
# the builders that needed generalising
# --------------------------------------------------------------------------


def test_in_file_inlines_the_primary_files_skeleton(multi_bodies, multi):
    carve, _inp = multi
    _intro, blocks, _stats = multi_bodies['in_file']
    joined = ''.join(blocks)
    assert carve.primary_relpath in joined
    assert 'raise NotImplementedError' in joined


def test_callees_living_in_a_carved_file_are_not_inlined(multi_bodies, multi):
    """A callee inside the carve set is carved: inlining it hands over the answer."""
    carve, _inp = multi
    for ct in ('callee_func', 'callee_sig'):
        _intro, blocks, _stats = multi_bodies[ct]
        for block in blocks:
            header = block.splitlines()[0]
            for rel in carve.carved_relpaths:
                assert rel not in header, f'{ct} inlined a callee from carved {rel}'


def test_project_and_retrieval_still_produce_content(multi_bodies):
    for ct in ('project', 'bm25', 'embedding', 'mix', 'repo_coder'):
        _intro, blocks, _stats = multi_bodies[ct]
        assert blocks, ct


def test_all_nine_builders_still_run(multi_bodies):
    assert set(multi_bodies) == set(CONTEXT_TYPES)


# --------------------------------------------------------------------------
# delete-whole-file
# --------------------------------------------------------------------------


@pytest.fixture(scope='module')
def deleted(repo):
    carve = build_carve_set(
        repo, CarveScope.FILE, carved_relpaths=(SECOND_FILE,), language='python',
        delete_whole_file=True,
    )
    funcs = collect_carved_functions('python', repo, carve.carved_relpaths,
                                     package_base='src/')
    graded = derive_graded_set('python', funcs, carve.carved_relpaths)
    return carve, ContextInputs.build(
        repo=repo, repo_name='python-a2a-python', target=carve.target_view,
        carve=carve, nodeids=graded.selectors, budget=100000, seed=0,
    )


def test_a_deleted_file_is_absent_from_the_surviving_corpus(deleted):
    carve, inp = deleted
    assert carve.deleted_relpaths == (SECOND_FILE,)
    assert SECOND_FILE not in {rel for rel, _ in inp.surviving}


def test_a_deleted_files_whole_content_becomes_tripwires(deleted):
    _carve, inp = deleted
    origins = {rel for rel, _l in inp.tripwires.values()}
    assert origins == {SECOND_FILE}
    assert len(inp.tripwires) > 5, 'a deleted file should yield many tripwires'


def test_deleted_file_builders_run_and_do_not_leak(deleted):
    from taskgen.contexts import assert_no_leakage

    _carve, inp = deleted
    for ct in CONTEXT_TYPES:
        intro, blocks, _stats = build_body(ct, inp)
        assert_no_leakage(intro + ''.join(blocks), inp.tripwires)
