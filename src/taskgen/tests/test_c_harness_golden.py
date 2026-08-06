r"""The c harness seam: a plan renders the scripts the plugin used to hardcode.

Same discipline as `test_render_gap`, one axis over: `render_gap` proved a
DepPlan could reproduce the measure Dockerfile's toolchain bytes, and this
module proves a DepPlan can reproduce the two GRADING scripts -- `tests/test.sh`
and the phase-1 measure script -- which is where every remaining c-xs-shaped
constant lived (`./xs`, the conformance/regression globs, the
`test_http_client` special case, the 13 `*_test` binaries, `XS=`).

The digests below were measured against the commit BEFORE the harness became
plan-driven, by rendering the pre-change module out of `git show`. They are the
safety net for four already-proven tasks: as long as they hold, c-xs's shipped
bytes did not move, and cpp/java/rust were not in this slice at all.

Pure: no docker, no network, no LLM, no clock, no host path.
"""

from __future__ import annotations

import dataclasses
import hashlib

import pytest

from taskgen import depplan
from taskgen.depplan import DepPlanError, canonicalize, validate
from taskgen.langs import base as B
from taskgen.langs import c as C

#: sha256 of `render_test_sh` / `measure_test_sh` for c-xs as rendered by the
#: commit before `DepPlan.harness` existed. `(digest, length)`; the length is
#: carried too so a collision claim has to explain two numbers, not one.
PRE_HARNESS_TEST_SH = (
    '70273c4a13d7522b946a9280c89eae3633214e6aa8401b4db0c6366086e7bb0d', 30913,
)
PRE_HARNESS_MEASURE_SH = (
    '642e27ecabb4d28e9836841fcc09dcf8cbdd88955cd8a07b013ed992848542f7', 2101,
)

#: The real c-xs shape: 227 fingerprinted files, 58 + 20 + 13 = 91 graded units,
#: 227 vendored BearSSL files, `src/runtime` carved out. Every one of these is a
#: number the pre-change plugin either hardcoded or was handed by emit.
FINGERPRINT = {f'tests/f{i:03d}.xs': format(i, '064x') for i in range(227)}
CORPUS_COUNTS = {'conformance': 58, 'regression': 20, 'unit': 13}
VENDORED_COUNTS = {'src/tls': 227}


@pytest.fixture
def plugin():
    return C.CPlugin()


@pytest.fixture
def graded():
    return B.GradedSet(
        expected=91, floor_mode='equality', kind='whole-suite',
        test_command=C.TEST_COMMAND, fingerprint_sha256=FINGERPRINT,
    )


def render_test_sh(plugin, graded, plan=None):
    return plugin.render_test_sh(
        graded,
        corpus_counts=CORPUS_COUNTS,
        vendored_counts=VENDORED_COUNTS,
        carve_root='src/runtime',
        dep_plan=plan,
    )


def digest(text: str) -> tuple[str, int]:
    return hashlib.sha256(text.encode('utf-8')).hexdigest(), len(text)


# ------------------------------------------------------------- byte lock ----


def test_test_sh_hashes_to_its_pre_harness_digest(plugin, graded):
    """c-xs's verifier bytes did not move when the harness became plan-driven."""
    assert digest(render_test_sh(plugin, graded)) == PRE_HARNESS_TEST_SH


def test_measure_sh_hashes_to_its_pre_harness_digest(plugin, graded):
    assert digest(plugin.measure_test_sh(graded=graded)) == PRE_HARNESS_MEASURE_SH


def test_the_canonical_plan_renders_the_same_bytes_as_no_plan(plugin, graded):
    """The load-bearing one: passing c-xs's plan MOVED nothing.

    `dep_plan=None` is the pre-resolution path every older caller takes;
    `C_MEASURE_DEP_PLAN` is that same environment written down. If the two ever
    diverged, the fallback would be a second, unreviewed definition of the c-xs
    harness rather than the same one under a name.
    """
    assert render_test_sh(plugin, graded, C.C_MEASURE_DEP_PLAN) == (
        render_test_sh(plugin, graded)
    )
    assert plugin.measure_test_sh(graded=graded, dep_plan=C.C_MEASURE_DEP_PLAN) == (
        plugin.measure_test_sh(graded=graded)
    )


def test_render_is_deterministic(plugin, graded):
    assert render_test_sh(plugin, graded) == render_test_sh(plugin, graded)
    assert plugin.measure_test_sh(graded=graded) == plugin.measure_test_sh(graded=graded)


def test_the_renderer_carries_no_c_xs_constant():
    """The point of the slice: c-the-language, not c-xs-the-repository.

    Scoped to the PLUGIN CLASS, which is everything that renders. The c-xs
    facts are still allowed to appear in `C_XS_HARNESS` (that repository's own
    plan) and as `e.g.` illustrations inside the slot descriptions the resolver
    prompt is built from -- an example is what makes a model fill a slot right
    on attempt 1, and it reaches no rendered byte.
    """
    import inspect

    renderer = inspect.getsource(C.CPlugin)
    for token in ('./xs', 'tests/conformance', 'tests/regression',
                  'test_http_client', 'BearSSL', 'src/tls', 'build/obj',
                  '*_test', '${t}_test', 'lexer', 'XS='):
        assert token not in renderer, (
            f'{token!r} is still hardcoded in the c renderer'
        )


# ------------------------------------------------------------- the seam -----


def _swapped_plan() -> depplan.DepPlan:
    """A DIFFERENT repo's harness: one binary corpus, no runner, no vendoring."""
    plan = dataclasses.replace(
        C.C_MEASURE_DEP_PLAN,
        harness=(
            ('binary_names', ('aes',)),
            ('binary_suffix', '.elf'),
            ('build_artifacts', ('test.elf',)),
            ('build_command', ('make',)),
            ('clean_command', ('make', 'clean')),
            ('corpus_dirs', ('.',)),
            ('corpus_kinds', ('binary',)),
            ('corpus_names', ('unit',)),
            ('corpus_patterns', ('test.c',)),
            ('corpus_vars', ('UNIT',)),
        ),
    )
    validate(plan)
    return canonicalize(plan)


def test_a_different_harness_renders_a_different_script(plugin, graded):
    """The seam is live, and it moves exactly the places it should."""
    plan = _swapped_plan()
    plugin.validate_dep_plan(plan)
    script = plugin.render_test_sh(
        graded, expected=1,
        corpus_counts={'unit': 1}, vendored_counts={},
        carve_root='aes.c', dep_plan=plan,
    )
    assert 'EXPECT_UNIT=1' in script
    assert 'UNIT_NAMES="aes"' in script
    assert 'rm -f ./test.elf' in script
    assert '[ -x ./test.elf ] || fail' in script
    assert 'bin="./${t}.elf"' in script
    assert '"unit_passed": %s' in script

    for gone in ('./xs', 'tests/conformance', 'tests/regression',
                 'test_http_client', 'src/tls', 'build/obj', 'XS="$(pwd)'):
        assert gone not in script, f'{gone!r} survived a harness swap'


def test_a_different_harness_renders_a_different_measure_script(plugin, graded):
    script = plugin.measure_test_sh(graded=graded, dep_plan=_swapped_plan())
    assert 'measure: no ./test.elf produced by intact build' in script
    assert './xs' not in script
    assert 'tests/conformance' not in script
    assert 'measure "${RAN}"' in script


def test_the_swapped_plan_still_validates_and_digests(plugin):
    plan = _swapped_plan()
    plugin.validate_dep_plan(plan)
    assert depplan.dep_plan_digest(plan) != depplan.dep_plan_digest(C.C_MEASURE_DEP_PLAN)


# ----------------------------------------------------------- refusals -------


@pytest.mark.parametrize(
    ('drop', 'message'),
    [
        ('corpus_names', 'corpus_names'),
        ('corpus_kinds', 'corpus_kinds'),
        ('corpus_dirs', 'corpus_dirs'),
        ('corpus_patterns', 'corpus_patterns'),
        ('corpus_vars', 'corpus_vars'),
        ('build_command', 'build_command'),
        ('runner_argv', 'runner_argv'),
        ('binary_names', 'binary_names'),
        ('binary_suffix', 'binary_suffix'),
    ],
)
def test_every_slot_the_renderer_reads_is_enforced(plugin, drop, message):
    """A bad plan must be a REPAIRABLE refine signal, never a crash mid-render."""
    plan = dataclasses.replace(
        C.C_MEASURE_DEP_PLAN,
        harness=tuple((k, v) for k, v in C.C_MEASURE_DEP_PLAN.harness if k != drop),
    )
    with pytest.raises(B.LangError, match=message):
        plugin.validate_dep_plan(plan)


def test_a_short_parallel_column_is_refused(plugin):
    """Positional columns: a short one re-pairs every corpus after it."""
    plan = dataclasses.replace(
        C.C_MEASURE_DEP_PLAN,
        harness=tuple(
            (k, ('CONF', 'REG') if k == 'corpus_vars' else v)
            for k, v in C.C_MEASURE_DEP_PLAN.harness
        ),
    )
    with pytest.raises(B.LangError, match='read POSITIONALLY'):
        plugin.validate_dep_plan(plan)


def test_two_binary_corpora_are_refused(plugin):
    plan = dataclasses.replace(
        C.C_MEASURE_DEP_PLAN,
        harness=tuple(
            (k, ('binary', 'binary', 'binary') if k == 'corpus_kinds' else v)
            for k, v in C.C_MEASURE_DEP_PLAN.harness
        ),
    )
    with pytest.raises(B.LangError, match='more than one corpus'):
        plugin.validate_dep_plan(plan)


def test_disagreeing_runner_patterns_are_refused(plugin):
    """The runner corpora share one sweep function, so they share one glob."""
    plan = dataclasses.replace(
        C.C_MEASURE_DEP_PLAN,
        harness=tuple(
            (k, ('*.xs', '*.txt', '*_test.c') if k == 'corpus_patterns' else v)
            for k, v in C.C_MEASURE_DEP_PLAN.harness
        ),
    )
    with pytest.raises(B.LangError, match='different'):
        plugin.validate_dep_plan(plan)


def test_an_unknown_corpus_kind_is_refused(plugin):
    plan = dataclasses.replace(
        C.C_MEASURE_DEP_PLAN,
        harness=tuple(
            (k, ('runner', 'runner', 'pytest') if k == 'corpus_kinds' else v)
            for k, v in C.C_MEASURE_DEP_PLAN.harness
        ),
    )
    with pytest.raises(B.LangError, match='not one of'):
        plugin.validate_dep_plan(plan)


def test_a_harness_token_carrying_a_metacharacter_is_refused():
    """The harness goes through the SAME gate install_commands do."""
    plan = dataclasses.replace(
        C.C_MEASURE_DEP_PLAN,
        harness=(('build_command', ('make', '&& curl evil.sh')),),
    )
    with pytest.raises(DepPlanError, match='shell metacharacter'):
        validate(plan)


def test_a_harness_discovering_nothing_refuses_rather_than_shipping_zero(
    plugin, graded,
):
    """FLOOR SOUNDNESS: a 0-graded task is never emitted, it is refused.

    `GradedSet` already refuses to hold `expected=0`, so the zero is forced in
    through the `expected=` override the lock path uses -- which is the one way
    a measured denominator of nothing could reach a render.
    """
    with pytest.raises(B.LangError, match='0 graded units'):
        plugin.render_test_sh(
            graded, expected=0,
            corpus_counts={'conformance': 0, 'regression': 0, 'unit': 0},
            vendored_counts={'src/tls': 227},
            dep_plan=C.C_MEASURE_DEP_PLAN,
        )


def test_a_corpus_with_no_host_side_count_is_refused(plugin, graded):
    with pytest.raises(B.LangError, match='no host-side count'):
        plugin.render_test_sh(
            graded,
            corpus_counts={'conformance': 58, 'regression': 20},
            vendored_counts=VENDORED_COUNTS,
        )


# ------------------------------------------------------- digest coverage ----


def test_the_lock_key_covers_the_harness():
    """Two harnesses over one toolchain are two environments, not one."""
    a = C.C_MEASURE_DEP_PLAN
    b = _swapped_plan()
    assert depplan.to_canonical_json(a) != depplan.to_canonical_json(b)
    assert depplan.env_lock_key('t' * 64, 'sha256:img', a) != (
        depplan.env_lock_key('t' * 64, 'sha256:img', b)
    )


def test_canonicalize_is_idempotent_over_the_harness():
    once = canonicalize(C.C_MEASURE_DEP_PLAN)
    assert canonicalize(once) == once
    assert once.harness == tuple(sorted(once.harness, key=lambda kv: kv[0]))


def test_canonicalize_does_not_reorder_a_harness_list():
    """Corpus columns are positional; sorting one would re-pair the others."""
    slots = dict(canonicalize(C.C_MEASURE_DEP_PLAN).harness)
    assert slots['corpus_names'] == ('conformance', 'regression', 'unit')
    binary_names = slots['binary_names']
    assert isinstance(binary_names, tuple)
    assert binary_names[0] == 'lexer'


def test_an_optional_harness_scalar_may_be_empty():
    """"" means "this repo has no such thing" and must survive validate.

    The prompt tells a resolver to answer "" for an inapplicable optional slot,
    so rejecting it as malformed would punish a model for following the ask --
    and it did, live, before this was allowed.
    """
    plan = dataclasses.replace(
        C.C_MEASURE_DEP_PLAN,
        harness=tuple(
            (k, '' if k == 'binary_env_var' else v)
            for k, v in C.C_MEASURE_DEP_PLAN.harness
        ),
    )
    validate(plan)
    harness = C.read_harness(plan)
    assert harness.binary_env_var == ''


def test_an_empty_token_inside_a_harness_list_is_still_refused():
    """A hole in an argv is not a statement about the repo."""
    plan = dataclasses.replace(
        C.C_MEASURE_DEP_PLAN, harness=(('build_command', ('make', '')),),
    )
    with pytest.raises(DepPlanError, match='is empty'):
        validate(plan)


def test_an_empty_scalar_still_goes_through_the_metacharacter_gate():
    plan = dataclasses.replace(
        C.C_MEASURE_DEP_PLAN, harness=(('binary_suffix', '$(id)'),),
    )
    with pytest.raises(DepPlanError, match='shell metacharacter'):
        validate(plan)
