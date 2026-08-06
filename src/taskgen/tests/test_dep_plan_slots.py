"""The lang-aware pre-render gate: what a plan must SAY before it is built.

`depplan.validate` is lang-agnostic on purpose -- it closes the enums and the
metacharacter gate for six languages at once, so it structurally cannot know
that c's gap interpolates `build_flags['make_version']` into a Dockerfile
comment. A real model exploited exactly that gap: it returned a SCHEMA-VALID
plan with no `make_version`, `render_gap` raised `LangError` halfway through
rendering, and because the refine loop only caught `MeasureError` the crash
escaped and aborted the whole generate.

Three properties close it, and each is proved by making its opposite observable:

  * ENFORCED -- `CPlugin.validate_dep_plan` rejects every plan `render_gap`
    could not render. Proved slot by slot, from the same enumeration the
    renderer reads, so a slot added to one and not the other fails here.
  * REPAIRED, NOT FATAL -- the rejection is fed back as the next attempt's
    repair message, the same channel `MeasureError.stderr` uses, and the
    rejected plan costs NO docker build. Cap exhaustion is a clean
    `ResolveRefused`, never an uncaught `LangError`.
  * ANNOUNCED -- the same slots are stated in the prompt, so attempt 1 already
    carries them. Leak-safe: slot names and shapes, never a repo body.

Offline and hermetic: no docker, no model, no socket.
"""

from __future__ import annotations

import dataclasses

import pytest

from taskgen import depplan, env_resolver as R, refine, resolver_inputs
from taskgen.env_resolver import ResolveRefused
from taskgen.langs import base as B
from taskgen.langs import c as C

GOOD_PLAN = C.C_MEASURE_DEP_PLAN

#: The live failure, as data: schema-valid, `render_gap`-fatal.
NO_MAKE_VERSION = dataclasses.replace(
    GOOD_PLAN, build_flags=(('link_libraries', '-lm -lpthread -ldl'),),
)


def strip_flag(key: str):
    return dataclasses.replace(
        GOOD_PLAN,
        build_flags=tuple((k, v) for k, v in GOOD_PLAN.build_flags if k != key),
    )


def blank_flag(key: str):
    return dataclasses.replace(
        GOOD_PLAN,
        build_flags=tuple(
            (k, '' if k == key else v) for k, v in GOOD_PLAN.build_flags
        ),
    )


# --------------------------------------------------- the plan the gap needs --


def test_the_canonical_libc_plan_is_still_renderable():
    """The fixed point: c's own environment must pass the gate it introduced."""
    plugin = B.get('c')
    plugin.validate_dep_plan(GOOD_PLAN)
    assert plugin.render_gap(GOOD_PLAN)


def test_the_generic_validator_accepts_the_plan_that_crashed_rendering():
    """The premise of the whole slice: schema-valid is NOT renderable."""
    depplan.validate(NO_MAKE_VERSION)

    with pytest.raises(B.LangError, match='make_version'):
        B.get('c').render_gap(NO_MAKE_VERSION)


@pytest.mark.parametrize('key', C.REQUIRED_BUILD_FLAGS)
def test_a_missing_or_empty_build_flag_is_rejected_by_name(key: str):
    plugin = B.get('c')
    for plan in (strip_flag(key), blank_flag(key)):
        with pytest.raises(B.LangError) as excinfo:
            plugin.validate_dep_plan(plan)
        message = str(excinfo.value)
        assert f'build_flags[{key!r}]' in message
        assert 'must be a non-empty string' in message


@pytest.mark.parametrize('key', C.REQUIRED_TEST_INVOCATION_KEYS)
def test_a_missing_or_empty_test_invocation_key_is_rejected_by_name(key: str):
    plugin = B.get('c')
    missing = dataclasses.replace(
        GOOD_PLAN,
        test_invocation=tuple(
            (k, v) for k, v in GOOD_PLAN.test_invocation if k != key
        ),
    )
    empty = dataclasses.replace(GOOD_PLAN, test_invocation=((key, ()),))
    for plan in (missing, empty):
        with pytest.raises(B.LangError) as excinfo:
            plugin.validate_dep_plan(plan)
        assert f'test_invocation[{key!r}]' in str(excinfo.value)


def test_a_one_component_toolchain_version_cannot_pin_gcc():
    with pytest.raises(B.LangError, match='major.minor'):
        B.get('c').validate_dep_plan(
            dataclasses.replace(GOOD_PLAN, toolchain_version='13')
        )


def test_a_plan_for_another_language_is_not_a_plan_for_this_gap():
    other = dataclasses.replace(GOOD_PLAN, lang='rust', package_manager='cargo')
    with pytest.raises(B.LangError, match='cannot render'):
        B.get('c').validate_dep_plan(other)


def test_the_gate_rejects_what_the_generic_validator_rejects_too():
    """The lang gate is a superset: it never accepts an inadmissible plan."""
    with pytest.raises(depplan.DepPlanError):
        B.get('c').validate_dep_plan(
            dataclasses.replace(GOOD_PLAN, package_manager='npm')
        )


def test_every_slot_the_gate_enforces_is_a_slot_the_renderer_reads():
    """No unannounced rejection: each enforced slot appears in the ask."""
    stated = '\n'.join(C.REQUIRED_PLAN_SLOTS)
    for key in (*C.REQUIRED_BUILD_FLAGS, *C.REQUIRED_TEST_INVOCATION_KEYS):
        assert key in stated
    assert 'toolchain_version' in stated


def test_a_rejection_names_the_slot_and_never_a_repo_body():
    """The message becomes a prompt, so it may carry no source and no path."""
    with pytest.raises(B.LangError) as excinfo:
        B.get('c').validate_dep_plan(NO_MAKE_VERSION)

    message = str(excinfo.value)
    assert 'make_version' in message
    assert '/' not in message
    assert 'xs_gc' not in message and '#include' not in message


def test_the_base_hook_is_a_no_op_so_other_languages_are_untouched():
    for lang in ('python', 'go', 'rust', 'cpp', 'java'):
        assert B.get(lang).validate_dep_plan(GOOD_PLAN) is None
        assert B.get(lang).required_plan_slots == ()


# ----------------------------------------------------------- the loop ------


class SlotResolver:
    """A queue of answers; records the repair message each ask carried."""

    def __init__(self, *answers) -> None:
        self._answers = list(answers)
        self.repairs: list[str | None] = []

    def __call__(self, *, lang, repo, base_image, repair=None):
        assert self._answers, 'the resolver was asked more times than were scripted'
        self.repairs.append(repair)
        return self._answers.pop(0)

    @property
    def calls(self) -> int:
        return len(self.repairs)


class CountingMeasure:
    """docker, minus docker. `builds` is what proves a rejected plan cost nothing."""

    def __init__(self, *outcomes) -> None:
        self._outcomes = list(outcomes)
        self.builds: list[depplan.DepPlan] = []

    def __call__(self, dep_plan):
        assert self._outcomes, 'the fake runner was asked to build an unscripted plan'
        self.builds.append(dep_plan)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def fake_lock(expected: int = 3):
    return {'expected': expected, 'graded': ['u::a', 'u::b', 'u::c']}


def run(resolver, measure_cb, repo, **kwargs):
    return refine.refine_dep_plan(
        resolver=resolver, lang='c', repo=repo, base_image='harbor-base:local',
        build_and_measure=measure_cb, echo=lambda *_: None,
        validate_candidate=B.get('c').validate_dep_plan, **kwargs,
    )


def test_an_unrenderable_plan_is_repaired_and_never_built(tmp_path):
    resolver = SlotResolver(NO_MAKE_VERSION, GOOD_PLAN)
    measure_cb = CountingMeasure(fake_lock())

    refined = run(resolver, measure_cb, tmp_path)

    assert resolver.calls == 2
    repair = resolver.repairs[1]
    assert repair is not None and 'make_version' in repair
    assert 'must be a non-empty string' in repair
    # The whole point: the plan that could not render cost NO docker build.
    assert measure_cb.builds == [depplan.canonicalize(GOOD_PLAN)]
    assert refined.dep_plan == depplan.canonicalize(GOOD_PLAN)
    assert refined.attempts == 2


def test_an_unrepaired_slot_exhausts_the_cap_as_a_clean_refusal(tmp_path):
    """No uncaught LangError, no exit 1: the loop REFUSES, and nothing is built."""
    answers = [
        dataclasses.replace(NO_MAKE_VERSION, apt_packages=(pkg,))
        for pkg in ('libffi-dev', 'libpq-dev', 'libssl-dev')
    ]
    resolver = SlotResolver(*answers)
    measure_cb = CountingMeasure()

    with pytest.raises(ResolveRefused) as excinfo:
        run(resolver, measure_cb, tmp_path)

    assert resolver.calls == refine.RESOLVE_ATTEMPT_CAP == 3
    assert measure_cb.builds == []
    assert 'exhausted 3 attempts' in excinfo.value.reason
    assert 'make_version' in excinfo.value.reason


def test_a_render_time_LangError_past_the_gate_is_repaired_not_fatal(tmp_path):
    """Defence in depth: the loop no longer lets ANY LangError escape it.

    `validate_dep_plan` is meant to make this unreachable, so the test reaches
    it the only honest way -- by letting the build itself raise what
    `render_measure_dockerfile` raises -- and proves the outcome is a repair
    rather than the abort that was observed live.
    """
    boom = B.LangError("the c gap needs build_flags['make_version']")
    resolver = SlotResolver(GOOD_PLAN, dataclasses.replace(GOOD_PLAN, apt_packages=('libpq-dev',)))
    measure_cb = CountingMeasure(boom, fake_lock())

    refined = run(resolver, measure_cb, tmp_path)

    assert resolver.repairs[1] is not None
    assert 'make_version' in resolver.repairs[1]
    assert refined.expected == 3


def test_the_loop_defaults_to_the_generic_gate_when_no_validator_is_injected(tmp_path):
    """`accept_any_plan` keeps every non-c caller on exactly today's behaviour."""
    resolver = SlotResolver(NO_MAKE_VERSION)
    measure_cb = CountingMeasure(fake_lock())

    refined = refine.refine_dep_plan(
        resolver=resolver, lang='c', repo=tmp_path, base_image='harbor-base:local',
        build_and_measure=measure_cb, echo=lambda *_: None,
    )

    assert refined.attempts == 1
    assert len(measure_cb.builds) == 1


# ----------------------------------------------------------- the prompt ----


def test_the_c_prompt_states_the_slots_the_gate_will_enforce():
    prompt = R.build_user_prompt(
        lang='c',
        toolchain_capabilities=list(C.BAKED_CAPABILITIES),
        manifest_files={'Makefile': 'all:\n\t$(CC) -o xs\n'},
        test_paths=['tests/unit/gc_test.c'],
        framework='make',
        required_slots=list(resolver_inputs.required_plan_slots(B.get('c'))),
    )

    assert '# Required plan slots for c' in prompt
    assert 'make_version' in prompt and 'GNU make version' in prompt
    assert 'link_libraries' in prompt
    assert 'test_invocation' in prompt and 'graded suite' in prompt
    assert '4.3' in prompt


def test_the_c_prompt_says_install_commands_do_not_compile_the_graded_sources():
    """The live quality failure, as an assertion on the ASK.

    A model resolved c with `install_commands=[{"tool":"make","args":[]}]`; the
    measure image ran `RUN make`, the build root has no default target, and all
    three refine attempts kept the step. Nothing in the prompt had ever said
    what install_commands are FOR, so the model had no reason to drop it. This
    is guidance, not the answer: the convention is stated, the plan is not.
    """
    prompt = R.build_user_prompt(
        lang='c',
        toolchain_capabilities=list(C.BAKED_CAPABILITIES),
        manifest_files={'Makefile': 'all:\n\t$(CC) -o xs\n'},
        test_paths=['tests/unit/gc_test.c'],
        framework='make',
        required_slots=list(resolver_inputs.required_plan_slots(B.get('c'))),
    )

    assert 'install_commands must NOT compile the graded sources' in prompt
    assert 'USUALLY EMPTY' in prompt
    assert 'never a bare "make" or "make all"' in prompt
    assert 'makes the whole plan unbuildable' in prompt
    assert 'apt_packages lists ONLY system libraries' in prompt


def test_the_system_prompt_states_the_install_versus_build_principle():
    """The general convention lives once, in the language-agnostic prompt."""
    system = R.SYSTEM_PROMPT

    assert 'INSTALL PREPARES, THE TEST COMMAND BUILDS' in system
    assert 'NOT for compiling the repository' in system
    assert 'USUALLY EMPTY' in system
    assert '`make` / `make all` / `cmake --build .`' in system
    assert 'makes the whole plan unbuildable' in system
    assert 'DELETE that step rather than patching around it' in system


def test_the_install_guidance_names_no_repo_and_no_carved_path():
    """Leak-safe: build CONVENTIONS, never this repo's answer."""
    blob = R.SYSTEM_PROMPT + '\n'.join(C.REQUIRED_PLAN_SLOTS)

    for secret in (
        'c-xs', 'src/runtime', 'xs_', '#include', 'test-conformance',
        'test-regression', 'test-unit', 'BearSSL', 'msgpack',
    ):
        assert secret not in blob


def test_the_guidance_does_not_dictate_the_canonical_plan():
    """Convergence must be EARNED. The prompt never spells c's own answer."""
    assert GOOD_PLAN.install_commands == ()
    for verbatim in ('-lm -lpthread -ldl', C.TEST_COMMAND):
        assert verbatim not in R.SYSTEM_PROMPT


def test_the_canonical_c_plan_still_satisfies_the_guidance_it_now_states():
    """The plan the guidance points at is the one the gap already renders."""
    plugin = B.get('c')
    assert GOOD_PLAN.install_commands == ()
    assert GOOD_PLAN.apt_packages == ()
    plugin.validate_dep_plan(GOOD_PLAN)
    assert 'RUN make\n' not in plugin.render_gap(GOOD_PLAN) + '\n'


def test_a_bare_make_install_step_is_what_the_guidance_warns_about():
    """The rejected shape stays SCHEMA-valid -- only guidance can prevent it.

    Proving this keeps the fix honest: no validator was quietly tightened to
    ban a `make` step, because a repo whose plan genuinely needs one must still
    be expressible. The prompt is the only thing that changed.
    """
    bare_make = dataclasses.replace(
        GOOD_PLAN, install_commands=(depplan.InstallCommand(tool='make', args=()),),
    )
    depplan.validate(bare_make)
    B.get('c').validate_dep_plan(bare_make)
    assert 'RUN make' in B.get('c').render_gap(bare_make)


def test_the_stated_slots_carry_no_repo_source():
    """The slot list comes off the PLUGIN, so no repo byte can ride along."""
    slots = resolver_inputs.required_plan_slots(B.get('c'))

    assert slots == tuple(sorted(slots))
    for slot in slots:
        assert '\n' not in slot
        assert '#include' not in slot and 'xs_' not in slot
        assert 'src/runtime' not in slot and 'tests/' not in slot


def test_a_language_with_no_declared_slots_adds_no_section():
    """The prompt for python is byte-identical to the one it sent before."""
    kwargs = {
        'lang': 'python',
        'toolchain_capabilities': ['python 3.12'],
        'manifest_files': {'pyproject.toml': '[project]\n'},
        'test_paths': ['tests/test_a.py'],
        'framework': 'pytest',
    }
    assert R.build_user_prompt(**kwargs) == R.build_user_prompt(
        **kwargs, required_slots=list(resolver_inputs.required_plan_slots(B.get('python'))),
    )
    assert '# Required plan slots' not in R.build_user_prompt(**kwargs)


def test_the_slot_section_is_order_insensitive():
    slots = list(resolver_inputs.required_plan_slots(B.get('c')))
    kwargs = {
        'lang': 'c',
        'toolchain_capabilities': list(C.BAKED_CAPABILITIES),
        'manifest_files': {'Makefile': 'all:\n'},
        'test_paths': ['tests/unit/gc_test.c'],
        'framework': 'make',
    }
    assert R.build_user_prompt(**kwargs, required_slots=slots) == R.build_user_prompt(
        **kwargs, required_slots=list(reversed(slots)),
    )
