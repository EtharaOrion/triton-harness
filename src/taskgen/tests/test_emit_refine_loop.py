"""The bounded refine loop: what it retries, what it refuses, what it never sends.

The loop is the one place where a model's answer gets a SECOND chance, so every
test here is about a limit rather than a feature. Three properties carry the
slice and each is proved by making its opposite observable:

  * BOUNDED -- the cap, the dedupe gate and the two wall-clock budgets are each
    proved by scripting MORE answers than the loop is allowed to take. An
    over-called `ScriptedResolver` raises, so "there was no fourth ask" is a
    failure the test cannot miss rather than an assertion it might forget.
  * EVIDENCE-DRIVEN -- the repair message the resolver receives is asserted to be
    the exact diagnostic the failed attempt produced, not a paraphrase. A loop
    that retried with an unchanged prompt would pass a call-count test and fail
    every one of these.
  * LEAK-SAFE -- the repair channel is the only new path from the repo back to
    the model, so it is asserted to carry scrubbed build output and nothing that
    names a file in the repository.

Offline and hermetic throughout: the resolver is a scripted queue, docker is a
scripted callable, and the clock is a dial the test turns. No container, no
model, no socket, and no sleeping to prove a timeout.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from taskgen import depplan, emit, refine
from taskgen import measure as M
from taskgen.depplan import DepPlan
from taskgen.env_resolver import Refuse, ResolveRefused
from taskgen.langs import base as B
from taskgen.langs import c as C
from taskgen.tests.test_emit_resolve_env import (  # noqa: F401  (imported fixtures)
    FAKE_DIGEST,
    MEASURE_JSON,
    c_repo,
    plan,
)

GOOD_PLAN = C.C_MEASURE_DEP_PLAN
OTHER_PLAN = dataclasses.replace(GOOD_PLAN, apt_packages=('libpq-dev',))
THIRD_PLAN = dataclasses.replace(GOOD_PLAN, apt_packages=('libffi-dev',))

#: The same environment written two ways: canonicalization sorts `apt_packages`,
#: so these are two spellings of ONE plan and must share a digest.
TWO_APT = dataclasses.replace(GOOD_PLAN, apt_packages=('libffi-dev', 'libpq-dev'))
TWO_APT_RESPELT = dataclasses.replace(GOOD_PLAN, apt_packages=('libpq-dev', ' libffi-dev'))

#: Rejected by `depplan.validate`: c is provisioned by cmake or make, never npm.
INVALID_PLAN = dataclasses.replace(GOOD_PLAN, package_manager='npm')

LINKER_ERROR = "undefined reference to `xs_gc_mark'"


def fake_lock(expected: int = 3, graded: tuple[str, ...] = ('u::a', 'u::b', 'u::c')):
    if expected < 1:
        return {'expected': expected, 'graded': list(graded)}
    return M.build_lock(
        lang='c', scope='folder', expected=expected, graded=graded,
        floor_mode='equality', fingerprint_sha256={},
        repo_sha256='11' * 32, intact_image='harbor-base:local',
        intact_image_digest=FAKE_DIGEST,
    )


class ScriptedResolver:
    """A queue of answers, and a transcript of the repair messages it was asked with.

    Running past the end of the queue is an AssertionError, not a StopIteration
    or a silent repeat: "the loop asked a fourth time" is the exact bug the cap
    exists to prevent, so it has to be impossible to script around.
    """

    def __init__(self, *answers: DepPlan | Refuse) -> None:
        self._answers = list(answers)
        self.repairs: list[str | None] = []

    def __call__(self, *, lang: str, repo: Path, base_image: str,
                 repair: str | None = None) -> DepPlan | Refuse:
        assert self._answers, (
            f'the resolver was asked {len(self.repairs) + 1} times but only '
            f'{len(self.repairs)} answers were scripted'
        )
        self.repairs.append(repair)
        return self._answers.pop(0)

    @property
    def calls(self) -> int:
        return len(self.repairs)


class ScriptedMeasure:
    """docker, minus docker: one scripted outcome per candidate plan.

    An outcome that is an exception is raised, so a `MeasureError` reaches the
    loop exactly as `measure.measure` would deliver it. `builds` is the record
    that proves a refused plan was never built.
    """

    def __init__(self, *outcomes, clock=None, cost: float = 0.0) -> None:
        self._outcomes = list(outcomes)
        self._clock = clock
        self._cost = cost
        self.builds: list[DepPlan] = []

    def __call__(self, dep_plan: DepPlan):
        assert self._outcomes, 'the fake runner was asked to build an unscripted plan'
        self.builds.append(dep_plan)
        if self._clock is not None:
            self._clock.advance(self._cost)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class ScriptedClock:
    """Monotonic time as a dial the test turns, so a timeout costs no seconds."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def run(resolver, measure_cb, *, repo: Path, **kwargs) -> refine.RefinedEnv:
    return refine.refine_dep_plan(
        resolver=resolver, lang='c', repo=repo, base_image='harbor-base:local',
        build_and_measure=measure_cb, echo=lambda *_: None, **kwargs,
    )


# ------------------------------------------------------------- the bound ----


def test_the_cap_is_one_resolve_plus_two_repairs():
    assert refine.RESOLVE_ATTEMPT_CAP == 3


def test_the_budgets_are_named_and_the_attempt_fits_inside_the_total():
    assert refine.RESOLVE_TOTAL_BUDGET_S == 900.0
    assert refine.RESOLVE_ATTEMPT_BUDGET_S == 600.0
    assert refine.RESOLVE_ATTEMPT_BUDGET_S < refine.RESOLVE_TOTAL_BUDGET_S


# ------------------------------------------------------- the happy path -----


def test_a_plan_that_builds_first_time_costs_exactly_one_resolve(tmp_path):
    resolver = ScriptedResolver(GOOD_PLAN)
    measure_cb = ScriptedMeasure(fake_lock())

    refined = run(resolver, measure_cb, repo=tmp_path)

    assert resolver.calls == 1
    assert resolver.repairs == [None]
    assert refined.dep_plan == depplan.canonicalize(GOOD_PLAN)
    assert refined.attempts == 1
    assert refined.expected == 3
    assert refined.graded == ('u::a', 'u::b', 'u::c')
    assert measure_cb.builds == [depplan.canonicalize(GOOD_PLAN)]


# ------------------------------------------------------------ the repairs ---


def test_a_build_failure_is_fed_back_verbatim_and_the_second_plan_wins(tmp_path):
    resolver = ScriptedResolver(GOOD_PLAN, OTHER_PLAN)
    measure_cb = ScriptedMeasure(
        M.MeasureError('build failed', stderr=LINKER_ERROR), fake_lock(),
    )

    refined = run(resolver, measure_cb, repo=tmp_path)

    assert resolver.calls == 2
    assert resolver.repairs == [None, LINKER_ERROR]
    assert refined.dep_plan == depplan.canonicalize(OTHER_PLAN)
    assert refined.attempts == 2
    assert measure_cb.builds == [
        depplan.canonicalize(GOOD_PLAN), depplan.canonicalize(OTHER_PLAN),
    ]


def test_a_schema_rejection_is_repaired_without_ever_building_it(tmp_path):
    resolver = ScriptedResolver(INVALID_PLAN, GOOD_PLAN)
    measure_cb = ScriptedMeasure(fake_lock())

    refined = run(resolver, measure_cb, repo=tmp_path)

    assert resolver.calls == 2
    repair = resolver.repairs[1]
    assert repair is not None
    assert "cannot be provisioned by 'npm'" in repair
    assert 'cmake, make' in repair
    # The rejected plan cost no container: validation is checked before a build.
    assert measure_cb.builds == [depplan.canonicalize(GOOD_PLAN)]
    assert refined.dep_plan == depplan.canonicalize(GOOD_PLAN)


def test_a_suite_that_collects_nothing_is_repaired_not_pinned(tmp_path):
    resolver = ScriptedResolver(GOOD_PLAN, OTHER_PLAN)
    measure_cb = ScriptedMeasure(fake_lock(expected=0, graded=()), fake_lock())

    refined = run(resolver, measure_cb, repo=tmp_path)

    assert resolver.repairs == [None, refine.NO_COLLECT_REPAIR]
    assert refined.dep_plan == depplan.canonicalize(OTHER_PLAN)
    assert refined.expected == 3


# ---------------------------------------------------------- the refusals ----


def test_three_failures_refuse_and_there_is_no_fourth_ask(tmp_path):
    resolver = ScriptedResolver(GOOD_PLAN, OTHER_PLAN, THIRD_PLAN)
    measure_cb = ScriptedMeasure(
        M.MeasureError('a', stderr='error one'),
        M.MeasureError('b', stderr='error two'),
        M.MeasureError('c', stderr='error three'),
    )

    with pytest.raises(ResolveRefused) as excinfo:
        run(resolver, measure_cb, repo=tmp_path)

    assert resolver.calls == refine.RESOLVE_ATTEMPT_CAP == 3
    assert len(measure_cb.builds) == 3
    assert 'exhausted 3 attempts' in excinfo.value.reason
    assert 'error three' in excinfo.value.reason


def test_an_identical_second_plan_is_refused_without_a_second_build(tmp_path):
    resolver = ScriptedResolver(GOOD_PLAN, GOOD_PLAN, GOOD_PLAN)
    measure_cb = ScriptedMeasure(M.MeasureError('build failed', stderr=LINKER_ERROR))

    with pytest.raises(ResolveRefused) as excinfo:
        run(resolver, measure_cb, repo=tmp_path)

    assert refine.NO_PROGRESS_REASON in excinfo.value.reason
    assert resolver.calls == 2
    assert len(measure_cb.builds) == 1


def test_a_plan_re_proposed_in_a_different_spelling_is_still_no_progress(tmp_path):
    """Dedupe is on the DIGEST, so respelling one environment is not progress."""
    assert TWO_APT != TWO_APT_RESPELT
    assert depplan.dep_plan_digest(TWO_APT) == depplan.dep_plan_digest(TWO_APT_RESPELT)

    resolver = ScriptedResolver(TWO_APT, TWO_APT_RESPELT)
    measure_cb = ScriptedMeasure(M.MeasureError('build failed', stderr=LINKER_ERROR))

    with pytest.raises(ResolveRefused, match='no progress'):
        run(resolver, measure_cb, repo=tmp_path)

    assert len(measure_cb.builds) == 1


def test_a_refusal_on_the_first_ask_is_never_retried(tmp_path):
    reason = 'tests need a postgres service, impossible under --network=none'
    resolver = ScriptedResolver(Refuse(reason=reason), GOOD_PLAN)
    measure_cb = ScriptedMeasure()

    with pytest.raises(ResolveRefused) as excinfo:
        run(resolver, measure_cb, repo=tmp_path)

    assert excinfo.value.reason == reason
    assert resolver.calls == 1
    assert measure_cb.builds == []


def test_a_refusal_on_a_repair_stops_the_loop_too(tmp_path):
    resolver = ScriptedResolver(GOOD_PLAN, Refuse(reason='no apt route exists'))
    measure_cb = ScriptedMeasure(M.MeasureError('build failed', stderr=LINKER_ERROR))

    with pytest.raises(ResolveRefused, match='no apt route exists'):
        run(resolver, measure_cb, repo=tmp_path)

    assert resolver.calls == 2
    assert len(measure_cb.builds) == 1


# ----------------------------------------------------------- the budgets ----


def test_the_total_budget_refuses_before_asking_again(tmp_path):
    clock = ScriptedClock()
    resolver = ScriptedResolver(GOOD_PLAN, OTHER_PLAN)
    measure_cb = ScriptedMeasure(
        M.MeasureError('slow build', stderr=LINKER_ERROR), clock=clock, cost=1000.0,
    )

    with pytest.raises(ResolveRefused) as excinfo:
        run(resolver, measure_cb, repo=tmp_path, clock=clock,
            total_budget_s=900.0, attempt_budget_s=None)

    assert 'wall-clock budget exceeded' in excinfo.value.reason
    assert resolver.calls == 1
    assert len(measure_cb.builds) == 1


def test_the_per_attempt_budget_refuses_before_asking_again(tmp_path):
    clock = ScriptedClock()
    resolver = ScriptedResolver(GOOD_PLAN, OTHER_PLAN)
    measure_cb = ScriptedMeasure(
        M.MeasureError('slow build', stderr=LINKER_ERROR), clock=clock, cost=700.0,
    )

    with pytest.raises(ResolveRefused) as excinfo:
        run(resolver, measure_cb, repo=tmp_path, clock=clock,
            total_budget_s=900.0, attempt_budget_s=600.0)

    assert 'per-attempt wall-clock budget exceeded' in excinfo.value.reason
    assert resolver.calls == 1


def test_a_slow_attempt_that_SUCCEEDED_is_kept_rather_than_refused(tmp_path):
    """A stopwatch never throws away a measurement that already worked."""
    clock = ScriptedClock()
    resolver = ScriptedResolver(GOOD_PLAN)
    measure_cb = ScriptedMeasure(fake_lock(), clock=clock, cost=5000.0)

    refined = run(resolver, measure_cb, repo=tmp_path, clock=clock,
                  total_budget_s=900.0, attempt_budget_s=600.0)

    assert refined.dep_plan == depplan.canonicalize(GOOD_PLAN)


# -------------------------------------------------------- the leak boundary --


def test_a_repair_message_carries_the_error_and_none_of_the_paths(tmp_path):
    """The repair channel is scrubbed build output, never a file in the repo."""
    repo = tmp_path / 'c-xs'
    repo.mkdir()
    noisy = '\n'.join([
        f'{repo}/src/runtime/gc.c:41: error: implicit declaration of xs_mark',
        '/opt/harbor/repo/tests/test_gc.c:12: error: expected an int',
        LINKER_ERROR,
    ])
    resolver = ScriptedResolver(GOOD_PLAN, OTHER_PLAN)
    measure_cb = ScriptedMeasure(M.MeasureError('failed', stderr=noisy), fake_lock())

    run(resolver, measure_cb, repo=repo)

    repair = resolver.repairs[1]
    assert repair == LINKER_ERROR
    assert str(repo) not in repair
    assert '/opt/harbor/repo' not in repair
    assert 'gc.c' not in repair and 'test_gc.c' not in repair


def test_a_diagnostic_that_is_nothing_but_paths_becomes_opaque_not_empty(tmp_path):
    """An empty repair would render as NO repair and silently re-ask unchanged."""
    repo = tmp_path / 'c-xs'
    repo.mkdir()
    all_paths = f'{repo}/src/runtime/gc.c:41: error: implicit declaration'
    resolver = ScriptedResolver(GOOD_PLAN, OTHER_PLAN)
    measure_cb = ScriptedMeasure(M.MeasureError('failed', stderr=all_paths), fake_lock())

    run(resolver, measure_cb, repo=repo)

    assert resolver.repairs[1] == refine._OPAQUE_REPAIR
    assert str(repo) not in refine._OPAQUE_REPAIR


@dataclasses.dataclass
class Answer:
    """Mutable `text` on purpose: `ResolverClient`'s response protocol is not read-only."""

    text: str


class RecordingClient:
    """Records the user prompt it was asked with; always answers the good plan."""

    def __init__(self) -> None:
        self.asks: list[str] = []

    def complete(self, system: str, user: str, *, max_tokens: int = 8192,
                 reasoning_effort: str | None = None) -> Answer:
        self.asks.append(user)
        return Answer(text=depplan.to_canonical_json(GOOD_PLAN))


def test_a_repaired_ask_differs_from_the_first_by_exactly_the_repair_block():
    """The other half of the thread: the loop supplies the text, the prompt places it.

    Asserting the two prompts are equal once the Repair section is removed is
    stronger than asserting the message appears: a repair that also perturbed
    the manifests or the test paths would pass a substring check and quietly
    change what the model was answering about.
    """
    from taskgen import env_resolver as R

    client = RecordingClient()
    for repair in (None, LINKER_ERROR):
        R.resolve_dep_plan(
            client, lang='c', toolchain_capabilities=['gcc 13.3.0'],
            manifest_files={'Makefile': 'all:\n'},
            test_paths=['tests/unit/gc_test.c'], framework='make', repair=repair,
        )

    plain, repaired = client.asks
    assert '# Repair' not in plain
    assert '# Repair' in repaired and LINKER_ERROR in repaired
    without_repair = '\n\n'.join(
        section for section in repaired.split('\n\n')
        if not section.startswith('# Repair')
    )
    assert without_repair == plain


# ---------------------------------------------- the loop, wired into emit ----


class FailThenBuildRunner:
    """docker for the emit-level tests: build #1 explodes, build #2 works."""

    instances: list[FailThenBuildRunner] = []

    def __init__(self, echo=print):
        self.builds: list[str] = []
        self.runs: list[str] = []
        self.removed: list[str] = []
        FailThenBuildRunner.instances.append(self)

    def image_digest(self, image: str) -> str:
        return FAKE_DIGEST

    def build_with_contexts(self, *, image, dockerfile, context, contexts):
        self.builds.append(Path(dockerfile).read_text())
        if len(self.builds) == 1:
            raise RuntimeError(LINKER_ERROR)
        return 'BUILD OK'

    def run(self, image: str, script: str) -> str:
        self.runs.append(image)
        return MEASURE_JSON

    def remove_image(self, image: str) -> None:
        self.removed.append(image)


@pytest.fixture(autouse=True)
def fake_docker(monkeypatch):
    FailThenBuildRunner.instances = []
    monkeypatch.setattr('taskgen.verify.DockerRunner', FailThenBuildRunner)
    return FailThenBuildRunner


def lock_path(out: Path, plan_obj) -> Path:
    return out.resolve() / '_staging' / plan_obj.staging_key / M.LOCK_FILENAME


def test_the_lock_pins_the_repaired_plan_and_never_the_failed_one(plan, tmp_path):
    out = tmp_path / 'out'
    resolver = ScriptedResolver(GOOD_PLAN, OTHER_PLAN)

    pinned = emit._measure_and_pin(
        plan, B.get('c'), out, echo=lambda *_: None,
        resolve_env=True, resolver=resolver,
    )

    assert resolver.calls == 2
    repair = resolver.repairs[1]
    assert repair is not None and 'undefined reference' in repair

    runner = FailThenBuildRunner.instances[-1]
    assert len(runner.builds) == 2
    assert 'libpq-dev' in runner.builds[1] and 'libpq-dev' not in runner.builds[0]

    env = json.loads(lock_path(out, plan).read_text())[M.LOCK_ENV_BLOCK]
    assert env['dep_plan_digest'] == depplan.dep_plan_digest(OTHER_PLAN)
    assert env['dep_plan_digest'] != depplan.dep_plan_digest(GOOD_PLAN)
    assert pinned.graded.expected == 3


def test_a_refine_that_refuses_writes_no_lock_at_all(plan, tmp_path):
    out = tmp_path / 'out'
    resolver = ScriptedResolver(GOOD_PLAN, GOOD_PLAN)

    with pytest.raises(ResolveRefused, match='no progress'):
        emit._measure_and_pin(
            plan, B.get('c'), out, echo=lambda *_: None,
            resolve_env=True, resolver=resolver,
        )

    assert not lock_path(out, plan).exists()
    assert len(FailThenBuildRunner.instances[-1].builds) == 1
