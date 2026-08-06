"""The bounded validate-and-refine loop: ask, build, collect, repair, or REFUSE.

`env_resolver` asks a model for an environment ONCE; `depplan` says what an
environment may be; this module is the part that does not believe either of them
until a container has agreed. It is the executable half of the plan's section 6:
a resolution is a HYPOTHESIS, and the only evidence that promotes it to a fact is
the repo's own suite collecting under it.

WHY A LOOP AT ALL. LLM output is non-deterministic even at temperature 0, so the
answer to "is this plan right" cannot be "the model said so". It is "it built and
it collected". A first answer that misses one apt package is not a failed
resolution, it is an under-specified one, and the cheapest repair is to hand the
model the exact linker error and ask for the SMALLEST edit. That is what
`build_user_prompt(repair=...)` was built for.

WHY IT IS BOUNDED, AND HOW. Three things stop it, all named constants, because
an unbounded repair loop is a way to spend an afternoon converging on a plan
nobody should ship:

  * `RESOLVE_ATTEMPT_CAP` -- one resolve plus two repairs. A model that has not
    fixed a build in two informed tries is not converging, it is guessing.
  * the DEDUPE gate -- a repaired plan whose `dep_plan_digest` was already tried
    is proof of exactly zero progress, and is refused WITHOUT rebuilding it. The
    digest, not the object: two spellings of one environment are one environment.
  * `RESOLVE_TOTAL_BUDGET_S` / `RESOLVE_ATTEMPT_BUDGET_S` -- wall clock, read off
    an INJECTED `clock`. Injected so the budget is testable without waiting for
    it, which is the only way a timeout ever gets a test at all.

NO IO LIVES HERE. The resolver, the build/measure step and the clock are all
arguments; this module imports nothing that can open a socket or a container.
That is what lets the whole loop -- including the docker-shaped half -- run in
the offline suite, and it is why `emit` can call it without importing docker.

LEAK-SAFETY. A repair message is the ONE new channel from the repo back to the
model, so it is the one that has to be narrow. Every message is passed through
`measure.scrub_stderr` with the repo path added to the redaction list, so what
crosses is bounded, deterministic, path-free build output -- never a source or
test body, which is the answer the benchmark hides.

REFUSE, NEVER DEGRADE. Every exit that is not a validated, collected plan is a
`ResolveRefused`. Intermediate plans are never returned, so nothing that failed
can reach a lock.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from . import depplan
from . import measure as measure_mod
from .depplan import DepPlan
from .env_resolver import EnvResolver, Refuse, ResolveRefused
from .langs.base import LangError

__all__ = [
    'MeasureAttempt',
    'NO_COLLECT_REPAIR',
    'NO_PROGRESS_REASON',
    'PlanValidator',
    'RESOLVE_ATTEMPT_BUDGET_S',
    'RESOLVE_ATTEMPT_CAP',
    'RESOLVE_TOTAL_BUDGET_S',
    'RefinedEnv',
    'accept_any_plan',
    'refine_dep_plan',
]

#: Total resolver calls: one initial resolution plus at most two repairs. Small
#: on purpose -- each attempt is a full image build plus a full suite run, and a
#: model that cannot fix a named linker error in two tries is not converging.
RESOLVE_ATTEMPT_CAP = 3

#: Wall clock for the WHOLE loop, seconds. Generous: three cold builds of a
#: compiled repo is minutes, not seconds, and a budget that trips on a slow
#: machine would refuse tasks that are perfectly resolvable.
RESOLVE_TOTAL_BUDGET_S = 900.0

#: Wall clock for ONE attempt, seconds. Catches the hang the total budget would
#: only catch after it had already eaten every remaining attempt's time.
RESOLVE_ATTEMPT_BUDGET_S = 600.0

#: A build that succeeded and a suite that collected nothing is not a working
#: environment, it is a silent floor of zero. Phrased as the model sees it.
NO_COLLECT_REPAIR = 'tests did not collect (expected=0)'

#: The dedupe refusal. Re-proposing a tried plan means the repair message taught
#: the model nothing, and a fourth identical build would teach US nothing.
NO_PROGRESS_REASON = 'refine made no progress: identical DepPlan re-proposed'

#: What goes to the model when the scrubber ate every line of a diagnostic --
#: a repo-path-only stderr, say. Better than an empty repair, which
#: `build_user_prompt` would render as NO repair and silently re-ask.
_OPAQUE_REPAIR = (
    'the previous plan failed to build or collect; the diagnostic was withheld '
    'because every line of it named a path inside the repository'
)


class MeasureAttempt(Protocol):
    """Build one candidate plan and measure the intact suite under it.

    Returns the lock `measure.measure` returns; raises `measure.MeasureError`
    when the image would not build or the suite would not collect. The loop
    knows nothing else about it, which is what keeps docker on the caller's side
    of this seam.
    """

    def __call__(self, dep_plan: DepPlan) -> dict: ...


class PlanValidator(Protocol):
    """Reject a candidate this language could not RENDER, before it is built.

    `depplan.validate` is lang-agnostic and cannot know that c's gap
    interpolates `build_flags['make_version']`, so a schema-valid plan can still
    be unrenderable. `langs.base.LangPlugin.validate_dep_plan` is what emit
    passes here; the loop only needs it to raise `LangError` with a message a
    model can act on, which is why it arrives as a callable rather than as a
    plugin -- this module stays free of the plugin registry, as it is of docker.

    Positional-only: the loop passes the candidate by position, so a plugin
    method is bindable here whatever it calls its own parameter.
    """

    def __call__(self, plan: DepPlan, /) -> None: ...


def accept_any_plan(plan: DepPlan, /) -> None:
    """The default `PlanValidator`: the generic schema gate is the only gate."""
    return None


@dataclass(frozen=True)
class RefinedEnv:
    """The one plan that earned its way out of the loop, and its evidence.

    `lock` and `graded` come from the SAME attempt as `dep_plan` -- they are the
    measurement of that plan, not of a sibling -- so a caller cannot pin a plan
    against another plan's collected ids.
    """

    dep_plan: DepPlan
    lock: dict
    graded: tuple[str, ...]
    expected: int
    attempts: int


def _repair_message(text: str, repo: Path) -> str:
    """A diagnostic, reduced to what may cross back to the model.

    `scrub_stderr` is reused rather than re-implemented: it already bounds the
    tail, drops every line naming the mounted repo, and keeps the deterministic
    signal lines. The repo's own path is added to the needles because a build run
    outside a container quotes THAT path, not the mount point.
    """
    scrubbed = measure_mod.scrub_stderr(text, extra_relpaths=(str(repo),))
    return scrubbed.strip() or _OPAQUE_REPAIR


def _check_budget(elapsed: float, budget: float | None, what: str) -> None:
    """Refuse the moment a budget is past. No retry: time does not come back."""
    if budget is not None and elapsed > budget:
        raise ResolveRefused(
            f'{what}wall-clock budget exceeded: {elapsed:.1f}s of {budget:.1f}s '
            'spent resolving the environment. A slower answer is not a better '
            'one, so the run refuses rather than spending the rest of the '
            'generate on a plan that has not built yet'
        )


def refine_dep_plan(
    *,
    resolver: EnvResolver,
    lang: str,
    repo: Path,
    base_image: str,
    build_and_measure: MeasureAttempt,
    validate_candidate: PlanValidator = accept_any_plan,
    echo: Callable[[str], object] = print,
    clock: Callable[[], float] = time.monotonic,
    attempt_cap: int = RESOLVE_ATTEMPT_CAP,
    total_budget_s: float | None = RESOLVE_TOTAL_BUDGET_S,
    attempt_budget_s: float | None = RESOLVE_ATTEMPT_BUDGET_S,
) -> RefinedEnv:
    """Resolve, build, collect; repair on failure; return the plan that worked.

    The signal order is FAIL-FAST and fixed, cheapest and most certain first:

      1. `Refuse`            -- the model declined on purpose. A refusal is an
                                ANSWER, not an error, so it is never repaired:
                                re-asking a model that just explained why it
                                cannot resolve is how a wrong plan gets born.
      2. canonicalize/validate -- the answer is not an admissible plan. Costs no
                                container, so it is checked before one is built,
                                and the validator's own text is the repair.
      3. dedupe              -- an already-tried plan. Checked before the build,
                                because rebuilding it would burn the budget to
                                re-derive a failure we have already seen.
      4. `validate_candidate` -- admissible, but not RENDERABLE by this
                                language: a slot its gap interpolates is missing
                                or empty. Lang-aware, so `depplan.validate`
                                structurally cannot catch it, and checked here
                                rather than at render time so the answer costs
                                no image and stays repairable.
      5. build/collect       -- `MeasureError.stderr`, already scrubbed once by
                                `measure`, is the repair. A `LangError` escaping
                                the renderer here is repaired the same way: (4)
                                should have caught it, and a gap between the two
                                is a bug to be repaired, never a crash that
                                aborts the generate.
      6. `expected < 1`      -- built, ran, collected nothing.

    Only (1), the dedupe gate, the budgets and a cap exhaustion leave this
    function as a `ResolveRefused`; only a plan that cleared ALL of them leaves
    it as a `RefinedEnv`.
    """
    started = clock()
    repair: str | None = None
    tried: dict[str, int] = {}
    last_failure = ''

    for attempt in range(1, attempt_cap + 1):
        _check_budget(clock() - started, total_budget_s, '')

        resolution = resolver(
            lang=lang, repo=repo, base_image=base_image, repair=repair,
        )
        if isinstance(resolution, Refuse):
            raise ResolveRefused(resolution.reason)

        try:
            candidate = depplan.canonicalize(resolution)
            digest = depplan.dep_plan_digest(candidate)
        except depplan.DepPlanError as exc:
            last_failure = _repair_message(str(exc), repo)
            repair = last_failure
            echo(f'resolve attempt {attempt}/{attempt_cap}: uncanonical plan; repairing')
            continue

        if digest in tried:
            raise ResolveRefused(
                f'{NO_PROGRESS_REASON} (attempt {attempt} re-proposed the plan '
                f'from attempt {tried[digest]}, {digest[:12]}). The repair '
                f'message was:\n{repair}'
            )
        tried[digest] = attempt

        try:
            depplan.validate(candidate)
        except depplan.DepPlanError as exc:
            last_failure = _repair_message(str(exc), repo)
            repair = last_failure
            echo(f'resolve attempt {attempt}/{attempt_cap}: plan rejected by validate; repairing')
            continue

        try:
            validate_candidate(candidate)
        except LangError as exc:
            last_failure = _repair_message(str(exc), repo)
            repair = last_failure
            echo(
                f'resolve attempt {attempt}/{attempt_cap}: plan is not renderable '
                f'by the {lang} gap; repairing without building it'
            )
            continue

        attempt_started = clock()
        try:
            lock = dict(build_and_measure(candidate))
        except measure_mod.MeasureError as exc:
            last_failure = _repair_message(exc.stderr or str(exc), repo)
            repair = last_failure
            echo(f'resolve attempt {attempt}/{attempt_cap}: measure failed; repairing')
        except LangError as exc:
            last_failure = _repair_message(str(exc), repo)
            repair = last_failure
            echo(
                f'resolve attempt {attempt}/{attempt_cap}: the {lang} gap refused '
                'to render the plan; repairing'
            )
        else:
            expected = int(lock.get('expected', 0) or 0)
            if expected >= 1:
                echo(
                    f'resolved env on attempt {attempt}/{attempt_cap}: '
                    f'{candidate.package_manager} toolchain '
                    f'{candidate.toolchain_version} ({digest[:12]}), expected={expected}'
                )
                return RefinedEnv(
                    dep_plan=candidate,
                    lock=lock,
                    graded=tuple(str(s) for s in lock.get('graded', ()) or ()),
                    expected=expected,
                    attempts=attempt,
                )
            last_failure = NO_COLLECT_REPAIR
            repair = last_failure
            echo(f'resolve attempt {attempt}/{attempt_cap}: {NO_COLLECT_REPAIR}; repairing')

        _check_budget(clock() - attempt_started, attempt_budget_s, 'per-attempt ')

    raise ResolveRefused(
        f'refine exhausted {attempt_cap} attempts without an environment that '
        f'builds and collects. The last failure was:\n{last_failure}'
    )
