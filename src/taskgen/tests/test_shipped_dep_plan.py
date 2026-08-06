r"""A pinned `DepPlan` drives the SHIPPED image, and moves nothing else in it.

`test_render_dockerfile_golden.py` freezes the plan-free bytes. This module is
the other half: it proves the seam is actually LIVE -- that a plan carrying apt
packages and an install command reaches the shipped Dockerfile -- and that it
reaches EXACTLY the toolchain slot and no other line.

That second half is the one worth testing hard. The shipped Dockerfile is the
file `verify` runs RED/GREEN and the layer archaeology against, so the blocks a
resolver must never be able to touch are named here one by one: the leak gate,
the plugin's `pre_leakgate_blocks`, the carve-receipt and oracle-absence
asserts, and the dependency stage. A seam that provisioned correctly while
dropping any of those would build a green image that ships the answer.

Every tool below is drawn from the language's OWN `TOOL_ALLOWLIST` -- `make` for
c and cpp, `gradle` for java, `cargo` for rust. A plan naming a tool outside it
never reaches a render at all, so one shared install command would have tested
the allowlist instead of the seam.

Pure: no docker, no network, no LLM, no clock, no host path.
"""

from __future__ import annotations

import dataclasses

import pytest

from taskgen import langs
from taskgen.depplan import DepPlan, DepPlanError, InstallCommand
from taskgen.langs import base as B
from taskgen.langs import c as C
from taskgen.langs import cpp as CPP
from taskgen.langs import java as JAVA
from taskgen.langs import rust as RUST

#: Two apt packages, empty in every canonical plan, so whatever they render is
#: unambiguously new bytes that came from the PLAN rather than from the plugin.
APT = ('libpq-dev', 'libssl-dev')
APT_LINE = (
    'RUN apt-get update && apt-get install -y --no-install-recommends '
    'libpq-dev libssl-dev && rm -rf /var/lib/apt/lists/*'
)

#: `(lang, repo, canonical plan, an ALLOWLISTED install command, its rendered
#: line, the measure-only comment that must not ship)`. The canonical plan
#: describes the environment the plugin already hardcodes, which is what makes
#: it the right baseline to perturb.
CASES = (
    ('c', 'c-xs', C.C_MEASURE_DEP_PLAN,
     InstallCommand(tool='make', args=('-C', 'vendor', 'all')),
     'RUN make -C vendor all', C.MEASURE_NO_WARM_COMMENT),
    ('cpp', 'cpp-Rux', CPP.CPP_MEASURE_DEP_PLAN,
     InstallCommand(tool='make', args=('-C', 'vendor', 'all')),
     'RUN make -C vendor all', CPP.MEASURE_NO_WARM_COMMENT),
    ('java', 'java-tamboui', JAVA.JAVA_MEASURE_DEP_PLAN,
     InstallCommand(tool='gradle', args=('--offline', 'dependencies')),
     'RUN gradle --offline dependencies', JAVA.MEASURE_NO_WARM_COMMENT),
    ('rust', 'rust-spacewasm', RUST.RUST_MEASURE_DEP_PLAN,
     InstallCommand(tool='cargo', args=('fetch', '--locked')),
     'RUN cargo fetch --locked', RUST.MEASURE_NO_WARM_COMMENT),
)
IDS = [name for name, *_ in CASES]

#: The same table minus the measure-only comment, which most tests do not read.
BASIC = [case[:5] for case in CASES]
BASIC_ARGS = ('lang', 'repo', 'plan', 'command', 'line')


def provisioning(plan: DepPlan, command: InstallCommand) -> DepPlan:
    return dataclasses.replace(plan, apt_packages=APT, install_commands=(command,))


def shipped(lang: str, repo: str, dep_plan: DepPlan | None = None) -> str:
    return langs.get(lang).render_dockerfile(
        B.EnvSpec(repo_name=repo), dep_plan=dep_plan,
    )


# ------------------------------------------------------- the seam is LIVE ---


@pytest.mark.parametrize(BASIC_ARGS, BASIC, ids=IDS)
def test_a_provisioning_plan_changes_the_shipped_bytes(lang, repo, plan, command, line):
    """The whole point: a resolved plan reaches the image all eleven entries build."""
    default = shipped(lang, repo)
    planned = shipped(lang, repo, provisioning(plan, command))

    assert planned != default
    assert APT_LINE in planned and APT_LINE not in default
    assert line in planned and line not in default


@pytest.mark.parametrize(BASIC_ARGS, BASIC, ids=IDS)
def test_the_canonical_plan_changes_nothing_at_all(lang, repo, plan, command, line):
    """The plan describing today's environment renders today's image, byte for byte.

    This is what makes the substitution reviewable: the seam is a no-op on the
    plan that means "what is already there", so any diff a REAL plan produces is
    the plan's doing and not the seam's.
    """
    assert shipped(lang, repo, plan) == shipped(lang, repo)


@pytest.mark.parametrize(BASIC_ARGS, BASIC, ids=IDS)
def test_the_plan_lands_in_exactly_one_block(lang, repo, plan, command, line):
    """Everything before and after the toolchain slot is untouched.

    Splitting the default rendering at its LAST hardcoded gap and re-joining the
    halves around the planned gap must reproduce the planned rendering exactly.
    That is a stronger claim than "the leak gate is still present": it says no
    OTHER byte moved, whitespace included. The LAST occurrence is the slot --
    java's warm stage renders the same toolchain earlier, and that stage is
    deliberately not plan-driven.
    """
    plugin = langs.get(lang)
    default = shipped(lang, repo)
    planned_gap = plugin._gap_body(provisioning(plan, command))

    head, sep, tail = default.rpartition(plugin.toolchain())
    assert sep, 'the hardcoded toolchain must appear verbatim in the default render'
    assert head + planned_gap + tail == shipped(lang, repo, provisioning(plan, command))


@pytest.mark.parametrize(
    ('lang', 'repo', 'plan', 'command', 'line', 'note'), CASES, ids=IDS,
)
def test_the_measure_only_note_never_ships(lang, repo, plan, command, line, note):
    """`render_gap`'s tail is measure-only and must not follow the plan across.

    It states that nothing was warmed, which is a fact about the MEASURE image.
    Asserted on the exact per-language sentence rather than on the phrase: c,
    cpp and rust also carry `dep_warm()`'s own "no warmed dependencies for this
    language" line, which is a different claim in a different block.
    """
    planned = shipped(lang, repo, provisioning(plan, command))

    assert note in langs.get(lang).render_gap(provisioning(plan, command))
    assert note not in planned
    assert note not in shipped(lang, repo)


# ----------------------- the scaffolding a plan may not reach, one by one ---


@pytest.mark.parametrize(BASIC_ARGS, BASIC, ids=IDS)
def test_the_leak_gate_survives_a_plan(lang, repo, plan, command, line):
    planned = shipped(lang, repo, provisioning(plan, command))

    assert 'bash /usr/local/bin/leakscan.sh' in planned
    assert 'TRIPWIRE_FILE=/tmp/.harbor-tripwires' in planned
    assert '--mount=type=bind,from=trip,' in planned


@pytest.mark.parametrize(BASIC_ARGS, BASIC, ids=IDS)
def test_the_carve_receipt_and_oracle_absence_asserts_survive_a_plan(
    lang, repo, plan, command, line,
):
    plugin = langs.get(lang)
    planned = shipped(lang, repo, provisioning(plan, command))

    assert 'find / -name carve_receipt.json' in planned
    assert 'test ! -e /opt/harbor-tooling/carve.py' in planned
    assert f'test ! -e {plugin.solution_mount}' in planned


@pytest.mark.parametrize(BASIC_ARGS, BASIC, ids=IDS)
def test_the_pre_leakgate_blocks_survive_a_plan(lang, repo, plan, command, line):
    plugin = langs.get(lang)
    planned = shipped(lang, repo, provisioning(plan, command))

    for block in plugin.pre_leakgate_blocks(B.EnvSpec(repo_name=repo)):
        assert block in planned


@pytest.mark.parametrize(BASIC_ARGS, BASIC, ids=IDS)
def test_the_dependency_stage_survives_a_plan_and_never_becomes_a_from(
    lang, repo, plan, command, line,
):
    """Invariant 7 -- `FROM warm` would inherit every layer the warm build touched.

    java is the language this bites on: it is the only one with a real warm
    stage, so it is the only one where a plan could have displaced a COPY.

    Matched per INSTRUCTION rather than as a substring: the block's own fixed
    comment says "`FROM warm` would inherit every layer", so a substring test
    would fail on the very prose explaining the rule.
    """
    plugin = langs.get(lang)
    planned = shipped(lang, repo, provisioning(plan, command))

    instructions = [
        stripped for stripped in (raw.strip() for raw in planned.splitlines())
        if stripped and not stripped.startswith('#')
    ]
    assert not [line for line in instructions if line.startswith('FROM warm')]
    assert plugin.dep_warm() in planned


@pytest.mark.parametrize(BASIC_ARGS, BASIC, ids=IDS)
def test_the_invariant_assert_still_runs_over_the_planned_bytes(
    lang, repo, plan, command, line,
):
    """A plan smuggling the oracle mount in is caught by the SAME assert.

    The tool is allowlisted and the plan is otherwise valid, so nothing earlier
    in the pipeline rejects it -- the refusal can only come from
    `_assert_dockerfile_invariants` re-reading the finished text, which is the
    proof that it still runs on the plan-driven path.
    """
    plugin = langs.get(lang)
    sneaky = dataclasses.replace(plan, install_commands=(
        dataclasses.replace(command, args=(plugin.solution_mount,)),
    ))

    with pytest.raises(B.LangError, match='oracle must reach the container'):
        shipped(lang, repo, sneaky)


# -------------------------------------------------- the gate before bytes ---


@pytest.mark.parametrize(BASIC_ARGS, BASIC, ids=IDS)
def test_a_plan_missing_a_rendered_slot_is_refused_not_rendered(
    lang, repo, plan, command, line,
):
    """`validate_dep_plan` runs FIRST, so a bad plan is a message, not a Dockerfile."""
    stripped = dataclasses.replace(plan, build_flags=())

    with pytest.raises(B.LangError):
        shipped(lang, repo, stripped)


@pytest.mark.parametrize(BASIC_ARGS, BASIC, ids=IDS)
def test_a_plan_for_another_language_is_refused(lang, repo, plan, command, line):
    """Handed a plan that is perfectly VALID -- for someone else.

    Swapping only the `lang` field would trip `depplan.validate` on the package
    manager and never reach the plugin, so the foreign plan is another
    language's canonical one, which validates on its own terms.
    """
    foreign = RUST.RUST_MEASURE_DEP_PLAN if lang != 'rust' else C.C_MEASURE_DEP_PLAN

    with pytest.raises(B.LangError):
        shipped(lang, repo, foreign)


@pytest.mark.parametrize(BASIC_ARGS, BASIC, ids=IDS)
def test_the_shipped_render_inherits_the_metacharacter_gate(
    lang, repo, plan, command, line,
):
    """A plan is tokens, never a command line -- the gate is depplan's, not the render's."""
    injected = dataclasses.replace(plan, apt_packages=('libpq-dev; rm -rf /',))

    with pytest.raises(DepPlanError):
        shipped(lang, repo, injected)


@pytest.mark.parametrize(
    ('lang', 'repo', 'manager'),
    [('python', 'python-a2a-python', 'pip'), ('go', 'go-multigres', 'go')],
)
def test_a_parser_backed_lang_refuses_a_plan_loudly(lang, repo, manager):
    """python and go have no gap, so a plan must raise rather than be ignored.

    Silently dropping it is the dangerous outcome: the caller would believe the
    environment was applied to an image that was built without it.
    """
    plan = dataclasses.replace(
        C.C_MEASURE_DEP_PLAN, lang=lang, package_manager=manager,
    )

    with pytest.raises(NotImplementedError, match='_gap_body'):
        shipped(lang, repo, plan)
