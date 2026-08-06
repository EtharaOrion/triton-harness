"""`--resolve-env`: the measure gap comes from a DepPlan, and the lock pins it.

The flag's whole promise is asymmetric, so this module tests it from both sides:

  * OFF is not "almost the same", it is THE SAME. The default measure Dockerfile
    and the default lock are pinned to digests captured from the tree before the
    flag existed, and the default path never touches the resolver.
  * ON changes the SOURCE of the gap without changing the gap. Fed the canonical
    libc-only plan, `render_measure_dockerfile` must emit the identical bytes --
    that is what makes the substitution reviewable rather than a leap of faith.
    What DOES change is the lock, which grows the pinned plan and its key.

Everything here is offline and hermetic: docker is a fake runner returning a
canned measure.json, the resolver is a stub returning a canned plan, and no
network, no LLM and no real container are reachable from any path under test.

Most tests here drive `_measure_and_pin` directly, which is the right unit for
what the gap is made of -- but reuse is not a property of that function alone.
It is a property of the ORDER `emit_all` runs its phases in, so the reuse claim
is additionally proved end to end through `emit_all` at the bottom of the file.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import ClassVar

import pytest

from taskgen import depplan, emit
from taskgen import measure as M
from taskgen.carve import CarveSet
from taskgen.depplan import DepPlan
from taskgen.emit import CarvePlan
from taskgen.env_resolver import Refuse, ResolveRefused
from taskgen.gradedset import GradedSelection
from taskgen.langs import base as B
from taskgen.langs import c as C
from taskgen.scope import CarveScope
from taskgen.staging import StagedTree

#: sha256 of `CPlugin.render_measure_dockerfile(EnvSpec('c-xs'))` with NO plan,
#: captured from the tree before `--resolve-env` existed. The flag is additive
#: iff this digest survives it -- on the default path AND, for c, on the flag
#: path, since the canonical plan describes exactly the hardcoded environment.
DEFAULT_MEASURE_SHA256 = (
    '080516e7d82252ee43a11c69123aac33dee0c4a0ebf14ff2e78985463342fe1f'
)

#: `render_lock(baseline_lock())`, captured the same way. Held
#: as the literal json rather than a digest: the ordering and the exact key set
#: ARE the regeneration contract, so a diff here should read as a diff.
DEFAULT_LOCK_JSON = """{
  "carve_set_sha256": "3333333333333333333333333333333333333333333333333333333333333333",
  "expected": 3,
  "fingerprint_sha256": {
    "Makefile": "abababababababababababababababababababababababababababababababab"
  },
  "floor_mode": "equality",
  "graded": [
    "unit::a",
    "unit::b",
    "unit::c"
  ],
  "lang": "c",
  "provenance": {
    "intact_image": "harbor-base:local",
    "intact_image_digest": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
    "repo_sha256": "1111111111111111111111111111111111111111111111111111111111111111"
  },
  "schema": 1,
  "scope": "folder",
  "tool_version": "taskgen/1"
}
"""

BASELINE_REPO_SHA = '11' * 32
BASELINE_IMAGE_DIGEST = 'sha256:' + '22' * 32


def baseline_lock(
    *,
    repo_sha256: str = BASELINE_REPO_SHA,
    intact_image_digest: str = BASELINE_IMAGE_DIGEST,
    dep_plan: DepPlan | None = None,
) -> dict:
    """The lock `DEFAULT_LOCK_JSON` was captured from, with the axes under test open."""
    return M.build_lock(
        lang='c',
        scope='folder',
        expected=3,
        graded=('unit::a', 'unit::b', 'unit::c'),
        floor_mode='equality',
        fingerprint_sha256={'Makefile': 'ab' * 32},
        repo_sha256=repo_sha256,
        intact_image='harbor-base:local',
        intact_image_digest=intact_image_digest,
        carve_set_sha256='33' * 32,
        dep_plan=dep_plan,
    )

MEASURE_JSON = '{"tests_total": 3, "graded": ["u::c", "u::a", "u::b"]}'
GRADED_SORTED = ['u::a', 'u::b', 'u::c']
FAKE_DIGEST = 'sha256:' + 'ab' * 32


# --------------------------------------------------------------- fixtures ---


class FakeRunner:
    """docker, minus docker. Records every call so the tests can assert none."""

    instances: ClassVar[list[FakeRunner]] = []

    def __init__(self, echo=print):
        self.builds: list[dict] = []
        self.runs: list[tuple[str, str]] = []
        self.removed: list[str] = []
        FakeRunner.instances.append(self)

    def image_digest(self, image: str) -> str:
        return FAKE_DIGEST

    def build_with_contexts(self, *, image, dockerfile, context, contexts):
        self.builds.append({'image': image, 'dockerfile': Path(dockerfile).read_text()})
        return 'BUILD OK'

    def run(self, image: str, script: str) -> str:
        self.runs.append((image, script))
        return MEASURE_JSON

    def remove_image(self, image: str) -> None:
        self.removed.append(image)


class StubResolver:
    """The injected seam, canned. `calls` is the point of the whole class."""

    def __init__(self, answer):
        self._answer = answer
        self.calls = 0

    def __call__(self, *, lang: str, repo: Path, base_image: str,
                 repair: str | None = None):
        self.calls += 1
        self.seen = {
            'lang': lang, 'repo': repo, 'base_image': base_image, 'repair': repair,
        }
        return self._answer


@pytest.fixture(autouse=True)
def no_docker(monkeypatch):
    """Every test in this module builds through the fake, or not at all."""
    FakeRunner.instances = []
    monkeypatch.setattr('taskgen.verify.DockerRunner', FakeRunner)
    return FakeRunner


@pytest.fixture
def c_repo(tmp_path) -> Path:
    repo = tmp_path / 'c-xs'
    (repo / 'tests').mkdir(parents=True)
    (repo / 'Makefile').write_text('all:\n\t$(CC) -o xs src/main.c\n', encoding='utf-8')
    (repo / 'tests' / 'run-all.sh').write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
    return repo


@pytest.fixture
def plan(c_repo, tmp_path) -> CarvePlan:
    """A c CarvePlan holding exactly what `_measure_and_pin` reads off one."""
    staged = StagedTree(
        out_dir=tmp_path / 'staging',
        ctx_dir=tmp_path / 'staging' / 'ctx',
        repo_dir=tmp_path / 'staging' / 'ctx' / 'repo',
        oracle_dir=tmp_path / 'staging' / 'oracle',
        receipt={},
        receipt_path=tmp_path / 'staging' / 'carve_receipt.json',
        tripwire_path=tmp_path / 'staging' / 'trip' / 'tripwires.txt',
        tripwire_digest_path=tmp_path / 'staging' / 'trip' / 'digests.txt',
        tripwires=(),
        staged_relpaths=(),
    )
    return CarvePlan(
        lang='c',
        scope=CarveScope.FOLDER,
        repo=c_repo,
        repo_name='c-xs',
        target=None,
        carve=CarveSet(
            scope=CarveScope.FOLDER,
            language='c',
            carved_relpaths=('src/runtime/gc.c',),
            deleted_relpaths=('src/runtime/gc.c',),
            overlay={},
            originals={},
            primary_relpath='src/runtime/gc.c',
        ),
        graded=GradedSelection(
            kind='whole_suite',
            selectors=(),
            packages=(),
            tests=(),
            carved_files=('src/runtime/gc.c',),
            carved_functions=(),
            test_command=C.TEST_COMMAND,
        ),
        staged=staged,
        staging_key='c-xs__folder',
    )


@pytest.fixture
def cplugin():
    return B.get('c')


def pin(plan_obj: CarvePlan, out: Path, **kwargs) -> CarvePlan:
    return emit._measure_and_pin(plan_obj, B.get('c'), out, echo=lambda *_: None, **kwargs)


def lock_path(out: Path, plan_obj: CarvePlan) -> Path:
    return out.resolve() / '_staging' / plan_obj.staging_key / M.LOCK_FILENAME


def dockerfile_path(out: Path, plan_obj: CarvePlan) -> Path:
    return out.resolve() / '_staging' / plan_obj.staging_key / 'measure' / 'Dockerfile'


# ------------------------------------------------- the default path, frozen ---


def test_default_measure_dockerfile_bytes_are_the_captured_baseline(cplugin):
    rendered = cplugin.render_measure_dockerfile(B.EnvSpec(repo_name='c-xs'))
    assert hashlib.sha256(rendered.encode()).hexdigest() == DEFAULT_MEASURE_SHA256


def test_default_lock_json_bytes_are_the_captured_baseline():
    assert M.render_lock(baseline_lock()) == DEFAULT_LOCK_JSON


def test_dep_plan_none_adds_no_key_and_reorders_none():
    without = baseline_lock()
    assert M.LOCK_ENV_BLOCK not in without
    assert list(without) == list(json.loads(DEFAULT_LOCK_JSON))


def test_default_path_emits_the_baseline_and_never_resolves(plan, tmp_path):
    resolver = StubResolver(C.C_MEASURE_DEP_PLAN)
    pinned = pin(plan, tmp_path / 'out', resolver=resolver)

    assert resolver.calls == 0
    assert pinned.graded.expected == 3
    written = dockerfile_path(tmp_path / 'out', plan).read_text()
    assert hashlib.sha256(written.encode()).hexdigest() == DEFAULT_MEASURE_SHA256
    lock = json.loads(lock_path(tmp_path / 'out', plan).read_text())
    assert M.LOCK_ENV_BLOCK not in lock
    assert lock['graded'] == GRADED_SORTED


# ----------------------------------------------------------- the flag path ---


def test_flag_renders_the_same_dockerfile_the_default_does(plan, tmp_path):
    resolver = StubResolver(C.C_MEASURE_DEP_PLAN)
    pin(plan, tmp_path / 'out', resolve_env=True, resolver=resolver)

    assert resolver.calls == 1
    # repair=None: attempt 1 asks exactly what the pre-loop single resolve asked.
    assert resolver.seen == {
        'lang': 'c', 'repo': plan.repo, 'base_image': B.get('c').toolchain_spec().base_image,
        'repair': None,
    }
    written = dockerfile_path(tmp_path / 'out', plan).read_text()
    assert hashlib.sha256(written.encode()).hexdigest() == DEFAULT_MEASURE_SHA256


def test_a_different_plan_renders_a_different_image(plan, tmp_path):
    """The byte-identity above is only meaningful if the plan is REACHING the render.

    Identical output is exactly what a dropped argument also produces, so the
    seam is proved live by mutating the plan and demanding the bytes move.
    """
    mutated = dataclasses.replace(
        C.C_MEASURE_DEP_PLAN, apt_packages=('libpq-dev',),
    )
    pin(plan, tmp_path / 'out', resolve_env=True, resolver=StubResolver(mutated))

    written = dockerfile_path(tmp_path / 'out', plan).read_text()
    assert hashlib.sha256(written.encode()).hexdigest() != DEFAULT_MEASURE_SHA256
    assert 'apt-get install -y --no-install-recommends libpq-dev' in written

    env = json.loads(lock_path(tmp_path / 'out', plan).read_text())[M.LOCK_ENV_BLOCK]
    assert 'libpq-dev' in env['dep_plan']
    assert env['dep_plan_digest'] != depplan.dep_plan_digest(C.C_MEASURE_DEP_PLAN)


def test_flag_pins_the_plan_the_ids_and_the_key(plan, tmp_path):
    pin(plan, tmp_path / 'out', resolve_env=True,
        resolver=StubResolver(C.C_MEASURE_DEP_PLAN))

    lock = json.loads(lock_path(tmp_path / 'out', plan).read_text())
    env = lock[M.LOCK_ENV_BLOCK]
    repo_sha = M.repo_tree_sha256(plan.repo)

    assert env['dep_plan'] == depplan.to_canonical_json(C.C_MEASURE_DEP_PLAN)
    assert env['dep_plan_digest'] == depplan.dep_plan_digest(C.C_MEASURE_DEP_PLAN)
    assert env['env_lock_key'] == depplan.env_lock_key(
        repo_sha, FAKE_DIGEST, C.C_MEASURE_DEP_PLAN,
    )
    # The collected ids are pinned ONCE, by the key that already pinned them.
    assert lock['graded'] == GRADED_SORTED
    assert 'graded' not in env


def test_flag_path_renders_twice_byte_identical(plan, tmp_path):
    first = tmp_path / 'a'
    second = tmp_path / 'b'
    for out in (first, second):
        pin(plan, out, resolve_env=True, resolver=StubResolver(C.C_MEASURE_DEP_PLAN))

    assert (lock_path(first, plan).read_bytes() == lock_path(second, plan).read_bytes())
    assert (dockerfile_path(first, plan).read_bytes()
            == dockerfile_path(second, plan).read_bytes())


def test_pinned_lock_round_trips_and_reuse_never_resolves(plan, tmp_path):
    out = tmp_path / 'out'
    resolver = StubResolver(C.C_MEASURE_DEP_PLAN)
    pin(plan, out, resolve_env=True, resolver=resolver)
    assert resolver.calls == 1

    measured = FakeRunner.instances[-1]
    assert len(measured.builds) == 1 and len(measured.runs) == 1

    reused_resolver = StubResolver(C.C_MEASURE_DEP_PLAN)
    pinned = pin(plan, out, resolve_env=True, resolver=reused_resolver)

    assert reused_resolver.calls == 0
    assert pinned.graded.expected == 3
    reusing = FakeRunner.instances[-1]
    assert reusing is not measured
    assert (reusing.builds, reusing.runs) == ([], [])

    lock = M.load_lock(lock_path(out, plan))
    assert M.check_provenance(lock, M.repo_tree_sha256(plan.repo), FAKE_DIGEST) == []


def test_legacy_lock_without_a_dep_plan_still_reuses_under_the_flag(plan, tmp_path):
    out = tmp_path / 'out'
    legacy = baseline_lock(
        repo_sha256=M.repo_tree_sha256(plan.repo), intact_image_digest=FAKE_DIGEST,
    )
    assert M.LOCK_ENV_BLOCK not in legacy
    M.write_lock(lock_path(out, plan), legacy)

    resolver = StubResolver(C.C_MEASURE_DEP_PLAN)
    pinned = pin(plan, out, resolve_env=True, resolver=resolver)

    assert resolver.calls == 0
    assert pinned.graded.expected == 3
    assert M.LOCK_ENV_BLOCK not in json.loads(lock_path(out, plan).read_text())


def test_a_lock_whose_pinned_plan_was_edited_is_re_measured(plan, tmp_path):
    out = tmp_path / 'out'
    pin(plan, out, resolve_env=True, resolver=StubResolver(C.C_MEASURE_DEP_PLAN))

    tampered = M.load_lock(lock_path(out, plan))
    tampered[M.LOCK_ENV_BLOCK]['dep_plan'] = depplan.to_canonical_json(
        dataclasses.replace(C.C_MEASURE_DEP_PLAN, toolchain_version='9.9.9'),
    )
    M.write_lock(lock_path(out, plan), tampered)

    resolver = StubResolver(C.C_MEASURE_DEP_PLAN)
    pin(plan, out, resolve_env=True, resolver=resolver)
    assert resolver.calls == 1


# ------------------------------------------------------------- the refusals ---


def test_a_refusal_raises_and_emits_neither_task_nor_lock(plan, tmp_path):
    out = tmp_path / 'out'
    reason = 'tests need a postgres service, which cannot exist under --network=none'

    with pytest.raises(ResolveRefused) as excinfo:
        pin(plan, out, resolve_env=True, resolver=StubResolver(Refuse(reason=reason)))

    assert excinfo.value.reason == reason
    assert not lock_path(out, plan).exists()
    assert not dockerfile_path(out, plan).exists()
    assert FakeRunner.instances[-1].builds == []
    assert FakeRunner.instances[-1].runs == []


def test_the_flag_refuses_a_language_with_no_rendered_gap(plan, tmp_path):
    rust_plan = dataclasses.replace(plan, lang='rust')
    with pytest.raises(B.LangError, match='only supported for c, cpp, java'):
        emit._measure_and_pin(
            rust_plan, B.get('rust'), tmp_path / 'out', echo=lambda *_: None,
            resolve_env=True, resolver=StubResolver(C.C_MEASURE_DEP_PLAN),
        )


def test_the_flag_refuses_a_parser_backed_language_before_it_carves(repo, tmp_path):
    with pytest.raises(B.LangError, match='only supported for c, cpp, java'):
        emit.emit_all(repo=repo, out=tmp_path / 'out', lang='python', resolve_env=True)


def test_the_flag_without_a_resolver_is_loud(plan, tmp_path):
    with pytest.raises(B.LangError, match='no resolver was injected'):
        pin(plan, tmp_path / 'out', resolve_env=True, resolver=None)


# ------------------------------------------------------ the key, at the unit ---


# ------------------------------------------- reuse through the REAL flow ---
#
# `pin()` above enters at `_measure_and_pin`, so it never runs the phase BEFORE
# it: `emit_all` carves first, and carving restages `_staging/<key>/` -- the
# directory the lock lives in. Every test above can pass while a real run finds
# no lock, re-resolves and rebuilds the measure image every time.

#: c-xs's harness list is 13 unit files, and the plugin refuses a tree whose
#: sub-counts disagree with it, so the fixture repo mirrors that shape exactly.
E2E_UNIT_FILES = 13
E2E_TOTAL = E2E_UNIT_FILES + 2
E2E_MEASURE_JSON = json.dumps(
    {'tests_total': E2E_TOTAL, 'graded': [f'unit::{i:02d}' for i in range(E2E_TOTAL)]}
)


class E2ERunner(FakeRunner):
    """`FakeRunner` whose canned count matches the end-to-end fixture's corpus."""

    def run(self, image: str, script: str) -> str:
        self.runs.append((image, script))
        return E2E_MEASURE_JSON


@pytest.fixture
def e2e_docker(monkeypatch, no_docker):
    monkeypatch.setattr('taskgen.verify.DockerRunner', E2ERunner)
    return E2ERunner


@pytest.fixture
def e2e_repo(tmp_path) -> Path:
    """A tree `emit_all` can carve, grade and measure without a container."""
    repo = tmp_path / 'c-xs-e2e'
    (repo / 'src' / 'runtime').mkdir(parents=True)
    for sub in ('conformance', 'regression', 'unit'):
        (repo / 'tests' / sub).mkdir(parents=True)
    (repo / 'Makefile').write_text(
        'test:\n\t$(CC) -o xs src/runtime/gc.c\n', encoding='utf-8')
    (repo / 'src' / 'runtime' / 'gc.c').write_text(
        'int gc_collect(int generation) {\n    return generation + 1;\n}\n',
        encoding='utf-8')
    (repo / 'tests' / 'conformance' / 'a.xs').write_text('1\n', encoding='utf-8')
    (repo / 'tests' / 'regression' / 'b.xs').write_text('2\n', encoding='utf-8')
    for i in range(E2E_UNIT_FILES):
        (repo / 'tests' / 'unit' / f'u{i:02d}_test.c').write_text(
            'int main(void) { return 0; }\n', encoding='utf-8')
    (repo / 'tests' / 'run-all.sh').write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
    return repo


def generate(repo: Path, out: Path, resolver, **kwargs) -> None:
    emit.emit_all(
        repo=repo, out=out, lang='c', carve_scope='folder',
        include=('src/runtime/**',), delete_whole_file=True,
        resolver=resolver, echo=lambda *_: None, **kwargs,
    )


def e2e_lock_path(out: Path) -> Path:
    return next(out.resolve().glob(f'_staging/*/{M.LOCK_FILENAME}'))


def test_a_warm_pinned_lock_survives_the_carve_and_is_reused(
        e2e_repo, tmp_path, e2e_docker):
    """The live regression: regenerate into a directory that already holds a
    valid pinned lock and NOTHING is asked and NOTHING is built."""
    out = tmp_path / 'out'
    cold = StubResolver(C.C_MEASURE_DEP_PLAN)
    generate(e2e_repo, out, cold, resolve_env=True)

    assert cold.calls == 1
    measured = FakeRunner.instances[-1]
    assert len(measured.builds) == 1 and len(measured.runs) == 1

    warm = StubResolver(C.C_MEASURE_DEP_PLAN)
    generate(e2e_repo, out, warm, resolve_env=True)

    assert warm.calls == 0, 'a warm pinned lock must never reach the resolver'
    reusing = FakeRunner.instances[-1]
    assert reusing is not measured
    assert (reusing.builds, reusing.runs) == ([], []), 'no measure image on a warm lock'


def test_the_carve_does_not_delete_the_lock_it_is_about_to_be_asked_for(
        e2e_repo, tmp_path, e2e_docker):
    """The mechanism, named: the pinned bytes survive the restage untouched."""
    out = tmp_path / 'out'
    generate(e2e_repo, out, StubResolver(C.C_MEASURE_DEP_PLAN), resolve_env=True)
    before = e2e_lock_path(out).read_bytes()

    emit.plan_carve(
        e2e_repo, out, lang='c', carve_scope='folder',
        include=('src/runtime/**',), delete_whole_file=True,
    )

    assert e2e_lock_path(out).read_bytes() == before


def test_the_warm_flag_run_emits_the_same_bytes_the_cold_one_did(
        e2e_repo, tmp_path, e2e_docker):
    """Reuse is a shortcut, not a different task: nothing shipped may move."""
    cold_out = tmp_path / 'cold'
    generate(e2e_repo, cold_out, StubResolver(C.C_MEASURE_DEP_PLAN), resolve_env=True)
    before = _entry_bytes(cold_out)

    generate(e2e_repo, cold_out, StubResolver(C.C_MEASURE_DEP_PLAN), resolve_env=True)

    assert _entry_bytes(cold_out) == before


def test_the_default_path_reuses_its_lock_through_the_real_flow_too(
        e2e_repo, tmp_path, e2e_docker):
    """Without the flag there is no resolver to skip, but there is still an image
    not to rebuild -- and `graded.lock.json` has always promised that."""
    out = tmp_path / 'out'
    generate(e2e_repo, out, None)
    assert len(FakeRunner.instances[-1].builds) == 1

    generate(e2e_repo, out, None)
    assert (FakeRunner.instances[-1].builds, FakeRunner.instances[-1].runs) == ([], [])


def _entry_bytes(out: Path) -> dict[str, bytes]:
    return {
        p.relative_to(out).as_posix(): p.read_bytes()
        for p in sorted(out.rglob('*'))
        if p.is_file() and '_staging' not in p.relative_to(out).parts
    }


# ------------------------------------------------------ the key, at the unit ---


def test_a_key_from_pinned_json_equals_the_key_from_the_plan():
    canonical = depplan.to_canonical_json(C.C_MEASURE_DEP_PLAN)
    assert depplan.env_lock_key_from_canonical_json('r', 'i', canonical) == (
        depplan.env_lock_key('r', 'i', C.C_MEASURE_DEP_PLAN)
    )


def test_a_lock_with_no_plan_passes_the_key_check_untouched():
    assert M.check_env_lock_key(baseline_lock(), 'r', 'i') == []


def test_a_moved_repo_invalidates_a_pinned_environment():
    lock = baseline_lock(dep_plan=C.C_MEASURE_DEP_PLAN)
    same = BASELINE_REPO_SHA
    digest = BASELINE_IMAGE_DIGEST

    assert M.check_env_lock_key(lock, same, digest) == []
    assert M.check_env_lock_key(lock, 'moved', digest) != []
    assert M.check_env_lock_key(lock, same, 'moved') != []
