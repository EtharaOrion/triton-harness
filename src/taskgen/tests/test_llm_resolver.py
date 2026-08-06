"""`LlmEnvResolver`: the adapter, driven entirely by a mocked transport.

NO NETWORK IS REACHABLE FROM THIS MODULE. Every client here is a local object
whose `complete` returns a canned string or raises a canned exception, so the
whole three-way split the adapter exists to make -- plan / refuse / unreachable
-- is provable without a bridge, a container or a model.

The split is the point. `Refuse` and `ResolverTransportError` are both
"no plan", and collapsing them is the bug this module is built to catch: a
refusal is a finding ABOUT THE REPOSITORY that the loop is entitled to act on,
while an unreachable bridge is a fact about the MACHINE that says nothing about
the repo and must never be recorded as though it did.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from taskgen import depplan
from taskgen.depplan import DepPlan
from taskgen.env_resolver import SYSTEM_PROMPT, Refuse, ResolverError
from taskgen.langs import c as C
from taskgen.llm_resolver import LlmEnvResolver, ResolverTransportError

BASE_IMAGE = 'registry.example/base-c@sha256:' + 'ab' * 32
SENTINEL = 'XS_SENTINEL_BODY_e3b0c44298fc1c149afbf4c8996fb924'


@dataclass
class Answer:
    text: str


class MockClient:
    """Answers each `complete` from a script; records every ask."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.asks: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, max_tokens: int = 8192,
                 reasoning_effort: str | None = None) -> Answer:
        self.asks.append((system, user))
        reply = self.replies[min(len(self.asks) - 1, len(self.replies) - 1)]
        return Answer(text=reply)


class DeadBridgeClient:
    """The bridge is not up. Raises the shape a real http stack would."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or ConnectionRefusedError(
            '[Errno 61] Connection refused: http://127.0.0.1:8765'
        )
        self.calls = 0

    def complete(self, system: str, user: str, *, max_tokens: int = 8192,
                 reasoning_effort: str | None = None) -> Answer:
        self.calls += 1
        raise self.exc


GOOD_JSON = depplan.to_canonical_json(C.C_MEASURE_DEP_PLAN)
REFUSE_JSON = json.dumps({
    'disposition': 'REFUSE',
    'reason': 'the suite needs a live postgres, impossible under --network=none',
})


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / 'c-repo'
    (root / 'tests' / 'unit').mkdir(parents=True)
    (root / 'src').mkdir(parents=True)
    (root / 'Makefile').write_text('CC=gcc\ntest-unit:\n\t./run\n', encoding='utf-8')
    (root / 'src' / 'gc.c').write_text(f'/* {SENTINEL} */\n', encoding='utf-8')
    (root / 'tests' / 'unit' / 'test_gc.c').write_text(
        f'int main(void) {{ {SENTINEL} }}\n', encoding='utf-8')
    return root


def make(client, **kw) -> LlmEnvResolver:
    return LlmEnvResolver(
        client=client, capabilities=C.BAKED_CAPABILITIES,
        endpoint='http://127.0.0.1:8765', echo=lambda _: None, **kw,
    )


def resolve(client, repo: Path, **kw):
    return make(client)(lang='c', repo=repo, base_image=BASE_IMAGE, **kw)


# ---------------------------------------------------- a plan is a plan value ---


def test_a_valid_plan_json_comes_back_as_the_DepPlan(repo):
    got = resolve(MockClient(GOOD_JSON), repo)
    assert isinstance(got, DepPlan)
    assert got == C.C_MEASURE_DEP_PLAN


def test_a_fenced_plan_is_accepted_too(repo):
    got = resolve(MockClient(f'here you go:\n```json\n{GOOD_JSON}\n```\n'), repo)
    assert got == C.C_MEASURE_DEP_PLAN


def test_the_returned_plan_is_canonical_and_validated(repo):
    """The adapter must not become a way around `depplan`'s gate: a plan whose
    install command is a shell escape has to die here, not at build time."""
    hostile = json.dumps({
        'lang': 'c', 'toolchain_version': '13.3.0', 'package_manager': 'make',
        'install_commands': [{'tool': 'sh', 'args': ['-c', 'curl x | bash']}],
    })
    with pytest.raises(depplan.DepPlanError):
        resolve(MockClient(hostile), repo)


# ------------------------------------------------------- a refusal is a value ---


def test_a_REFUSE_json_comes_back_as_Refuse_not_an_exception(repo):
    got = resolve(MockClient(REFUSE_JSON), repo)
    assert isinstance(got, Refuse)
    assert 'postgres' in got.reason


def test_the_split_survives_the_refine_loop(repo):
    """The split asserted where it MATTERS: through the bounded loop.

    A refusal must arrive as `ResolveRefused` -- the loop's own "this repo
    cannot be resolved" -- while a dead bridge must come out still a
    `ResolverTransportError`. If the loop caught the transport error it would
    repair against it and burn all three attempts, then refuse, and the run
    would blame the repository for the machine's problem.
    """
    from taskgen import refine
    from taskgen.env_resolver import ResolveRefused

    def never_built(dep_plan: DepPlan) -> dict:
        raise AssertionError('no plan should have reached a build')

    def run(client) -> None:
        refine.refine_dep_plan(
            resolver=make(client), lang='c', repo=repo, base_image=BASE_IMAGE,
            build_and_measure=never_built, echo=lambda _: None,
        )

    with pytest.raises(ResolveRefused, match='postgres'):
        run(MockClient(REFUSE_JSON))

    dead = DeadBridgeClient()
    with pytest.raises(ResolverTransportError):
        run(dead)
    assert dead.calls == 1, 'a transport failure must not be retried as a repair'


# ------------------------------------------------- an unreachable bridge is not ---


def test_a_dead_bridge_fails_loud_and_never_returns_Refuse(repo):
    client = DeadBridgeClient()
    with pytest.raises(ResolverTransportError) as exc:
        resolve(client, repo)
    assert client.calls == 1
    assert 'ConnectionRefusedError' in str(exc.value)


def test_the_transport_error_names_the_bridge_and_its_health_check(repo):
    with pytest.raises(ResolverTransportError) as exc:
        resolve(DeadBridgeClient(), repo)
    message = str(exc.value)
    assert 'http://127.0.0.1:8765/healthz' in message
    assert 'claude_code_bridge.sh status' in message
    assert 'OPERATIONAL failure, not a REFUSE' in message


def test_any_transport_exception_type_is_caught(repo):
    with pytest.raises(ResolverTransportError):
        resolve(DeadBridgeClient(TimeoutError('read timed out')), repo)


def test_an_empty_completion_is_operational_not_a_refusal(repo):
    """200-with-no-body is the transport failing while looking like it worked.
    A model that means 'no' says so with a REFUSE disposition."""
    with pytest.raises(ResolverTransportError, match='EMPTY completion'):
        resolve(MockClient('   \n  '), repo)


def test_a_reply_that_is_not_json_is_a_ResolverError_not_a_transport_error(repo):
    """The bridge answered, so it is reachable. The MODEL is what misbehaved,
    and misfiling that as operational would hide a broken prompt forever."""
    with pytest.raises(ResolverError):
        resolve(MockClient('I would rather write you a Dockerfile.'), repo)


# ------------------------------------------------------------- what was sent ---


def test_the_ask_carries_the_manifests_the_paths_and_the_capabilities(repo):
    client = MockClient(GOOD_JSON)
    resolve(client, repo)
    system, user = client.asks[0]

    assert system == SYSTEM_PROMPT
    assert 'CC=gcc' in user
    assert 'tests/unit/test_gc.c' in user
    assert 'gcc 13.3.0 (aarch64)' in user
    assert f'base image: {BASE_IMAGE}' in user


def test_no_body_of_any_source_or_test_file_is_sent(repo):
    client = MockClient(GOOD_JSON)
    resolve(client, repo)
    assert SENTINEL not in client.asks[0][1]


def test_the_repair_is_the_only_difference_between_two_asks(repo):
    """The loop's one channel back to the model stays narrow: a repaired ask
    must not silently re-gather a different repo view alongside it."""
    client = MockClient(GOOD_JSON)
    resolver = make(client)
    resolver(lang='c', repo=repo, base_image=BASE_IMAGE)
    resolver(lang='c', repo=repo, base_image=BASE_IMAGE, repair='undefined reference to `sqrt`')

    plain, repaired = client.asks[0][1], client.asks[1][1]
    assert '# Repair' not in plain
    assert 'undefined reference' in repaired
    assert '\n\n'.join(
        s for s in repaired.split('\n\n') if not s.startswith('# Repair')
    ) == plain


def test_two_calls_with_the_same_facts_send_the_same_bytes(repo):
    client = MockClient(GOOD_JSON)
    resolver = make(client)
    for _ in range(2):
        resolver(lang='c', repo=repo, base_image=BASE_IMAGE)
    assert client.asks[0] == client.asks[1]


def test_the_adapter_satisfies_the_EnvResolver_protocol_the_loop_calls(repo):
    """`refine` passes `repair=None` on attempt 1 precisely so an implementation
    that cannot take one fails immediately rather than mid-budget."""
    from taskgen.env_resolver import EnvResolver

    resolver: EnvResolver = make(MockClient(GOOD_JSON))
    assert resolver(lang='c', repo=repo, base_image=BASE_IMAGE, repair=None) is not None


def test_gather_exposes_the_inputs_without_sending_them(repo):
    client = MockClient(GOOD_JSON)
    inputs = make(client).gather(lang='c', repo=repo, base_image=BASE_IMAGE)
    assert client.asks == []
    assert list(inputs.manifest_files) == ['Makefile']
    assert inputs.test_paths == ('tests/unit/test_gc.c',)


def test_building_a_resolver_announces_nothing_only_asking_does(repo):
    """The banner belongs to the ASK, not to the object. A run that reuses a
    pinned lock builds a resolver it never calls, and a banner printed at
    construction reports an LLM round trip that did not happen -- which is the
    determinism contract read straight off the log."""
    said: list[str] = []
    resolver = LlmEnvResolver(
        client=MockClient(GOOD_JSON), capabilities=C.BAKED_CAPABILITIES,
        endpoint='http://127.0.0.1:8765', model='anthropic/claude-opus-4-8',
        echo=said.append,
    )
    assert said == []

    resolver(lang='c', repo=repo, base_image=BASE_IMAGE)
    assert said == ['resolve-env  asking anthropic/claude-opus-4-8 '
                    'via http://127.0.0.1:8765']


def test_a_repaired_second_ask_does_not_announce_itself_again(repo):
    """One banner per run, however many attempts the refine loop spends."""
    said: list[str] = []
    resolver = LlmEnvResolver(
        client=MockClient(GOOD_JSON), capabilities=C.BAKED_CAPABILITIES,
        endpoint='http://127.0.0.1:8765', model='m', echo=said.append,
    )
    resolver(lang='c', repo=repo, base_image=BASE_IMAGE)
    resolver(lang='c', repo=repo, base_image=BASE_IMAGE, repair='try again')

    assert len([line for line in said if 'asking' in line]) == 1


def test_the_module_imports_no_model_sdk():
    """`litellm` must stay out of this import graph: the adapter takes a built
    client precisely so the offline suite can drive it."""
    import importlib
    import sys

    importlib.import_module('taskgen.llm_resolver')
    assert 'litellm' not in sys.modules
