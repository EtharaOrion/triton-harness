"""The LLM half of environment resolution: prompt in, `DepPlan` or `Refuse` out.

`depplan` owns what a plan may BE; this module owns how one is ASKED FOR and how
a model's answer is turned back into that closed record. The split matters: the
validation gate lives in `depplan.validate`, so nothing here has to be trusted to
notice that a model asked for `bash -c`. A resolution that survives this module
has already been through the allowlist and the metacharacter gate.

NO NETWORK, NO MODEL SDK. The client is injected and only has to answer
`complete(system, user, ...)` -- the shape `verifier.generators.model_client`
already speaks, so `MockClient` drives every test here and `litellm` is never
imported (not even lazily) on this path.

WHAT THE RESOLVER MAY SEE. `build_user_prompt` takes manifest CONTENTS but only
test PATHS. That is not a convention, it is the benchmark's integrity: a test
body is the answer the task hides, so a body must not be expressible in this
function's arguments at all. The signature is the enforcement.

ONE SHOT. `resolve_dep_plan` asks once and raises on an invalid answer. A
build/collect repair loop needs docker, which this module does not have; the
`repair` slot on `build_user_prompt` is the seam that loop will use later.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from taskgen.depplan import (
    DepPlan,
    FlagValue,
    InstallCommand,
    TestValue,
    canonicalize,
    validate,
)

__all__ = [
    'MAX_RESPONSE_TOKENS',
    'Refuse',
    'ResolverClient',
    'ResolverError',
    'SYSTEM_PROMPT',
    'build_user_prompt',
    'parse_resolution',
    'resolve_dep_plan',
]

#: A DepPlan is a few hundred tokens; a cap keeps a rambling model from paying
#: for prose the parser will throw away anyway.
MAX_RESPONSE_TOKENS = 4096


class ResolverError(RuntimeError):
    """A model answer that is not a resolution: unparseable, or malformed.

    Distinct from `DepPlanError`, which means the answer WAS a plan and the plan
    was inadmissible. Keeping them apart lets a caller tell "the model did not
    answer the question" from "the model answered with a forbidden environment".
    """


@dataclass(frozen=True)
class Refuse:
    """The resolver declined, on purpose. A precise refusal is a success."""

    reason: str


class _Response(Protocol):
    """Just the field this module reads off a completion."""

    text: str


class ResolverClient(Protocol):
    """The injected model client, mirroring `model_client.ModelClient`."""

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 8192,
        reasoning_effort: str | None = None,
    ) -> _Response: ...


# --------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """ROLE
You are a build-environment resolver for a reproducible code-evaluation harness. Given a single source repository, you output a STRUCTURED DepPlan (JSON) that lets the harness build ONE Docker image (on a pinned base) in which the repository's own test suite runs and PASSES. You do NOT write a Dockerfile or any shell script; you fill typed slots. A deterministic template turns your DepPlan into the image.

WHAT YOU CAN SEE (and nothing else)
- The repo's dependency/build MANIFEST files (pyproject.toml, uv.lock, requirements*.txt, poetry.lock, setup.cfg, go.mod/go.sum, Cargo.toml/Cargo.lock, CMakeLists.txt, pom.xml/build.gradle, .python-version, tox.ini, CI workflow).
- The base image's baked capabilities (interpreters/toolchains available via `mise`, preinstalled compilers/build tools) — given explicitly.
- The LIST OF TEST FILE PATHS and the detected test framework. You will NEVER be shown test or source bodies and must never ask for them: the behavior under test is the answer the benchmark hides. Manifests + paths are sufficient.

PRINCIPLES (grounded in prior art)
1. INFER FROM CONFIG, IN PRECEDENCE (repo2docker; DockerizeMe). Prefer an existing LOCKFILE over re-resolving. Detect native/system (apt) deps that language manifests omit.
2. PIN FOR REPRODUCIBILITY (SWE-bench install specs; Nix/Bazel; lockfiles). Exact versions, never "latest". Prefer the repo's declared toolchain, but ONLY if the base provides it; else REFUSE.
3. VALIDATE-BY-RUNNING is the harness's job. Choose the MINIMAL install that makes the suite collect and pass (SWE-bench PASS_TO_PASS). Minimal extras; add one only if tests need it.
4. OFFLINE AT GRADE TIME. Network exists only while building; the graded run has none. Every dependency must be fully installed at build time; anything fetching/serving at test time is disqualifying.
5. STRUCTURED, NOT ARBITRARY. install_commands are {tool, args[]} with tool in the language allowlist. No pipes, no curl|bash, no sh -c, no shell metacharacters. System packages via apt only, sorted and pinned.

REFUSE (do not guess) -> {"disposition":"REFUSE","reason":"..."} when: no recognized manifest; required toolchain not provided by the base; tests need an external service (DB/network/browser/docker) impossible under --network=none; the only viable install needs a non-apt source / source build / PPA / curl; multiple test frameworks or an ambiguous package root. A precise REFUSE is a SUCCESS; a plausible-but-wrong plan corrupts the benchmark.

OUTPUT — emit ONLY canonical JSON, keys sorted, lists sorted+deduped, versions normalized, package names lowercased:
{ "schema":1, "lang":"...", "toolchain_version":"...", "package_manager":"<enum>", "manifest_files":[...], "apt_packages":[...], "install_commands":[{"tool":"<allowlisted>","args":["..."]}], "build_flags":{...}, "test_invocation":{"framework":"...","collect_args":[...],"run_args":[...]}, "needs_git_metadata":<bool> }
Be minimal and canonical: two runs on the same repo must produce the same plan; prefer the fewest packages, the lockfile path, one obvious install sequence; between equivalent options pick the lexicographically smallest. If given a repair message (build/collect stderr), change the SMALLEST set of slots that fixes the specific error."""


def _reject_multiline(value: str, where: str) -> str:
    """A "path" spanning lines is a body wearing a path's clothes.

    The signature already excludes test sources, but a caller assembling
    `test_paths` from a loose parse could smuggle one through as a single
    newline-bearing entry, so the leak boundary is checked, not assumed.
    """
    if '\n' in value or '\r' in value:
        raise ResolverError(
            f'{where} spans multiple lines: {value!r}. The resolver is shown test '
            'PATHS only -- a multi-line entry is a test body, which is the answer '
            'the benchmark hides and must never reach the model'
        )
    return value


def build_user_prompt(
    *,
    lang: str,
    toolchain_capabilities: list[str],
    manifest_files: dict[str, str],
    test_paths: list[str],
    framework: str,
    repair: str | None = None,
) -> str:
    """The user half of the resolution request. Deterministic, and body-free.

    Every list is sorted, so two callers holding the same facts in a different
    order send the same bytes and (for a temperature-0 client) get the same plan.
    """
    sections = [
        f'# Language\n{lang}',
        '# Base image capabilities\n'
        + ('\n'.join(f'- {c}' for c in sorted(set(toolchain_capabilities))) or '- (none given)'),
    ]

    manifests = [
        f'## {name}\n```\n{content}\n```'
        for name, content in sorted(manifest_files.items())
    ]
    sections.append(
        '# Manifest files\n' + ('\n\n'.join(manifests) or '(no manifest found)')
    )

    paths = sorted({_reject_multiline(p, 'a test path') for p in test_paths})
    sections.append(
        '# Test file paths (paths only -- bodies are never shown)\n'
        + ('\n'.join(f'- {p}' for p in paths) or '- (none found)')
    )
    sections.append(f'# Detected test framework\n{framework}')

    if repair:
        sections.append(
            '# Repair\nThe previous plan failed. Change the SMALLEST set of slots '
            f'that fixes this specific error:\n```\n{repair}\n```'
        )

    sections.append('Emit ONLY the canonical DepPlan JSON object now.')
    return '\n\n'.join(sections)


# --------------------------------------------------------------------------
# parsing a resolution back into the closed record
# --------------------------------------------------------------------------

#: Same fenced-block shape `pytest_gen.extract_code` reads, retargeted at json.
_FENCE = re.compile(r'```(?:json)?\s*\n(.*?)```', re.DOTALL)
_BARE_OBJECT = re.compile(r'\{.*\}', re.DOTALL)


def _extract_json_object(text: str) -> dict[str, object]:
    """The one json object in a model answer, fenced or bare, else `ResolverError`."""
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        bare = _BARE_OBJECT.search(text)
        candidate = bare.group(0) if bare else None
    if candidate is None:
        raise ResolverError(f'no json object found in the resolver answer: {text[:200]!r}')
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ResolverError(f'resolver answer is not valid json: {exc}') from exc
    if not isinstance(data, dict):
        raise ResolverError(
            f'resolver answer is a {type(data).__name__}, not a json object'
        )
    return data


def _require(data: dict[str, object], key: str) -> object:
    if key not in data:
        raise ResolverError(f'resolution is missing required key {key!r}')
    return data[key]


def _as_str(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise ResolverError(f'{where} must be a string, got {type(value).__name__}')
    return value


def _as_str_tuple(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ResolverError(f'{where} must be a list, got {type(value).__name__}')
    return tuple(_as_str(item, f'an item of {where}') for item in value)


def _as_install_commands(value: object) -> tuple[InstallCommand, ...]:
    if not isinstance(value, list):
        raise ResolverError(
            f'install_commands must be a list, got {type(value).__name__}'
        )
    commands: list[InstallCommand] = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ResolverError(
                'each install command must be a {"tool":...,"args":[...]} object'
            )
        tool = _as_str(_require(entry, 'tool'), 'an install command tool')
        commands.append(
            InstallCommand(tool=tool, args=_as_str_tuple(entry.get('args', []), 'args'))
        )
    return tuple(commands)


def _as_build_flags(value: object) -> tuple[tuple[str, FlagValue], ...]:
    if not isinstance(value, dict):
        raise ResolverError(f'build_flags must be an object, got {type(value).__name__}')
    flags: list[tuple[str, FlagValue]] = []
    for key, raw in value.items():
        # bool before int: `isinstance(True, int)` is true, and a flag that
        # canonicalises `true` into `1` is a different environment on paper.
        if isinstance(raw, bool) or isinstance(raw, (int, str)):
            flags.append((_as_str(key, 'a build flag name'), raw))
            continue
        raise ResolverError(
            f'build flag {key!r} must be a string, integer or boolean, '
            f'got {type(raw).__name__}'
        )
    return tuple(sorted(flags, key=lambda pair: pair[0]))


def _as_test_invocation(value: object) -> tuple[tuple[str, TestValue], ...]:
    if not isinstance(value, dict):
        raise ResolverError(
            f'test_invocation must be an object, got {type(value).__name__}'
        )
    entries: list[tuple[str, TestValue]] = []
    for key, raw in value.items():
        name = _as_str(key, 'a test_invocation key')
        parsed: TestValue = (
            raw if isinstance(raw, str) else _as_str_tuple(raw, f'test_invocation {name!r}')
        )
        entries.append((name, parsed))
    return tuple(sorted(entries, key=lambda pair: pair[0]))


def parse_resolution(text: str, lang: str) -> DepPlan | Refuse:
    """A model answer as either a validated, canonical `DepPlan` or a `Refuse`."""
    data = _extract_json_object(text)

    disposition = data.get('disposition')
    if isinstance(disposition, str) and disposition.strip().upper() == 'REFUSE':
        reason = _as_str(_require(data, 'reason'), 'reason')
        if not reason.strip():
            raise ResolverError('a REFUSE carries no reason; a bare refusal is not actionable')
        return Refuse(reason=reason)

    declared_lang = data.get('lang', lang)
    if _as_str(declared_lang, 'lang') != lang:
        raise ResolverError(
            f'resolution declares lang {declared_lang!r} but was asked about {lang!r}; '
            'a plan for another language is not a plan for this repo'
        )

    schema = data.get('schema', 1)
    if not isinstance(schema, int) or isinstance(schema, bool):
        raise ResolverError(f'schema must be an integer, got {type(schema).__name__}')

    needs_git = data.get('needs_git_metadata', False)
    if not isinstance(needs_git, bool):
        raise ResolverError(
            f'needs_git_metadata must be a boolean, got {type(needs_git).__name__}'
        )

    plan = DepPlan(
        lang=lang,
        toolchain_version=_as_str(_require(data, 'toolchain_version'), 'toolchain_version'),
        package_manager=_as_str(_require(data, 'package_manager'), 'package_manager'),
        schema=schema,
        manifest_files=_as_str_tuple(data.get('manifest_files', []), 'manifest_files'),
        apt_packages=_as_str_tuple(data.get('apt_packages', []), 'apt_packages'),
        install_commands=_as_install_commands(data.get('install_commands', [])),
        build_flags=_as_build_flags(data.get('build_flags', {})),
        test_invocation=_as_test_invocation(data.get('test_invocation', {})),
        needs_git_metadata=needs_git,
    )
    validate(plan)
    return canonicalize(plan)


def resolve_dep_plan(
    client: ResolverClient,
    *,
    lang: str,
    toolchain_capabilities: list[str],
    manifest_files: dict[str, str],
    test_paths: list[str],
    framework: str,
) -> DepPlan | Refuse:
    """One resolution round trip: ask the injected client, parse, validate.

    No repair loop: repairing means building, building means docker, and this
    module is offline by construction. An inadmissible answer raises here rather
    than being silently softened into a plausible plan.
    """
    user = build_user_prompt(
        lang=lang,
        toolchain_capabilities=toolchain_capabilities,
        manifest_files=manifest_files,
        test_paths=test_paths,
        framework=framework,
    )
    response = client.complete(SYSTEM_PROMPT, user, max_tokens=MAX_RESPONSE_TOKENS)
    return parse_resolution(response.text, lang)
