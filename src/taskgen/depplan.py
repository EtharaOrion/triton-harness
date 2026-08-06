"""The structured, per-language build/test environment plan -- and its lock key.

A `DepPlan` is what an environment resolver (later, an LLM) is allowed to
produce: a CLOSED, typed description of how a repo's toolchain, dependencies and
test runner are obtained. It is deliberately NOT a Dockerfile and NOT a shell
string. A free-form Dockerfile from a model is unreviewable and unsandboxable;
a `DepPlan` is a finite record whose every tool comes from an allowlist and whose
every argument is a separate token that carries no shell metacharacter. The
renderer that turns a plan into an image is downstream and is not this module's
business -- but a plan that reaches it cannot smuggle `$(curl ...)` through an
argument, because such a plan does not validate here.

WHY A LOCK KEY OVER THE INTENT, NOT THE BYTES. `env_lock_key` hashes the repo
tree digest, the base image digest and the canonical plan. It never hashes
downloaded dependency bytes: those are not reproducible across a registry's
lifetime, and a key that changes when an unrelated mirror re-tars a wheel is a
cache key, not a provenance key. The three inputs are exactly the three things
that, if any one moves, mean "this environment is a different environment" --
mirroring `measure.check_provenance`, which invalidates a floor when either the
tree or the toolchain image moves underneath it.

DETERMINISM. `canonicalize` is idempotent and total, `to_canonical_json` sorts
keys and emits no timestamp and no host path, so two hosts holding the same plan
compute the same digest. Everything here is pure: no clock, no I/O, no network,
no docker, no LLM.

NAMING. `langs.base.EnvSpec` is a placement record (where things live in the
image); a `DepPlan` is a provisioning record (how they get there). They are
different axes and deliberately do not share a name.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace

__all__ = [
    'APT_TOOL',
    'DepPlan',
    'DepPlanError',
    'InstallCommand',
    'LANGS',
    'PACKAGE_MANAGERS',
    'SCHEMA',
    'SHELL_METACHARACTERS',
    'TOOL_ALLOWLIST',
    'canonicalize',
    'dep_plan_digest',
    'env_lock_key',
    'to_canonical_json',
    'validate',
]

SCHEMA = 1

#: The six languages with a shipped plugin. `csharp` is planned, not resolvable.
LANGS: tuple[str, ...] = ('python', 'go', 'rust', 'c', 'cpp', 'java')

#: Closed enum: the ONE package manager a plan may declare, per language.
PACKAGE_MANAGERS: dict[str, frozenset[str]] = {
    'python': frozenset({'uv', 'pip', 'poetry'}),
    'go': frozenset({'go'}),
    'rust': frozenset({'cargo'}),
    'c': frozenset({'cmake', 'make'}),
    'cpp': frozenset({'cmake', 'make'}),
    'java': frozenset({'gradle', 'maven'}),
}

#: `apt-get` is the one system-dependency escape hatch, and it is allowed for
#: every language because every language's native deps land through it.
APT_TOOL = 'apt-get'

#: Closed enum: the executables an `InstallCommand` may name. The package
#: manager enum is about DECLARED intent; this is about what may actually run,
#: so `maven` declares the manager while `mvn` is the binary. Nothing else --
#: notably no `bash`, `sh`, `curl`, `wget`, `git` or `sudo` -- is reachable.
TOOL_ALLOWLIST: dict[str, frozenset[str]] = {
    'python': frozenset({'uv', 'pip', 'poetry', APT_TOOL}),
    'go': frozenset({'go', APT_TOOL}),
    'rust': frozenset({'cargo', APT_TOOL}),
    'c': frozenset({'cmake', 'make', APT_TOOL}),
    'cpp': frozenset({'cmake', 'make', APT_TOOL}),
    'java': frozenset({'gradle', 'mvn', APT_TOOL}),
}

#: Rejected anywhere a token can reach a command line. The first thirteen are
#: the plan's stated set; the control characters and the quotes are added
#: because a token carrying either defeats the same naive renderer for the same
#: reason -- quoting is the renderer's job, never the plan's.
SHELL_METACHARACTERS: frozenset[str] = frozenset('|&;$`><\\(){}\n\r\t\x00\'"')

#: A run of `=` collapses to one: `name`, `name=version` and `name==version`
#: are the same apt intent written three ways, and three spellings of one
#: intent are three digests of one environment.
_APT_PIN = re.compile(r'=+')

FlagValue = str | int | bool
TestValue = tuple[str, ...] | str


class DepPlanError(RuntimeError):
    """A plan that must not be rendered, pinned or shipped."""


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InstallCommand:
    """One allowlisted executable and its already-tokenised arguments.

    `args` is a tuple, never a string, so there is no place for a renderer to
    have to split -- and therefore no place for a separator to be smuggled in.
    """

    tool: str
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class DepPlan:
    """A resolved, per-language build/test environment, as data.

    `build_flags` and `test_invocation` are mappings by nature, but a dict is
    neither hashable nor ordered-by-content, and this record is both a cache key
    and a frozen value. They are therefore carried as key-sorted tuples of
    pairs, which `canonicalize` establishes and `to_canonical_json` renders back
    out as JSON objects.
    """

    lang: str
    toolchain_version: str
    package_manager: str
    schema: int = SCHEMA
    manifest_files: tuple[str, ...] = ()
    apt_packages: tuple[str, ...] = ()
    install_commands: tuple[InstallCommand, ...] = ()
    build_flags: tuple[tuple[str, FlagValue], ...] = ()
    test_invocation: tuple[tuple[str, TestValue], ...] = ()
    needs_git_metadata: bool = False


# --------------------------------------------------------------------------
# validation: the closed enums and the metacharacter gate
# --------------------------------------------------------------------------


def _reject_metacharacters(token: str, where: str) -> None:
    bad = sorted({c for c in token if c in SHELL_METACHARACTERS})
    if bad:
        raise DepPlanError(
            f'{where} contains shell metacharacter(s) {"".join(bad)!r}: {token!r}. '
            'A plan describes tokens, not a command line; anything that could '
            'change the shape of a rendered command is refused at the plan level '
            'rather than trusted to a downstream quoter'
        )


def _require_nonempty(value: str, where: str) -> str:
    if not value.strip():
        raise DepPlanError(f'{where} is empty')
    return value


def _require_unique_keys(pairs: tuple[tuple[str, object], ...], where: str) -> None:
    keys = [k for k, _ in pairs]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise DepPlanError(
            f'{where} declares {", ".join(repr(k) for k in dupes)} more than once. '
            'The canonical json renders these as one object, so a duplicate key '
            'would silently drop a value and give two environments one digest'
        )


def _apt_parts(package: str) -> tuple[str, str]:
    """`(name, version)` of an apt entry in any accepted pin spelling."""
    name, _, version = _APT_PIN.sub('=', package.strip()).partition('=')
    return name, version


def validate(plan: DepPlan) -> None:
    """Raise `DepPlanError` unless every field is inside the closed enums."""
    if plan.schema != SCHEMA:
        raise DepPlanError(f'unknown DepPlan schema {plan.schema!r}; expected {SCHEMA}')
    if plan.lang not in LANGS:
        raise DepPlanError(
            f'unknown lang {plan.lang!r}; expected one of {", ".join(LANGS)}'
        )
    if plan.package_manager not in PACKAGE_MANAGERS[plan.lang]:
        allowed = ', '.join(sorted(PACKAGE_MANAGERS[plan.lang]))
        raise DepPlanError(
            f'{plan.lang} cannot be provisioned by {plan.package_manager!r}; '
            f'expected one of {allowed}'
        )
    _require_nonempty(plan.toolchain_version, 'toolchain_version')
    _reject_metacharacters(plan.toolchain_version, 'toolchain_version')

    allowed_tools = TOOL_ALLOWLIST[plan.lang]
    for command in plan.install_commands:
        if command.tool not in allowed_tools:
            raise DepPlanError(
                f'{command.tool!r} is not an allowlisted tool for {plan.lang}; '
                f'expected one of {", ".join(sorted(allowed_tools))}. The allowlist '
                'is closed on purpose: a shell or a downloader in it would make '
                'every other check in this module decorative'
            )
        for arg in command.args:
            _require_nonempty(arg, f'an argument of {command.tool!r}')
            _reject_metacharacters(arg, f'an argument of {command.tool!r}')

    for package in plan.apt_packages:
        _require_nonempty(package, 'an apt package')
        _reject_metacharacters(package, 'an apt package')
        name, version = _apt_parts(package)
        if not name or (version == '' and '=' in package):
            raise DepPlanError(
                f'apt package {package!r} is not `name` or `name=version`'
            )

    for path in plan.manifest_files:
        _require_nonempty(path, 'a manifest file')
        _reject_metacharacters(path, 'a manifest file')

    _require_unique_keys(plan.build_flags, 'build_flags')
    for key, value in plan.build_flags:
        _require_nonempty(key, 'a build flag name')
        _reject_metacharacters(key, 'a build flag name')
        if isinstance(value, str):
            _reject_metacharacters(value, f'the value of build flag {key!r}')

    _require_unique_keys(plan.test_invocation, 'test_invocation')
    for key, value in plan.test_invocation:
        _require_nonempty(key, 'a test_invocation key')
        _reject_metacharacters(key, 'a test_invocation key')
        for token in (value,) if isinstance(value, str) else value:
            _require_nonempty(token, f'a token of test_invocation {key!r}')
            _reject_metacharacters(token, f'a token of test_invocation {key!r}')


# --------------------------------------------------------------------------
# canonicalisation
# --------------------------------------------------------------------------


def _canonical_apt(package: str) -> str:
    name, version = _apt_parts(package.lower())
    return f'{name}={version}' if version else name


def _canonical_flag_value(value: FlagValue) -> FlagValue:
    return value.strip() if isinstance(value, str) else value


def _canonical_test_value(value: TestValue) -> TestValue:
    if isinstance(value, str):
        return value.strip()
    return tuple(token.strip() for token in value)


def canonicalize(plan: DepPlan) -> DepPlan:
    """One spelling per environment. Idempotent: `c(c(x)) == c(x)`.

    Set-like fields are sorted and deduped; `install_commands` keep the order
    they were given, because install order is semantic (a `pip install` after an
    `apt-get` is not the same environment as the reverse) and sorting it would
    canonicalise away a real difference.
    """
    return replace(
        plan,
        manifest_files=tuple(sorted({p.strip() for p in plan.manifest_files})),
        apt_packages=tuple(sorted({_canonical_apt(p) for p in plan.apt_packages})),
        install_commands=tuple(
            InstallCommand(
                tool=command.tool.strip(),
                args=tuple(arg.strip() for arg in command.args),
            )
            for command in plan.install_commands
        ),
        build_flags=tuple(
            sorted(
                ((k.strip(), _canonical_flag_value(v)) for k, v in plan.build_flags),
                key=lambda pair: pair[0],
            )
        ),
        test_invocation=tuple(
            sorted(
                ((k.strip(), _canonical_test_value(v)) for k, v in plan.test_invocation),
                key=lambda pair: pair[0],
            )
        ),
    )


# --------------------------------------------------------------------------
# digest and lock key
# --------------------------------------------------------------------------


def to_canonical_json(plan: DepPlan) -> str:
    """The canonical plan as compact, key-sorted json. No host path, no clock."""
    canonical = canonicalize(plan)
    payload = {
        'apt_packages': list(canonical.apt_packages),
        'build_flags': {k: v for k, v in canonical.build_flags},
        'install_commands': [
            {'args': list(command.args), 'tool': command.tool}
            for command in canonical.install_commands
        ],
        'lang': canonical.lang,
        'manifest_files': list(canonical.manifest_files),
        'needs_git_metadata': bool(canonical.needs_git_metadata),
        'package_manager': canonical.package_manager,
        'schema': int(canonical.schema),
        'test_invocation': {
            k: (v if isinstance(v, str) else list(v)) for k, v in canonical.test_invocation
        },
        'toolchain_version': canonical.toolchain_version,
    }
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def dep_plan_digest(plan: DepPlan) -> str:
    """sha256 of the canonical json -- equal for two spellings of one plan."""
    return hashlib.sha256(to_canonical_json(plan).encode('utf-8')).hexdigest()


def env_lock_key(repo_tree_sha256: str, base_image_digest: str, plan: DepPlan) -> str:
    """The content key an environment lock is stored under.

    Composite of the three things whose movement means a different environment:
    the tree that was resolved, the base image it was resolved against, and the
    INTENT that resolution produced. Downloaded dependency bytes are never
    hashed -- see the module docstring.
    """
    material = f'{repo_tree_sha256}\n{base_image_digest}\n{dep_plan_digest(plan)}'
    return hashlib.sha256(material.encode('utf-8')).hexdigest()
