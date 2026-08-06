"""What the resolver is allowed to know about a repo, gathered from disk.

`env_resolver.build_user_prompt` takes manifest CONTENTS but only test PATHS,
and says so in its signature because "the signature is the enforcement". This
module is the other half of that promise: the thing that actually walks a
checkout and produces arguments for it. A signature cannot stop a caller from
passing a test body as a manifest, so the gathering is written once, here, and
the leak boundary is a property of the WALK rather than of each caller's care.

THE RULE, restated as code. A stage that has the sources has the answer, so:

  * a file's CONTENT is read only when its name is a recognised dependency or
    build manifest (`_ROOT_CATALOG` / `LANG_MANIFESTS`). Never a source file,
    never a test file -- not even one that happens to sit next to a manifest.
  * a test file contributes its repo-relative PATH and nothing else. The list is
    built by walking, so no read ever happens on that branch at all.

DETERMINISM. Two runs on one checkout must send the model the same bytes, so
every list is sorted, every mapping is built in sorted order, `rglob`'s
filesystem order is never trusted, and CRLF is normalised away. The gatherer
takes no clock, no environment and no host path into its output.

WHERE THE CAPABILITIES COME FROM. They are the BASE IMAGE's, not the repo's, so
they are not gathered here at all -- `base_capabilities` reads them off the
language plugin, which is where the baked toolchain is already declared.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

__all__ = [
    'CI_WORKFLOW_DIR',
    'CI_WORKFLOW_SUFFIXES',
    'LANG_MANIFESTS',
    'MAX_MANIFEST_BYTES',
    'MAX_MANIFEST_DEPTH',
    'MAX_MANIFEST_FILES',
    'ROOT_CATALOG',
    'ResolverInputs',
    'ResolverInputsError',
    'SKIP_DIRS',
    'TEST_DIR_NAMES',
    'TEST_NAME_PATTERNS',
    'base_capabilities',
    'detect_framework',
    'gather_resolver_inputs',
    'required_plan_slots',
]


class ResolverInputsError(RuntimeError):
    """The checkout cannot be reduced to a leak-safe resolver input set."""


# --------------------------------------------------------------------------
# what counts as a manifest
# --------------------------------------------------------------------------

#: Per-language dependency/build manifests, by exact basename. This is the set a
#: NESTED directory may contribute: a monorepo's second `pyproject.toml` is a
#: real fact about a python build, while the `Cargo.toml` of an editor plugin
#: vendored inside a C repository is noise that would invite the model to
#: resolve the wrong toolchain.
LANG_MANIFESTS: Mapping[str, frozenset[str]] = {
    'python': frozenset({
        '.python-version', 'Pipfile', 'Pipfile.lock', 'environment.yml',
        'poetry.lock', 'pyproject.toml', 'setup.cfg', 'setup.py', 'tox.ini',
        'uv.lock',
    }),
    'go': frozenset({'go.mod', 'go.sum', 'go.work', 'go.work.sum'}),
    'rust': frozenset({
        'Cargo.lock', 'Cargo.toml', 'rust-toolchain', 'rust-toolchain.toml',
    }),
    'c': frozenset({
        'CMakeLists.txt', 'GNUmakefile', 'Makefile', 'configure.ac',
        'conanfile.txt', 'meson.build', 'vcpkg.json',
    }),
    'cpp': frozenset({
        'CMakeLists.txt', 'GNUmakefile', 'Makefile', 'configure.ac',
        'conanfile.txt', 'meson.build', 'vcpkg.json',
    }),
    'java': frozenset({
        'build.gradle', 'build.gradle.kts', 'gradle.properties',
        'gradle-wrapper.properties', 'pom.xml', 'settings.gradle',
        'settings.gradle.kts',
    }),
    'csharp': frozenset({
        'Directory.Build.props', 'Directory.Packages.props', 'global.json',
        'packages.lock.json',
    }),
}

#: Basename GLOBS that are manifests too. Kept apart from the exact sets because
#: a glob cannot be a frozenset membership test, and `requirements-dev.txt` is
#: as much a manifest as `requirements.txt`.
_MANIFEST_GLOBS: Mapping[str, tuple[str, ...]] = {
    'python': ('requirements*.txt',),
    'csharp': ('*.csproj', '*.sln'),
}

#: At the repo ROOT every language's manifests are eligible, whatever `--lang`
#: says: a python repo driven by a top-level Makefile, or a C repo carrying a
#: `.python-version` for its test harness, are facts the resolver needs. The
#: narrowing to one language applies only BELOW the root.
ROOT_CATALOG: frozenset[str] = frozenset().union(*LANG_MANIFESTS.values())

_ROOT_GLOBS: tuple[str, ...] = tuple(
    sorted({g for globs in _MANIFEST_GLOBS.values() for g in globs})
)

#: CI describes how upstream itself builds and tests, which is the most direct
#: statement of intent a repo makes. Always eligible, at its fixed location.
CI_WORKFLOW_DIR = '.github/workflows'
CI_WORKFLOW_SUFFIXES: tuple[str, ...] = ('.yml', '.yaml')

#: Root-level CI configs that are single files rather than a directory.
_ROOT_CI_FILES: frozenset[str] = frozenset({'.gitlab-ci.yml', '.travis.yml'})

#: Never descended into. Two reasons, and both matter: a vendored dependency's
#: manifests describe SOMEONE ELSE's build, and a build output directory is
#: nondeterministic, so either one would make the prompt depend on whether the
#: repo had been built on this machine before.
SKIP_DIRS: frozenset[str] = frozenset({
    '.git', '.gradle', '.hg', '.idea', '.mypy_cache', '.pytest_cache', '.svn',
    '.tox', '.venv', '__pycache__', 'build', 'dist', 'node_modules',
    'target', 'third_party', 'vendor', 'venv',
})

#: How deep a nested manifest may live. A manifest below this is describing a
#: sub-sub-component, not the build the harness is about to run.
MAX_MANIFEST_DEPTH = 4

#: Per-file content cap. A lockfile can be a megabyte, and a megabyte of pinned
#: hashes tells the model nothing the first few thousand lines did not.
MAX_MANIFEST_BYTES = 65_536

#: How many manifests may be sent. Generous; a repo over it is a monorepo, and
#: the overflow is REPORTED (`ResolverInputs.omitted_manifests`) rather than
#: dropped in silence.
MAX_MANIFEST_FILES = 60

_TRUNCATION_MARKER = '\n... [truncated by taskgen at {n} bytes]\n'


# --------------------------------------------------------------------------
# what counts as a test
# --------------------------------------------------------------------------

#: A directory whose name says everything under it is a test. Matched at any
#: depth, because `src/test/java/...` and `tests/unit/...` are the same claim.
TEST_DIR_NAMES: frozenset[str] = frozenset({
    '__tests__', 'spec', 'specs', 'test', 'testing', 'tests',
})

#: Basename globs that make a file a test wherever it lives.
TEST_NAME_PATTERNS: Mapping[str, tuple[str, ...]] = {
    'python': ('conftest.py', 'test_*.py', '*_test.py'),
    'go': ('*_test.go',),
    'rust': ('*_test.rs', 'test_*.rs'),
    'c': ('*_test.c', 'test_*.c', 'test_*.sh'),
    'cpp': ('*_test.cc', '*_test.cpp', 'test_*.cc', 'test_*.cpp'),
    'java': ('*Test.java', '*TestCase.java', '*Tests.java'),
    'csharp': ('*Test.cs', '*Tests.cs'),
}


@dataclass(frozen=True)
class ResolverInputs:
    """Everything the resolver may see, and nothing else, in canonical order.

    `manifest_files` maps a repo-relative POSIX path to that file's text.
    `test_paths` are repo-relative POSIX paths and carry NO content -- the type
    is `tuple[str, ...]` and not a mapping precisely so a body has nowhere to go.
    """

    manifest_files: Mapping[str, str]
    test_paths: tuple[str, ...]
    framework: str
    capabilities: tuple[str, ...] = ()
    omitted_manifests: tuple[str, ...] = ()


def base_capabilities(plugin) -> tuple[str, ...]:
    """The BASE IMAGE's baked toolchains, read off the language plugin.

    Not probed from a container and not inferred from the repo: the plugin
    already had to declare what its base ships in order to render a Dockerfile
    against it, so re-deriving the list anywhere else is how the prompt and the
    image start disagreeing about what is installed.
    """
    declared = tuple(str(c) for c in getattr(plugin, 'baked_capabilities', ()))
    env = plugin.toolchain_spec().env
    return tuple(sorted(set(declared) | {f'ENV {k}={v}' for k, v in env.items()}))


def required_plan_slots(plugin) -> tuple[str, ...]:
    """The plan slots this language's gap cannot render without.

    Read off the plugin for the same reason `base_capabilities` is: the plugin
    already had to know them to write `render_gap`, and `validate_dep_plan`
    enforces exactly this list. A second spelling anywhere else is how the model
    starts being rejected for a requirement it was never shown.

    Not gathered from the repo, so nothing here can carry a body.
    """
    return tuple(sorted({str(s) for s in getattr(plugin, 'required_plan_slots', ())}))


# --------------------------------------------------------------------------
# the walk
# --------------------------------------------------------------------------


def _walk(root: Path) -> list[tuple[str, ...]]:
    """Every regular file under `root`, as repo-relative parts, sorted.

    `os.walk(followlinks=False)` rather than `rglob`: a symlinked directory
    would otherwise be walked twice under two names -- or forever, if it points
    at an ancestor -- and a prompt whose content depends on a symlink cycle is
    not a deterministic prompt. Pruned directories are dropped in place so the
    walk never pays to descend a `node_modules`.
    """
    found: list[tuple[str, ...]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        here = Path(dirpath)
        for name in sorted(filenames):
            entry = here / name
            if entry.is_symlink() or not entry.is_file():
                continue
            found.append(entry.relative_to(root).parts)
    found.sort()
    return found


def _matches_glob(name: str, globs: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(name, g) for g in globs)


def _is_manifest(parts: tuple[str, ...], lang: str) -> bool:
    name = parts[-1]
    depth = len(parts)
    posix = '/'.join(parts)

    if posix in _ROOT_CI_FILES:
        return True
    if posix.startswith(CI_WORKFLOW_DIR + '/'):
        return name.endswith(CI_WORKFLOW_SUFFIXES)

    if depth == 1:
        return name in ROOT_CATALOG or _matches_glob(name, _ROOT_GLOBS)
    if depth > MAX_MANIFEST_DEPTH:
        return False
    return (
        name in LANG_MANIFESTS.get(lang, frozenset())
        or _matches_glob(name, _MANIFEST_GLOBS.get(lang, ()))
    )


def _is_test_path(parts: tuple[str, ...], lang: str) -> bool:
    if any(part in TEST_DIR_NAMES for part in parts[:-1]):
        return True
    return _matches_glob(parts[-1], TEST_NAME_PATTERNS.get(lang, ()))


def _read_manifest(path: Path) -> str:
    """A manifest's text, bounded and newline-normalised.

    `errors='replace'` rather than a raise: a lockfile with one stray byte is
    still the lockfile, and refusing the whole resolution over an encoding
    detail would turn a cosmetic defect into an unresolvable repo.
    """
    raw = path.read_bytes()
    truncated = len(raw) > MAX_MANIFEST_BYTES
    text = raw[:MAX_MANIFEST_BYTES].decode('utf-8', errors='replace')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    if truncated:
        text += _TRUNCATION_MARKER.format(n=MAX_MANIFEST_BYTES)
    return text


def detect_framework(lang: str, manifest_files: Mapping[str, str],
                     test_paths: tuple[str, ...]) -> str:
    """The test runner, named from the manifests -- never from a test body.

    Deliberately coarse. The resolver is told what the repo DECLARES it runs
    tests with; deciding the exact invocation is the model's job and the
    validator's gate, not a heuristic's.
    """
    names = {p.rsplit('/', 1)[-1] for p in manifest_files}
    if lang == 'go':
        return 'go test'
    if lang == 'rust':
        return 'cargo test'
    if lang == 'python':
        blob = '\n'.join(manifest_files.values())
        if 'pytest' in blob or any(p.endswith('conftest.py') for p in test_paths):
            return 'pytest'
        return 'unittest'
    if lang == 'java':
        if 'pom.xml' in names:
            return 'maven-surefire'
        if {'build.gradle', 'build.gradle.kts'} & names:
            return 'gradle'
        return 'junit'
    if lang in ('c', 'cpp'):
        if 'CMakeLists.txt' in names:
            return 'ctest'
        if {'Makefile', 'GNUmakefile'} & names:
            return 'make'
        return 'unknown'
    if lang == 'csharp':
        return 'dotnet test'
    return 'unknown'


def _assert_no_bodies(manifest_files: Mapping[str, str], test_paths: tuple[str, ...],
                      lang: str) -> None:
    """The leak boundary, checked rather than assumed.

    Everything above is already supposed to make this impossible; a mistake in
    `_is_manifest` would be a benchmark-corrupting one, so it is asserted at the
    point the two collections meet instead of being left to a code review.
    """
    tests = set(test_paths)
    for name in manifest_files:
        if name in tests:
            raise ResolverInputsError(
                f'{name} was collected as BOTH a manifest and a test file. Its '
                'content would reach the model, and a test body is the answer '
                'the benchmark hides'
            )
        if _is_test_path(tuple(name.split('/')), lang):
            raise ResolverInputsError(
                f'{name} reads as a test file but was collected for its CONTENT. '
                'The resolver is shown manifests and test PATHS, never bodies'
            )


def gather_resolver_inputs(*, repo: Path, lang: str,
                           capabilities: tuple[str, ...] = ()) -> ResolverInputs:
    """Walk one checkout into the arguments `build_user_prompt` accepts.

    One walk, two collectors, and only the manifest collector ever opens a file.

    The TEST test runs FIRST, and that order is the leak boundary: a `Makefile`
    under `tests/` is build configuration by name and a test-directory member by
    location, and when the two rules disagree the one that does not read the
    file wins. Losing a little build signal is a cost; reading something out of
    a test directory is a benchmark defect.
    """
    root = Path(repo).resolve()
    if not root.is_dir():
        raise ResolverInputsError(f'not a repository checkout: {root}')

    manifest_paths: list[tuple[str, Path]] = []
    tests: list[str] = []
    for parts in _walk(root):
        posix = '/'.join(parts)
        if _is_test_path(parts, lang):
            tests.append(posix)
        elif _is_manifest(parts, lang):
            manifest_paths.append((posix, root.joinpath(*parts)))

    kept = manifest_paths[:MAX_MANIFEST_FILES]
    omitted = tuple(name for name, _ in manifest_paths[MAX_MANIFEST_FILES:])

    manifest_files = {name: _read_manifest(path) for name, path in kept}
    test_paths = tuple(sorted(tests))
    _assert_no_bodies(manifest_files, test_paths, lang)

    return ResolverInputs(
        manifest_files=manifest_files,
        test_paths=test_paths,
        framework=detect_framework(lang, manifest_files, test_paths),
        capabilities=tuple(sorted(set(capabilities))),
        omitted_manifests=omitted,
    )
