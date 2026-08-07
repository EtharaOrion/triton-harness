"""The rust plugin: whole-suite grading, measured floor, vendored offline build.

Rust is the first plugin whose graded set is NOT derived by a tree-sitter
parser. python and go link individual tests to a carved function; rust here
deletes the WHOLE `src/` tree of an integration crate, so nothing linkable
remains and the "which tests are graded" question collapses to "the whole
integration suite". The denominator therefore has to be MEASURED against the
intact tree once (measure.py, phase 1) and PINNED in `graded.lock.json` so the
shipped test.sh grades against a number that is not derivable from the carved
image (which by construction cannot see it). The plan's two-phase build exists
for exactly this shape.

FLOOR MODE: equality. Rust's libtest reports one `test result:` line per test
binary and does not abort a whole binary on a panic the way Go does (spike G1),
so `observed == EXPECTED` is a real assertion rather than a mis-fire.

TOOLCHAIN: rustc 1.87.0 via rustup. harbor-base ships an older rustc (1.79 per
the reference Dockerfile); the crate under test declares edition 2024 /
rust-version 1.87. wabt (`wast2json`) 1.0.41 is the second unpinned dependency:
tests/util/spectest.rs shells out to it for every .wast file and always passes
`--enable-custom-page-sizes`, which Ubuntu's 1.0.34 rejects. Both come in via
the pre-leakgate blocks so the leak scan sees the warmed target/ directory.

VENDORING via empty-lib-stub. `cargo vendor` resolves dependencies from
Cargo.toml + Cargo.lock; it does NOT need `src/` to contain any real code. A
one-byte `src/lib.rs` therefore reproduces the intact tree's vendor set exactly
(same 12 crates measured in the spike) while keeping the carved sources out of
every layer. The stub warms `target/deps/*` for the vendored crates so the
graded run does not rebuild serde from scratch; the spacewasm-specific
fingerprints and incremental artifacts are then explicitly deleted.

STRINGS-TARGET LEAK ASSERT. rustc leaves mangled symbol names and debug-info
paths in target/*.o. `strings target/... | grep -E '<crate>|<workdir>/src'`
would recover the carved API without touching src/ at all, so the assert makes
that fail-fast at build time rather than a nasty surprise after `docker save`.

DOCKERFILE INVARIANTS. `git init` is NOT synthesised (unlike python), so the
base's ban on it stays in force. The empty-lib-stub uses `mkdir src; :>src/lib.rs`
which contains neither `repos-src`/`repo-src` nor `git init` -- verify the
composed Dockerfile against `_assert_dockerfile_invariants` in the test file.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar, Mapping

from ..depplan import (
    DepPlan,
    FlagValue,
    HarnessValue,
    TestValue,
    canonicalize,
    validate,
)
from . import base as B
from .base import DepWarmSpec, EnvSpec, GradedSet, ToolchainSpec

__all__ = [
    'BAKED_CAPABILITIES',
    'HARNESSES',
    'Harness',
    'INTACT_VENDOR_BLOCK',
    'INTEGRITY_HARNESS_FILES',
    'MEASURE_NO_WARM_COMMENT',
    'MIN_WAST',
    'BASE_IMAGE',
    'REQUIRED_BUILD_FLAGS',
    'REQUIRED_PLAN_SLOTS',
    'REQUIRED_TEST_INVOCATION_KEYS',
    'RUST_MEASURE_DEP_PLAN',
    'RUST_SPACEWASM_HARNESS',
    'RUST_VERSION',
    'RustPlugin',
    'SHIMS_PATH',
    'TEST_COMMAND',
    'TOOLCHAIN_SOURCE',
    'ToolSpec',
    'WABT_VERSION',
    'read_harness',
]

RUST_VERSION = '1.87.0'

#: Where the base image's mise keeps the shims that resolve the selected
#: toolchain. The zz- profile.d file prepends exactly this, so a base that moved
#: its shims would leave the pin asserting a rustc nothing on PATH points at.
SHIMS_PATH = '/opt/mise/shims'

#: The CDN a `rustup toolchain install` would reach, named in the comment that
#: explains why this image does not: the toolchain is baked, so no task build
#: pays a download it cannot repeat under `--network=none`.
TOOLCHAIN_SOURCE = 'static.rust-lang.org'

#: Per-language base carrying rustc/cargo and wasm2wat already, so no task build
#: reaches static.rust-lang.org for a compiler the image can simply hold.
BASE_IMAGE = '426628337772.dkr.ecr.ap-south-2.amazonaws.com/triton/base-rust@sha256:9bb61cf220efe0c7c8236841c3cb985aa2ce744c414e1fbeaab858832daae911'
WABT_VERSION = '1.0.41'

#: rust-spacewasm's own harness as a `DepPlan.harness` section -- the fixed
#: point of the refactor: rendered, it reproduces the scripts and the
#: pre-leakgate blocks this plugin used to hardcode byte for byte. Every value
#: is a fact about the rust-spacewasm REPOSITORY, which is exactly why none of
#: it belongs in a language plugin.
#:
#: The four `harness_names` are the graded integration binaries. They survive
#: the src/ carve because every symbol they touch is either a compilable
#: test-side helper (in tests/util/) or a .wast file lifted from the official
#: spec conformance corpus. Deleting src/ makes them fail to link; restoring
#: src/ makes them pass again. That is the whole point of the task.
#:
#: `exclude_names` carries the ONE case in custom_page_sizes_integration that
#: declares two ~4 GiB linear memories and is killed by the cgroup OOM killer
#: at 7 GiB; grading it would make the reward a function of host RAM.
#:
#: The HARNESS COUNT is a plan claim, but never a TRUSTED one: the names are
#: derivable from test PATHS (which is all the resolver is shown), and
#: `emit._rust_grader_metadata` enumerates the intact tree's own cargo
#: integration targets host-side and refuses when the two sets differ.
RUST_SPACEWASM_HARNESS: tuple[tuple[str, HarnessValue], ...] = (
    ('corpus_dir', 'tests'),
    ('corpus_label', 'wast'),
    ('corpus_min', 90),
    ('corpus_pattern', '*.wast'),
    ('exclude_names', ('memory_max',)),
    ('harness_files', (
        'tests/core_integration.rs',
        'tests/regression_integration.rs',
        'tests/statistics_integration.rs',
        'tests/custom_page_sizes_integration.rs',
    )),
    ('harness_names', (
        'core_integration',
        'regression_integration',
        'statistics_integration',
        'custom_page_sizes_integration',
    )),
    ('prune_manifest_entries', ('crates/*', 'fuzz')),
    ('prune_manifest_keys', ('members', 'exclude')),
    ('prune_paths', ('crates', 'fuzz')),
    ('support_files', (
        'tests/util/mod.rs',
        'tests/util/spectest.rs',
        'tests/util/inspector.rs',
    )),
    ('tool_consumers', ('tests/util/spectest.rs',)),
    ('tool_names', ('wast2json',)),
    ('tool_packages', ('wabt',)),
    ('tool_versions', (WABT_VERSION,)),
)

_LOGS_DEFAULT = '${VERIFIER_DIR:-' + B.LOGS_DIR + '}'
_MEASURE_LOGS_DEFAULT = '${MEASURE_DIR:-' + B.LOGS_DIR + '}'

#: The measure image's dep-warm slot. rust has no warm stage at all (see
#: `dep_warm_spec`: `cargo vendor` needs a resolvable [lib] target, so vendoring
#: happens inside the graded image), so this comment IS the slot's content.
MEASURE_NO_WARM_COMMENT = '# no warmed dependencies (rust vendors in the graded image)'

#: The measure image's vendor step, verbatim and FIXED. It is scaffolding, not
#: provisioning a resolver may vary: it exists because the measure image holds
#: the INTACT tree, where the shipped image's empty-lib-stub would half-carve
#: src/ and break every compile. It also cannot live in `render_gap`'s string --
#: it has to run AFTER the repo COPY, since `cargo vendor` reads Cargo.toml,
#: and the gap slot sits above that COPY.
INTACT_VENDOR_BLOCK = '\n'.join([
    '# Vendor against the INTACT src/. No empty-lib-stub here: overwriting',
    '# src/lib.rs while the other 30 .rs files remain would half-carve the',
    '# tree and every test would fail to compile. The intact src/ is the',
    '# real one, so cargo vendor resolves and downloads normally.',
    'RUN set -eux; \\',
    '    CARGO_NET_OFFLINE=false cargo vendor > /tmp/vendor-config.toml; \\',
    '    mkdir -p .cargo; \\',
    '    cp /tmp/vendor-config.toml .cargo/config.toml; \\',
    '    rm -f /tmp/vendor-config.toml',
])

#: What the base image already provides, as `--resolve-env` states it to a
#: model. Sorted and version-exact, so the same prompt is built on every run.
#: Declared, never probed: the same facts the pin assert in `render_gap` checks
#: at build time.
BAKED_CAPABILITIES: tuple[str, ...] = (
    'apt-get (build time only; the graded run has no network)',
    f'cargo {RUST_VERSION}, with CARGO_NET_OFFLINE=true already exported',
    'mise, whose shims resolve the selected toolchain under a login shell',
    f'rustc {RUST_VERSION} (edition 2024 capable)',
)

#: Every `build_flags` key `render_gap` interpolates. Enumerated once, read by
#: BOTH the renderer and `RustPlugin.validate_dep_plan`, so a flag added to the
#: rendered text without being added here is the only way the two can disagree.
REQUIRED_BUILD_FLAGS: tuple[str, ...] = ('shims_path', 'toolchain_source')

#: Every `test_invocation` key the rust measure path needs. `render_gap` does
#: not interpolate it -- the measure script runs the graded command itself --
#: but a plan that does not state what it was resolved FOR is a plan nobody can
#: check against the image it produced, so it is required at the same gate.
REQUIRED_TEST_INVOCATION_KEYS: tuple[str, ...] = ('command',)

#: The same requirements as the resolver is told them, before it answers. Kept
#: adjacent to the checks above so the ask and the rejection cannot drift apart.
#: Leak-safe: slot names, shapes and toolchain versions only -- never a repo
#: path, a crate name or a source body.
REQUIRED_PLAN_SLOTS: tuple[str, ...] = (
    'build_flags["shims_path"] must be a non-empty string: the directory the '
    f'base image keeps its mise shims in, e.g. "{SHIMS_PATH}". The gap writes a '
    'zz- profile.d file prepending it to PATH, which is the ONLY reason a login '
    'shell resolves the pinned toolchain rather than the base default',
    'build_flags["toolchain_source"] must be a non-empty string: the CDN a '
    f'toolchain install would otherwise reach, e.g. "{TOOLCHAIN_SOURCE}". The '
    'gap names it in the comment explaining why this image downloads nothing',
    'install_commands must NOT build or test the graded sources -- '
    'test_invocation runs the suite itself, and the image vendors its crates in '
    'a later fixed step -- so for a cargo repo this list is USUALLY EMPTY. Add a '
    'step only to PREPARE the environment; never a bare "cargo build". A build '
    'step that cannot succeed makes the whole plan unbuildable, not merely '
    'suboptimal',
    'apt_packages lists ONLY system libraries the sources need that the base '
    'image lacks; rustc, cargo and mise are already baked in, so naming them '
    'here buys nothing and costs a network fetch',
    'package_manager must be cargo: the gap pins the toolchain cargo drives and '
    'the measure image vendors its dependencies with it',
    'test_invocation["command"] must be a non-empty list of argv tokens: the '
    'command that runs the whole graded suite, and it must NAME EVERY entry of '
    'harness["harness_names"] with its own `--test` flag, e.g. ["cargo", '
    '"test", "--offline", "--test", "alpha", "--test", "beta"]. Do NOT use '
    '"--workspace", "--all" or a bare "cargo test": those select targets by a '
    'rule rather than by name, so the run that MEASURES the denominator and the '
    'run that GRADES against it can drift apart as soon as the workspace '
    'changes. Append "--", then "--skip <name>" once per harness["exclude_names"] '
    'entry',
    'toolchain_version must carry at least major.minor.patch, e.g. "1.87.0": it '
    'is the rustc version, and the build-time login-shell pin matches it '
    'verbatim against `rustc --version`',
    'harness is REQUIRED for rust and describes THIS repository\'s own '
    'integration-test harness; it is what the graded verifier script and the '
    'measure script are rendered from. State it from the build manifests and '
    'the TEST FILE PATHS you were given, and from nothing else.',
    'harness["harness_names"] is the list of cargo INTEGRATION-TEST target '
    'names the graded suite runs -- one per `--test` flag, e.g. '
    '["core_integration", "regression_integration"]. A target is a .rs file '
    'directly under tests/ (its name is the filename without .rs), a '
    'tests/<dir>/main.rs (named for the directory), or a [[test]] entry in '
    'Cargo.toml. ENUMERATE EVERY ONE OF THEM. Unit tests inside src/ are NOT '
    'integration targets and must not be listed. Listing fewer targets than the '
    'repository has is the single worst error you can make here: it pins the '
    'floor over only the part you named, every later gate still passes, and the '
    'rest of the suite goes ungraded. The harness counts the targets itself and '
    'will name any you missed',
    'harness["harness_files"] is the repo-relative root source file of each '
    'target in harness_names, SAME ORDER and SAME LENGTH, e.g. '
    '["tests/core_integration.rs"]. They are checked for existence before the '
    'suite is scored, so a solver cannot delete the oracle to make it pass',
    'harness["support_files"] lists the shared test-side files those targets '
    'import but that are not targets themselves, e.g. ["tests/util/mod.rs"]. '
    'Use [] when there are none',
    'harness["corpus_dir"], harness["corpus_pattern"], harness["corpus_label"] '
    'and harness["corpus_min"] describe a DATA corpus the suite reads (e.g. '
    'dir "tests", pattern "*.wast", label "wast", min 90). The verifier refuses '
    'to score a truncated corpus. OMIT ALL FOUR when the suite reads no data '
    'files -- most crates do not have one, and declaring a corpus that does not '
    'exist renders a gate that can never pass',
    'harness["tool_names"], harness["tool_versions"], harness["tool_packages"] '
    'and harness["tool_consumers"] describe EXTERNAL binaries the tests shell '
    'out to, as parallel lists (e.g. names ["wast2json"], versions ["1.0.41"], '
    'packages ["wabt"], consumers ["tests/util/spectest.rs"]). OMIT ALL FOUR '
    'unless a test genuinely execs a program that is not cargo: an asserted '
    'tool the base image lacks fails every build',
    'harness["prune_paths"], harness["prune_manifest_keys"] and '
    'harness["prune_manifest_entries"] remove out-of-scope cargo WORKSPACE '
    'members before vendoring, e.g. paths ["crates", "fuzz"], keys ["members", '
    '"exclude"], entries ["crates/*", "fuzz"] for a Cargo.toml declaring '
    'members = ["crates/*"] and exclude = ["fuzz"]. keys and entries are '
    'parallel; each key must be "members" or "exclude". OMIT ALL THREE for a '
    'single-package crate with no [workspace] section',
)

#: The one package manager rust's gap has prose for. `depplan.PACKAGE_MANAGERS`
#: already closes rust to exactly this, so the check is a belt on a brace -- but
#: it is the check that would fire first if that enum ever widened.
_SUPPORTED_MANAGERS: frozenset[str] = frozenset({'cargo'})


def _flag_str(build_flags: tuple[tuple[str, FlagValue], ...], key: str) -> str:
    """A build flag the gap's rendered text needs, or a refusal to render at all."""
    for name, value in build_flags:
        if name == key:
            if isinstance(value, str) and value:
                return value
            break
    raise B.LangError(
        f'the rust gap needs build_flags[{key!r}]: it must be a non-empty '
        'string. A plan without it would render a Dockerfile pin with a hole in '
        'it, and a hole in a toolchain pin is how a base swap goes unnoticed'
    )


def _test_tokens(
    test_invocation: tuple[tuple[str, TestValue], ...], key: str,
) -> tuple[str, ...]:
    """A test_invocation entry the rust measure path needs, as argv tokens."""
    for name, value in test_invocation:
        if name == key:
            tokens = (value,) if isinstance(value, str) else tuple(value)
            if tokens and all(token.strip() for token in tokens):
                return tokens
            break
    raise B.LangError(
        f'the rust plan needs test_invocation[{key!r}]: it must be a non-empty '
        'list of argv tokens naming the command that runs the graded suite. A '
        'plan that does not state what it was resolved to RUN cannot be checked '
        'against the image it produced'
    )


def _rustc_version(toolchain_version: str) -> str:
    """The rustc version the login-shell pin matches verbatim, or a refusal.

    The pin is `case "$v" in *"<version>"*)`, so a version that is not the exact
    string `rustc --version` prints either matches nothing -- turning every
    build into an exit 42 -- or, for something as short as a bare major, matches
    far too much and passes on a toolchain nobody asked for.
    """
    parts = toolchain_version.split('.')
    if len(parts) < 3 or not all(part.isdigit() for part in parts[:2]):
        raise B.LangError(
            f'toolchain_version {toolchain_version!r} must carry at least '
            'major.minor.patch, e.g. "1.87.0"; the rust gap matches it verbatim '
            'against what `rustc --version` prints under a login shell, and a '
            'looser pin either never matches or matches a toolchain nobody asked '
            'for'
        )
    return toolchain_version


#: The Cargo.toml `[workspace]` arrays a prune may rewrite. Closed because the
#: rendered edit is a literal string replacement, and a key cargo does not
#: define would render an edit that silently matches nothing.
_WORKSPACE_ARRAYS: frozenset[str] = frozenset({'members', 'exclude'})


def _slot_error(key: str, want: str) -> B.LangError:
    return B.LangError(
        f'the rust harness needs harness[{key!r}]: {want}. Without it the '
        'rendered verifier would either grade nothing or grade a suite nobody '
        'described, and a denominator nobody described is not a floor'
    )


def _tuple_slot(values: Mapping[str, HarnessValue], key: str,
                want: str, *, required: bool = False) -> tuple[str, ...]:
    raw = values.get(key)
    if raw is None:
        if required:
            raise _slot_error(key, want)
        return ()
    if isinstance(raw, (bool, int)):
        raise _slot_error(key, f'{want} (a list of strings, not a scalar)')
    tokens = (raw,) if isinstance(raw, str) else tuple(raw)
    if required and not tokens:
        raise _slot_error(key, want)
    return tokens


def _str_slot(values: Mapping[str, HarnessValue], key: str,
              want: str, *, required: bool = False) -> str:
    raw = values.get(key)
    if raw is None or raw == '':
        if required:
            raise _slot_error(key, want)
        return ''
    if not isinstance(raw, str):
        raise _slot_error(key, f'{want} (a single string)')
    return raw


def _int_slot(values: Mapping[str, HarnessValue], key: str, want: str) -> int:
    raw = values.get(key)
    if raw is None:
        return 0
    if isinstance(raw, bool):
        raise _slot_error(key, f'{want} (a whole number, not a boolean)')
    if isinstance(raw, int):
        parsed = raw
    elif isinstance(raw, str) and raw.strip().isdigit():
        parsed = int(raw)
    else:
        raise _slot_error(key, f'{want} (a whole number)')
    if parsed < 0:
        raise _slot_error(key, f'{want} (never negative)')
    return parsed


def _require_relative(path: str, key: str) -> str:
    """A repo-relative path, or a refusal. `..` escapes the tree it describes."""
    if path.startswith('/') or path.split('/')[0] == '..' or '/../' in path:
        raise _slot_error(
            key,
            f'a repo-relative path, but {path!r} is absolute or climbs out of '
            'the checkout; the verifier resolves it against ${REPO} and a path '
            'that escapes there names something no carve controls',
        )
    return path


def _parallel(columns: tuple[tuple[str, tuple[str, ...]], ...]) -> None:
    """Refuse a set of positional columns whose lengths disagree.

    Parallel tuples are read by INDEX, so a short column does not raise -- it
    silently re-pairs every later row and describes a harness nobody wrote.
    """
    lengths = {key: len(value) for key, value in columns}
    if len(set(lengths.values())) > 1:
        spelled = ', '.join(f'{key}={n}' for key, n in lengths.items())
        raise B.LangError(
            f'the rust harness reads {", ".join(k for k, _ in columns)} '
            f'positionally against each other, so they must be the same length; '
            f'got {spelled}. A short column does not fail loudly on its own -- it '
            're-pairs every row after it and describes a harness nobody wrote'
        )


@dataclass(frozen=True)
class _Scaffold:
    """The image blocks that run BELOW the gap slot, kept apart by NAME.

    The measure image takes the tool asserts and the prune but replaces the
    vendor step (it holds the intact tree, where the empty-lib-stub would
    half-carve src/) and drops the leak assert entirely. Selecting those by
    position from a flat tuple worked only while every repo contributed exactly
    the same blocks; a repo needing no external tool shifts every index.
    """

    tool_asserts: tuple[str, ...]
    prune: str
    vendor: str
    strings_assert: str


def _english_list(harness: 'Harness') -> str:
    """The pruned entries as prose: `crates/* and fuzz/`.

    A glob entry is shown verbatim; a plain directory gains a trailing slash, so
    the comment reads as the set of DIRECTORIES the prune removes rather than as
    a list of manifest tokens.
    """
    labels = [
        entry if entry.endswith('*') else f'{entry}/'
        for _key, entry in harness.prune_manifest
    ] or [f'{path}/' for path in harness.prune_paths]
    if len(labels) < 2:
        return ''.join(labels)
    return f'{", ".join(labels[:-1])} and {labels[-1]}'


@dataclass(frozen=True)
class ToolSpec:
    """One external binary the graded harness shells out to, and its pin.

    `package` is what ships the binary (`wabt` ships `wast2json`) and defaults
    to the binary's own name; `consumer` is the repo-relative test-side file
    that calls it, and is prose only.
    """

    name: str
    version: str
    package: str
    consumer: str


@dataclass(frozen=True)
class Harness:
    """A rust integration-test harness as data: run it, guard it, size its corpus.

    Everything the two rendered scripts and the pre-leakgate blocks used to
    hardcode about the rust-spacewasm REPO, read off `DepPlan.harness`.
    `read_harness` is the only constructor, so a slot that is missing, mistyped
    or inconsistent with its siblings becomes a `LangError` before any container
    exists rather than a shell script that greps an empty log and reports a
    floor of zero.
    """

    names: tuple[str, ...]
    files: tuple[str, ...]
    support_files: tuple[str, ...]
    exclude_names: tuple[str, ...]
    corpus_dir: str
    corpus_pattern: str
    corpus_label: str
    corpus_min: int
    tools: tuple[ToolSpec, ...]
    prune_paths: tuple[str, ...]
    prune_manifest: tuple[tuple[str, str], ...]

    @property
    def integrity_files(self) -> tuple[str, ...]:
        """Every file the graded verifier refuses to score without.

        The support files come FIRST because they are the shared oracle the
        per-target files import; the order is the one the reference test.sh
        checked them in.
        """
        return self.support_files + self.files

    @property
    def has_corpus(self) -> bool:
        return bool(self.corpus_dir and self.corpus_pattern)

    @property
    def corpus_var(self) -> str:
        return f'{self.corpus_label.upper()}_COUNT'

    @property
    def corpus_suffix(self) -> str:
        """`*.wast` -> `.wast`, for the prose that counts "N .wast files"."""
        return self.corpus_pattern.lstrip('*')

    @property
    def graded_argv(self) -> tuple[str, ...]:
        """The cargo invocation that runs exactly the declared targets."""
        argv = ['cargo', 'test', '--offline']
        for name in self.names:
            argv += ['--test', name]
        if self.exclude_names:
            argv.append('--')
            for name in self.exclude_names:
                argv += ['--skip', name]
        return tuple(argv)

    @property
    def graded_command(self) -> str:
        return ' '.join(self.graded_argv)


def read_harness(plan: DepPlan) -> Harness:
    """`plan.harness` as the record the renderers read, or a `LangError`.

    Every check here is reachable from the refine loop, so every message names
    the slot and says what a correct value looks like: a resolver that gets one
    wrong must be able to repair it without being shown the repository.
    """
    values = {key: value for key, value in canonicalize(plan).harness}

    names = _tuple_slot(
        values, 'harness_names',
        'the cargo integration-test target names the graded suite runs, one per '
        '`--test` flag, e.g. ["core_integration", "regression_integration"]',
        required=True,
    )
    files = _tuple_slot(
        values, 'harness_files',
        'the repo-relative root source file of each target named in '
        'harness_names, same order and same length, e.g. '
        '["tests/core_integration.rs"]',
        required=True,
    )
    _parallel((('harness_names', names), ('harness_files', files)))

    tool_names = _tuple_slot(
        values, 'tool_names',
        'the external binaries the graded harness shells out to, e.g. '
        '["wast2json"], or omit the slot entirely when it needs none',
    )
    tool_versions = _tuple_slot(
        values, 'tool_versions',
        'one exact version string per entry in tool_names, same order',
    )
    tool_packages = _tuple_slot(
        values, 'tool_packages',
        'one distribution name per entry in tool_names, same order',
    )
    tool_consumers = _tuple_slot(
        values, 'tool_consumers',
        'one repo-relative test-side file per entry in tool_names, same order',
    )
    if tool_names:
        _parallel((('tool_names', tool_names), ('tool_versions', tool_versions)))
    for optional, label in ((tool_packages, 'tool_packages'),
                            (tool_consumers, 'tool_consumers')):
        if optional:
            _parallel((('tool_names', tool_names), (label, optional)))

    prune_paths = _tuple_slot(
        values, 'prune_paths',
        'the out-of-scope workspace directories to remove before vendoring, '
        'e.g. ["crates", "fuzz"], or omit the slot for a single-package repo',
    )
    prune_keys = _tuple_slot(
        values, 'prune_manifest_keys',
        'the Cargo.toml workspace array each pruned entry is listed under, one '
        'of "members" or "exclude", parallel to prune_manifest_entries',
    )
    prune_entries = _tuple_slot(
        values, 'prune_manifest_entries',
        'the workspace array entry to drop, exactly as Cargo.toml spells it '
        '(e.g. "crates/*"), parallel to prune_manifest_keys',
    )
    _parallel((
        ('prune_manifest_keys', prune_keys),
        ('prune_manifest_entries', prune_entries),
    ))
    for key in prune_keys:
        if key not in _WORKSPACE_ARRAYS:
            raise _slot_error(
                'prune_manifest_keys',
                f'one of {", ".join(sorted(_WORKSPACE_ARRAYS))}, but {key!r} is '
                'neither. The prune rewrites a cargo WORKSPACE array, and a key '
                'cargo does not define is an edit no manifest would ever match',
            )

    harness = Harness(
        names=names,
        files=tuple(_require_relative(p, 'harness_files') for p in files),
        support_files=tuple(
            _require_relative(p, 'support_files')
            for p in _tuple_slot(
                values, 'support_files',
                'the shared test-side oracle files the targets import, e.g. '
                '["tests/util/mod.rs"]',
            )
        ),
        exclude_names=_tuple_slot(
            values, 'exclude_names',
            'the individual test names to pass to `--skip`, for cases whose '
            'result depends on the host rather than on the code under test',
        ),
        corpus_dir=_str_slot(
            values, 'corpus_dir',
            'the repo-relative directory holding the data corpus the suite '
            'reads, e.g. "tests"; omit it when the suite reads no corpus',
        ),
        corpus_pattern=_str_slot(
            values, 'corpus_pattern',
            'the filename glob of that corpus, e.g. "*.wast"',
        ),
        corpus_label=_str_slot(
            values, 'corpus_label',
            'a short lowercase word naming that corpus, e.g. "wast"',
        ),
        corpus_min=_int_slot(
            values, 'corpus_min',
            'the smallest corpus size worth grading, below which the verifier '
            'refuses rather than scoring a truncated set',
        ),
        tools=tuple(
            ToolSpec(
                name=name,
                version=tool_versions[i],
                package=tool_packages[i] if tool_packages else name,
                consumer=tool_consumers[i] if tool_consumers else '',
            )
            for i, name in enumerate(tool_names)
        ),
        prune_paths=tuple(_require_relative(p, 'prune_paths') for p in prune_paths),
        prune_manifest=tuple(zip(prune_keys, prune_entries)),
    )

    if harness.has_corpus:
        _require_relative(harness.corpus_dir, 'corpus_dir')
        if not harness.corpus_label.isidentifier():
            raise _slot_error(
                'corpus_label',
                'a short word usable as a shell variable stem, but '
                f'{harness.corpus_label!r} is not; the verifier builds '
                f'{harness.corpus_label.upper()}_COUNT out of it',
            )
    elif harness.corpus_min:
        raise _slot_error(
            'corpus_min',
            f'a floor on a corpus that harness["corpus_dir"] and '
            f'harness["corpus_pattern"] actually describe, but it is '
            f'{harness.corpus_min} with no corpus declared. A floor over '
            'nothing is a gate that can never be evaluated',
        )
    return harness


def _rust_measure_dep_plan() -> DepPlan:
    """rust's own environment as the record a resolver would have to produce.

    The fixed point of the exercise: fed to `RustPlugin.render_gap`, this plan
    reproduces the gap bytes `render_measure_dockerfile` hardcodes today.

    `apt_packages` and `install_commands` are EMPTY and must stay empty. rustc,
    cargo and mise are all BAKED into the per-language base -- that is exactly
    why the install block SELECTS a toolchain instead of reaching
    `static.rust-lang.org` -- and these two fields mean "what the gap INSTALLS".
    Listing a baked component would make the gap emit an `apt-get` line today's
    bytes do not contain, and that line would want network in the measure build.

    What is NOT described here: the wabt install, the workspace prune, the
    vendor step and the strings leak-assert. Those are fixed scaffolding (and
    for the shipped image, the leak proof), they run BELOW the repo COPY rather
    than in the gap slot, and they stay hardcoded.
    """
    plan = DepPlan(
        lang='rust',
        toolchain_version=RUST_VERSION,
        package_manager='cargo',
        manifest_files=('Cargo.lock', 'Cargo.toml'),
        apt_packages=(),
        install_commands=(),
        build_flags=(
            ('shims_path', SHIMS_PATH),
            ('toolchain_source', TOOLCHAIN_SOURCE),
        ),
        test_invocation=(
            ('command', read_harness(_bare(RUST_SPACEWASM_HARNESS)).graded_argv),
        ),
        harness=RUST_SPACEWASM_HARNESS,
        needs_git_metadata=False,
    )
    validate(plan)
    return canonicalize(plan)


def _bare(harness: tuple[tuple[str, HarnessValue], ...]) -> DepPlan:
    """A schema-valid plan carrying only a harness, so `read_harness` can run.

    `_rust_measure_dep_plan` needs the graded argv the harness implies BEFORE it
    can state `test_invocation`, and `read_harness` takes a whole plan. Spelling
    the argv a second time instead would be a second authority on what the
    grader runs, which is exactly the drift the harness section removes.
    """
    return DepPlan(
        lang='rust',
        toolchain_version=RUST_VERSION,
        package_manager='cargo',
        harness=harness,
    )


#: rust's canonical, validated environment plan. Module-level so a test can
#: assert the rendered gap against it without re-deriving the facts it states.
RUST_MEASURE_DEP_PLAN: DepPlan = _rust_measure_dep_plan()

#: rust-spacewasm's harness as the record the renderers read. The fallback for
#: every caller written before the harness was plan-driven.
RUST_SPACEWASM_HARNESS_SPEC: Harness = read_harness(RUST_MEASURE_DEP_PLAN)

#: The reference repo's facts, still exported under the names they had when they
#: WERE the plugin's definition of a rust suite -- but now DERIVED from the
#: harness above rather than spelled out. A caller reading these is reading
#: rust-spacewasm's plan, which is what they always were.
HARNESSES: tuple[str, ...] = RUST_SPACEWASM_HARNESS_SPEC.names
INTEGRITY_HARNESS_FILES: tuple[str, ...] = RUST_SPACEWASM_HARNESS_SPEC.integrity_files
MIN_WAST: int = RUST_SPACEWASM_HARNESS_SPEC.corpus_min

#: The graded cargo invocation as the one prose line instruction.md shows.
#: Derived from the plan rather than spelled again: the argv is already the
#: authority the two scripts render from, and a second spelling of it is how the
#: instruction the model reads starts describing a command the grader does not
#: run. A resolved plan replaces it through `test_command_from_plan`.
TEST_COMMAND: str = RUST_SPACEWASM_HARNESS_SPEC.graded_command


class RustPlugin(B.LangPlugin):
    """rustc 1.87 + wabt 1.0.41, whole-suite equality floor, measured denominator."""

    name: ClassVar[str] = 'rust'
    toml_family: ClassVar[str] = 'A'
    floor_mode: ClassVar[str] = 'equality'
    parser_backed: ClassVar[bool] = False
    synthesizes_git: ClassVar[bool] = False

    #: The same facts `toolchain_spec().install_block` asserts at build time,
    #: as the capability list `--resolve-env` shows the model. apt is listed
    #: because principle 5 allows apt and ONLY apt, and a model not told so
    #: reaches for curl.
    baked_capabilities: ClassVar[tuple[str, ...]] = BAKED_CAPABILITIES

    #: What `render_gap` reads and `depplan.validate` cannot know about. Stated
    #: to the model in the prompt, enforced by `validate_dep_plan` before a
    #: build; the two read the same tuple.
    required_plan_slots: ClassVar[tuple[str, ...]] = REQUIRED_PLAN_SLOTS

    rust_version: ClassVar[str] = RUST_VERSION
    wabt_version: ClassVar[str] = WABT_VERSION
    test_command: ClassVar[str] = TEST_COMMAND

    # --- axis 1 -----------------------------------------------------------

    def toolchain_spec(self) -> ToolchainSpec:
        """Select the rustc the base bakes, and prove it under a login shell.

        harbor-base ships rustc 1.79 per the reference Dockerfile, but the
        crate under test declares edition 2024 / rust-version 1.87. The install
        is `rustup toolchain install ... --profile minimal`, which reaches
        static.rust-lang.org -- allowed at BUILD time (harbor-base blocks
        github.com but not the rust CDN), forbidden at grade time (network=none
        in task.toml).

        `CARGO_NET_OFFLINE=true` is deliberately NOT set here: harbor-base
        already exports it, and the vendor RUN below overrides it locally for
        the one step that needs the network. Baking it as ENV would silently
        make it un-overridable and break the vendor step.
        """
        install = (
            f'# rustc {self.rust_version} is baked into the base; select it rather\n'
            f'# than reaching {TOOLCHAIN_SOURCE} on every task build.\n'
            '# profile.d is sourced in sorted order and the base re-exports its own\n'
            '# PATH ahead of the mise shims, so only a zz- file keeps the pin in front\n'
            "# for the login shells harbor's test.sh and solve.sh run as.\n"
            'RUN set -eux; \\\n'
            f'    printf \'export PATH="{SHIMS_PATH}:$PATH"\\n\' > /etc/profile.d/zz-harbor-toolchain-pin.sh; \\\n'
            '    chmod 0644 /etc/profile.d/zz-harbor-toolchain-pin.sh\n'
            'RUN set -eux; \\\n'
            '    v="$(bash -lc \'rustc --version\')"; \\\n'
            '    case "$v" in \\\n'
            f'        *"{self.rust_version}"*) echo "TOOLCHAIN PIN OK (login shell): $v" ;; \\\n'
            f'        *) echo "TOOLCHAIN PIN FAILED (login shell): got $v want {self.rust_version}" >&2; exit 42 ;; \\\n'
            '    esac'
        )
        return ToolchainSpec(
            base_image=BASE_IMAGE,
            install_block=install,
            env={
                'CARGO_TERM_COLOR': 'never',
                'RUST_BACKTRACE': '1',
            },
            workdir=B.WORKDIR,
        )

    # --- axis 2 -----------------------------------------------------------

    def dep_warm_spec(self) -> DepWarmSpec:
        """No warm stage: vendoring runs in the graded image (see pre_leakgate).

        Rust cannot warm dependencies the way go does. `cargo vendor` needs a
        resolvable `[lib]` target in Cargo.toml -- which for spacewasm means
        SOME `src/lib.rs` on disk. A warm stage that carried a real one would
        have the answer (invariant 7); an empty-stub one carried alongside the
        real repo would still need the carved Cargo.toml in the warm image.
        The pattern that works is: vendor in the graded image, WITH the carved
        tree already in place, using an empty `src/lib.rs` stub that lives for
        the duration of that one RUN and is then deleted.
        """
        return DepWarmSpec(
            stage_block='',
            files_needed=(),
            copy_paths=(),
        )

    # --- axes 3-6 ---------------------------------------------------------

    def _harness(self, dep_plan: DepPlan | None) -> tuple[DepPlan, Harness]:
        """The plan the harness comes from, and the harness itself.

        `dep_plan=None` is the pre-resolution path (`--no-resolve-env`, and
        every caller written before the harness was plan-driven). It falls back
        to rust-spacewasm's own canonical plan, which is exactly the environment
        those callers were hardcoded against -- so the fallback is not a guess,
        it is the same bytes under a name.
        """
        plan = canonicalize(RUST_MEASURE_DEP_PLAN if dep_plan is None else dep_plan)
        return plan, read_harness(plan)

    def test_command_from_plan(self, dep_plan: DepPlan | None) -> str:
        """What the rendered scripts ACTUALLY run, for instruction.md to quote."""
        _plan, harness = self._harness(dep_plan)
        return harness.graded_command

    def render_test_sh(
        self,
        graded: GradedSet,
        *,
        expected: int | None = None,
        fingerprint: Mapping[str, str] | None = None,
        integration_targets: int | None = None,
        dep_plan: DepPlan | None = None,
    ) -> str:
        expected = graded.expected if expected is None else int(expected)
        fingerprint = graded.fingerprint_sha256 if fingerprint is None else fingerprint
        _plan, harness = self._harness(dep_plan)
        # The HARNESS is the single authority on what runs, not
        # `graded.test_command`. The two agree by construction -- emit's
        # `_retitle_test_command` derives the latter from the former -- but
        # `SUMMARIES -eq n` counts the harness's targets, so reading the command
        # from anywhere else leaves a way for the run and its structural gate to
        # describe different suites.
        test_cmd = harness.graded_command

        # THE STRUCTURAL NET. `SUMMARIES -eq <n>` used to read a hardcoded 4.
        # Made plan-fed it would be a number the plan could shrink, so the
        # count comes from the HOST's own scan of the intact tree
        # (`emit._rust_grader_metadata`) and the plan is only ever checked
        # against it -- see `assert_repo_agrees`.
        targets = len(harness.names) if integration_targets is None else int(
            integration_targets
        )
        if targets != len(harness.names):
            raise B.LangError(
                f'the plan runs {len(harness.names)} integration target(s) but '
                f'the intact tree declares {targets}; render_test_sh must never '
                'reconcile that difference silently -- assert_repo_agrees is '
                'where it gets repaired'
            )

        harness_check = '\n'.join(
            [
                'for f in \\',
                ' \\\n'.join(f'        {rel}' for rel in harness.integrity_files) + '; do',
                '    [ -f "$f" ] || fail "graded harness file missing: $f"',
                'done',
            ]
        )

        corpus_block: list[str] = []
        corpus_prose = ''
        if harness.has_corpus:
            var, label = harness.corpus_var, harness.corpus_label
            corpus_block = [
                f'{var}=$(find {harness.corpus_dir} -name '
                f"'{harness.corpus_pattern}' | wc -l | tr -d ' ')",
                f'[ "${{{var}}}" -ge {harness.corpus_min} ] \\',
                f'    || fail "{label} corpus truncated: found ${{{var}}}, '
                f'expected >={harness.corpus_min}"',
                '',
            ]
            corpus_prose = f', ${{{var}}} {harness.corpus_suffix} files'

        tool_block = [
            f'command -v {tool.name} >/dev/null 2>&1 '
            f'|| fail "{tool.name} missing from image"'
            for tool in harness.tools
        ]
        if tool_block:
            tool_block.append('')
        tool_prose = ''.join(
            f', {tool.name} $({tool.name} --version)' for tool in harness.tools
        )

        return '\n'.join([
            '#!/usr/bin/env bash',
            f'# Harbor verifier -- rust, {expected} pinned test(s) across '
            f'{len(harness.names)} harnesses.',
            '#',
            '# EQUALITY floor. Rust libtest reports one `test result:` line per test',
            '# binary and does NOT abort the whole binary on a panic (that is a Go',
            '# hazard, spike G1), so observed==EXPECTED is a real assertion. The',
            f'# denominator ({expected}) was measured once against the intact tree in',
            '# phase 1 (measure.py) and pinned in graded.lock.json.',
            '#',
            "# Harbor ignores this script's exit code; /logs/verifier/reward.json is",
            '# the single source of truth and is written on EVERY path.',
            '',
            'set -uo pipefail',
            '',
            f'REPO=${{REPO:-{B.WORKDIR}}}',
            B.reward_emitter_block(_LOGS_DEFAULT),
            'CARGO_LOG="${VERIFIER_DIR}/cargo-test.log"',
            '',
            B.fail_closed_preamble(expected, compiled='0.0'),
            '',
            'cd "${REPO}" || fail "no ${REPO}"',
            '',
            f'# Integrity guards. The {len(harness.integrity_files)} harness files ARE '
            'the oracle (backed by the',
            f'# {harness.corpus_suffix or "graded"} conformance corpus); refusing to '
            'score if any of them was deleted',
            '# or rewritten stops a solver from making the suite trivially pass.',
            harness_check,
            '',
            *corpus_block,
            *tool_block,
            B.fingerprint_gate_block(fingerprint, repo_var='${REPO}'),
            '',
            f'echo "== cargo test: {expected} pinned test(s) across '
            f'{len(harness.names)} harnesses (offline) =="',
            f'echo "rustc $(rustc --version){tool_prose}{corpus_prose}"',
            '',
            # --offline defeats the network; -count/rerun is not a rust concept.
            # A cached PASS still recompiles on `cargo test` because src/ was
            # replaced by the oracle, invalidating the incremental cache.
            f'{test_cmd} > "${{CARGO_LOG}}" 2>&1',
            'STATUS=$?',
            'tail -40 "${CARGO_LOG}"',
            '',
            '# Count the "test result:" lines. Each harness prints exactly one; a',
            '# missing summary means that harness failed to LINK, which is different',
            '# from "linked but tests failed" and the reward schema keeps them apart.',
            'SUMMARIES=$(grep -c \'^test result:\' "${CARGO_LOG}" || true)',
            'SUMMARIES=${SUMMARIES:-0}',
            '',
            'if [ "${SUMMARIES}" -eq 0 ]; then',
            '    echo "no \'test result:\' lines -- crate did not build (compiled=0)"',
            '    emit 0.0 0 "${EXPECTED}" 0.0 0.0',
            '    exit 0',
            'fi',
            '',
            '# Every harness that printed a summary linked successfully.',
            'COMPILED=1.0',
            '',
            '# libtest layout: `<n> passed; <n> failed; <n> ignored; ...`. The count',
            '# for each keyword sits in the field immediately BEFORE its label, so',
            '# awk peeks at $(i+1) and sums $i across every "test result:" line.',
            'sum_field() {',
            '    awk -v want="$1" \'',
            '        /^test result:/ {',
            '            for (i = 1; i < NF; i++)',
            '                if ($(i+1) == want) s += $i',
            '        }',
            '        END { printf "%d", s + 0 }\' "${CARGO_LOG}"',
            '}',
            'PASSED=$(sum_field \'passed;\')',
            'FAILED=$(sum_field \'failed;\')',
            'IGNORED=$(sum_field \'ignored;\')',
            'PASSED=${PASSED:-0}',
            'FAILED=${FAILED:-0}',
            'IGNORED=${IGNORED:-0}',
            'TOTAL=$((PASSED + FAILED))',
            '',
            'echo "cargo exit=${STATUS} harnesses=${SUMMARIES}/'
            + str(targets)
            + ' passed=${PASSED} failed=${FAILED} ignored=${IGNORED} '
            'total=${TOTAL}/${EXPECTED} compiled=${COMPILED}"',
            '',
            '# Structural anti-gaming gates (equality floor is the third).',
            f'[ "${{SUMMARIES}}" -eq {targets} ] \\',
            f'    || fail "only ${{SUMMARIES}}/{targets} harnesses produced a '
            'summary -- refusing partial credit"',
            '[ "${IGNORED}" -eq 0 ] \\',
            '    || fail "${IGNORED} tests were #[ignore]d -- refusing to grade a '
            'suite that opted out"',
            '',
            B.floor_gate_block(
                self.floor_mode, expected,
                passed_var='PASSED', total_var='TOTAL', compiled_var='COMPILED',
            ),
            'exit 0',
            '',
        ])

    def measure_test_sh(
        self,
        *,
        graded: GradedSet | None = None,
        dep_plan: DepPlan | None = None,
        **kwargs,
    ) -> str:
        """Phase 1: run the WHOLE integration suite and count `passed + failed`.

        Floor-FREE by construction: it counts, it asserts nothing. A floor
        cannot be enforced by the run that is supposed to discover it. Unlike
        python (`--collect-only`) and go (`-list`), rust's libtest has no
        "select + count without running" mode, so this actually runs the tests
        -- against the intact tree, where they must all pass.
        """
        _plan, harness = self._harness(dep_plan)
        test_cmd = (
            graded.test_command
            if graded and graded.test_command
            else harness.graded_command
        )
        return '\n'.join([
            '#!/usr/bin/env bash',
            '# Harbor MEASURE (phase 1) -- rust. Floor-FREE by construction; the',
            '# pinned denominator is what THIS run measures.',
            '',
            'set -uo pipefail',
            '',
            f'REPO=${{REPO:-{B.WORKDIR}}}',
            B.measure_emitter_block(_MEASURE_LOGS_DEFAULT),
            'MEASURE_LOG="${MEASURE_DIR}/cargo-measure.log"',
            '',
            'cd "${REPO}" || { echo "no ${REPO}" >&2; exit 1; }',
            '',
            f'{test_cmd} > "${{MEASURE_LOG}}" 2>&1',
            'STATUS=$?',
            'tail -40 "${MEASURE_LOG}"',
            'echo "cargo exit=${STATUS}"',
            '',
            'sum_field() {',
            '    awk -v want="$1" \'',
            '        /^test result:/ {',
            '            for (i = 1; i < NF; i++)',
            '                if ($(i+1) == want) s += $i',
            '        }',
            '        END { printf "%d", s + 0 }\' "${MEASURE_LOG}"',
            '}',
            'PASSED=$(sum_field \'passed;\')',
            'FAILED=$(sum_field \'failed;\')',
            'PASSED=${PASSED:-0}',
            'FAILED=${FAILED:-0}',
            'TOTAL=$((PASSED + FAILED))',
            '',
            'measure "${TOTAL}" \'\'',
            'exit 0',
            '',
        ])

    # --- axis 7 -----------------------------------------------------------

    def post_restore_block(self) -> str:
        """Nothing: the oracle restore into src/ invalidates cargo's incremental.

        cargo detects src/ changes by mtime and re-builds -- the incremental
        cache under target/debug/incremental gets stale-marked automatically.
        No `cargo clean` needed; the test.sh doesn't ship one either (`--offline`
        does not defeat caching, but the src/ mtime change does).
        """
        return ''

    # --- axis 8 + the image ----------------------------------------------

    def extra_ctx_assets(self) -> tuple[tuple[Path, str], ...]:
        """Nothing. The base carries wabt, so the tooling context holds only
        leakscan.sh. The vendored tarball it used to stage was named for one
        architecture, which pinned every rust task image to arm64.
        """
        return ()

    def pre_leakgate_blocks_for(
        self, env: EnvSpec, dep_plan: DepPlan | None,
    ) -> tuple[str, ...]:
        """`pre_leakgate_blocks`, with the wabt assert and prune scoped by the plan.

        The vendor stub and the strings leak-assert are NOT scoped: they are the
        leak proof, they are a fact about how rustc works rather than about any
        repository, and a plan that could narrow them could narrow the thing
        that proves no carved symbol reached a layer.
        """
        _plan, harness = self._harness(dep_plan)
        return self._pre_leakgate(env, harness)

    def pre_leakgate_blocks(self, env: EnvSpec) -> tuple[str, ...]:
        return self._pre_leakgate(env, RUST_SPACEWASM_HARNESS_SPEC)

    def _scaffold(self, env: EnvSpec, harness: Harness) -> '_Scaffold':
        """The four RUN blocks that turn a bare rustc image into a graded one.

        Order matters: wabt first (so wast2json is present before any test
        script or vendor step references it), then the workspace prune (so
        `cargo vendor` sees the trimmed member list), then the empty-lib-stub
        vendor (so target/deps/* is warm for the vendored crates), then the
        strings-target leak assert (which needs target/ to exist).

        Everything runs BEFORE the leak gate because the leak scan is what
        should be catching a mistake here: a stray carved symbol in target/,
        a leftover incremental fingerprint, a receipt fragment. Putting these
        after the gate would hide exactly the things the gate is looking for.
        """
        corpus_noun = f'every {harness.corpus_suffix} file' if harness.has_corpus else 'it'
        tool_asserts = [
            '\n'.join([
                f'# {tool.package} {tool.version}: {tool.name} is a HARD runtime '
                'dependency of the graded',
                *(
                    [f'# harness ({tool.consumer} shells out to it for {corpus_noun}).']
                    if tool.consumer
                    else []
                ),
                f'# The base ships the {tool.package} suite at {tool.version}, so this '
                'ASSERTS the',
                '# dependency rather than installing it. Asserted under a login shell because',
                "# that is how harbor's test.sh resolves it at grade time.",
                'RUN set -eux; \\',
                f"    v=\"$(bash -lc '{tool.name} --version')\"; \\",
                f'    case "$v" in *{tool.version}*) echo '
                f'"{tool.package.upper()} OK (login shell): $v" ;; \\',
                '      *) echo "TOOLCHAIN PIN FAILED (login shell): '
                f'{tool.name} $v want {tool.version}" >&2; exit 42 ;; \\',
                '    esac',
            ])
            for tool in harness.tools
        ]

        # Prune out-of-scope workspace members so cargo vendor covers only the
        # graded package. A LITERAL string replacement, not a toml rewrite: no
        # toml emitter in the image preserves key order and comments, and
        # reformatting the whole manifest would perturb far more of the graded
        # tree than the one array entry being dropped.
        prune = ''
        if harness.prune_paths or harness.prune_manifest:
            edits = [
                f"s = s.replace('{key} = [\"{entry}\"]', '{key} = []')"
                for key, entry in harness.prune_manifest
            ]
            rewrite = [
                " && python3 - <<'PY'",
                'import pathlib',
                f'p = pathlib.Path("{env.workdir}/Cargo.toml")',
                's = p.read_text()',
                *edits,
                'p.write_text(s)',
                'PY',
            ] if edits else []
            removal = (
                f'RUN rm -rf {" ".join(harness.prune_paths)}'
                if harness.prune_paths
                else 'RUN true'
            )
            prune = '\n'.join([
                f'# Prune out-of-scope workspace members: {_english_list(harness)} are '
                'neither',
                '# carved nor graded, and their dependency graphs would bloat the vendor',
                '# set and the offline runtime.',
                f'{removal} \\' if rewrite else removal,
                *rewrite,
            ])

        # The empty-lib-stub trick. cargo vendor needs a resolvable [lib]
        # target; it does NOT need that target to contain any code. A one-byte
        # src/lib.rs lets vendor run, warms target/deps for the resolved
        # crates, and is deleted before the RUN ends. Nothing carved is
        # disclosed -- src/ ends up empty on disk.
        vendor = '\n'.join([
            '# Vendor via empty-lib-stub. cargo vendor needs a resolvable [lib] target,',
            '# which the carve deleted along with the rest of src/. It does NOT need',
            '# that target to hold real code: vendoring is dependency RESOLUTION driven',
            '# by Cargo.toml + Cargo.lock, and a one-byte src/lib.rs reproduces the',
            '# intact vendor set exactly. The same stub warms target/deps/* for the',
            '# resolved crates so the graded run does not rebuild them from scratch;',
            '# the carved-crate fingerprints and incremental artifacts are then',
            '# explicitly deleted (see the strings-assert RUN below).',
            "# CARGO_NET_OFFLINE=true is inherited from harbor-base; it is relaxed HERE",
            '# (and only here) so the lockfile can be re-resolved for the pruned',
            '# workspace and the crates fetched.',
            'RUN set -eux; \\',
            '    CRATE=$(sed -n \'/^\\[package\\]/,/^\\[/p\' Cargo.toml \\',
            '            | sed -n \'s/^name[[:space:]]*=[[:space:]]*"\\([^"]*\\)".*/\\1/p\' \\',
            '            | head -1); \\',
            '    echo "vendoring for crate: ${CRATE}"; \\',
            '    mkdir -p src; \\',
            '    : > src/lib.rs; \\',
            f'    CARGO_NET_OFFLINE=false cargo vendor {env.workdir}/vendor > /tmp/vendor-config.toml; \\',
            f'    mkdir -p {env.workdir}/.cargo; \\',
            f'    cp /tmp/vendor-config.toml {env.workdir}/.cargo/config.toml; \\',
            '    rm -f /tmp/vendor-config.toml; \\',
            '    (cargo test --offline --no-run 2>&1 | tail -5) || true; \\',
            '    cargo clean --offline -p "${CRATE}" 2>/dev/null || true; \\',
            '    rm -f src/lib.rs; \\',
            '    test -z "$(ls -A src)"; \\',
            '    rm -rf target/debug/incremental; \\',
            '    find target/debug/.fingerprint -maxdepth 1 -name "${CRATE}-*" -exec rm -rf {} + 2>/dev/null || true; \\',
            '    find target/debug/deps -maxdepth 1 -name "${CRATE}-*" -exec rm -rf {} + 2>/dev/null || true; \\',
            '    find target/debug/deps -maxdepth 1 -name "lib${CRATE}-*" -exec rm -rf {} + 2>/dev/null || true',
        ])

        # target/ is the one place where a build could have left a structural
        # fossil of the carved API. `strings <target-obj> | grep <crate>` would
        # recover mangled symbol names, private method names, unit-test names;
        # `grep <workdir>/src` would recover every carved file path. Both fail
        # closed here.
        strings_assert = '\n'.join([
            '# strings-target leak assert. target/ is the one place where compiling',
            '# the crate could have left a structural fossil of the carved API. Require',
            '# that the crate name and every "<workdir>/src" debug-info path appear',
            '# NOWHERE in target/: the first catches every mangled symbol, the second',
            '# catches every debug-info path that would enumerate the carved file list.',
            '# Vendored crates reference "<workdir>/vendor/<crate>/src/..." and so do',
            '# not trip this. This is BELT-AND-BRACES: the generic leak scan below is',
            '# what catches content leaks, but a symbol-only leak in an .o would slip',
            '# past a content grep and be recoverable with `strings`.',
            'RUN set -eu; \\',
            '    CRATE=$(sed -n \'/^\\[package\\]/,/^\\[/p\' Cargo.toml \\',
            '            | sed -n \'s/^name[[:space:]]*=[[:space:]]*"\\([^"]*\\)".*/\\1/p\' \\',
            '            | head -1); \\',
            f'    hits=$(find {env.workdir}/target -type f -print0 2>/dev/null \\',
            '             | xargs -0 -r strings -a 2>/dev/null \\',
            f'             | grep -E "${{CRATE}}|{env.workdir}/src" || true); \\',
            '    if [ -n "$hits" ]; then \\',
            '        echo "LEAK: rustc artifacts under target/ reveal carved internals:" >&2; \\',
            '        echo "$hits" | sort -u | head -40 >&2; \\',
            '        exit 1; \\',
            '    fi; \\',
            f'    echo "ASSERT: target/ carries no ${{CRATE}} symbols and no {env.workdir}/src paths"',
        ])

        return _Scaffold(
            tool_asserts=tuple(tool_asserts),
            prune=prune,
            vendor=vendor,
            strings_assert=strings_assert,
        )

    def _pre_leakgate(self, env: EnvSpec, harness: Harness) -> tuple[str, ...]:
        """The scaffold as the blank-line-separated block list the image takes.

        A repo that needs no external tool and has no workspace to prune simply
        contributes fewer blocks; the vendor stub and the leak assert are always
        present, because they are the proof rather than the provisioning.
        """
        scaffold = self._scaffold(env, harness)
        blocks = [
            *scaffold.tool_asserts,
            *([scaffold.prune] if scaffold.prune else []),
            scaffold.vendor,
            scaffold.strings_assert,
        ]
        separated: list[str] = []
        for block in blocks:
            if separated:
                separated.append('')
            separated.append(block)
        return tuple(separated)

    def validate_dep_plan(self, plan: DepPlan) -> None:
        """Every precondition `render_gap` has, checked before a container exists.

        The same set of checks `render_gap` makes, hoisted to where they cost
        nothing and can still be repaired: same helpers, same messages, so a
        plan accepted here cannot then fail to render. Enumerated from the
        renderer below -- the rustc version the pin matches, the manager whose
        toolchain it pins, the two rendered flags and the graded command --
        because a slot the renderer reads and this gate does not is exactly the
        crash this exists to prevent.
        """
        validate(plan)
        plan = canonicalize(plan)
        self._reject_foreign(plan)
        _rustc_version(plan.toolchain_version)
        for key in REQUIRED_BUILD_FLAGS:
            _flag_str(plan.build_flags, key)
        for key in REQUIRED_TEST_INVOCATION_KEYS:
            _test_tokens(plan.test_invocation, key)
        harness = read_harness(plan)
        self._assert_command_runs_the_harness(plan, harness)

    @staticmethod
    def _assert_command_runs_the_harness(plan: DepPlan, harness: Harness) -> None:
        """`test_invocation["command"]` must name every target the harness claims.

        The two are rendered from different slots -- the measure script runs the
        stated command, while `SUMMARIES -eq n` counts the harness's targets --
        so a command that runs three of four declared targets would measure a
        denominator the graded run could never reproduce, and the equality floor
        would fail every solver forever.
        """
        tokens = _test_tokens(plan.test_invocation, 'command')
        missing = [name for name in harness.names if name not in tokens]
        if missing:
            raise B.LangError(
                f'test_invocation["command"] is {" ".join(tokens)!r}, which does '
                f'not name {", ".join(missing)}. Every target in '
                'harness["harness_names"] must appear in the command that runs '
                'the suite, or the run that MEASURES the denominator and the run '
                'that GRADES against it are executing different test sets'
            )

    def assert_repo_agrees(
        self, plan: DepPlan, *, integration_targets, corpus_count: int,
    ) -> None:
        """`validate_dep_plan` plus what only the INTACT tree can answer."""
        harness = read_harness(plan)
        self._assert_targets_agree(harness, tuple(integration_targets))
        self._assert_corpus_exists(harness, int(corpus_count))

    @staticmethod
    def _assert_corpus_exists(harness: Harness, corpus_count: int) -> None:
        """A declared corpus has to BE there, and the floor has to be reachable.

        The plan says WHERE its corpus lives; the host counts what is actually
        there. Both failures render a gate nobody can pass: a corpus_dir/pattern
        matching nothing makes the verifier refuse every run, and a corpus_min
        above the real size does the same thing more quietly.
        """
        if not harness.has_corpus:
            return
        where = f'{harness.corpus_dir}/{harness.corpus_pattern}'
        if not corpus_count:
            raise B.LangError(
                f'harness["corpus_dir"]/harness["corpus_pattern"] describe {where}, '
                'but the intact tree holds no such file. The verifier would count '
                'zero and refuse every run, so a corpus that is not there is worse '
                'than no corpus at all -- omit corpus_dir, corpus_pattern, '
                'corpus_label and corpus_min entirely when the suite reads no data '
                'files'
            )
        if harness.corpus_min > corpus_count:
            raise B.LangError(
                f'harness["corpus_min"] is {harness.corpus_min}, but the intact '
                f'tree holds only {corpus_count} file(s) matching {where}. The '
                'floor is checked against the SHIPPED tree, which can only be '
                'smaller, so a floor above the intact size fails every run '
                'including the oracle'
            )

    @staticmethod
    def _assert_targets_agree(harness: Harness, declared: tuple[str, ...]) -> None:
        """THE UNDER-ENUMERATION GUARD, and what replaced `SUMMARIES -eq 4`.

        `SUMMARIES -eq 4` was rust's structural net: it caught any plan that
        described less of the suite than the repository has. Once the count
        comes from the plan the net is gone, and a plan that under-enumerates is
        the dangerous failure -- a shrunken denominator is SELF-CONSISTENT at
        the wrong number, so measure, the equality floor and RED/GREEN all pass
        while part of the suite is silently ungraded.

        The replacement compares NAMES, not a count. `emit._rust_grader_metadata`
        enumerates the intact crate's own cargo integration targets host-side
        (`tests/*.rs`, `tests/*/main.rs` and any `[[test]]` in Cargo.toml)
        without asking the resolver anything. Names are strictly harder to fool
        than a number: a plan cannot satisfy it by dropping one target and
        inventing another, and the refusal can say exactly which target went
        missing so the refine loop repairs in one attempt.
        """
        want, got = set(declared), set(harness.names)
        if want == got:
            return
        parts = []
        if want - got:
            parts.append(
                f'the intact tree declares {", ".join(sorted(want - got))} which '
                'the plan does not run'
            )
        if got - want:
            parts.append(
                f'the plan runs {", ".join(sorted(got - want))} which the intact '
                'tree does not declare'
            )
        raise B.LangError(
            f'harness["harness_names"] does not match the repository: '
            f'{"; and ".join(parts)}. The cargo integration targets were counted '
            'host-side from tests/*.rs, tests/*/main.rs and any [[test]] in '
            'Cargo.toml. A plan that under-enumerates pins a floor over only the '
            'part it described, and a shrunken denominator is self-consistent at '
            'the wrong number -- every later gate would pass while the rest of '
            f'the suite went ungraded. Restate harness["harness_names"] as '
            f'{sorted(want)} (with harness["harness_files"] in the same order)'
        )

    def _gap_body(self, plan: DepPlan) -> str:
        """rust's toolchain bytes, rendered from a plan instead of a literal.

        The part BOTH the measure and the shipped render take; the measure-only
        no-warm note is `render_gap`'s (see `LangPlugin._gap_body`).

        The SCAFFOLDING stays fixed and stays hardcoded: the zz- profile.d file
        that keeps the mise shims ahead of the base's own PATH, the login-shell
        pin assert's shape, the ENV/WORKDIR placement tail -- and, below the
        repo COPY where this string cannot reach, the wabt install, the
        workspace prune, `INTACT_VENDOR_BLOCK` and the strings leak-assert.
        Those are how harbor resolves a baked toolchain and proves no carved
        byte survives, not facts about this repo. What comes from the PLAN is
        the rustc version the pin matches, the shims directory it prepends, the
        CDN the comment says it avoids, and the apt/install lines a repo needing
        system libraries would add.

        The apt and install blocks are unreachable for `RUST_MEASURE_DEP_PLAN`
        (both fields are empty by construction, because the base BAKES the
        toolchain) and are rendered from the plan for the case where a resolved
        plan does declare them. They are the only lines here that are not in
        today's image.
        """
        validate(plan)
        plan = canonicalize(plan)
        self._reject_foreign(plan)

        version = _rustc_version(plan.toolchain_version)
        shims = _flag_str(plan.build_flags, 'shims_path')
        lines = [
            f'# rustc {version} is baked into the base; select it rather',
            f'# than reaching {_flag_str(plan.build_flags, "toolchain_source")} '
            'on every task build.',
            '# profile.d is sourced in sorted order and the base re-exports its own',
            '# PATH ahead of the mise shims, so only a zz- file keeps the pin in front',
            "# for the login shells harbor's test.sh and solve.sh run as.",
            'RUN set -eux; \\',
            f'    printf \'export PATH="{shims}:$PATH"\\n\' > '
            '/etc/profile.d/zz-harbor-toolchain-pin.sh; \\',
            '    chmod 0644 /etc/profile.d/zz-harbor-toolchain-pin.sh',
            'RUN set -eux; \\',
            '    v="$(bash -lc \'rustc --version\')"; \\',
            '    case "$v" in \\',
            f'        *"{version}"*) echo "TOOLCHAIN PIN OK (login shell): $v" ;; \\',
            f'        *) echo "TOOLCHAIN PIN FAILED (login shell): got $v want '
            f'{version}" >&2; exit 42 ;; \\',
            '    esac',
        ]
        if plan.apt_packages:
            lines.append(
                'RUN apt-get update && apt-get install -y --no-install-recommends '
                + ' '.join(plan.apt_packages)
                + ' && rm -rf /var/lib/apt/lists/*'
            )
        lines += [
            ' '.join((f'RUN {command.tool}', *command.args)).rstrip()
            for command in plan.install_commands
        ]

        return replace(
            self.toolchain_spec(), install_block='\n'.join(lines),
        ).render()

    def render_gap(self, plan: DepPlan) -> str:
        """The measure image's gap: the shared body plus the no-warm note.

        The note is the one line the shipped image must NOT carry -- it COPYs
        the warmed cargo registry from a separate stage.
        """
        return '\n'.join([self._gap_body(plan), '', MEASURE_NO_WARM_COMMENT])

    def _reject_foreign(self, plan: DepPlan) -> None:
        if plan.lang != self.name:
            raise B.LangError(
                f'the rust gap cannot render a {plan.lang!r} plan; a gap is the '
                'one part of a Dockerfile that is language-specific by definition'
            )
        if plan.package_manager not in _SUPPORTED_MANAGERS:
            raise B.LangError(
                f'the rust gap has no prose for package_manager '
                f'{plan.package_manager!r}; expected one of '
                f'{", ".join(sorted(_SUPPORTED_MANAGERS))}'
            )

    def render_measure_dockerfile(
        self, env: EnvSpec, *, dep_plan: DepPlan | None = None,
    ) -> str:
        """The stripped Dockerfile for the never-ship measure image.

        Same toolchain + wabt + prune as the shipped image, but the vendor
        step is SIMPLER: src/ contains the real crate here, so the empty-lib
        stub of the shipped image is inapplicable (would overwrite lib.rs
        without touching the other 30 files, leaving a broken half-carve).
        `cargo vendor` runs directly against the intact tree.

        NO leak gate / NO strings-assert / NO tripwire scan / NO carve-receipt
        check. On the INTACT tree the strings-assert and tripwires WOULD fire
        (they exist to catch carved bytes); the measure image is built ONLY to
        count `tests_total` and is deleted in `measure.py`'s finally block.

        `dep_plan` swaps the hardcoded gap for `render_gap(dep_plan)` in the
        SAME slot and touches nothing else -- notably NOT the wabt install, the
        prune or `INTACT_VENDOR_BLOCK` below, which are scaffolding and which
        run after the repo COPY the gap slot sits above. `dep_plan=None` is what
        emit.py passes and renders the bytes it always did.
        """
        base_image = self.toolchain_spec().base_image
        gap = (
            '\n'.join([self.toolchain(), '', MEASURE_NO_WARM_COMMENT])
            if dep_plan is None
            else self.render_gap(dep_plan)
        )
        _plan, harness = self._harness(dep_plan)
        scaffold = self._scaffold(env, harness)

        return '\n'.join([
            '# syntax=docker/dockerfile:1.7',
            f'# Harbor MEASURE image -- {env.repo_name} ({self.name}). NEVER SHIP.',
            '#',
            '# Built by measure.py phase 1 to count the intact test suite. Contains',
            '# the intact tree by construction -- an escaped measure image is the',
            '# whole answer, not a partial leak. `measure_image_tag` marks it as',
            '# never-ship and `measure.py` deletes it in a finally block.',
            '',
            f'FROM {base_image} AS measure',
            '',
            gap,
            '',
            f'# The measure phase points {env.repo_context} at the INTACT repo directly,',
            f'# not a staging tree, so there is no repo/ prefix to copy from. That is',
            f'# the only structural difference from the shipped Dockerfile.',
            f'COPY --from={env.repo_context} . {env.workdir}/',
            f'RUN mkdir -p {env.logs_dir}',
            '',
            *(
                [
                    f'# --- {", ".join(t.package for t in harness.tools)} '
                    f'(bind-mounted from the {env.tooling_context} context) ---',
                    *scaffold.tool_asserts,
                    '',
                ]
                if harness.tools
                else []
            ),
            *(
                ['# --- prune out-of-scope workspace members ---', scaffold.prune, '']
                if scaffold.prune
                else []
            ),
            '# --- vendor deps against the intact tree ---',
            INTACT_VENDOR_BLOCK,
            '',
            '# --- measure script (COPYed, not carved-tree, so it lives in a layer) ---',
            f'COPY measure.sh /opt/harbor/tests/measure.sh',
            f'RUN chmod 0555 /opt/harbor/tests/measure.sh',
            '',
        ])


B.register(RustPlugin())
