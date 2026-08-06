"""Assemble one complete harbor task entry per context type, per language, per scope.

Layout written per entry (directory name = the uuid5 entry id):

    <entry_id>/
      task.toml                          the plugin's family (A for python/go)
      instruction.md                     only the pre-loaded context block and
                                         the condition differ between the eleven
      environment/Dockerfile             from the plugin; consumes the STAGED
                                         carved tree, never the intact one
      environment/Dockerfile.dockerignore
      environment/contexts.json          which named build contexts to pass
      environment/carve/<relpath>        the stubbed/skeletonised files
      tests/test.sh                      from the plugin; common reward schema
      tests/graded.json                  selectors, packages, pinned denominator
      tests/allowlist.txt                pytest only: the graded node ids
      tests/harbor_filter.py             pytest only: vendored verbatim
      solution/solve.sh                  from the plugin; restores from the
                                         RUN-TIME mount, never from a layer
      solution/carved/<relpath>          the INTACT originals

and, ONCE per carve rather than once per entry, under `<out>/_staging/<key>/`:

    ctx/repo/                            the carved tree  -> --build-context repoctx=
    trip/tripwires.txt                   the leak gate    -> --build-context trip=
    tooling/leakscan.sh                  harbor's scanner -> --build-context tooling=
    carve_receipt.json                   HOST ONLY: names every carved path
    oracle/<relpath>                     HOST ONLY: the intact originals

Why the oracle is correct by construction: `solution/carved/<relpath>` holds the
untouched originals, so `solve.sh` returns the working tree to a state that is
byte-identical to the pristine checkout. reward=1 then reduces to "the graded
tests pass on the pristine repo AND the selector selects exactly them", which is
a property of the target selection, not of the emission.

Why the staged tree is shared: all eleven conditions of one target carve the
same files and share one image and one oracle. Staging eleven copies of a 33MB
repository would be eleven times the disk for zero additional proof.

DETERMINISM: no timestamps, no host paths, no dict iteration order in output.
The token budget handed to all eleven context builders is computed from the
WIDEST of the eleven headers, so every condition packs against the same budget.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import _tooling_path

from gen_context import FRAMING_BLOCK, assert_no_leakage  # noqa: E402

from . import refine  # noqa: E402
from . import scope as scope_mod  # noqa: E402
from . import templates  # noqa: E402
from .carve import CarveSet, build_carve_set  # noqa: E402
from .contexts import (  # noqa: E402
    CONTEXT_TYPES,
    ContextInputs,
    build_body,
    common_root,
    fence,
    stub_phrase,
)
from .depplan import DepPlan  # noqa: E402
from .env_resolver import EnvResolver  # noqa: E402
from .gradedset import (  # noqa: E402
    GradedSelection,
    derive_graded_set,
    parse_linked,
    select_carved,
)
from .ids import entry_id, slug as make_slug  # noqa: E402
from .langs import base as langs_base  # noqa: E402
from .scope import CarveScope  # noqa: E402
from .staging import (  # noqa: E402
    StagedTree,
    merge_bundle_tripwires,
    stage_carved_tree,
)

DEFAULT_BUDGET = 100000
DEFAULT_SEED = 0
TOKENIZER = 'chars4'
UV_VERSION = '0.12.1'

#: Mirrors verifier.generators.verifier_loop.MIN_SOUND_CRITERIA. Duplicated as a
#: literal so `--verifier-min-criteria`'s default costs no import: the verifier
#: package (and litellm behind it) must stay unimported unless --verifier is set.
#: test_emit_verifier.py asserts the two never drift.
DEFAULT_VERIFIER_MIN_CRITERIA = 6

#: Directory the bundle lands in, inside each entry. Under solution/, which is
#: never COPYed into a layer, so the bundle inherits the oracle's exclusion.
VERIFIER_SUBDIR = 'solution/verifier'

#: Which languages `--resolve-env` will render a resolved gap for. It is a
#: whitelist rather than a `hasattr(plugin, 'render_gap')` probe because every
#: plugin HAS the method -- `langs.base` defines it to raise -- so a probe would
#: green-light a language whose only implementation is the refusal.
RESOLVE_ENV_LANGS: frozenset[str] = frozenset({'c'})


def assert_resolve_env_supported(lang: str) -> None:
    """Refuse `--resolve-env` for a language whose gap cannot be rendered yet.

    Checked at the TOP of `emit_all`, not only in the measure phase: a
    parser-backed language never reaches that phase, so a check living only
    there would accept `--resolve-env --lang python` and then quietly ignore it.
    A flag that silently does nothing is worse than one that refuses.
    """
    if lang in RESOLVE_ENV_LANGS:
        return
    raise langs_base.LangError(
        f'--resolve-env is only supported for '
        f'{", ".join(sorted(RESOLVE_ENV_LANGS))} in this slice, not {lang!r}: '
        "rendering a measure image from a DepPlan needs the plugin's "
        'render_gap, and only c has one proven byte-identical to the gap it '
        'hardcodes today'
    )

#: Test/dev hook: `module:callable` returning a ModelClient, used INSTEAD of the
#: configured proxy. It exists so the offline suite can drive the whole pipeline
#: with a MockClient; it announces itself loudly on every run precisely because a
#: bundle authored by anything but a real model is not a real bundle.
VERIFIER_FAKE_CLIENT_ENV = 'TASKGEN_VERIFIER_FAKE_CLIENT'

#: uuid5 namespace for the shared staging directory name. Keyed on the carve,
#: so two scopes over one repo stage side by side instead of overwriting.
_STAGING_NS = uuid.uuid5(uuid.NAMESPACE_URL, 'triton.taskgen.staging')


@dataclass(frozen=True)
class Entry:
    entry_id: str
    context_type: str
    slug: str
    path: Path
    target_relpath: str
    nodeids: tuple[str, ...]
    lang: str = 'python'
    carve_scope: str = 'function'
    carved_relpaths: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# Repo metadata (offline, from pyproject.toml)
# --------------------------------------------------------------------------


def repo_metadata(repo: Path) -> dict:
    """Upstream slug and license, read off pyproject.toml. No network.

    Falls back to the directory name when the project declares nothing usable,
    so a repo without pyproject metadata still generates -- with a visibly
    approximate `repo` field rather than a silent wrong one.
    """
    repo = Path(repo)
    name = repo.name
    out = {'upstream': name, 'license': 'UNKNOWN'}
    pyproject = repo / 'pyproject.toml'
    if not pyproject.is_file():
        return out
    with open(pyproject, 'rb') as fh:
        data = tomllib.load(fh)
    project = data.get('project', {})

    lic = project.get('license')
    if isinstance(lic, str):
        out['license'] = lic
    elif isinstance(lic, dict) and isinstance(lic.get('text'), str):
        out['license'] = lic['text']

    urls = {k.lower(): v for k, v in (project.get('urls') or {}).items()}
    for key in ('repository', 'source', 'homepage'):
        url = urls.get(key)
        if isinstance(url, str) and 'github.com/' in url:
            tail = url.split('github.com/', 1)[1].strip('/')
            parts = tail.split('/')
            if len(parts) >= 2:
                out['upstream'] = f'{parts[0]}/{parts[1]}'
                break
    return out


def _image_name(repo_name: str) -> str:
    return f'taskgen-{repo_name.lower()}:local'


# --------------------------------------------------------------------------
# Planning: target -> carve set -> carve -> graded set -> staged tree
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """Where the carved checkout came from, when it came from a clone.

    Absent (`None`) for a directory handed to `--repo`: that path is a host
    detail, not a reproducible source, and recording it would put a host path in
    every emitted task.toml.
    """

    repo_url: str
    commit: str
    clone_kind: str

    @property
    def pinned(self) -> bool:
        return self.clone_kind == 'pinned'


def make_provenance(repo_url, commit, clone_kind) -> Provenance | None:
    if repo_url is None:
        return None
    kind = clone_kind or ('pinned' if commit else 'floating')
    if kind not in ('pinned', 'floating'):
        raise SystemExit(f'unknown clone_kind {kind!r}; expected pinned or floating')
    if kind == 'pinned' and not commit:
        raise SystemExit(
            'a pinned clone must record its commit; that sha is the only thing '
            'that makes the emitted task reproducible'
        )
    return Provenance(
        repo_url=str(repo_url), commit=str(commit or ''), clone_kind=kind,
    )


@dataclass(frozen=True)
class CarvePlan:
    lang: str
    scope: CarveScope
    repo: Path
    repo_name: str
    target: object
    carve: CarveSet
    graded: GradedSelection
    staged: StagedTree
    staging_key: str
    provenance: Provenance | None = None


def _resolve_target(lang: str, repo: Path, *, package_base, file, func, cls, receiver,
                    project):
    if lang == 'python':
        from .select import select_target

        return select_target(
            repo, package_base=package_base, file=file, func=func, cls=cls,
        )
    if lang == 'go':
        from .langs.go_select import select_go_target

        return select_go_target(
            repo, project, file=file, func=func, receiver=receiver,
        )
    raise langs_base.LangError(
        f'function scope needs a target resolver for {lang!r}; python and go have one'
    )


def _staging_key(repo_name, lang, scope, carved, primary) -> str:
    digest = uuid.uuid5(
        _STAGING_NS,
        '|'.join((repo_name, lang, scope.value, primary, *sorted(carved))),
    )
    return f'{repo_name}__{scope.value}__{digest.hex[:12]}'


def _copy_leakscan(dest: Path, plugin=None) -> None:
    """Vendor harbor's scanner (and any per-plugin static assets) into the tooling context.

    The shipped Dockerfile bind-mounts leakscan.sh from the `tooling` named
    context; extra per-plugin assets (rust's wabt tarball) land alongside so
    the plugin's pre_leakgate_blocks can bind-mount them from the same context.
    """
    dest.mkdir(parents=True, exist_ok=True)
    source = _tooling_path.TOOLING / 'leakscan.sh'
    if source.is_file():
        shutil.copyfile(source, dest / 'leakscan.sh')
        (dest / 'leakscan.sh').chmod(0o755)
    if plugin is not None:
        for asset_src, asset_name in plugin.extra_ctx_assets():
            asset_src = Path(asset_src)
            if not asset_src.is_file():
                raise langs_base.LangError(
                    f'{plugin.name!r} plugin asset missing on host: {asset_src}'
                )
            shutil.copyfile(asset_src, dest / asset_name)


def plan_carve(
    repo,
    out,
    *,
    lang='python',
    carve_scope='function',
    package_base='src/',
    file=None,
    func=None,
    cls=None,
    receiver=None,
    project=None,
    include=(),
    exclude=(),
    delete_whole_file=False,
    provenance: Provenance | None = None,
) -> CarvePlan:
    """Everything that is independent of the context type, done exactly once."""
    plugin = langs_base.get(lang)
    scope = CarveScope.parse(carve_scope)
    repo = Path(repo).resolve()
    repo_name = repo.name

    target = None
    if scope is CarveScope.FUNCTION:
        target = _resolve_target(
            lang, repo, package_base=package_base, file=file, func=func, cls=cls,
            receiver=receiver, project=project,
        )

    carved = scope_mod.resolve_carved_set(
        repo, scope,
        target_relpath=target.relpath if target is not None else None,
        include=include, exclude=exclude,
        test_globs=scope_mod.DEFAULT_TEST_GLOBS.get(lang, ()),
    )

    if plugin.parser_backed:
        parsed = parse_linked(
            lang, repo, package_base=package_base, project=project,
            scope_dirs=sorted({Path(r).parent.as_posix() for r in carved}),
        )
        only = {target.name} if target is not None else None
        carved_functions = select_carved(parsed, carved, only=only)

        carve = build_carve_set(
            repo, scope, carved_relpaths=carved, language=lang, target=target,
            delete_whole_file=delete_whole_file, carved_functions=carved_functions,
            all_functions=parsed,
        )
        # A skeleton carve drops files that hold no function body at all, so the
        # graded set must be derived from what was ACTUALLY carved -- deriving
        # it from the glob match would union in tests of code that still exists.
        if carve.carved_relpaths != carved:
            carved_functions = select_carved(parsed, carve.carved_relpaths, only=only)

        graded = derive_graded_set(lang, carved_functions, carve.carved_relpaths)
    else:
        # Whole-suite: no parser, no per-test selectors. The carve set comes from
        # the include globs, the graded set is a placeholder here (expected=1)
        # and the real denominator is measured against the intact tree in
        # `emit_all` before test.sh is rendered.
        carve = build_carve_set(
            repo, scope, carved_relpaths=carved, language=lang, target=target,
            delete_whole_file=delete_whole_file, carved_functions=(),
        )
        from .gradedset import whole_suite_selection

        fingerprint_globs = getattr(plugin, 'grader_fingerprint_globs', ())
        fingerprint_relpaths: tuple[str, ...] = ()
        if fingerprint_globs:
            gs = scope_mod.GlobSet(fingerprint_globs)
            fingerprint_relpaths = tuple(sorted(
                rel for rel in (
                    p.relative_to(repo).as_posix()
                    for p in repo.rglob('*') if p.is_file()
                )
                if gs.matches(rel)
            ))
            if not fingerprint_relpaths:
                raise langs_base.LangError(
                    f'{lang!r} plugin declared grader_fingerprint_globs='
                    f'{fingerprint_globs!r} but no files under {repo} match; '
                    'the fingerprint would be empty and shrink-locking the grader '
                    'is exactly what these globs are meant to enable'
                )

        graded = whole_suite_selection(
            lang, expected=1, test_command=plugin.test_command,
            carved_files=carve.carved_relpaths,
            fingerprint_relpaths=fingerprint_relpaths,
        )

    scope_mod.assert_carve_set_safe(
        carve.carved_relpaths,
        fingerprint_relpaths=graded.fingerprint_relpaths,
        test_globs=scope_mod.DEFAULT_TEST_GLOBS.get(lang, ()),
    )

    key = _staging_key(repo_name, lang, scope, carve.carved_relpaths, carve.primary_relpath)
    staging_dir = Path(out).resolve() / '_staging' / key
    staged = stage_carved_tree(
        repo, carve.carved_relpaths, carve.overlay, carve.deleted_relpaths, staging_dir,
    )
    _copy_leakscan(staging_dir / 'tooling', plugin=plugin)

    return CarvePlan(
        lang=lang, scope=scope, repo=repo, repo_name=repo_name, target=carve.target_view,
        carve=carve, graded=graded, staged=staged, staging_key=key,
        provenance=provenance,
    )


# --------------------------------------------------------------------------
# instruction.md
# --------------------------------------------------------------------------


def _target_signature_spec(inp: ContextInputs) -> str:
    """sigextract's rendering of the target symbol only. Signature, never body."""
    want = inp.target.name
    for sym in inp.spec.symbols:
        if sym.kind in ('function', 'method') and f' {want}(' in sym.text:
            return '\n'.join(sym.render())
        if sym.kind == 'class' and sym.text.startswith(
            f'class {getattr(inp.target, "class_name", "")}'
        ):
            for child in sym.children:
                if f' {want}(' in child.text:
                    return '\n'.join(child.render())
    return inp.carve.signature.rstrip() + ' ...'


def _render_grading(inp: ContextInputs, plan: CarvePlan, a) -> None:
    n = plan.graded.expected if plan.graded.kind == 'whole-suite' else len(inp.nodeids)
    a('## How you are graded\n')
    a('Test command, run from the repository root inside the task container:\n')
    if plan.graded.kind == 'go-run':
        pkgs = ' '.join(plan.graded.packages)
        run = '^(' + '|'.join(plan.graded.selectors) + ')$'
        a(f"```bash\ngo test -short -count=1 -json -run '{run}' {pkgs}\n```\n")
        a(f'The graded scope is exactly these {n} test function(s):\n')
        a('```')
        for nid in inp.nodeids:
            a(nid)
        a('```\n')
    elif plan.graded.kind == 'whole-suite':
        a(f'```bash\n{plan.graded.test_command}\n```\n')
        a(
            f'The graded scope is the WHOLE integration suite ({n} tests). '
            'There are no per-test selectors; the denominator was measured once '
            'against the intact tree and pinned in `tests/graded.json`.\n'
        )
    else:
        a(
            '```bash\nuv run --no-sync --offline pytest -o addopts=\'\' -p no:xdist '
            '-p no:cacheprovider -p harbor_filter\n```\n'
        )
        a(f'The graded scope is exactly these {n} node id(s):\n')
        a('```')
        for nid in inp.nodeids:
            a(nid)
        a('```\n')
    a(
        f'- Reward is **fractional**: passed / {n}, over that pinned denominator '
        f'and never over whatever happened to run. `binary` is 1.0 only when all '
        f'{n} pass.\n'
        '- A run that collects nothing, or that was trimmed to a passing subset, '
        'scores 0 -- never a flattering fraction of a shrunken denominator.\n'
        '- Editing, deleting, skipping or `xfail`-ing tests earns no credit; the '
        'graded test files are sha256-locked.\n'
        '- Do not modify surviving source to make tests pass. Only the stubbed '
        'bodies are yours to write.\n'
    )


def render_head(inp: ContextInputs, context_type: str, plan: CarvePlan) -> str:
    """Sections 1-6: everything above the pre-loaded context block."""
    carve = inp.carve
    lang = plan.lang
    stub = stub_phrase(lang)
    parts: list[str] = []
    a = parts.append

    if plan.scope is CarveScope.FUNCTION:
        a(f'# Task: regenerate `{inp.target.qualname}` in `{inp.target_relpath}` '
          f'of `{inp.repo_name}`\n')
        a('## What happened\n')
        a(
            f'The body of one function -- `{inp.target.qualname}`, in '
            f'`{inp.target_relpath}` -- was deleted from this repository and replaced '
            f'with `{stub}`. Everything else is still on disk '
            f'exactly as it was: the enclosing file, its imports, its other '
            f'definitions, every other module, and the whole test suite.\n'
        )
        a(
            'Your job is to write that one body again, from scratch, so that the '
            'tests which exercise it pass. Do not delete or rewrite the surrounding '
            'file; replace the stub and nothing else. The rest of the repository is '
            'your ground truth -- read whatever you need from disk.\n'
        )
    else:
        a(f'# Task: regenerate the carved code of `{inp.repo_name}` '
          f'({len(carve.carved_relpaths)} file(s))\n')
        a('## What happened\n')
        if carve.deleted_relpaths:
            a(
                f'{len(carve.deleted_relpaths)} file(s) were DELETED from this '
                'repository outright. Everything else -- every other module, and the '
                'whole test suite -- is still on disk exactly as it was.\n'
            )
        else:
            a(
                f'Every function body in {len(carve.carved_relpaths)} file(s) was '
                f'deleted from this repository and replaced with `{stub}`. The '
                'imports, module-level code and every signature in those files '
                'survive, and so does the rest of the repository, including the '
                'whole test suite.\n'
            )
        a(
            'Your job is to write those bodies again, from scratch, so that the '
            'tests which exercise them pass. The rest of the repository is your '
            'ground truth -- read whatever you need from disk.\n'
        )

    a('## Context condition\n')
    a(f'This instance runs under condition **`{context_type}`**.\n')
    a('> ' + FRAMING_BLOCK + '\n')

    if plan.scope is CarveScope.FUNCTION:
        r = carve.receipt
        a('## Function you must regenerate\n')
        a(f'File: `{inp.target_relpath}`\n')
        a(
            f'Carved byte range: `[{r["body_start_byte"]}:{r["body_end_byte"]}]` '
            f'({r["carved_bytes"]} bytes of function body) inside a '
            f'{r["original_bytes"]}-byte file.\n'
        )
        a('Signature, exactly as it still appears on disk:\n')
        a(f'```{fence(lang)}')
        a(carve.signature.rstrip())
        a('```\n')
        if carve.docstring:
            a('Its docstring, which is the specification you must implement:\n')
            a('```text')
            a(carve.docstring)
            a('```\n')
    else:
        a('## Files you must regenerate\n')
        verb = 'Deleted' if carve.deleted_relpaths else 'Skeletonised'
        a(f'{verb} carve root: `{common_root(carve.carved_relpaths)}`\n')
        a('```')
        for rel in carve.carved_relpaths:
            a(rel)
        a('```\n')
        a(
            f'{len(carve.carved_functions)} function(s) across those files lost '
            'their bodies. Their signatures are still on disk unless the file was '
            'deleted outright.\n'
        )

    _render_grading(inp, plan, a)

    if plan.scope is CarveScope.FUNCTION:
        a('## Signature spec (public symbols of the carved function)\n')
        a(
            'Extracted exactly via the Python `ast` module. Signature only -- the '
            'body is the task.\n'
        )
        a('```')
        a(_target_signature_spec(inp))
        a('```\n')

    return '\n'.join(parts) + '\n'


def render_footer(inp: ContextInputs, context_type: str, stats: dict,
                  plan: CarvePlan) -> str:
    """Section 8: the generation-metadata fenced block."""
    carve = inp.carve
    rows = [
        ('repo', inp.repo_name),
        ('condition', context_type),
        ('language', plan.lang),
        ('carve_scope', plan.scope.value),
        ('budget_tokens', inp.budget),
        ('tokenizer', inp.counter.method),
        ('seed', inp.seed),
        ('target_file', inp.target_relpath),
        ('carved_files', len(carve.carved_relpaths)),
        ('carved_functions', len(carve.carved_functions)),
        ('deleted_files', len(carve.deleted_relpaths)),
        ('linked_tests', len(inp.nodeids)),
        ('context_tokens', stats.get('context_tokens', 0)),
        ('selection', stats.get('selection', 'none')),
    ]
    if plan.scope is CarveScope.FUNCTION:
        rows.insert(8, ('target_func', inp.target.qualname))
        rows.insert(9, ('carved_bytes', carve.receipt['carved_bytes']))

    named = {k for k, _ in rows}
    parts = ['## Generation metadata\n', '```']
    parts.extend(f'{k} = {v}' for k, v in rows)
    parts.extend(f'{k} = {stats[k]}' for k in sorted(stats) if k not in named)
    parts.append(f'surviving_files_considered = {len(inp.surviving)}')
    parts.append(f'leak_tripwires_checked = {len(inp.tripwires)}')
    parts.append('```\n')
    if plan.scope is not CarveScope.FUNCTION:
        parts.append('Carved files:\n')
        parts.append('```')
        parts.extend(carve.carved_relpaths)
        parts.append('```\n')
    # NOT counter.describe(): it blames a missing tiktoken, but taskgen pins
    # chars4 outright rather than probing for it.
    parts.append(
        f'Token counts above use the {TOKENIZER} heuristic (ceil(len/4)), chosen '
        'unconditionally for reproducibility rather than probed for: a tokenizer '
        'that varies with the generation host would break byte-identical reruns. '
        'It carries roughly +-15% error against a real BPE tokenizer on source '
        'code.\n'
    )
    parts.append(
        'Leakage check: the generator verified that no distinctive line of any '
        'carved body appears anywhere in this document. The signature '
        'and the docstring are quoted deliberately -- they are the prompt, not '
        'the answer.\n'
    )
    return '\n'.join(parts) + '\n'


def render_instruction(inp: ContextInputs, context_type: str,
                       plan: CarvePlan) -> tuple[str, dict]:
    intro, blocks, stats = build_body(context_type, inp)
    head = render_head(inp, context_type, plan)

    def assemble(chosen: list[str]) -> str:
        mid = intro + '\n' + ''.join(chosen) if chosen else intro + '\n'
        return head + '\n' + mid + '\n' + render_footer(inp, context_type, stats, plan)

    document = assemble(blocks)
    # The footer quotes selection stats, so its true size is only knowable after
    # packing; drop trailing blocks if the reserve turned out to be too small.
    while blocks and inp.counter.count(document) > inp.budget:
        blocks.pop()
        stats['context_tokens'] = sum(inp.counter.count(b) for b in blocks)
        document = assemble(blocks)

    total = inp.counter.count(document)
    if total > inp.budget:
        raise SystemExit(
            f'{context_type}: document is {total} tokens, over the {inp.budget} budget '
            'even with no context blocks. Raise --budget or shrink the header.'
        )
    assert_no_leakage(document, inp.tripwires)
    if FRAMING_BLOCK not in document:
        raise SystemExit('internal error: required framing block missing from document')

    stats['total_tokens'] = total
    return document, stats


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def _write(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _graded_spec(plan: CarvePlan, plugin) -> langs_base.GradedSet:
    fingerprints = {}
    for rel in plan.graded.fingerprint_relpaths:
        path = plan.repo / rel
        if path.is_file():
            fingerprints[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return langs_base.GradedSet(
        expected=plan.graded.expected,
        floor_mode=plugin.floor_mode,
        fingerprint_sha256=fingerprints,
        kind=plan.graded.kind,
        selectors=plan.graded.selectors,
        packages=plan.graded.packages,
        test_command=plan.graded.test_command,
    )


def _test_file_counts(plan: CarvePlan) -> dict[str, int]:
    """`_test.go` count per graded package directory -- go's anti-shrink lock."""
    counts: dict[str, int] = {}
    for rel in plan.graded.fingerprint_relpaths:
        directory = Path(rel).parent
        counts[directory.as_posix()] = len(
            list((plan.repo / directory).glob('*_test.go'))
        )
    return counts


def _c_grader_metadata(plan: CarvePlan) -> dict:
    """Sub-counts and tls_count for c-xs, computed HOST-SIDE from the intact tree.

    Threaded into the plugin's render_test_sh so no magic numbers appear in the
    plugin source; the reference test.sh reads the tls count off an image file
    (/opt/harbor/tls_file_count), we inline it as a bash literal at generate
    time. Both paths refuse to grade if the runtime count drifts.
    """
    repo = plan.repo
    conformance = len(sorted((repo / 'tests' / 'conformance').glob('*.xs')))
    regression = len(sorted((repo / 'tests' / 'regression').glob('*.xs')))
    unit_dir = repo / 'tests' / 'unit'
    unit = len(sorted(
        p for p in unit_dir.iterdir()
        if p.is_file() and p.name.endswith('_test.c')
    )) if unit_dir.is_dir() else 0
    tls_count = len([
        p for p in (repo / 'src' / 'tls').rglob('*') if p.is_file()
    ]) if (repo / 'src' / 'tls').is_dir() else 0
    return {
        'corpus_counts': {
            'conformance': conformance,
            'regression': regression,
            'unit': unit,
        },
        'tls_count': tls_count,
    }


def _render_test_sh(plugin, plan: CarvePlan, graded_spec) -> str:
    if plan.lang == 'go':
        return plugin.render_test_sh(
            graded_spec, test_file_counts=_test_file_counts(plan),
        )
    if plan.lang == 'c':
        return plugin.render_test_sh(graded_spec, **_c_grader_metadata(plan))
    return plugin.render_test_sh(graded_spec)


def _write_entry(inp: ContextInputs, context_type: str, out: Path, meta: dict,
                 plan: CarvePlan, plugin, graded_spec) -> Entry:
    repo_name = inp.repo_name
    carve = inp.carve
    relpath = carve.primary_relpath
    func_key = (
        inp.target.name if plan.scope is CarveScope.FUNCTION
        else common_root(carve.carved_relpaths)
    )
    cls_key = getattr(inp.target, 'class_name', '') or ''
    eid = entry_id(
        repo_name, relpath, cls_key, func_key, context_type,
        scope=plan.scope.value, lang=plan.lang,
    )
    slug = make_slug(repo_name, func_key, context_type)
    path = out / eid

    document, _stats = render_instruction(inp, context_type, plan)
    image = _image_name(repo_name)

    _write(path / 'instruction.md', document)
    _write(path / 'task.toml', templates.render_task_toml(
        slug=slug,
        n_conditions=len(CONTEXT_TYPES),
        func=inp.target.qualname,
        upstream=meta['upstream'],
        condition=context_type,
        language=plan.lang,
        entry_id=eid,
        license=meta['license'],
        budget=inp.budget,
        tokenizer=TOKENIZER,
        seed=inp.seed,
        target_file=relpath,
        target_func=inp.target.qualname,
        graded_tests=json.dumps(list(plan.graded.selectors)),
        carve_scope=plan.scope.value,
        carved_files=json.dumps(list(carve.carved_relpaths)),
        docker_image=image,
        provenance=templates.render_provenance(plan.provenance),
    ))

    _write(path / 'tests/graded.json',
           json.dumps(plan.graded.to_dict(), indent=2, sort_keys=True) + '\n')
    if plan.graded.kind == 'pytest-allowlist':
        _write(path / 'tests/allowlist.txt',
               ''.join(f'{n}\n' for n in plan.graded.selectors))
        _write(path / 'tests/harbor_filter.py', templates.HARBOR_FILTER_PY)
    _write(path / 'tests/test.sh', _render_test_sh(plugin, plan, graded_spec),
           executable=True)

    _write(path / 'solution/solve.sh',
           plugin.render_solve_sh(carve.carved_relpaths), executable=True)
    for rel, text in sorted(carve.originals.items()):
        _write(path / f'solution/carved/{rel}', text)

    env = langs_base.EnvSpec(repo_name=repo_name)
    _write(path / 'environment/Dockerfile', plugin.render_dockerfile(env))
    _write(path / 'environment/Dockerfile.dockerignore', templates.DOCKERIGNORE)
    _write(path / 'environment/contexts.json', json.dumps({
        'image': image,
        'stage': env.stage,
        'build_contexts': {
            env.repo_context: f'../../_staging/{plan.staging_key}/ctx',
            env.trip_context: f'../../_staging/{plan.staging_key}/trip',
            env.tooling_context: f'../../_staging/{plan.staging_key}/tooling',
            env.entry_context: '..',
        },
    }, indent=2, sort_keys=True) + '\n')
    for rel, text in sorted(carve.overlay.items()):
        _write(path / f'environment/carve/{rel}', text)

    return Entry(
        entry_id=eid,
        context_type=context_type,
        slug=slug,
        path=path,
        target_relpath=relpath,
        nodeids=tuple(plan.graded.selectors),
        lang=plan.lang,
        carve_scope=plan.scope.value,
        carved_relpaths=tuple(carve.carved_relpaths),
    )


def emit_all(repo, out, package_base: str = 'src/', file: str | None = None,
             func: str | None = None, cls: str | None = None,
             budget: int = DEFAULT_BUDGET, seed: int = DEFAULT_SEED,
             context_types=CONTEXT_TYPES, *,
             lang: str = 'python', carve_scope: str = 'function',
             receiver: str | None = None, project: str | None = None,
             include=(), exclude=(), delete_whole_file: bool = False,
             repo_url: str | None = None, commit: str | None = None,
             clone_kind: str | None = None,
             verifier: bool = False, llm_config: str | Path | None = None,
             verifier_min_criteria: int = DEFAULT_VERIFIER_MIN_CRITERIA,
             resolve_env: bool = False, resolver: EnvResolver | None = None,
             echo=print) -> list[Entry]:
    """Select, carve, stage, and write one full harbor entry per context type.

    `verifier=False` is the whole existing behaviour, byte-for-byte: nothing in
    the verifier package is imported, no LLM is contacted, and the run stays
    offline and `diff -r`-reproducible. `resolve_env=False` is the same promise
    for the measure phase's environment -- see `_measure_and_pin`.
    """
    out = Path(out)
    if resolve_env:
        assert_resolve_env_supported(lang)
    plugin = langs_base.get(lang)
    plan = plan_carve(
        repo, out, lang=lang, carve_scope=carve_scope, package_base=package_base,
        file=file, func=func, cls=cls, receiver=receiver, project=project,
        include=include, exclude=exclude, delete_whole_file=delete_whole_file,
        provenance=make_provenance(repo_url, commit, clone_kind),
    )

    if not plugin.parser_backed:
        plan = _measure_and_pin(
            plan, plugin, out, echo=echo,
            resolve_env=resolve_env, resolver=resolver,
        )

    inp = ContextInputs.build(
        repo=plan.repo, repo_name=plan.repo_name, target=plan.target,
        carve=plan.carve, nodeids=plan.graded.selectors, budget=budget, seed=seed,
        tokenizer=TOKENIZER, language=lang,
    )
    # One budget for all eleven: reserve the WIDEST header so a longer condition
    # name can never buy a condition fewer context tokens than its neighbours.
    inp.header_tokens = max(
        inp.counter.count(render_head(inp, ct, plan)) for ct in context_types
    )

    meta = repo_metadata(plan.repo)
    graded_spec = _graded_spec(plan, plugin)
    out.mkdir(parents=True, exist_ok=True)
    entries = [
        _write_entry(inp, ct, out, meta, plan, plugin, graded_spec)
        for ct in context_types
    ]
    if verifier:
        emit_verifier_bundle(
            plan, entries, llm_config=llm_config,
            min_criteria=verifier_min_criteria, echo=echo,
        )
    return entries


# --------------------------------------------------------------------------
# --verifier: ONE bundle per invocation, copied into all eleven entries
# --------------------------------------------------------------------------


def _load_fake_client(spec: str, echo):
    """Resolve `module:callable` from VERIFIER_FAKE_CLIENT_ENV into a ModelClient."""
    import importlib

    module_name, _, attr = spec.partition(':')
    if not module_name or not attr:
        raise SystemExit(
            f'{VERIFIER_FAKE_CLIENT_ENV}={spec!r} is not "module:callable"'
        )
    try:
        factory = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise SystemExit(
            f'{VERIFIER_FAKE_CLIENT_ENV}={spec!r} could not be resolved: {exc}'
        ) from exc
    echo(
        f'verifier: WARNING -- authoring with the FAKE client hook {spec!r} '
        f'({VERIFIER_FAKE_CLIENT_ENV}), NOT a model. The resulting bundle is a '
        'test fixture and must not be shipped as a real verifier.'
    )
    return factory()


def resolve_verifier_client(llm_config, *, echo=print):
    """The authoring client, or a LOUD failure (PLAN V4).

    An unreachable or unconfigured proxy when `--verifier` was explicitly asked
    for is NOT the non-blocking case: the non-blocking case is "the soundness
    floor was not met", which is a real answer. A missing config is no answer at
    all, and quietly shipping the task without a bundle would hide it.
    """
    hook = os.environ.get(VERIFIER_FAKE_CLIENT_ENV)
    if hook:
        return _load_fake_client(hook, echo)

    from verifier.llm_config import LLMConfigError, client_from_config, resolve_config

    try:
        cfg = resolve_config(llm_config)
    except LLMConfigError as exc:
        raise SystemExit(f'--verifier: {exc}') from exc
    return client_from_config(cfg)


def _copy_bundle(src_dir: Path, dest_dir: Path) -> list[str]:
    """Copy one frozen bundle into an entry, through the emitter's own discipline.

    `_write` is reused rather than `shutil.copy` so the copies land with LF
    endings and no mode/timestamp carried over from the temporary build.
    """
    written: list[str] = []
    for path in sorted(src_dir.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(src_dir).as_posix()
        _write(dest_dir / rel, path.read_text(encoding='utf-8'))
        written.append(rel)
    return written


def emit_verifier_bundle(plan: CarvePlan, entries, *, llm_config=None,
                         min_criteria: int = DEFAULT_VERIFIER_MIN_CRITERIA,
                         echo=print) -> bool:
    """Generate ONE bundle for the carve and copy it into every entry. (D4/D6)

    Everything here is imported lazily: `--verifier` is the only path that may
    pull litellm and the generator stack into a taskgen process.

    Two failure shapes, deliberately different:
      * the soundness floor was not met -> log, ship the task WITHOUT a bundle,
        return False (D5: additive, non-blocking);
      * the client/config/proxy is unusable, or the language has no soundness
        executor -> raise. An unproven bundle is worse than no bundle, and a
        silent skip would make `--verifier` a no-op that looks like success.
    """
    client = resolve_verifier_client(llm_config, echo=echo)

    from verifier.bundle import build_verifier_bundle
    from verifier.exec_env import make_exec_env
    from verifier.inputs import build_truth_inputs, solution_code

    inputs = build_truth_inputs(
        plan.carve, plan.target, plan.lang,
        repo_name=plan.repo_name, fail_to_pass=list(plan.graded.selectors),
    )
    code = solution_code(plan.carve)

    with tempfile.TemporaryDirectory(prefix='taskgen-verifier-') as tmp:
        tmp_root = Path(tmp)
        try:
            env = make_exec_env(
                plan.lang, repo=plan.repo, carve=plan.carve,
                work_dir=tmp_root / 'trees',
            )
        except NotImplementedError as exc:
            raise SystemExit(f'--verifier: {exc}') from exc

        base_dir = tmp_root / 'bundle'
        try:
            shipped = build_verifier_bundle(
                inputs, client, base_dir,
                exec_runner=env.as_run_pytest(),
                golden_code=code.golden, stub_code=code.stub,
                min_criteria=min_criteria, echo=echo,
            )
        except SystemExit:
            raise
        except Exception as exc:
            raise SystemExit(
                f'--verifier: bundle generation FAILED '
                f'({type(exc).__name__}: {exc}). This is not a soundness abort -- '
                'nothing was written, and a task is never shipped with a bundle '
                'that could not be authored and validated.'
            ) from exc

        if not shipped:
            echo('verifier: shipping the task WITHOUT a bundle (non-blocking)')
            return False

        for entry in entries:
            _copy_bundle(base_dir, entry.path / VERIFIER_SUBDIR)
        echo(f'verifier: bundle copied into {len(entries)} entry solution/verifier/')

        added = merge_bundle_tripwires(plan.staged, base_dir)
        echo(f'verifier: merged {len(added)} bundle tripwire line(s) into the leak gate')
    return True


def _resolve_env_plan(resolver: EnvResolver | None, plan: CarvePlan,
                      base_image: str, *, build_and_measure: refine.MeasureAttempt,
                      validate_candidate: refine.PlanValidator,
                      echo, clock=time.monotonic) -> refine.RefinedEnv:
    """The refine loop, wired to this repo. A validated, MEASURED plan, or a raise.

    A `Refuse` becomes a raise rather than a fallback to the hardcoded gap: the
    fallback would emit a task whose environment nobody vouched for, which is
    the silent degradation the whole disposition rule exists to forbid.

    The loop itself is `refine.refine_dep_plan` and knows nothing about docker;
    everything container-shaped arrives as `build_and_measure`, so the bound and
    the budget stay testable offline. It knows nothing about languages either,
    which is why `validate_candidate` -- the plugin's own pre-render gate -- is
    injected the same way: it is the check that a schema-valid plan is one THIS
    gap can actually render, and it runs before the build so an unrenderable
    answer is repaired rather than paid for in docker time.

    What this wrapper adds is the one thing only emit can answer -- that a flag
    with no resolver behind it is a caller error, not a refusal to be retried.
    """
    if resolver is None:
        raise langs_base.LangError(
            '--resolve-env was passed but no resolver was injected, so there is '
            'nothing to resolve the environment WITH. emit does not construct '
            'one itself -- that would make a generate run reach the network from '
            'a call chain whose whole contract is that it does not -- so the '
            'caller passes one in; binding the flag to a real model client is a '
            'later slice'
        )
    return refine.refine_dep_plan(
        resolver=resolver,
        lang=plan.lang,
        repo=plan.repo,
        base_image=base_image,
        build_and_measure=build_and_measure,
        validate_candidate=validate_candidate,
        echo=echo,
        clock=clock,
    )


def _measure_and_pin(plan: CarvePlan, plugin, out: Path, *, echo=print,
                     resolve_env: bool = False,
                     resolver: EnvResolver | None = None,
                     clock=time.monotonic) -> CarvePlan:
    """Phase 1 of the two-phase build for whole-suite languages.

    Builds a never-ship measure image against the INTACT tree, runs the
    plugin's floor-FREE measure_test_sh, reads the count, deletes the image,
    and returns a new CarvePlan whose graded set carries the measured
    denominator. Reuses `graded.lock.json` from a prior run when the tree hash
    and base-image digest still match, so a second `generate` is fast and
    byte-identical.

    `resolve_env=False` is every byte of that, unchanged: no resolver is called,
    no plan is rendered and the lock grows no key -- the measurement is the same
    single call it always was, now reached through `_run_measure(None)`. With it
    on, the GAP of the measure Dockerfile comes from the injected resolver's
    `DepPlan` instead of the plugin's hardcoded lines, that measurement may
    happen more than once (the refine loop rebuilds a repaired plan), and only
    the plan that finally built and collected is pinned into the lock so the
    next run reuses the environment rather than re-resolving it.
    """
    import dataclasses
    from . import measure as measure_mod
    from . import verify as verify_mod

    if resolve_env:
        assert_resolve_env_supported(plan.lang)

    staging_dir = Path(out).resolve() / '_staging' / plan.staging_key
    measure_ctx = staging_dir / 'measure'
    lock_path = staging_dir / measure_mod.LOCK_FILENAME

    env = langs_base.EnvSpec(repo_name=plan.repo_name)
    dockerfile_text = plugin.render_measure_dockerfile(env)
    measure_sh_text = plugin.measure_test_sh(graded=plan.graded)

    runner = verify_mod.DockerRunner(echo=echo)
    base_image = plugin.toolchain_spec().base_image
    try:
        intact_image_digest = runner.image_digest(base_image)
    except verify_mod.VerifyError as exc:
        raise langs_base.LangError(
            f'measure phase needs {base_image} loaded locally to derive '
            f'provenance for graded.lock.json: {exc}'
        ) from exc

    repo_sha = measure_mod.repo_tree_sha256(plan.repo)
    if lock_path.is_file():
        try:
            existing = measure_mod.load_lock(lock_path)
            reasons = measure_mod.check_provenance(existing, repo_sha, intact_image_digest)
            if not reasons and int(existing.get('expected', 0)) >= 1:
                echo(f'reusing measure lock {lock_path} (expected={existing["expected"]})')
                new_graded = dataclasses.replace(
                    plan.graded, _expected=int(existing['expected']),
                )
                return dataclasses.replace(plan, graded=new_graded)
            if reasons:
                echo(f're-measuring: {"; ".join(reasons)}')
        except measure_mod.MeasureError as exc:
            echo(f're-measuring: {lock_path} unreadable ({exc})')

    tooling_context_dir = staging_dir / 'tooling'
    runner_with_tooling = _MeasureRunnerAdapter(runner, tooling_context_dir)

    def _run_measure(dep_plan: DepPlan | None) -> dict:
        """One build-and-measure of ONE candidate environment.

        A closure rather than a second copy of the request assembly: the refine
        loop calls this once per attempt and the default path calls it exactly
        once, so both paths write the same context out of the same lines and a
        drift between them is not expressible.
        """
        text = (
            dockerfile_text if dep_plan is None
            else plugin.render_measure_dockerfile(env, dep_plan=dep_plan)
        )
        measure_ctx.mkdir(parents=True, exist_ok=True)
        dockerfile = measure_ctx / 'Dockerfile'
        dockerfile.write_text(text, encoding='utf-8')
        (measure_ctx / 'measure.sh').write_text(measure_sh_text, encoding='utf-8')

        request = measure_mod.MeasureRequest(
            repo=plan.repo,
            repo_name=plan.repo_name,
            lang=plan.lang,
            intact_image=base_image,
            dockerfile=dockerfile,
            context=measure_ctx,
            scope=plan.scope.value,
            floor_mode=plugin.floor_mode,
            fingerprint_sha256={},
            repo_url=plan.provenance.repo_url if plan.provenance else None,
            commit=plan.provenance.commit if plan.provenance else None,
            dep_plan=dep_plan,
        )
        return measure_mod.measure(request, plugin, runner_with_tooling, echo=echo)

    if resolve_env:
        refined = _resolve_env_plan(
            resolver, plan, base_image,
            build_and_measure=_run_measure,
            validate_candidate=plugin.validate_dep_plan,
            echo=echo, clock=clock,
        )
        lock = refined.lock
    else:
        lock = _run_measure(None)
    measure_mod.write_lock(lock_path, lock)

    expected = int(lock['expected'])
    if expected < 1:
        raise langs_base.LangError(
            f'measure phase reported expected={expected}; refusing a floor of zero'
        )
    echo(f'measured intact denominator: expected={expected}')

    new_graded = dataclasses.replace(plan.graded, _expected=expected)
    return dataclasses.replace(plan, graded=new_graded)


class _MeasureRunnerAdapter:
    """Wraps DockerRunner so `build_with_contexts` also mounts the tooling context.

    measure.py's contract is a runner with `image_digest`, `build_with_contexts`,
    `run`, `remove_image`. It passes only `contexts={REPO_CONTEXT: ...}`; for
    rust we also need the wabt tarball via the `tooling` context, so this
    adapter unions the plugin-managed contexts into every build.
    """

    def __init__(self, runner, tooling_dir: Path):
        self._runner = runner
        self._tooling_dir = Path(tooling_dir)

    def image_digest(self, image):
        return self._runner.image_digest(image)

    def build_with_contexts(self, *, image, dockerfile, context, contexts):
        merged = dict(contexts)
        if self._tooling_dir.is_dir() and langs_base.TOOLING_CONTEXT not in merged:
            merged[langs_base.TOOLING_CONTEXT] = str(self._tooling_dir)
        return self._runner.build_with_contexts(
            image=image, dockerfile=dockerfile, context=context, contexts=merged,
        )

    def run(self, image, script):
        return self._runner.run(image, script)

    def remove_image(self, image):
        return self._runner.remove_image(image)
