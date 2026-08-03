"""Assemble one complete harbor task entry per context type, per language, per scope.

Layout written per entry (directory name = the uuid5 entry id):

    <entry_id>/
      task.toml                          the plugin's family (A for python/go)
      instruction.md                     only the pre-loaded context block and
                                         the condition differ between the nine
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

Why the staged tree is shared: all nine conditions of one target carve the same
files and share one image and one oracle. Staging nine copies of a 33MB
repository would be nine times the disk for zero additional proof.

DETERMINISM: no timestamps, no host paths, no dict iteration order in output.
The token budget handed to all nine context builders is computed from the WIDEST
of the nine headers, so every condition packs against the same budget.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import _tooling_path

from gen_context import FRAMING_BLOCK, assert_no_leakage  # noqa: E402

from . import scope as scope_mod  # noqa: E402
from . import templates  # noqa: E402
from .carve import build_carve_set  # noqa: E402
from .contexts import (  # noqa: E402
    CONTEXT_TYPES,
    ContextInputs,
    build_body,
    common_root,
    fence,
    stub_phrase,
)
from .gradedset import derive_graded_set, parse_linked, select_carved  # noqa: E402
from .ids import entry_id, slug as make_slug  # noqa: E402
from .langs import base as langs_base  # noqa: E402
from .scope import CarveScope  # noqa: E402
from .staging import stage_carved_tree  # noqa: E402

DEFAULT_BUDGET = 100000
DEFAULT_SEED = 0
TOKENIZER = 'chars4'
UV_VERSION = '0.12.1'

#: uuid5 namespace for the shared staging directory name. Keyed on the carve,
#: so two scopes over one repo stage side by side instead of overwriting.
_STAGING_NS = uuid.uuid5(uuid.NAMESPACE_URL, 'mrgctx.taskgen.staging')


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
    return f'taskgen-{repo_name}:local'


# --------------------------------------------------------------------------
# Planning: target -> carve set -> carve -> graded set -> staged tree
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CarvePlan:
    lang: str
    scope: CarveScope
    repo: Path
    repo_name: str
    target: object
    carve: object
    graded: object
    staged: object
    staging_key: str


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


def _copy_leakscan(dest: Path) -> None:
    """Vendor harbor's scanner beside the staged tree.

    The Dockerfile mounts it from a named context; copying it here is what keeps
    that context relative to the output directory instead of a host path.
    """
    dest.mkdir(parents=True, exist_ok=True)
    source = _tooling_path.TOOLING / 'leakscan.sh'
    if source.is_file():
        shutil.copyfile(source, dest / 'leakscan.sh')
        (dest / 'leakscan.sh').chmod(0o755)


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

    parsed = parse_linked(
        lang, repo, package_base=package_base, project=project,
        scope_dirs=sorted({Path(r).parent.as_posix() for r in carved}),
    )
    only = {target.name} if target is not None else None
    carved_functions = select_carved(parsed, carved, only=only)

    carve = build_carve_set(
        repo, scope, carved_relpaths=carved, language=lang, target=target,
        delete_whole_file=delete_whole_file, carved_functions=carved_functions,
    )
    # A skeleton carve drops files that hold no function body at all, so the
    # graded set must be derived from what was ACTUALLY carved -- deriving it
    # from the glob match would union in tests of code that still exists.
    if carve.carved_relpaths != carved:
        carved_functions = select_carved(parsed, carve.carved_relpaths, only=only)

    graded = derive_graded_set(lang, carved_functions, carve.carved_relpaths)
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
    _copy_leakscan(staging_dir / 'tooling')

    del plugin
    return CarvePlan(
        lang=lang, scope=scope, repo=repo, repo_name=repo_name, target=carve.target_view,
        carve=carve, graded=graded, staged=staged, staging_key=key,
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
    n = len(inp.nodeids)
    a('## How you are graded\n')
    a('Test command, run from the repository root inside the task container:\n')
    if plan.graded.kind == 'go-run':
        pkgs = ' '.join(plan.graded.packages)
        run = '^(' + '|'.join(plan.graded.selectors) + ')$'
        a(f"```bash\ngo test -short -count=1 -json -run '{run}' {pkgs}\n```\n")
        a(f'The graded scope is exactly these {n} test function(s):\n')
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


def _render_test_sh(plugin, plan: CarvePlan, graded_spec) -> str:
    if plan.lang == 'go':
        return plugin.render_test_sh(
            graded_spec, test_file_counts=_test_file_counts(plan),
        )
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
             include=(), exclude=(), delete_whole_file: bool = False) -> list[Entry]:
    """Select, carve, stage, and write one full harbor entry per context type."""
    out = Path(out)
    plugin = langs_base.get(lang)
    plan = plan_carve(
        repo, out, lang=lang, carve_scope=carve_scope, package_base=package_base,
        file=file, func=func, cls=cls, receiver=receiver, project=project,
        include=include, exclude=exclude, delete_whole_file=delete_whole_file,
    )

    inp = ContextInputs.build(
        repo=plan.repo, repo_name=plan.repo_name, target=plan.target,
        carve=plan.carve, nodeids=plan.graded.selectors, budget=budget, seed=seed,
        tokenizer=TOKENIZER, language=lang,
    )
    # One budget for all nine: reserve the WIDEST header so a longer condition
    # name can never buy a condition fewer context tokens than its neighbours.
    inp.header_tokens = max(
        inp.counter.count(render_head(inp, ct, plan)) for ct in context_types
    )

    meta = repo_metadata(plan.repo)
    graded_spec = _graded_spec(plan, plugin)
    out.mkdir(parents=True, exist_ok=True)
    return [
        _write_entry(inp, ct, out, meta, plan, plugin, graded_spec)
        for ct in context_types
    ]
