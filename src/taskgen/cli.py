"""Command line entry point.

    python -m taskgen.cli generate (--repo <checkout> | --repo-url <src> \
        [--commit <sha> | --allow-floating]) --out <dir> \
        [--file src/pkg/mod.py] [--func name] [--class Cls] \
        [--package-base src/] [--budget 100000] [--seed 0] \
        [--contexts all|bm25,mix,...]

    python -m taskgen.cli verify --entry <dir> [--repo <repo>] [--keep-image]
    python -m taskgen.cli verify --all <out-dir> [--repo <repo>] [--keep-image]

`generate` is offline and deterministic for python and go, and for any run that
reuses a pinned `graded.lock.json`: run it twice into two directories and
`diff -r` them. A COLD whole-suite run (c, cpp, java, rust) resolves its
environment from a model first, which is a normal step of generating those
tasks; `--no-resolve-env` opts out to the plugin's hardcoded environment.

`verify` needs docker: it builds the entry's carved image and reads the reward
out of two real container runs -- untouched (must be 0.0) and after the oracle
(must be 1.0). It exits non-zero if either bar is missed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import clone, verify
from . import emit as emit_mod
from .contexts import CONTEXT_TYPES
from .emit import DEFAULT_BUDGET, DEFAULT_SEED, emit_all
from .env_resolver import EnvResolver, ResolveRefused
from .langs import base
from .llm_resolver import LlmEnvResolver, ResolverTransportError
from .resolver_inputs import base_capabilities, required_plan_slots
from .scope import CarveScope

#: Where `--repo-url` clones land by default: the same directory `verify`
#: already searches when `--repo` is omitted (verify.REPOS_SRC_RELPATH), so a
#: cloned task verifies with no extra flags.
DEFAULT_REPOS_CACHE = Path('../../harbor-tasks/repos-src')


def _parse_contexts(raw: str) -> tuple[str, ...]:
    if raw == 'all':
        return CONTEXT_TYPES
    wanted = [c.strip() for c in raw.split(',') if c.strip()]
    unknown = [c for c in wanted if c not in CONTEXT_TYPES]
    if unknown:
        raise SystemExit(
            f'unknown context type(s): {", ".join(unknown)}. '
            f'Choose from: {", ".join(CONTEXT_TYPES)}'
        )
    # Canonical order regardless of how they were listed, so --contexts changes
    # WHICH entries are written, never their content.
    return tuple(c for c in CONTEXT_TYPES if c in set(wanted))


def resolve_source(args, echo=print) -> tuple[Path, str | None, str | None, str | None]:
    """`(local checkout, repo_url, commit, clone_kind)` for one generate run.

    `--repo` is used DIRECTLY and records no provenance, so its emitted bytes are
    exactly what they were before cloning existed. `--repo-url` goes through
    `clone.ensure_repo`, which itself falls back to direct use for a plain
    directory snapshot that has no `.git` to clone.
    """
    if args.repo is not None:
        return args.repo, None, None, None

    source = args.repo_url
    name = args.repo_name or clone.derive_name(source)
    repo = clone.ensure_repo(
        source=source,
        commit=args.commit,
        cache_dir=Path(args.repos_cache),
        name=name,
        allow_floating=args.allow_floating,
    )
    head = clone.head_commit(repo)
    pinned = head is not None and args.commit is not None and not args.allow_floating
    if pinned:
        echo(f'source      {source} @ {head}')
        return repo, source, head, 'pinned'
    echo(f'source      {source} (FLOATING: not reproducible)')
    echo(
        'warning     this task records no commit, so upstream can move under it '
        'and verify will start failing. Pass --commit <sha> to pin it.'
    )
    return repo, source, '', 'floating'


def no_llm_config_message(lang: str, detail: str) -> str:
    """The refusal a cold whole-suite run gets when no model is reachable.

    Both remedies by name, because the two are genuinely different decisions and
    neither is the tool's to make: a resolved environment is what the task's
    floor is measured in, so degrading quietly to the hardcoded one would ship a
    task whose environment nobody chose and nobody could tell apart afterwards.
    """
    return (
        f'resolving the environment for --lang {lang}: {detail}\n'
        'Resolving is a normal step of generating a whole-suite task -- the '
        'DepPlan it produces builds the measure image AND the environment the '
        'task ships -- and this run has no valid pinned environment in '
        'graded.lock.json to reuse, so there is nothing to fall back on that '
        'anybody vouched for. Either:\n'
        '  * point --llm-config at the JSON naming your bridge, e.g. '
        '--llm-config .llm_config/claude-code-oauth.json (or put one at the '
        'repo-root .llm_config, which is discovered automatically); or\n'
        f'  * pass --no-resolve-env to generate against the hardcoded {lang} '
        'environment instead.\n'
        'The run refuses rather than silently degrading to an environment '
        'nobody resolved.'
    )


def build_env_resolver(args, *, echo=print) -> EnvResolver:
    """The real resolver: config in, client out.

    The ONLY place a generate run learns how to reach a model, at the cli
    boundary so `emit`'s import graph stays free of anything that opens a socket.

    `--llm-config` is discovered exactly the way `--verifier`'s is, through
    `verifier.llm_config.resolve_config`: one repo has one bridge, and making
    the user retype its path for a step that now runs by default would be
    friction with nothing behind it. What is NOT shared with `--verifier` is the
    consequence of not finding one -- see `no_llm_config_message`.
    """
    from verifier.llm_config import LLMConfigError, client_from_config, resolve_config

    try:
        cfg = resolve_config(args.llm_config)
    except LLMConfigError as exc:
        raise SystemExit(no_llm_config_message(args.lang, str(exc))) from exc

    endpoint = str(cfg['base_url'])
    plugin = base.get(args.lang)
    return LlmEnvResolver(
        client=client_from_config(cfg),
        capabilities=base_capabilities(plugin),
        required_slots=required_plan_slots(plugin),
        endpoint=endpoint,
        model=str(cfg['model']),
        echo=echo,
    )


class DeferredEnvResolver:
    """A resolver that discovers its config on the FIRST ASK, not before.

    Deferred because a WARM lock never asks. `_measure_and_pin` returns the
    pinned environment before it reaches a resolver, so a regeneration over a
    valid lock needs no config, no bridge and no model -- and building the
    client up front would demand all three of a run that provably uses none of
    them, and would drag the model sdk into a process that contacts nothing.

    The first ask is also the first moment the failure is REAL, which is why the
    loud message lives here rather than in argument parsing: "cold" is not
    knowable until the lock has been read.
    """

    def __init__(self, args, *, echo=print):
        self._args = args
        self._echo = echo
        self._delegate: EnvResolver | None = None

    def built(self) -> EnvResolver:
        if self._delegate is None:
            self._delegate = build_env_resolver(self._args, echo=self._echo)
        return self._delegate

    def __call__(self, *, lang: str, repo: Path, base_image: str,
                 repair: str | None = None):
        return self.built()(
            lang=lang, repo=repo, base_image=base_image, repair=repair,
        )


def env_resolution_mode(args) -> bool | None:
    """`--no-resolve-env` off, `--resolve-env` forced, neither the default."""
    if args.no_resolve_env:
        return False
    return True if args.resolve_env else None


def build_generate_resolver(args, *, echo=print) -> EnvResolver | None:
    """The resolver this run will use, or None because it will not resolve.

    None is returned for the opted-out run AND for a parser-backed language,
    which runs no measure phase at all: constructing a resolver for a run that
    structurally cannot use one would be a promise the pipeline cannot keep.
    """
    try:
        wanted = emit_mod.resolution_wanted(args.lang, env_resolution_mode(args))
    except base.LangError as exc:
        raise SystemExit(f'--resolve-env: {exc}') from exc
    return DeferredEnvResolver(args, echo=echo) if wanted else None


def cmd_generate(args) -> int:
    repo, repo_url, commit, clone_kind = resolve_source(args)
    resolver = build_generate_resolver(args)
    try:
        entries = _emit(args, repo, repo_url, commit, clone_kind, resolver)
    except ResolveRefused as exc:
        # SHIP or REFUSE: the measure phase raises before any entry or lock is
        # written, so this path leaves the output directory empty on purpose.
        print(f'REFUSE({exc.reason})', file=sys.stderr)
        return 3
    except ResolverTransportError as exc:
        raise SystemExit(f'environment resolution: {exc}') from exc

    first = entries[0]
    print(f'language    {first.lang}')
    print(f'carve scope {first.carve_scope}')
    print(f'target      {first.target_relpath}::{first.slug.split("__")[1]}')
    print(f'carved      {len(first.carved_relpaths)} file(s)')
    for rel in first.carved_relpaths:
        print(f'  - {rel}')
    print(f'graded ids  {len(first.nodeids)}')
    print(f'out         {Path(args.out).resolve()}')
    for e in entries:
        print(f'  {e.entry_id}  {e.context_type}')
    print(f'{len(entries)} entries written')
    return 0


def _emit(args, repo, repo_url, commit, clone_kind, resolver):
    return emit_all(
        repo=repo,
        out=args.out,
        package_base=args.package_base,
        file=args.file,
        func=args.func,
        cls=getattr(args, 'class'),
        budget=args.budget,
        seed=args.seed,
        context_types=_parse_contexts(args.contexts),
        lang=args.lang,
        carve_scope=args.carve_scope,
        receiver=args.receiver,
        project=args.project,
        include=args.include,
        exclude=args.exclude,
        delete_whole_file=args.delete_whole_file,
        repo_url=repo_url,
        commit=commit,
        clone_kind=clone_kind,
        verifier=args.verifier,
        llm_config=args.llm_config,
        verifier_min_criteria=args.verifier_min_criteria,
        resolve_env=env_resolution_mode(args),
        resolver=resolver,
    )


def cmd_verify(args) -> int:
    try:
        return verify.main(
            entry=args.entry,
            out=getattr(args, 'all'),
            repo=args.repo,
            keep_image=args.keep_image,
            lang=args.lang,
            carve_scope=args.carve_scope,
        )
    except verify.VerifyError as exc:
        raise SystemExit(f'verify: {exc}') from exc


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='taskgen', description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)

    gen = sub.add_parser('generate', help='emit one harbor entry per context type')
    source = gen.add_mutually_exclusive_group(required=True)
    source.add_argument('--repo', type=Path, default=None,
                        help='repository checkout to carve, already on disk')
    source.add_argument('--repo-url', default=None, metavar='SRC',
                        help='clone source instead of a ready checkout: a public '
                             'https/ssh/git/file:// URL, or a local git repository. A '
                             'plain directory with no .git is used directly, since '
                             'there is nothing to clone')
    gen.add_argument('--commit', default=None, metavar='SHA',
                     help='--repo-url: the commit to check out. Required unless '
                          '--allow-floating: verify pins the task to exact upstream '
                          'bytes, so an unpinned clone stops verifying when upstream '
                          'moves')
    gen.add_argument('--allow-floating', action='store_true',
                     help='--repo-url: accept the default-branch HEAD instead of a '
                          'pinned commit. The emitted task is NOT reproducible')
    gen.add_argument('--repo-name', default=None, metavar='NAME',
                     help='--repo-url: override the checkout basename derived from the '
                          'source. It is the identity: it seeds the entry ids, the slug '
                          'and the image tag')
    gen.add_argument('--repos-cache', type=Path, default=DEFAULT_REPOS_CACHE,
                     metavar='DIR',
                     help=f'--repo-url: where clones land (default: {DEFAULT_REPOS_CACHE})')
    gen.add_argument('--out', required=True, type=Path, help='output directory')
    gen.add_argument('--package-base', default='src/',
                     help="import root inside the repo (default: 'src/')")
    gen.add_argument('--file', default=None,
                     help='target file, repo-relative; default: first eligible')
    gen.add_argument('--func', default=None, help='target function name')
    gen.add_argument('--class', default=None, dest='class',
                     help='target class name; omit for module-level functions')
    gen.add_argument('--lang', default='python', choices=list(base.PLANNED_LANGS),
                     help='language plugin to render the entry with (default: python)')
    gen.add_argument('--carve-scope', default=CarveScope.FUNCTION.value,
                     choices=[s.value for s in CarveScope],
                     help='how much to carve: one function body, or every function '
                          'body in a glob-selected set of files (default: function)')
    gen.add_argument('--include', action='append', default=[], metavar='GLOB',
                     help='file/folder scope: a carve glob; repeatable')
    gen.add_argument('--exclude', action='append', default=[], metavar='GLOB',
                     help='file/folder scope: subtract a glob from --include; repeatable')
    gen.add_argument('--delete-whole-file', action='store_true',
                     help='file/folder scope: delete the carved files outright '
                          '(harbor-faithful) instead of skeleton-stubbing them. Opt-in: '
                          'a skeleton keeps the tree importable and collectable, so a '
                          'delete risks a repo-wide collection cascade the solver '
                          'cannot fix')
    gen.add_argument('--receiver', default=None,
                     help='go function scope: the method receiver type, required when '
                          'one file declares the same method name on two receivers')
    gen.add_argument('--project', default=None,
                     help='go: the go.mod project segment the parser matches imports '
                          'on; derived from go.mod when omitted')
    gen.add_argument('--budget', type=int, default=DEFAULT_BUDGET)
    gen.add_argument('--seed', type=int, default=DEFAULT_SEED)
    gen.add_argument('--contexts', default='all',
                     help=f"'all' or a comma list of: {', '.join(CONTEXT_TYPES)}")
    gen.add_argument('--verifier', action='store_true',
                     help='additionally generate ONE per-task verifier bundle and copy '
                          "it into every entry's solution/verifier/. OFF by default: the "
                          'bundle is LLM-authored, so a --verifier run is NOT '
                          'diff -r-reproducible and needs an LLM proxy. Everything else '
                          'the run emits is unchanged')
    gen.add_argument('--llm-config', default=None, metavar='PATH',
                     help='the JSON config naming the LLM proxy (model, base_url, '
                          'api_key). Both --verifier and environment resolution '
                          'default it to the git-ignored .llm_config at the repo '
                          'root. Missing or unreachable is a LOUD failure, never a '
                          'silently empty bundle or an unvouched-for lock -- but a '
                          'run that reuses a pinned graded.lock.json resolves nothing '
                          'and needs no config at all')
    gen.add_argument('--verifier-min-criteria', type=int,
                     default=emit_mod.DEFAULT_VERIFIER_MIN_CRITERIA, metavar='INT',
                     help='--verifier: how many rubric criteria must survive '
                          'golden-pass AND stub-fail before a bundle ships (default: '
                          f'{emit_mod.DEFAULT_VERIFIER_MIN_CRITERIA}). Below the floor '
                          'the task ships WITHOUT a bundle -- additive and non-blocking')
    env = gen.add_mutually_exclusive_group()
    env.add_argument('--no-resolve-env', action='store_true',
                     help='do NOT resolve the environment: build the measure image '
                          "and ship the environment from the plugin's hardcoded "
                          'toolchain and dependency lines, exactly as they were '
                          'before resolution existed. The emitted bytes and the lock '
                          'are byte-identical to that behaviour, and no model config '
                          'is looked for or needed')
    env.add_argument('--resolve-env', action='store_true',
                     help='ON by default for c, cpp, java and rust, so this flag asks '
                          'for nothing extra: resolving the toolchain and dependency '
                          "block from a structured DepPlan -- instead of the plugin's "
                          'hardcoded lines -- and pinning that plan into '
                          'graded.lock.json is a normal step of generating those '
                          'tasks. Passing it explicitly makes an unsupported --lang a '
                          'LOUD error instead of the silent no-op it is by default. A '
                          'reused lock never re-resolves')
    gen.set_defaults(func_impl=cmd_generate)

    ver = sub.add_parser(
        'verify',
        help='docker-build an emitted entry and prove stub=0.0 / oracle=1.0',
    )
    scope = ver.add_mutually_exclusive_group(required=True)
    scope.add_argument('--entry', type=Path, help='one emitted entry directory')
    scope.add_argument('--all', type=Path, dest='all',
                       help='an output directory: verify every entry it holds through '
                            'the one image they share')
    ver.add_argument('--repo', type=Path, default=None,
                     help='source checkout to build from; default: located from the '
                          'entry under harbor-tasks/repos-src/')
    ver.add_argument('--keep-image', action='store_true',
                     help='keep the built image instead of deleting it afterwards')
    ver.add_argument('--lang', default='python', choices=list(base.PLANNED_LANGS),
                     help="the entry's language; selects the floor mode and reward "
                          'parser (default: python)')
    ver.add_argument('--carve-scope', default=CarveScope.FUNCTION.value,
                     choices=[s.value for s in CarveScope],
                     help='how much of the repo the entry carves (default: function)')
    ver.set_defaults(func_impl=cmd_verify)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func_impl(args)


if __name__ == '__main__':
    sys.exit(main())
