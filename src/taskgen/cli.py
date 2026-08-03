"""Command line entry point.

    python -m taskgen.cli generate --repo <repo> --out <dir> \
        [--file src/pkg/mod.py] [--func name] [--class Cls] \
        [--package-base src/] [--budget 100000] [--seed 0] \
        [--contexts all|bm25,mix,...]

    python -m taskgen.cli verify --entry <dir> [--repo <repo>] [--keep-image]
    python -m taskgen.cli verify --all <out-dir> [--repo <repo>] [--keep-image]

`generate` is offline and deterministic: run it twice into two directories and
`diff -r` them.

`verify` needs docker: it builds the entry's carved image and reads the reward
out of two real container runs -- untouched (must be 0.0) and after the oracle
(must be 1.0). It exits non-zero if either bar is missed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import verify
from .contexts import CONTEXT_TYPES
from .emit import DEFAULT_BUDGET, DEFAULT_SEED, emit_all
from .langs import base
from .scope import CarveScope


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


def cmd_generate(args) -> int:
    entries = emit_all(
        repo=args.repo,
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
    )
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
    gen.add_argument('--repo', required=True, type=Path, help='repository checkout to carve')
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
