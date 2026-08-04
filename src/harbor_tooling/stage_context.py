#!/usr/bin/env python3
"""Build a PRE-CARVED docker build context on the host.

Why this exists
---------------
Every Harbor task image used to `COPY` the intact repository and then carve it
in a later `RUN`.  That is wrong in two independent ways:

  1. The intact tree survives forever in the earlier image layer.  It is not
     reachable from inside a running container, but `docker save <tag>` (or any
     registry pull) hands the complete answer key to anybody who asks.
  2. It makes it tempting to also ship `solution/` into the image "so solve.sh
     can find it", which is a direct, trivially exploitable leak
     (`cp -r /opt/harbor/solution/carved/* $REPO/`).

The fix is to carve on the HOST, into a throwaway staging directory, and make
THAT the build context.  The deleted source then never enters the build context
at all, so no layer -- earliest or latest -- can contain it.

What this script produces under --out:

    repo/                 the pre-carved repository (no carve_receipt.json)
    leak-tripwires.txt    one distinctive source line per carved file, used by
                          the Dockerfile's build-time content assertion

`leak-tripwires.txt` is deliberately NOT copied into the image; the Dockerfile
consumes it through `RUN --mount=type=bind`, so the tripwire text (which is
itself carved source) never lands in a layer either.

A tripwire line is accepted only if it does not occur anywhere in the carved
tree, so a hit during the image build is unambiguous evidence of leakage rather
than a false positive from a surviving file.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# A tripwire must be long enough that colliding by chance with an unrelated
# file somewhere in a 5 GiB image is implausible.  32 is the target; a few
# carved files (rust's src/util/mod.rs is nothing but `pub use foo::*;` lines)
# genuinely have no line that long, so the floor is 24 and the picker always
# takes the LONGEST qualifying line in the file.
MIN_TRIPWIRE_CHARS = 24

# Directories that must never be copied into a build context: VCS metadata (it
# contains the carved files' full history, which is a far bigger leak than the
# working tree) and any build detritus.
PRUNE_DIRS = {'.git', '.gradle', 'target', 'Build', 'Bin', '__pycache__', 'node_modules'}


def norm(line: str) -> str:
    return ' '.join(line.split())


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding='utf-8')
    except (UnicodeDecodeError, OSError):
        return None


def copy_repo(src: Path, dest: Path) -> None:
    """Copy src -> dest, skipping VCS metadata and build outputs."""

    def ignore(directory: str, names: list[str]) -> set[str]:
        del directory
        return {n for n in names if n in PRUNE_DIRS}

    shutil.copytree(src, dest, symlinks=True, ignore=ignore)


def oracle_files(oracle_root: Path, repo_src: Path, carve_root: str) -> list[tuple[Path, str]]:
    """Pair each oracle file with its path relative to the repository root.

    The carved trees are not rooted consistently across repos, so resolve
    against the pristine tree and fail loudly rather than guessing.
    """
    out: list[tuple[Path, str]] = []
    for path in sorted(oracle_root.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(oracle_root).as_posix()
        for candidate in (rel, f'{carve_root}/{rel}'):
            if (repo_src / candidate).is_file():
                out.append((path, candidate))
                break
        else:
            raise SystemExit(f'oracle file has no counterpart in {repo_src}: {rel}')
    return out


def surviving_lines(repo: Path) -> set[str]:
    """Every normalised line present in the carved staging tree."""
    seen: set[str] = set()
    for path in repo.rglob('*'):
        if not path.is_file() or path.is_symlink():
            continue
        text = read_text(path)
        if text is None:
            continue
        for line in text.splitlines():
            seen.add(norm(line))
    return seen


def pick_tripwires(oracle: list[tuple[Path, str]], carved_repo: Path) -> list[str]:
    """One distinctive line per carved file, guaranteed absent from the carve."""
    survivors = surviving_lines(carved_repo)
    tripwires: list[str] = []
    chosen: set[str] = set()
    missed: list[str] = []

    for path, rel in oracle:
        text = read_text(path)
        if text is None:
            missed.append(rel)
            continue
        candidates = []
        for line in text.splitlines():
            stripped = line.strip()
            n = norm(line)
            if len(stripped) < MIN_TRIPWIRE_CHARS:
                continue
            if n in survivors or n in chosen:
                continue
            # A tripwire is fed to `grep -F`, so any byte is legal, but keep the
            # set free of characters that make shell/CI logs unreadable.
            if '\t' in stripped or '\x00' in stripped:
                continue
            candidates.append(stripped)
        if not candidates:
            missed.append(rel)
            continue
        # Longest line == most distinctive == least likely to collide by chance.
        best = max(candidates, key=len)
        tripwires.append(best)
        chosen.add(norm(best))

    if missed:
        raise SystemExit(f'could not derive a tripwire for {len(missed)} file(s): {missed[:5]}')
    return tripwires


def assert_carve_is_content_clean(carved_repo: Path, tripwires: list[str]) -> None:
    """Host-side pre-flight: the staged context must already be free of answers."""
    joined = set(tripwires)
    for path in carved_repo.rglob('*'):
        if not path.is_file() or path.is_symlink():
            continue
        text = read_text(path)
        if text is None:
            continue
        for line in text.splitlines():
            hit = line.strip()
            if hit in joined:
                raise SystemExit(
                    f'staged context still contains carved content: {path}\n  {hit[:120]}'
                )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--repo-src', required=True, type=Path, help='pristine repos-src/<repo>')
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--oracle', required=True, type=Path, help='solution/carved/ ground truth')
    ap.add_argument('--out', required=True, type=Path, help='staging context directory to create')
    ap.add_argument('--expect-carved', type=int, required=True)
    ap.add_argument(
        '--prune-empty-dirs',
        action='store_true',
        help='collapse directories left hollow by the carve (cpp-Rux needs this)',
    )
    args = ap.parse_args(argv)

    manifest = tomllib.loads(args.manifest.read_text())
    carve_root = manifest['carve_root'].strip('/')

    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    repo = out / 'repo'

    print(f'[stage] copying {args.repo_src} -> {repo}')
    copy_repo(args.repo_src.resolve(), repo)

    print('[stage] carving')
    subprocess.run(
        [sys.executable, str(HERE / 'carve.py'), '--dest', str(repo), '--manifest', str(args.manifest)],
        check=True,
    )

    receipt = repo / 'carve_receipt.json'
    if receipt.exists():
        receipt.unlink()
        print('[stage] removed carve_receipt.json (it lists every deleted filename)')

    if args.prune_empty_dirs:
        # carve.py's _prune_empty_dirs inspects a `dirnames` list captured
        # before its children were removed, so a directory whose only contents
        # were themselves-empty directories survives one pass.  Depth-first
        # re-testing collapses the whole skeleton, which would otherwise hand
        # the agent a free map of the deleted subsystem's layout.
        removed = True
        while removed:
            removed = False
            for dirpath, dirnames, filenames in os.walk(repo / carve_root, topdown=False):
                if not dirnames and not filenames:
                    Path(dirpath).rmdir()
                    removed = True

    oracle = oracle_files(args.oracle.resolve(), args.repo_src.resolve(), carve_root)
    if len(oracle) != args.expect_carved:
        raise SystemExit(f'oracle has {len(oracle)} files, expected {args.expect_carved}')

    # Every oracle file must be gone from the staged tree, by path.
    for _, rel in oracle:
        if (repo / rel).exists():
            raise SystemExit(f'carve missed {rel}')

    print(f'[stage] deriving tripwires from {len(oracle)} oracle files')
    tripwires = pick_tripwires(oracle, repo)
    assert_carve_is_content_clean(repo, tripwires)

    (out / 'leak-tripwires.txt').write_text('\n'.join(tripwires) + '\n', encoding='utf-8')
    print(f'[stage] wrote {len(tripwires)} tripwires; context ready at {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
