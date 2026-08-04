"""Build a PRE-CARVED docker build context on the host.

Why this exists
---------------
Carving inside the image is unfixably leaky. Even if the carve RUN deletes the
files, the `COPY repos-src/<repo>` layer that precedes it still contains 100%
of the answer, and `docker save` recovers it. The only way the answer cannot be
extracted from a distributed image is for the answer never to enter the build
context in the first place.

So: copy the pristine repo to a scratch tree, carve THAT on the host, consume
and delete carve_receipt.json (it enumerates carve_root and every removed
filename, which is itself a substantial hint), and hand the result to
`docker build --build-context <name>=<scratch>`. The image then only ever sees
a tree from which the subsystem is already absent.

Also emits the tripwire file used by the Dockerfile's build-time leak gate.
That file IS answer material; it is passed to the build over a BuildKit
`--mount=type=bind`, which leaves no layer, and is deleted by --clean.

repos-src/ and repos-test/ are never touched: carve.py refuses any --dest whose
path contains either name, and this script only ever writes under --staging.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from carve import carve  # noqa: E402
from manifest import Manifest  # noqa: E402
import tripwires  # noqa: E402

# repo -> (carve_root, expected carved file count)
REPOS = {
    'python-a2a-python': ('src/a2a/server', 48),
    'go-multigres': ('go/common/pgprotocol', 39),
    'c-xs': ('src/runtime', 58),
}

# Never copied into the build context. .git is the big one: python-a2a-python's
# image manufactures its own throwaway repo AFTER the carve, and an inherited
# upstream .git would reintroduce every carved file as a reachable object.
PRUNE = shutil.ignore_patterns(
    '.git', '.venv', '__pycache__', '*.pyc', '.pytest_cache', '.ruff_cache',
    '*.egg-info', 'node_modules', '.gradle', 'target',
)


def stage(repo: str, staging_root: Path) -> Path:
    """Materialise <staging>/<repo>/ctx/repo, the pre-carved build context.

    The tree deliberately sits one level DOWN from the context root. Several of
    these repos ship their own .dockerignore (go-multigres' excludes `*.md`,
    `bin/` and `build/`), and BuildKit applies the .dockerignore found at the
    ROOT of a named context. Rooting the context at ctx/ -- which contains
    nothing but repo/ -- means the repo's own ignore file is just another file
    being copied, not a filter on the copy.
    """
    carve_root, expected = REPOS[repo]
    src = ROOT / 'repos-src' / repo
    if not src.is_dir():
        raise SystemExit(f'missing pristine source tree: {src}')

    dest = staging_root / repo / 'ctx' / 'repo'
    if dest.parent.exists():
        shutil.rmtree(dest.parent)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, symlinks=True, ignore=PRUNE)
    if (dest.parent / '.dockerignore').exists():
        raise SystemExit('a .dockerignore at the named-context root would filter the carve')

    manifest = Manifest.load(ROOT / 'shared' / 'manifests' / f'{repo}.carve.toml')
    removed = carve(dest, manifest)

    # Consume the receipt on the HOST, then delete it. Shipping it names
    # carve_root and every single removed filename inside the task image.
    receipt_path = dest / 'carve_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt['removed_count'] != expected:
        raise SystemExit(
            f'{repo}: carved {receipt["removed_count"]} files, expected {expected}'
        )
    if receipt['carve_root'] != carve_root:
        raise SystemExit(f'{repo}: carve_root {receipt["carve_root"]!r} != {carve_root!r}')
    receipt_path.unlink()

    # carve.py's _prune_empty_dirs reads os.walk's `dirnames`, already stale by
    # the time the children are unlinked, so carve_root survives as an empty
    # dir. An empty src/a2a/server is still importable as a PEP 420 namespace
    # package, which masks the intended ImportError.
    while True:
        empties = [
            d for d in (dest / carve_root).rglob('*') if d.is_dir() and not any(d.iterdir())
        ]
        root_dir = dest / carve_root
        if root_dir.is_dir() and not any(root_dir.iterdir()):
            empties.append(root_dir)
        if not empties:
            break
        for d in empties:
            d.rmdir()

    # The oracle on the host must match the carve exactly; if it does not, the
    # RED->GREEN oracle restore cannot work and the task is unshippable.
    oracle = ROOT / 'shared' / 'repo-assets' / repo / 'solution' / 'carved'
    n_oracle = sum(1 for p in oracle.rglob('*') if p.is_file())
    if n_oracle != expected:
        raise SystemExit(f'{repo}: host oracle has {n_oracle} files, expected {expected}')

    # Independent re-verification: nothing matching the carve globs survives,
    # and no receipt survives, anywhere in the staged tree.
    from manifest import resolve_carved
    leftover = resolve_carved(dest, manifest)
    if leftover:
        raise SystemExit(f'{repo}: {len(leftover)} carved paths survived staging: {leftover[:5]}')
    strays = [p for p in dest.rglob('carve_receipt.json')]
    if strays:
        raise SystemExit(f'{repo}: carve_receipt.json survived: {strays}')

    n_staged = sum(1 for p in dest.rglob('*') if p.is_file())
    print(f'{repo}: staged {n_staged} files at {dest} '
          f'({len(removed)} carved out, receipt consumed and deleted)')
    return dest


def emit_tripwires(repo: str, staging_root: Path, count: int) -> Path:
    """Write the leak-gate patterns to <staging>/<repo>/trip/tripwires.txt.

    Kept OUTSIDE ctx/ so it can never be swept into the image by a COPY. It
    reaches the build only through `--mount=type=bind`, which leaves no layer.
    """
    carve_root, _ = REPOS[repo]
    out = staging_root / repo / 'trip' / 'tripwires.txt'
    out.parent.mkdir(parents=True, exist_ok=True)
    picks = tripwires.pick(repo, carve_root, count, min_chars=40)
    out.write_text('\n'.join(line for line, _ in picks) + '\n', encoding='utf-8')
    print(f'{repo}: {len(picks)} tripwires -> {out}')
    for line, origin in picks:
        print(f'    {origin}')
    return out


def verify_pristine_untouched(repo: str) -> None:
    """repos-src/ is a benchmark-wide invariant. Prove we did not disturb it."""
    src = ROOT / 'repos-src' / repo
    carve_root, expected = REPOS[repo]
    manifest = Manifest.load(ROOT / 'shared' / 'manifests' / f'{repo}.carve.toml')
    from manifest import resolve_carved
    still_there = resolve_carved(src, manifest)
    if len(still_there) != expected:
        raise SystemExit(
            f'{repo}: repos-src/ was disturbed -- {len(still_there)} carved files '
            f'present, expected the pristine {expected}'
        )
    if (src / 'carve_receipt.json').exists():
        raise SystemExit(f'{repo}: a receipt was written into pristine repos-src/!')
    print(f'{repo}: repos-src/ intact ({len(still_there)} carve-set files still present)')
    _ = carve_root


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--repo', required=True, choices=sorted(REPOS))
    ap.add_argument('--staging', required=True, type=Path)
    ap.add_argument('--tripwire-count', type=int, default=5)
    ap.add_argument('--clean', action='store_true',
                    help='delete the staging tree for this repo and exit')
    args = ap.parse_args(argv)

    staging_root = args.staging.resolve()
    if args.clean:
        target = staging_root / args.repo
        if target.exists():
            shutil.rmtree(target)
            print(f'{args.repo}: removed staging tree {target}')
        return 0

    staging_root.mkdir(parents=True, exist_ok=True)
    stage(args.repo, staging_root)
    emit_tripwires(args.repo, staging_root, args.tripwire_count)
    verify_pristine_untouched(args.repo)
    _ = subprocess
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
