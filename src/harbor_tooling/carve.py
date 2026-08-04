"""Delete the carved subsystem from a DESTINATION COPY of a repository.

Safety contract, enforced at runtime and not merely documented:

  * carve.py never reads-and-writes in place. It takes --dest, which must
    already exist and must NOT be inside a protected pristine tree.
  * Any --dest under repos-src/ or repos-test/ is refused outright. Those are
    the immutable originals; the whole benchmark depends on them staying that
    way.
  * The carve set is resolved against --dest, so a manifest glob can never
    reach outside it.

Emits carve_receipt.json next to the carved tree so the grader can verify
exactly which files were removed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manifest import Manifest, resolve_carved  # noqa: E402

PROTECTED_DIR_NAMES = ('repos-src', 'repos-test')


def assert_dest_is_safe(dest: Path) -> None:
    dest = dest.resolve()
    if not dest.is_dir():
        raise SystemExit(f'--dest does not exist or is not a directory: {dest}')
    for part in dest.parts:
        if part in PROTECTED_DIR_NAMES:
            raise SystemExit(
                f'refusing to carve inside a protected pristine tree: {dest}\n'
                f'({part}/ is read-only for this tooling; copy the repo out first)'
            )
    if dest == Path(dest.anchor) or len(dest.parts) <= 2:
        raise SystemExit(f'refusing to carve a filesystem root or near-root path: {dest}')


def carve(dest: Path, manifest: Manifest, dry_run: bool = False) -> list[str]:
    assert_dest_is_safe(dest)
    carved = resolve_carved(dest, manifest)
    if not carved:
        raise SystemExit(
            f'manifest matched zero files under {dest} -- refusing to write an empty carve'
        )
    removed: list[str] = []
    for rel in carved:
        target = dest / rel
        if not target.is_file():
            continue
        if not dry_run:
            target.unlink()
        removed.append(rel)
    if not dry_run:
        _prune_empty_dirs(dest / manifest.carve_root)
        receipt = {
            'repo': manifest.repo,
            'carve_root': manifest.carve_root,
            'test_command': manifest.test_command,
            'removed_files': removed,
            'removed_count': len(removed),
        }
        (dest / 'carve_receipt.json').write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8'
        )
    return removed


def _prune_empty_dirs(root: Path) -> None:
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        if not dirnames and not filenames:
            Path(dirpath).rmdir()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dest', required=True, type=Path, help='destination COPY to carve')
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)

    manifest = Manifest.load(args.manifest)
    removed = carve(args.dest, manifest, dry_run=args.dry_run)
    verb = 'would remove' if args.dry_run else 'removed'
    print(f'{verb} {len(removed)} files from {args.dest}/{manifest.carve_root}')
    for rel in removed[:10]:
        print(f'  {rel}')
    if len(removed) > 10:
        print(f'  ... and {len(removed) - 10} more')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
