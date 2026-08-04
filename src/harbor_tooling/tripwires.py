"""Pick distinctive body lines from a carved oracle tree.

A "tripwire" is a line that appears in a CARVED file and in NO surviving file of
the pristine repository. If such a line is found anywhere inside a task image,
the answer leaked -- by any route, including a renamed copy, a tarball, a git
object, a compiled binary or an embedded string. This is content-based, so it
cannot be fooled by moving the files somewhere the path-based checks do not look.

Selection criteria, in order:
  * long (>= --min-chars after whitespace normalisation), so a grep for it is
    not going to collide with unrelated code;
  * not present in ANY surviving (non-carved) file of repos-src/<repo>;
  * no characters that are awkward for `grep -F` / shell round-tripping;
  * spread across distinct carved files, so one missed file cannot hide a leak.

The chosen lines are written verbatim to --out. THAT FILE IS ANSWER MATERIAL and
must never be COPYed into an image; the Dockerfiles consume it through a
BuildKit `--mount=type=bind`, which leaves no layer behind.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SKIP_DIRS = {'.git', 'node_modules', 'target', 'vendor', '__pycache__', '.venv', 'build'}


def norm(line: str) -> str:
    return ' '.join(line.split())


def read(p: Path) -> str | None:
    try:
        return p.read_text(encoding='utf-8')
    except (UnicodeDecodeError, OSError):
        return None


def carved_relpaths(carved_root: Path, src_root: Path, carve_root: str) -> dict[Path, str]:
    """Map each oracle file to its repo-root-relative path in repos-src/."""
    out: dict[Path, str] = {}
    for p in sorted(carved_root.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(carved_root).as_posix()
        for cand in (f'{carve_root}/{rel}', rel):
            if (src_root / cand).is_file():
                out[p] = cand
                break
        else:
            raise SystemExit(f'cannot locate oracle file {rel} under {src_root}')
    return out


def surviving_lines(src_root: Path, carved_rel: set[str]) -> set[str]:
    seen: set[str] = set()
    for p in src_root.rglob('*'):
        if not p.is_file() or p.is_symlink():
            continue
        if any(s in p.parts for s in SKIP_DIRS):
            continue
        if p.relative_to(src_root).as_posix() in carved_rel:
            continue
        text = read(p)
        if text is None:
            continue
        for line in text.splitlines():
            seen.add(norm(line))
    return seen


def greppable(line: str) -> bool:
    """Reject anything that would not survive a `grep -F -f <file>` round trip.

    Patterns travel to grep in a FILE, never on a command line, so quotes and
    backslashes are safe -- -F makes every byte literal. Only non-printable and
    non-ASCII bytes are rejected, because those are what differ between the
    build-time check and a later re-check under a different locale.
    """
    return bool(line) and all(32 <= ord(c) <= 126 for c in line)


def pick(repo: str, carve_root: str, count: int, min_chars: int) -> list[tuple[str, str]]:
    src_root = ROOT / 'repos-src' / repo
    carved_root = ROOT / 'shared' / 'repo-assets' / repo / 'solution' / 'carved'
    mapping = carved_relpaths(carved_root, src_root, carve_root)
    survivors = surviving_lines(src_root, set(mapping.values()))

    # One candidate per carved file, longest first: long unique lines are the
    # least likely to collide and the most likely to survive reformatting.
    per_file: list[tuple[int, str, str]] = []
    for path, rel in sorted(mapping.items(), key=lambda kv: kv[1]):
        text = read(path)
        if text is None:
            continue
        best: tuple[int, str, str] | None = None
        for i, raw in enumerate(text.splitlines(), 1):
            n = norm(raw)
            if len(n) < min_chars or n in survivors or not greppable(n):
                continue
            if best is None or len(n) > best[0]:
                best = (len(n), n, f'{rel}:{i}')
        if best is not None:
            per_file.append(best)

    if len(per_file) < count:
        raise SystemExit(f'{repo}: only {len(per_file)} candidate files, wanted {count}')

    # Spread the picks evenly across the carved file list rather than taking the
    # first N, so a leak confined to one corner of the subsystem still trips.
    step = len(per_file) / count
    chosen = [per_file[int(i * step)] for i in range(count)]
    return [(line, origin) for _, line, origin in chosen]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--repo', required=True)
    ap.add_argument('--carve-root', required=True)
    ap.add_argument('--count', type=int, default=5)
    ap.add_argument('--min-chars', type=int, default=40)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)

    picks = pick(args.repo, args.carve_root.strip('/'), args.count, args.min_chars)
    args.out.write_text('\n'.join(line for line, _ in picks) + '\n', encoding='utf-8')
    for line, origin in picks:
        print(f'  {origin}  ({len(line)} chars)', file=sys.stderr)
    print(f'wrote {len(picks)} tripwires -> {args.out}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
