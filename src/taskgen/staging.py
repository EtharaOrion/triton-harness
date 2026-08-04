"""Host-side carve staging: the intact tree NEVER enters a build context.

This module writes NO new carve, prune or tripwire algorithm. It DRIVES the
tooling harbor already ships and already proved, adding only the one case harbor
does not have (an *overlay* of skeleton-stubbed text, where harbor only deletes)
and the §(f) tripwire fallback ladder that multi-file carve needs.

Wrapped, by file:line
---------------------
  shared/tooling/stage_context.py
      :58  PRUNE_DIRS                    (via copy_repo)
      :61  norm
      :72  copy_repo                     -> the staged tree
      :102 surviving_lines               (via pick_tripwires)
      :116 pick_tripwires                -> the line tripwire (rung 1)
      :154 assert_carve_is_content_clean -> host-side pre-flight
  shared/tooling/stage_carved.py
      :52  PRUNE                         -> the extra ignore patterns
                                           (.venv/.pytest_cache/*.egg-info/...)
      :78  the "no .dockerignore at the named-context root" rule
      :86  receipt consumed-and-deleted on the HOST
      :125 stray-receipt assertion
  shared/tooling/carve.py
      :29  PROTECTED_DIR_NAMES
      :32  assert_dest_is_safe           -> repos-src/repos-test refusal

API mismatches we had to adapt around (documented, not hidden)
-------------------------------------------------------------
  * `stage_carved.stage()` and `tripwires.pick()` are hardwired to a 3-repo
    table, to `repos-src/<repo>` and to `shared/repo-assets/<repo>/solution/
    carved`. taskgen stages an ARBITRARY repo from an in-memory carve set, so
    neither entry point is callable; their reusable parts (`PRUNE`,
    `greppable`) are wrapped and the tree-relative `stage_context.pick_tripwires`
    is used as the picker.
  * `pick_tripwires` raises `SystemExit` for the WHOLE batch and its message
    truncates the miss list, so a failure cannot be attributed to a file. It is
    therefore called batch-first as the fast path, and the ladder re-drives it
    one file at a time only when the batch fails.
  * harbor's picker floor is 24 chars and it always takes the LONGEST qualifying
    line, so what it returns is re-checked against STRONG_TRIPWIRE_CHARS here
    and a sub-strong line is rejected in favour of the digest rung.

The ladder is two rungs, not the four §(f) proposed, because `leakscan.sh`
consumes `grep -F -f`, which reads its pattern file as ONE PATTERN PER LINE. A
multi-line "window" pattern cannot be expressed to it: it silently becomes N
independent single-line patterns, each weaker than the line that was rejected,
while the coverage count still reads as full. So a carved file yields either a
strong single line or its content digest, and nothing in between.
  * `copy_repo` uses `shutil.copytree` (dest must not exist) while
    `assert_dest_is_safe` requires the path to exist -- so the staging root is
    created and safety-checked first, and the tree is copied one level below it.

Layout produced under `out_dir` -- only `ctx/` is ever shipped:

    ctx/repo/               the carved tree      -> --build-context repoctx=
    trip/tripwires.txt      grep patterns        -> RUN --mount=type=bind
    trip/tripwire-digests.txt
    oracle/<relpath>        intact originals     HOST ONLY
    carve_receipt.json      the receipt          HOST ONLY

The receipt names every carved path and its digest, i.e. it is a free, precise
description of what to regenerate. It is produced, consumed and kept OUTSIDE
`ctx/`, exactly as harbor consumes-and-unlinks it at `stage_carved.py:86-94`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import _tooling_path  # noqa: F401  (side effect: sys.path -> harbor tooling)

import carve as harbor_carve  # noqa: E402
import stage_carved as harbor_stage_carved  # noqa: E402
import stage_context as harbor_stage_context  # noqa: E402

__all__ = [
    'StagedTree',
    'StagingError',
    'Tripwire',
    'TripwireError',
    'stage_carved_tree',
]

#: harbor's candidate floor (stage_context.py:53). NOT the acceptance bar.
MIN_TRIPWIRE_CHARS = harbor_stage_context.MIN_TRIPWIRE_CHARS

#: Acceptance bar for a line handed to leakscan.sh. harbor's 24 is safe for
#: harbor's whole-file deletes, which choose from every line in the file; a
#: skeleton stub removes only bodies, whose entire candidate pool can be one
#: line of boilerplate. Measured on the real folder image: a2a errors.py
#: offered only `super().__init__(self.message)` (30 chars, over 24) and
#: leakscan hit it in 13 unrelated files (urllib3, litellm, redis, pip).
#: Do not lower this to match upstream.
STRONG_TRIPWIRE_CHARS = 40

#: harbor's `PRUNE_DIRS` carries cpp-Rux's capitalised `Build/` and `Bin/` but no
#: lowercase `build/`, which plan §(e).8 lists explicitly. Added here rather than
#: edited into the harbor constant, so the wrapped tooling stays untouched.
#:
#: Applied WITH `_should_prune_extra_dir` below. `build/` unconditionally would
#: strip java-tamboui's legitimate `dev.tamboui.build` Java package at
#: `buildSrc/src/main/java/dev/tamboui/build/` -- the reference author flagged
#: that exact failure mode with a specific error signature ("Unresolved reference
#: 'dev'" at :buildSrc:compileKotlin). The refined predicate keeps `build/` when
#: it sits inside a Java-style source root (`**/src/main/**` or `**/src/test/**`),
#: where the name can only be a package identifier; everywhere else it stays
#: pruned. python/rust/c/cpp/go carry `build/` at repo-top or beside project
#: manifests, both of which are correctly outside `**/src/main|test/**`, so their
#: pruning is byte-identical to before.
EXTRA_PRUNE_DIRS = {'build'}
_JAVA_SOURCE_ROOT_MARKERS = ('main', 'test')


def _should_prune_extra_dir(dirpath: str, name: str) -> bool:
    """Refine EXTRA_PRUNE_DIRS: `build/` inside Java src roots is a package."""
    if name not in EXTRA_PRUNE_DIRS:
        return False
    if name == 'build':
        parts = Path(dirpath).parts
        for i, part in enumerate(parts):
            if (
                part == 'src'
                and i + 1 < len(parts)
                and parts[i + 1] in _JAVA_SOURCE_ROOT_MARKERS
            ):
                return False
    return True


class StagingError(RuntimeError):
    """A staged tree that must not be shipped. Always raised, never warned."""


class TripwireError(StagingError):
    """No tripwire could be derived. The leak gate is never downgraded."""


@dataclass(frozen=True)
class Tripwire:
    relpath: str
    rung: int
    kind: str  # 'line' | 'sha256'
    pattern: str
    origin: str = ''


@dataclass(frozen=True)
class StagedTree:
    out_dir: Path
    ctx_dir: Path
    repo_dir: Path
    oracle_dir: Path
    receipt: dict
    receipt_path: Path
    tripwire_path: Path
    tripwire_digest_path: Path
    tripwires: tuple[Tripwire, ...]
    staged_relpaths: tuple[str, ...]

    @property
    def tripwire_patterns(self) -> tuple[str, ...]:
        """Fixed-string patterns for the leak scan (`leakscan.sh` / `grep -F`)."""
        return tuple(t.pattern for t in self.tripwires if t.kind != 'sha256')

    @property
    def tripwire_digests(self) -> tuple[str, ...]:
        """Content digests, for the files that had no greppable tripwire."""
        return tuple(t.pattern for t in self.tripwires if t.kind == 'sha256')


# --------------------------------------------------------------------------
# safety
# --------------------------------------------------------------------------


def _assert_out_dir_is_safe(out_dir: Path, repo: Path) -> None:
    for part in out_dir.parts:
        if part in harbor_carve.PROTECTED_DIR_NAMES:
            raise StagingError(
                f'refusing to stage inside a protected pristine tree: {out_dir} '
                f'({part}/ is read-only for this tooling)'
            )
    if out_dir == repo or repo in out_dir.parents or out_dir in repo.parents:
        raise StagingError(f'staging dir {out_dir} overlaps the source repo {repo}')


def _assert_dest_is_safe(out_dir: Path) -> None:
    """harbor `carve.assert_dest_is_safe` (:32), with SystemExit adapted."""
    try:
        harbor_carve.assert_dest_is_safe(out_dir)
    except SystemExit as exc:
        raise StagingError(str(exc)) from exc


# --------------------------------------------------------------------------
# the staged tree
# --------------------------------------------------------------------------


def _prune_extra(root: Path) -> None:
    """Apply harbor `stage_carved.PRUNE` (:52) on top of `copy_repo`'s PRUNE_DIRS.

    `copy_repo` drops directories only; PRUNE additionally removes `.venv`,
    `.pytest_cache`, `.ruff_cache`, `*.egg-info` and loose `*.pyc`.
    """
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        here = Path(dirpath)
        names = list(dirnames) + list(filenames)
        doomed = harbor_stage_carved.PRUNE(dirpath, names) | {
            n for n in dirnames if _should_prune_extra_dir(dirpath, n)
        }
        for name in doomed:
            victim = here / name
            if victim.is_dir():
                shutil.rmtree(victim, ignore_errors=True)
            elif victim.exists():
                victim.unlink()


def _prune_empty_parents(start: Path, stop: Path) -> None:
    """Collapse directories left hollow by a delete (harbor stage_carved:100-110)."""
    current = start
    while current != stop and stop in current.parents:
        if any(current.iterdir()):
            return
        current.rmdir()
        current = current.parent


def _relpaths(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file())
    )


# --------------------------------------------------------------------------
# tripwires: the §(f) ladder, fail-closed
# --------------------------------------------------------------------------


def _is_strong(line: str) -> bool:
    return len(harbor_stage_context.norm(line)) >= STRONG_TRIPWIRE_CHARS


def _rung_line(path: Path, rel: str, repo_dir: Path, chosen: set[str]) -> Tripwire | None:
    """Rung 1: harbor's own picker, driven one file at a time, strong lines only."""
    try:
        picked = harbor_stage_context.pick_tripwires([(path, rel)], repo_dir)
    except SystemExit:
        return None
    if not picked:
        return None
    line = picked[0]
    if not _is_strong(line) or harbor_stage_context.norm(line) in chosen:
        return None
    return Tripwire(rel, 1, 'line', line, rel)


def _staged_digests(repo_dir: Path) -> set[str]:
    return {
        hashlib.sha256(p.read_bytes()).hexdigest()
        for p in repo_dir.rglob('*')
        if p.is_file() and not p.is_symlink()
    }


def _rung_sha256(text: str, rel: str, digests: set[str]) -> Tripwire | None:
    """Rung 2: the file's content digest, matched by digesting the image."""
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    if not text or digest in digests:
        # A surviving file already has these exact bytes: the digest would fire
        # on a legitimate file, so it is not a tripwire at all.
        return None
    return Tripwire(rel, 4, 'sha256', digest, rel)


def _derive_tripwires(
    originals: dict[str, str], repo_dir: Path, oracle_dir: Path
) -> tuple[Tripwire, ...]:
    pairs = [(oracle_dir / rel, rel) for rel in sorted(originals)]

    # FAST PATH: harbor's batch picker. One survivors() scan for the whole set.
    try:
        picked = harbor_stage_context.pick_tripwires(pairs, repo_dir)
    except SystemExit:
        picked = None
    if picked is not None and len(picked) == len(pairs) and all(map(_is_strong, picked)):
        return tuple(
            Tripwire(rel, 1, 'line', line, rel)
            for (_, rel), line in zip(pairs, picked)
        )

    # LADDER: at least one file had no strong line. Re-drive per file and fall
    # back to the digest -- never silently under-cover, never emit a weak line.
    digests = _staged_digests(repo_dir)
    chosen: set[str] = set()
    out: list[Tripwire] = []
    for path, rel in pairs:
        tw = (
            _rung_line(path, rel, repo_dir, chosen)
            or _rung_sha256(originals[rel], rel, digests)
        )
        if tw is None:
            raise TripwireError(
                f'{rel}: no tripwire could be derived (no >={STRONG_TRIPWIRE_CHARS}-char '
                'distinctive line, and its content digest already matches a surviving '
                'file) -- the carved content is indistinguishable from the surviving '
                'tree, so a leak could not be detected. Refusing to stage. Narrow the '
                'carve set so this file survives, or record it as tripwire_exempt WITH '
                'a compensating path-and-digest assertion.'
            )
        out.append(tw)
        chosen.add(harbor_stage_context.norm(tw.pattern) if tw.kind == 'line' else tw.pattern)
    return tuple(out)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def stage_carved_tree(
    repo: Path,
    carved_relpaths,
    stubbed_texts: dict[str, str],
    deleted_relpaths,
    out_dir: Path,
) -> StagedTree:
    """Materialise the carved tree on the HOST. The intact tree is never shipped."""
    repo = Path(repo).resolve()
    out_dir = Path(out_dir).resolve()
    if not repo.is_dir():
        raise StagingError(f'repo is not a directory: {repo}')

    carved = tuple(sorted({Path(r).as_posix() for r in carved_relpaths}))
    overlay = {Path(k).as_posix(): v for k, v in dict(stubbed_texts).items()}
    deleted = tuple(sorted({Path(r).as_posix() for r in deleted_relpaths}))

    if not carved:
        raise StagingError('refusing to stage an empty carve set')
    both = sorted(set(overlay) & set(deleted))
    if both:
        raise StagingError(f'file(s) both deleted and overlaid: {both}')
    unaccounted = sorted(set(carved) - set(overlay) - set(deleted))
    if unaccounted:
        raise StagingError(
            f'carved file(s) with neither a stub nor a delete: {unaccounted}'
        )
    stray = sorted((set(overlay) | set(deleted)) - set(carved))
    if stray:
        raise StagingError(f'staging asked to touch non-carved file(s): {stray}')

    _assert_out_dir_is_safe(out_dir, repo)

    originals: dict[str, str] = {}
    for rel in carved:
        path = repo / rel
        if not path.is_file():
            raise StagingError(f'carved file is not present in {repo}: {rel}')
        try:
            originals[rel] = path.read_text(encoding='utf-8')
        except UnicodeDecodeError as exc:
            raise StagingError(f'{rel}: not utf-8 text, refusing to stage') from exc

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    _assert_dest_is_safe(out_dir)

    ctx_dir = out_dir / 'ctx'
    ctx_dir.mkdir()
    repo_dir = ctx_dir / 'repo'
    harbor_stage_context.copy_repo(repo, repo_dir)
    _prune_extra(repo_dir)
    if (ctx_dir / '.dockerignore').exists():
        raise StagingError('a .dockerignore at the named-context root would filter the carve')

    for rel in deleted:
        victim = repo_dir / rel
        if victim.is_file():
            victim.unlink()
        _prune_empty_parents(victim.parent, repo_dir)
    for rel in sorted(overlay):
        target = repo_dir / rel
        if not target.is_file():
            raise StagingError(f'overlay target missing from the staged tree: {rel}')
        target.write_text(overlay[rel], encoding='utf-8')

    for rel in deleted:
        if (repo_dir / rel).exists():
            raise StagingError(f'carve missed {rel}')
    for rel, text in overlay.items():
        if (repo_dir / rel).read_text(encoding='utf-8') != text:
            raise StagingError(f'overlay did not take effect for {rel}')

    # HOST-ONLY: the intact originals, so the oracle and the tripwire picker
    # have something to compare against. Deliberately outside ctx/.
    oracle_dir = out_dir / 'oracle'
    for rel, text in originals.items():
        dest = oracle_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding='utf-8')

    tripwires = _derive_tripwires(originals, repo_dir, oracle_dir)
    if len(tripwires) != len(carved):
        raise TripwireError(
            f'tripwire coverage {len(tripwires)} != {len(carved)} carved files'
        )

    # harbor's host-side pre-flight: the staged context must already be clean.
    try:
        harbor_stage_context.assert_carve_is_content_clean(
            repo_dir, [t.pattern for t in tripwires if t.kind == 'line']
        )
    except SystemExit as exc:
        raise StagingError(str(exc)) from exc

    trip_dir = out_dir / 'trip'
    trip_dir.mkdir()
    tripwire_path = trip_dir / 'tripwires.txt'
    patterns = [t.pattern for t in tripwires if t.kind != 'sha256']
    tripwire_path.write_text('\n'.join(patterns) + ('\n' if patterns else ''), encoding='utf-8')
    digest_path = trip_dir / 'tripwire-digests.txt'
    digests = [f'{t.pattern}  {t.relpath}' for t in tripwires if t.kind == 'sha256']
    digest_path.write_text('\n'.join(digests) + ('\n' if digests else ''), encoding='utf-8')

    staged_relpaths = _relpaths(repo_dir)
    receipt = {
        'repo': repo.name,
        'carved_relpaths': list(carved),
        'removed_files': list(deleted),
        'removed_count': len(deleted),
        'overlaid_relpaths': sorted(overlay),
        'original_sha256': {
            rel: hashlib.sha256(text.encode('utf-8')).hexdigest()
            for rel, text in sorted(originals.items())
        },
        'stubbed_sha256': {
            rel: hashlib.sha256(text.encode('utf-8')).hexdigest()
            for rel, text in sorted(overlay.items())
        },
        'staged_file_count': len(staged_relpaths),
        'tripwire_rungs': {t.relpath: t.rung for t in tripwires},
    }
    receipt_path = out_dir / 'carve_receipt.json'
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )

    # harbor stage_carved.py:125-127 -- the receipt must not have leaked into
    # anything that becomes a build context.
    strays = sorted(p.as_posix() for p in ctx_dir.rglob('carve_receipt.json'))
    if strays:
        raise StagingError(f'carve_receipt.json survived inside the shipped context: {strays}')

    return StagedTree(
        out_dir=out_dir,
        ctx_dir=ctx_dir,
        repo_dir=repo_dir,
        oracle_dir=oracle_dir,
        receipt=receipt,
        receipt_path=receipt_path,
        tripwire_path=tripwire_path,
        tripwire_digest_path=digest_path,
        tripwires=tripwires,
        staged_relpaths=staged_relpaths,
    )
