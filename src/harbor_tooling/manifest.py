"""Carve-manifest parsing and read-only repository walking.

Shared by gen_context.py (never writes to the repo) and carve.py (writes only
to an explicitly-passed destination copy).

Glob semantics are implemented here rather than borrowed from pathlib because
PurePath.match does not give `**` recursive semantics on Python 3.12, and the
manifests depend on `**` meaning "any number of directories".
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

BINARY_SNIFF_BYTES = 4096
DEFAULT_MAX_FILE_BYTES = 2_000_000

ALWAYS_EXCLUDE = (
    '.git/**',
    '**/__pycache__/**',
    '**/node_modules/**',
    '**/.venv/**',
    '**/target/debug/**',
    '**/target/release/**',
)


def _glob_to_regex(pattern: str) -> re.Pattern:
    i = 0
    out = ['^']
    n = len(pattern)
    while i < n:
        c = pattern[i]
        if c == '*':
            if pattern.startswith('**/', i):
                out.append('(?:.*/)?')
                i += 3
                continue
            if pattern.startswith('**', i):
                out.append('.*')
                i += 2
                continue
            out.append('[^/]*')
            i += 1
        elif c == '?':
            out.append('[^/]')
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    out.append('$')
    return re.compile(''.join(out))


class GlobSet:
    def __init__(self, patterns) -> None:
        self.patterns = list(patterns)
        self._res = [_glob_to_regex(p) for p in self.patterns]

    def matches(self, relpath: str) -> bool:
        return any(r.match(relpath) for r in self._res)

    def __bool__(self) -> bool:
        return bool(self._res)


@dataclass
class Manifest:
    path: Path
    repo: str
    language: str
    carve_root: str
    test_command: str
    description: str
    include: list[str]
    carve_exclude: list[str] = field(default_factory=list)
    context_exclude: list[str] = field(default_factory=list)
    parent_package: str | None = None
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES

    @classmethod
    def load(cls, path: Path) -> 'Manifest':
        with open(path, 'rb') as fh:
            data = tomllib.load(fh)
        missing = [
            k for k in ('repo', 'language', 'carve_root', 'test_command') if k not in data
        ]
        if missing:
            raise SystemExit(f'{path}: manifest missing required key(s): {", ".join(missing)}')
        carve = data.get('carve', {})
        ctx = data.get('context', {})
        include = list(carve.get('include', []))
        if not include:
            raise SystemExit(f'{path}: [carve].include must list at least one glob')
        return cls(
            path=path,
            repo=str(data['repo']),
            language=str(data['language']),
            carve_root=str(data['carve_root']).strip('/'),
            test_command=str(data['test_command']),
            description=str(data.get('description', '')).strip(),
            include=include,
            carve_exclude=list(carve.get('exclude', [])),
            context_exclude=list(ctx.get('exclude', [])),
            parent_package=ctx.get('parent_package'),
            max_file_bytes=int(ctx.get('max_file_bytes', DEFAULT_MAX_FILE_BYTES)),
        )

    @property
    def folder_root(self) -> str:
        """Root for the longctx-folder condition: carve_root's parent package."""
        if self.parent_package is not None:
            return str(self.parent_package).strip('/')
        parent = str(Path(self.carve_root).parent)
        return '' if parent == '.' else parent


def resolve_carved(root: Path, manifest: Manifest) -> list[str]:
    """POSIX relpaths matched by the carve globs. Sorted, deduplicated."""
    include = GlobSet(manifest.include)
    exclude = GlobSet(manifest.carve_exclude)
    found: set[str] = set()
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if include.matches(rel) and not exclude.matches(rel):
            found.add(rel)
    return sorted(found)


def _looks_binary(raw: bytes) -> bool:
    return b'\x00' in raw[:BINARY_SNIFF_BYTES]


def walk_repo(root: Path, manifest: Manifest) -> list[tuple[str, str]]:
    """Read every eligible text file. Returns sorted [(relpath, text)].

    Read-only: this never opens a file for writing and never follows symlinks
    out of the tree.
    """
    exclude = GlobSet(list(ALWAYS_EXCLUDE) + list(manifest.context_exclude))
    out: list[tuple[str, str]] = []
    for path in sorted(root.rglob('*'), key=lambda p: p.as_posix()):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        if exclude.matches(rel):
            continue
        try:
            if path.stat().st_size > manifest.max_file_bytes:
                continue
            raw = path.read_bytes()
        except OSError:
            continue
        if _looks_binary(raw):
            continue
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            continue
        out.append((rel, text))
    out.sort(key=lambda p: p[0])
    return out
