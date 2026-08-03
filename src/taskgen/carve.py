"""One carve representation for all three scopes.

`carve_fn.py` stubs one function body; `carve_file.py` skeletonises or deletes
whole files. They produce different shapes, and every consumer downstream --
staging, the nine context builders, the oracle, instruction.md -- needs the same
questions answered regardless of which one ran:

    what did we remove          `carved_relpaths` / `deleted_relpaths`
    what does the solver see    `overlay` (relpath -> stubbed text)
    what does the oracle put back `originals` (relpath -> intact text)
    which file is the subject   `primary_relpath`

`CarveSet` is that shape, and `build_carve_set` is the only thing that knows
which underlying carver to call. Function scope keeps its `CarveResult` intact
on `function_carve`, because instruction.md quotes the target's signature,
docstring and byte range and none of that exists for a file-level carve.

A NOTE ON WHAT COUNTS AS CARVED. Function scope removes ONE body from a file
that otherwise survives whole, so every other function in that file is still
readable. File and folder scope remove EVERY body in the carve set. That
difference is why `carved_bodies` exists as its own question: the context
builders must not inline a callee whose body is gone, and for function scope
that is one function, not one file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from .carve_file import (
    CarveFileError,
    CarveFileMode,
    carve_files,
    default_stub_body,
    has_carveable_functions,
)
from .carve_fn import CarveResult, carve_function
from .scope import CarveScope

__all__ = [
    'CarveError',
    'CarveSet',
    'TargetView',
    'build_carve_set',
]


class CarveError(RuntimeError):
    """A carve that must not be shipped. Always raised, never downgraded."""


@dataclass(frozen=True)
class TargetView:
    """What the prompt and the context builders need to call "the target".

    Function scope passes the real `select.Target` / `go_select.GoTarget`
    through untouched. File and folder scope have no single target function, so
    this stands in for one: the subject is a set of files, and the callees are
    the union of what those files' functions reach outside the carve set.
    """

    qualname: str
    name: str
    class_name: str
    relpath: str
    callees: tuple = ()
    kind: str = 'files'

    def callee_qualnames(self) -> list[str]:
        return [
            f'{c.class_name}.{c.name}' if getattr(c, 'class_name', '') else c.name
            for c in self.callees
        ]


@dataclass(frozen=True)
class CarveSet:
    scope: CarveScope
    language: str
    carved_relpaths: tuple[str, ...]
    deleted_relpaths: tuple[str, ...]
    overlay: dict[str, str]
    originals: dict[str, str]
    primary_relpath: str
    skipped_relpaths: tuple[str, ...] = ()
    function_carve: CarveResult | None = None
    target_view: TargetView | None = None
    carved_functions: tuple = field(default=(), repr=False)

    # --- the shape contexts.py and emit.py read -------------------------

    @property
    def relpath(self) -> str:
        return self.primary_relpath

    @property
    def original_text(self) -> str:
        return self.originals[self.primary_relpath]

    @property
    def stubbed_text(self) -> str:
        return self.overlay[self.primary_relpath]

    @property
    def is_function_scope(self) -> bool:
        return self.scope is CarveScope.FUNCTION

    def is_deleted(self, relpath: str) -> bool:
        return relpath in set(self.deleted_relpaths)

    def surviving_text(self, relpath: str, on_disk: str) -> str | None:
        """What the solver sees at `relpath`. None means the file is gone."""
        if relpath in self.overlay:
            return self.overlay[relpath]
        if self.is_deleted(relpath):
            return None
        return on_disk

    def carved_body_owner(self, relpath: str, func_name: str) -> bool:
        """Is THIS function's body among the ones the carve removed?

        Function scope removed exactly one; file/folder scope removed every
        function body in every carved file.
        """
        if self.is_function_scope:
            fc = self.function_carve
            return (
                fc is not None
                and relpath == self.primary_relpath
                and func_name == fc.receipt['func_name']
            )
        return relpath in set(self.carved_relpaths)

    # --- prompt material (function scope only) ---------------------------

    @property
    def signature(self) -> str:
        return self.function_carve.signature if self.function_carve else ''

    @property
    def docstring(self) -> str:
        return self.function_carve.docstring if self.function_carve else ''

    @property
    def docstring_lines(self) -> set[int]:
        return self.function_carve.docstring_lines if self.function_carve else set()

    @property
    def body_lines(self) -> set[int]:
        return self.function_carve.body_lines if self.function_carve else set()

    @property
    def receipt(self) -> dict:
        return self.function_carve.receipt if self.function_carve else {}

    @property
    def original_sha256(self) -> dict[str, str]:
        return {
            rel: hashlib.sha256(text.encode('utf-8')).hexdigest()
            for rel, text in sorted(self.originals.items())
        }

    @property
    def stubbed_sha256(self) -> dict[str, str]:
        return {
            rel: hashlib.sha256(text.encode('utf-8')).hexdigest()
            for rel, text in sorted(self.overlay.items())
        }

    @classmethod
    def from_function_carve(cls, carve: CarveResult, *, language: str = 'python',
                            target=None) -> 'CarveSet':
        """Adapt a bare `CarveResult`, so callers that predate scopes still work."""
        return cls(
            scope=CarveScope.FUNCTION,
            language=language,
            carved_relpaths=(carve.relpath,),
            deleted_relpaths=(),
            overlay={carve.relpath: carve.stubbed_text},
            originals={carve.relpath: carve.original_text},
            primary_relpath=carve.relpath,
            function_carve=carve,
            target_view=target,
        )


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------


def _primary(carved: tuple[str, ...], target) -> str:
    """The file instruction.md is written about.

    The target's file when there is one; otherwise the first carved path in
    sort order, which is stable and therefore reproducible.
    """
    if target is not None and getattr(target, 'relpath', None):
        return Path(target.relpath).as_posix()
    return carved[0]


def _callees_outside(carved_functions, carved_relpaths, repo) -> tuple:
    """Callees of the carved functions whose bodies still exist on disk."""
    carved = set(carved_relpaths)
    repo = Path(repo).resolve()
    seen: dict[tuple[str, str], object] = {}
    for fn in carved_functions:
        fd = getattr(fn, 'fd', None)
        if fd is None:
            continue
        for callee in getattr(fd, 'callee', ()) or ():
            try:
                rel = Path(callee.file_path).resolve().relative_to(repo).as_posix()
            except ValueError:
                rel = Path(callee.file_path).as_posix()
            if rel in carved:
                continue
            seen[(rel, callee.name)] = callee
    return tuple(v for _k, v in sorted(seen.items()))


def _function_target_view(target, repo):
    """Give a function-scope target the callee surface the contexts read.

    `select.Target` already has one. `go_select.GoTarget` deliberately does not:
    it models a Go method's identity (receiver + package + `-run` selector) and
    has no business also modelling MRG-Bench's callee-context notion. Rather
    than widen it, the callee view is derived here from the parser's own graph.
    """
    if hasattr(target, 'callees') and hasattr(target, 'callee_qualnames'):
        return target
    fd = getattr(target, 'fd', None)
    callees = _callees_outside([_FdHolder(fd)], (), repo) if fd is not None else ()
    return TargetView(
        qualname=target.qualname,
        name=target.name,
        class_name=getattr(target, 'receiver', '') or '',
        relpath=target.relpath,
        callees=callees,
        kind='function',
    )


@dataclass(frozen=True)
class _FdHolder:
    fd: object


def _files_target_view(carved, carved_functions, repo, scope) -> TargetView:
    root = _carve_root(carved)
    return TargetView(
        qualname=root,
        name=root,
        class_name='',
        relpath=carved[0],
        callees=_callees_outside(carved_functions, carved, repo),
        kind=scope.value,
    )


def _carve_root(carved: tuple[str, ...]) -> str:
    parents = [Path(rel).parent for rel in carved]
    common = parents[0].parts
    for parent in parents[1:]:
        parts = parent.parts
        keep = 0
        while keep < min(len(common), len(parts)) and common[keep] == parts[keep]:
            keep += 1
        common = common[:keep]
    return '/'.join(common) or '.'


def build_carve_set(
    repo,
    scope,
    *,
    carved_relpaths=(),
    language: str = 'python',
    target=None,
    delete_whole_file: bool = False,
    stub_body: str | None = None,
    carved_functions=(),
) -> CarveSet:
    """Carve, whichever scope this is. Deterministic; writes nothing to disk."""
    scope = CarveScope.parse(scope)
    repo = Path(repo)

    if scope is CarveScope.FUNCTION:
        if target is None:
            raise CarveError('function scope needs a resolved target')
        carve = carve_function(repo, target, language=language, stub_body=stub_body)
        return CarveSet(
            scope=scope,
            language=language,
            carved_relpaths=(carve.relpath,),
            deleted_relpaths=(),
            overlay={carve.relpath: carve.stubbed_text},
            originals={carve.relpath: carve.original_text},
            primary_relpath=carve.relpath,
            function_carve=carve,
            target_view=_function_target_view(target, repo),
            carved_functions=tuple(carved_functions),
        )

    rels = tuple(sorted({Path(r).as_posix() for r in carved_relpaths}))
    if not rels:
        raise CarveError(f'{scope.value} scope was given an empty carve set')

    mode = CarveFileMode.DELETE if delete_whole_file else CarveFileMode.SKELETON
    stub = stub_body if stub_body is not None else default_stub_body(language)

    keep, skipped = rels, ()
    if mode is CarveFileMode.SKELETON:
        keep, skipped = _partition_carveable(repo, rels, language)
        if not keep:
            raise CarveError(
                f'none of the {len(rels)} glob-matched file(s) holds a {language} '
                'function body, so a skeleton carve would remove nothing. Narrow '
                '--include, or pass --delete-whole-file if removing the files '
                'outright is the intent.'
            )

    result = carve_files(repo, keep, mode=mode, stub_body=stub, language=language)
    view = (
        target if target is not None
        else _files_target_view(result.carved_relpaths, carved_functions, repo, scope)
    )
    return CarveSet(
        scope=scope,
        language=language,
        carved_relpaths=result.carved_relpaths,
        deleted_relpaths=result.deleted_relpaths,
        overlay=dict(result.overlay),
        originals=dict(result.originals),
        primary_relpath=_primary(result.carved_relpaths, target),
        skipped_relpaths=skipped,
        target_view=view,
        carved_functions=tuple(carved_functions),
    )


def _partition_carveable(repo, rels, language) -> tuple[tuple[str, ...], tuple[str, ...]]:
    keep, skipped = [], []
    for rel in rels:
        path = Path(repo) / rel
        if not path.is_file():
            raise CarveError(f'carve target is not a file in {repo}: {rel}')
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            skipped.append(rel)
            continue
        try:
            carveable = has_carveable_functions(text, language)
        except CarveFileError:
            carveable = False
        (keep if carveable else skipped).append(rel)
    return tuple(keep), tuple(skipped)
