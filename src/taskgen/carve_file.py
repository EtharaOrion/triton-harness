"""File-level carve: skeleton-stub every function in a file, or delete the file.

Two modes, because the two failure surfaces are different:

  * SKELETON (python default) -- keep imports, module-level code, class bodies,
    decorators and every signature; replace only the *bodies*. The file still
    imports and pytest still collects it, so a multi-file carve does not cascade
    into a repo-wide collection error that the solver cannot fix. That cascade
    would be a fake RED: reward 0 for a reason unrelated to the task.
  * DELETE -- harbor-faithful whole-file removal (`--delete-whole-file`), which
    is what rust/c/cpp want and what `shared/tooling/carve.py` does.

Two invariants make the skeleton edit correct, and both are load-bearing:

  1. OUTERMOST ONLY. Every collected definition whose byte range is *contained*
     in another collected definition is dropped BEFORE any editing. A nested
     `def`, a closure, a decorator factory's inner function and a class defined
     inside a function body all live inside an enclosing function body; editing
     them as well as their parent either corrupts the file or reinstates text
     that was already stubbed. Methods of a *module-level* class stay outermost
     -- a class body is not a function body -- and are stubbed individually,
     which is the intent.
  2. DESCENDING BYTE ORDER. Surviving ranges are applied highest-offset-first so
     an earlier edit never shifts a later range.

Reuses `carve_fn.py`'s body-range choice: `body_node.start_byte` is already past
the newline and the block indent, so the replacement carries no indent of its
own and decorators/signature survive automatically because they sit before it.

Nothing here writes to the repository; the caller gets text back.
"""

from __future__ import annotations

import enum
import hashlib
import importlib
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

# G5 (S-GO spike): src/taskgen must NEVER go on sys.path -- `taskgen/select.py`
# would shadow the stdlib `select` module and break `tqdm`'s `import socket`.
# The vendored tree-sitter shim is therefore reached by package-relative import
# when we are imported as part of the package, and BY FILE PATH otherwise.
try:  # pragma: no cover - exercised by whichever import style is in play
    from . import _ts_compat
except ImportError:  # pragma: no cover
    _spec = importlib.util.spec_from_file_location(
        'taskgen_ts_compat', Path(__file__).with_name('_ts_compat.py')
    )
    _ts_compat = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_ts_compat)

__all__ = [
    'DELETED',
    'CarveFileError',
    'CarveFileMode',
    'CarveFileResult',
    'carve_files',
    'default_stub_body',
    'delete_whole_file',
    'has_carveable_functions',
    'skeleton_stub_file',
]


class CarveFileError(RuntimeError):
    """A carve that must not be shipped. Always raised, never downgraded."""


class CarveFileMode(enum.Enum):
    SKELETON = 'skeleton'
    DELETE = 'delete'

    @classmethod
    def parse(cls, value: 'str | CarveFileMode') -> 'CarveFileMode':
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError:
            raise CarveFileError(f'unknown carve mode {value!r}') from None


#: Sentinel returned for a file that is removed outright rather than rewritten.
DELETED = None

#: language -> (tree-sitter binding module, definition node types)
_LANGUAGES: dict[str, tuple[str, tuple[str, ...]]] = {
    'python': ('tree_sitter_python', ('function_definition',)),
    'go': ('tree_sitter_go', ('function_declaration', 'method_declaration')),
    'java': ('tree_sitter_java', ('method_declaration', 'constructor_declaration')),
}

#: A stub must be a body the surviving code can still *parse*, and one that can
#: never pass a test by accident. S-GO gotcha G2: a zero-value Go stub scored
#: reward 0.25 because one test happened to expect the zero value, so Go stubs
#: panic. Braces are part of the Go/Java body node, hence part of the stub.
_DEFAULT_STUB_BODY = {
    'python': 'raise NotImplementedError',
    'go': '{\n\tpanic("not implemented")\n}',
    'java': '{\n    throw new UnsupportedOperationException();\n}',
    # rust always carves --delete-whole-file (whole src/ removed for whole-suite
    # grading), so this stub is never actually spliced in; it exists only so
    # the language check in build_carve_set does not need a rust special case.
    'rust': '{\n    unimplemented!()\n}',
    # c is likewise --delete-whole-file (entire src/runtime removed), so this
    # never spliced. Kept so default_stub_body(language='c') resolves rather
    # than raising, mirroring the rust special-case-avoidance.
    'c': '{\n    /* not implemented */\n}',
    # cpp is likewise --delete-whole-file (entire Compiler/{Semantic,Ir,CodeGen}
    # removed), so this stub is never spliced. Kept so default_stub_body('cpp')
    # resolves rather than raising, matching rust/c.
    'cpp': '{\n    /* not implemented */\n}',
}


def default_stub_body(language: str = 'python') -> str:
    try:
        return _DEFAULT_STUB_BODY[language]
    except KeyError:
        raise CarveFileError(
            f'no default stub body for language {language!r}; '
            f'known: {", ".join(sorted(_DEFAULT_STUB_BODY))}'
        ) from None


def _parser_for(language: str):
    if language not in _LANGUAGES:
        raise CarveFileError(
            f'unsupported language {language!r}; known: {", ".join(sorted(_LANGUAGES))}'
        )
    _ts_compat.install()
    import tree_sitter

    modname, node_types = _LANGUAGES[language]
    try:
        binding = importlib.import_module(modname)
    except ImportError as exc:
        raise CarveFileError(
            f'language {language!r} needs {modname}; install taskgen/requirements-dev.txt'
        ) from exc
    return tree_sitter.Parser(tree_sitter.Language(binding.language())), node_types


def _definition_nodes(root, node_types: tuple[str, ...]) -> list:
    """Every definition node in the tree, pre-order, with a `body` field."""
    found = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in node_types and node.child_by_field_name('body') is not None:
            found.append(node)
        stack.extend(reversed(node.children))
    return found


def _outermost(nodes: list) -> list:
    """Drop every node contained in another node. Returns them start-ascending.

    Containment is tested on the DEFINITION range, not the body range: a nested
    definition sits inside its parent's body, which sits inside its parent's
    definition, so the two tests agree -- and the definition range is the one a
    reader can check against the source.
    """
    ordered = sorted(nodes, key=lambda n: (n.start_byte, -n.end_byte))
    kept: list = []
    frontier = -1
    for node in ordered:
        if node.start_byte >= frontier:
            kept.append(node)
            frontier = node.end_byte
    return kept


def skeleton_stub_file(
    source: str,
    relpath: str,
    *,
    stub_body: str,
    language: str = 'python',
) -> str:
    """Replace the body of every OUTERMOST function/method with `stub_body`."""
    parser, node_types = _parser_for(language)
    raw = source.encode('utf-8')
    tree = parser.parse(raw)

    outermost = _outermost(_definition_nodes(tree.root_node, node_types))
    ranges = []
    for node in outermost:
        body = node.child_by_field_name('body')
        start, end = body.start_byte, body.end_byte
        if not (node.start_byte <= start < end <= len(raw)):
            raise CarveFileError(
                f'{relpath}: implausible byte ranges for {node.type} '
                f'def={node.start_byte}..{node.end_byte} body={start}..{end} '
                f'(file is {len(raw)} bytes)'
            )
        ranges.append((start, end))

    if not ranges:
        raise CarveFileError(
            f'{relpath}: no {language} function bodies found -- refusing a no-op skeleton '
            '(use whole-file delete, or drop the file from the carve set)'
        )

    stub = stub_body.encode('utf-8')
    out = raw
    # DESCENDING, so an earlier replacement cannot shift a later range.
    for start, end in sorted(ranges, key=lambda r: r[0], reverse=True):
        out = out[:start] + stub + out[end:]

    stubbed = out.decode('utf-8')
    if stubbed == source:
        raise CarveFileError(f'{relpath}: skeleton carve was a no-op')
    return stubbed


def has_carveable_functions(source: str, language: str = 'python') -> bool:
    """Would a skeleton carve of this file remove anything?

    A folder glob routinely sweeps up `__init__.py`, a constants module or a
    generated stub table -- files with no function body at all. Skeletonising
    them is a no-op, and `skeleton_stub_file` rightly refuses a no-op. The
    caller needs to know that BEFORE it decides whether the carve set is empty,
    so this asks the question instead of catching the refusal.
    """
    parser, node_types = _parser_for(language)
    tree = parser.parse(source.encode('utf-8'))
    return bool(_outermost(_definition_nodes(tree.root_node, node_types)))


def delete_whole_file(source: str, relpath: str, **_kwargs) -> None:
    """Whole-file delete (harbor-faithful). Returns the `DELETED` sentinel."""
    del source, relpath
    return DELETED


@dataclass(frozen=True)
class CarveFileResult:
    """Everything staging and the oracle need, and nothing that touches disk."""

    mode: CarveFileMode
    language: str
    carved_relpaths: tuple[str, ...]
    deleted_relpaths: tuple[str, ...]
    overlay: dict[str, str] = field(default_factory=dict)
    originals: dict[str, str] = field(default_factory=dict)
    original_sha256: dict[str, str] = field(default_factory=dict)
    stubbed_sha256: dict[str, str] = field(default_factory=dict)

    def is_deleted(self, relpath: str) -> bool:
        return relpath in self.deleted_relpaths


def carve_files(
    repo: Path,
    carved_relpaths,
    *,
    mode: 'CarveFileMode | str' = CarveFileMode.SKELETON,
    stub_body: str | None = None,
    language: str = 'python',
) -> CarveFileResult:
    """Carve every file in `carved_relpaths`. Deterministic; writes nothing."""
    mode = CarveFileMode.parse(mode)
    repo = Path(repo)
    rels = tuple(sorted({Path(r).as_posix() for r in carved_relpaths}))
    if not rels:
        raise CarveFileError('refusing to carve an empty file set')
    if stub_body is None:
        stub_body = default_stub_body(language)

    originals: dict[str, str] = {}
    original_sha: dict[str, str] = {}
    overlay: dict[str, str] = {}
    stubbed_sha: dict[str, str] = {}
    deleted: list[str] = []

    for rel in rels:
        path = repo / rel
        if not path.is_file():
            raise CarveFileError(f'carve target is not a file in {repo}: {rel}')
        raw = path.read_bytes()
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise CarveFileError(f'{rel}: not utf-8 text, refusing to carve') from exc
        originals[rel] = text
        original_sha[rel] = hashlib.sha256(raw).hexdigest()

        if mode is CarveFileMode.DELETE:
            if delete_whole_file(text, rel) is not DELETED:  # pragma: no cover
                raise CarveFileError(f'{rel}: delete helper did not return the sentinel')
            deleted.append(rel)
            continue

        stubbed = skeleton_stub_file(text, rel, stub_body=stub_body, language=language)
        overlay[rel] = stubbed
        stubbed_sha[rel] = hashlib.sha256(stubbed.encode('utf-8')).hexdigest()

    return CarveFileResult(
        mode=mode,
        language=language,
        carved_relpaths=rels,
        deleted_relpaths=tuple(deleted),
        overlay=overlay,
        originals=originals,
        original_sha256=original_sha,
        stubbed_sha256=stubbed_sha,
    )
