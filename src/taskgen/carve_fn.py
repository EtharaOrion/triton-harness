"""Function-level carve: stub ONE function body, leave the rest of the file alone.

The harbor tooling's `carve.py` deletes whole FILES. MRG-Bench tasks are
function-granular, so this module replaces the target's body byte range with
`raise NotImplementedError` instead.

Why the BODY range and not the whole definition:

  * decorators and the `def` line survive automatically -- they sit before
    `body_node.start_byte` -- so the module still imports and only the target's
    own tests go RED. Deleting the definition would break every importer and
    turn a function-level task into a whole-suite collection error.
  * `body_node.start_byte` is already past the newline and the block indent, so
    the replacement text carries NO leading indent of its own.

The docstring is INSIDE the body and is therefore removed by the carve, but it
is also the task prompt (MRG-Bench's "annotation" field), so it is captured
separately and re-shown in instruction.md. `docstring_lines` records where it
was, so the leakage checker can tell "deliberately quoted prompt" apart from
"leaked answer".

Nothing here writes to the repository: the caller gets text back and decides
where it lands.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from . import _tooling_path  # noqa: F401  (side effect: sys.path + tree-sitter shim)

STUB_BODY = 'raise NotImplementedError'


@dataclass
class CarveResult:
    relpath: str
    original_text: str
    stubbed_text: str
    signature: str
    docstring: str
    docstring_lines: set[int] = field(default_factory=set)
    body_lines: set[int] = field(default_factory=set)
    receipt: dict = field(default_factory=dict)

    @property
    def carved_body(self) -> str:
        return self.receipt['carved_body']


def _line_span(text: str, start_byte: int, end_byte: int) -> tuple[int, int]:
    """1-based inclusive line range covered by a byte range of `text` (utf-8)."""
    raw = text.encode('utf-8')
    first = raw[:start_byte].count(b'\n') + 1
    last = raw[:end_byte].count(b'\n') + 1
    return first, last


def _decl_start_byte(func_node) -> int:
    """Start of the declaration INCLUDING decorators.

    The harness query captures `function_definition`, whose parent is a
    `decorated_definition` when decorators are present.
    """
    parent = func_node.parent
    if parent is not None and parent.type == 'decorated_definition':
        return parent.start_byte
    return func_node.start_byte


def _docstring_span(fd) -> tuple[int, int] | None:
    node = fd.comment_node
    if node is None:
        return None
    if isinstance(node, list):
        if not node:
            return None
        return node[0].start_byte, node[-1].end_byte
    return node.start_byte, node.end_byte


def carve_function(repo: Path, target, *, language: str = 'python',
                   stub_body: str | None = None) -> CarveResult:
    """Stub `target`'s body. Returns both texts plus a receipt; writes nothing.

    `stub_body` defaults to the language's own. Go must NOT get a zero-value or
    a `NotImplementedError` stub: spike gotcha G2 measured reward 0.25 from a
    zero-value stub, because one linked test happened to expect the zero value
    and passed by accident, which breaks the RED bar.
    """
    from .carve_file import default_stub_body

    if stub_body is None:
        stub_body = default_stub_body(language)
    repo = Path(repo)
    path = repo / target.relpath
    raw = path.read_bytes()
    original = raw.decode('utf-8')

    fd = target.fd
    body_start, body_end = fd.body_node.start_byte, fd.body_node.end_byte
    func_start, func_end = fd.func_node.start_byte, fd.func_node.end_byte
    decl_start = _decl_start_byte(fd.func_node)

    if not (decl_start <= func_start < body_start < body_end <= len(raw)):
        raise ValueError(
            f'{target.relpath}:{target.qualname}: implausible byte ranges '
            f'decl={decl_start} func={func_start}..{func_end} body={body_start}..{body_end} '
            f'(file is {len(raw)} bytes)'
        )

    carved_body = raw[body_start:body_end].decode('utf-8')
    stubbed = (raw[:body_start] + stub_body.encode('utf-8') + raw[body_end:]).decode('utf-8')
    if stubbed == original:
        raise ValueError(f'{target.relpath}:{target.qualname}: carve was a no-op')

    signature = raw[decl_start:body_start].decode('utf-8').rstrip()

    doc_span = _docstring_span(fd)
    docstring = raw[doc_span[0]:doc_span[1]].decode('utf-8') if doc_span else ''
    doc_lines: set[int] = set()
    if doc_span:
        lo, hi = _line_span(original, *doc_span)
        doc_lines = set(range(lo, hi + 1))

    body_lo, body_hi = _line_span(original, body_start, body_end)

    receipt = {
        'relpath': target.relpath,
        'class_name': getattr(target, 'class_name', '') or getattr(target, 'receiver', ''),
        'func_name': target.name,
        'language': language,
        'decl_start_byte': decl_start,
        'func_start_byte': func_start,
        'func_end_byte': func_end,
        'body_start_byte': body_start,
        'body_end_byte': body_end,
        'carved_bytes': body_end - body_start,
        'carved_body': carved_body,
        'stub_body': stub_body,
        'original_sha256': hashlib.sha256(raw).hexdigest(),
        'stubbed_sha256': hashlib.sha256(stubbed.encode('utf-8')).hexdigest(),
        'original_bytes': len(raw),
        'stubbed_bytes': len(stubbed.encode('utf-8')),
    }

    return CarveResult(
        relpath=target.relpath,
        original_text=original,
        stubbed_text=stubbed,
        signature=signature,
        docstring=docstring,
        docstring_lines=doc_lines,
        body_lines=set(range(body_lo, body_hi + 1)),
        receipt=receipt,
    )
