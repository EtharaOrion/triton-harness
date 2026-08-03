"""Compat shim making tree_sitter >= 0.23 behave like 0.22.x for the harness parser.

VENDORED ON PURPOSE. `triton/harness/src/parser/*.py` was written against
tree_sitter 0.22.x, whose `Query.matches()` returned

    {capture_name: Node}        for un-quantified captures
    {capture_name: [Node, ...]} for quantified (`*`, `+`) captures

0.23+ always returns lists, so `match[1]["func_name"].text` blows up with
`AttributeError: 'list' object has no attribute 'text'`. Rather than fork the
harness parser (which is an upstream vendored tree), we re-wrap `Language.query`
and unwrap the single-element lists again.

Quantified captures are detected by scanning the query SOURCE for `*@name` /
`+@name` / `?@name`, because that is the only place the quantifier survives --
the compiled Query object does not expose it.

Pinned working set on arm64 macOS: tree_sitter==0.24.0, tree_sitter_python==0.23.6.
`install()` is idempotent and MUST be called before `parser.py_parser` is imported;
`taskgen._tooling_path` does that for you.
"""

from __future__ import annotations

import re

_QUANT = re.compile(r'[*+?]\s*@([A-Za-z_][A-Za-z0-9_]*)')


class _MatchesShim:
    """Wraps a compiled Query, restoring 0.22.x `matches()` semantics."""

    def __init__(self, inner, quantified: set[str]) -> None:
        self._inner = inner
        self._quantified = quantified

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def matches(self, node):
        out = []
        for pat_idx, caps in self._inner.matches(node):
            fixed = {}
            for name, nodes in caps.items():
                if name in self._quantified:
                    fixed[name] = list(nodes)
                else:
                    fixed[name] = nodes[0] if len(nodes) == 1 else list(nodes)
            out.append((pat_idx, fixed))
        return out

    def captures(self, node):
        return self._inner.captures(node)


def install() -> None:
    """Idempotently give `tree_sitter.Language` a 0.22-compatible `.query()`."""
    import tree_sitter

    language = tree_sitter.Language
    if getattr(language, '_taskgen_shim', False):
        return
    query_cls = getattr(tree_sitter, 'Query', None)
    if query_cls is None:
        raise RuntimeError(
            'tree_sitter exposes no Query class; taskgen needs tree_sitter>=0.23 '
            '(pinned: 0.24.0). Install taskgen/requirements-dev.txt.'
        )

    def query(self, source):
        return _MatchesShim(query_cls(self, source), set(_QUANT.findall(source)))

    language.query = query
    language._taskgen_shim = True
