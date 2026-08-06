"""Deterministic harbor-format task generation at FUNCTION granularity.

Given a repository checkout, `taskgen` carves one function body out of one file
and emits a complete harbor task entry per context-provisioning condition --
eleven of them: the nine `triton/harness` enums (`eval_llm.py`'s context_type
plus `eval_rag.py`'s rag_type) and the caller_* pair taskgen inverts out of the
call graph.

No LLM and no network are used at generation time. Every ordering is a total
order over content, every id is a uuid5, and the only tokenizer is chars4, so
two runs on the same input produce byte-identical trees.

    python -m taskgen.cli generate --repo <repo> --out <dir>

See taskgen/README.md for the runtime requirements (tree-sitter pins).
"""

from __future__ import annotations

from .ids import CONTEXT_TYPES, entry_id

__all__ = ['CONTEXT_TYPES', 'entry_id']
