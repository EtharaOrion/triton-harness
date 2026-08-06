"""Deterministic entry identifiers.

An entry id is a UUIDv5, so it is a pure function of what the entry IS -- repo,
file, class, function, context type -- with no clock, no counter and no host
state. Two generation runs on the same input therefore produce the same
directory names, which is what makes `diff -r` a meaningful determinism proof.
"""

from __future__ import annotations

import uuid

#: Fixed namespace for every id this package mints.
NS = uuid.uuid5(uuid.NAMESPACE_URL, 'mrgctx.taskgen')

#: The eleven harness context-provisioning conditions, in canonical order.
#: Five are `eval_llm.py`'s context_type enum and four `eval_rag.py`'s rag_type
#: enum; `caller_func`/`caller_sig` are taskgen's own, and sit next to the
#: callee pair they invert (gap B1) rather than at the end, because the reading
#: order of the tuple IS the documented order of the conditions.
CONTEXT_TYPES = (
    'no_context',
    'callee_func',
    'callee_sig',
    'caller_func',
    'caller_sig',
    'in_file',
    'project',
    'bm25',
    'embedding',
    'mix',
    'repo_coder',
)


#: The key shape that existed before `--lang`/`--carve-scope`. Both are appended
#: to the key ONLY when they differ from these, so the single-function python
#: entries keep the ids they have always had. Widening the key unconditionally
#: would renumber every shipped entry for no semantic change.
DEFAULT_LANG = 'python'
DEFAULT_SCOPE = 'function'


def entry_id(repo: str, relpath: str, cls: str, func: str, context_type: str,
             *, scope: str = DEFAULT_SCOPE, lang: str = DEFAULT_LANG) -> str:
    """uuid5 over `repo|relpath|cls|func|context_type[|scope|lang]`.

    `cls` is '' for module-level functions; it stays in the key so a method and
    a module-level function of the same name in the same file never collide.
    `scope` and `lang` join the key only when non-default, so two scopes over
    one file get different ids while the default path keeps its own.
    """
    if context_type not in CONTEXT_TYPES:
        raise ValueError(
            f'unknown context_type {context_type!r}; expected one of {CONTEXT_TYPES}'
        )
    key = [repo, relpath, cls, func, context_type]
    if scope != DEFAULT_SCOPE or lang != DEFAULT_LANG:
        key += [scope, lang]
    return str(uuid.uuid5(NS, '|'.join(key)))


def slug(repo: str, func: str, context_type: str) -> str:
    """Human-readable entry name: `<repo>__<func>__<context_type>`."""
    return f'{repo}__{func}__{context_type}'
