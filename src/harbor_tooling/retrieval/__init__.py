"""Retrieval helpers shared by the BM25 and dense-LSA conditions.

The tokenizer lives here so both retrievers index the corpus identically --
otherwise the two conditions would differ in preprocessing as well as in
scoring, and the comparison between them would be confounded.
"""

from __future__ import annotations

import re

# Split on non-alphanumerics, then split camelCase / PascalCase runs. Source
# identifiers carry most of the retrieval signal, so `TaskUpdater` must match a
# query for `task updater`.
_WORD = re.compile(r'[A-Za-z_][A-Za-z0-9_]*|\d+')
_CAMEL = re.compile(r'[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+')

MIN_TOKEN_LEN = 2
MAX_TOKEN_LEN = 40


def tokenize(text: str) -> list[str]:
    """Deterministic identifier-aware tokenization. Lowercased, order-preserving.

    `snake_case_name` -> snake, case, name, snake_case_name
    `TaskUpdater`     -> task, updater, taskupdater
    """
    out: list[str] = []
    for raw in _WORD.findall(text):
        parts = [p.lower() for p in _CAMEL.findall(raw) if p]
        pieces = [p for p in raw.lower().split('_') if p]
        # Ordered dedupe: both splitters can emit the same sub-token, and
        # double-counting it would skew term frequency.
        for tok in dict.fromkeys(parts + pieces + [raw.lower()]):
            if MIN_TOKEN_LEN <= len(tok) <= MAX_TOKEN_LEN:
                out.append(tok)
    return out
