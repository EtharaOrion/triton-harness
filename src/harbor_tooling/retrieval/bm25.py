"""Okapi BM25 over the chunked surviving repository. Pure stdlib.

Scoring is the standard Robertson/Sparck-Jones form:

    score(D,Q) = sum_{t in Q} IDF(t) * f(t,D)*(k1+1)
                              / (f(t,D) + k1*(1 - b + b*|D|/avgdl))

    IDF(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))

The +1 inside the log is the standard non-negative variant; without it, terms
appearing in more than half the corpus get negative weight, which on a
single-project corpus (where `import`, `self`, `def` are near-universal) makes
scores swing wildly on boilerplate.

Determinism: ranking ties break on chunk_id, so identical inputs always yield
an identical ordering regardless of dict iteration.
"""

from __future__ import annotations

import math
from collections import Counter as _Counter

from . import tokenize

K1 = 1.5
B = 0.75


class BM25Index:
    def __init__(self, chunks: list, k1: float = K1, b: float = B) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._tf: list[dict[str, int]] = []
        self._len: list[int] = []
        df: dict[str, int] = {}
        for ch in chunks:
            toks = tokenize(ch.text)
            tf = dict(_Counter(toks))
            self._tf.append(tf)
            self._len.append(len(toks))
            for term in tf:
                df[term] = df.get(term, 0) + 1
        self.n_docs = len(chunks)
        self.avgdl = (sum(self._len) / self.n_docs) if self.n_docs else 0.0
        self.idf = {
            t: math.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }

    def score(self, query: str) -> list[tuple[float, int]]:
        """Return [(score, chunk_index)] sorted by score desc, chunk_id asc."""
        q_terms = dict(_Counter(tokenize(query)))
        out: list[tuple[float, int]] = []
        for i in range(self.n_docs):
            tf = self._tf[i]
            dl = self._len[i]
            denom_len = self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0.0))
            s = 0.0
            for term, qf in q_terms.items():
                f = tf.get(term)
                if not f:
                    continue
                s += self.idf.get(term, 0.0) * (f * (self.k1 + 1.0)) / (f + denom_len)
            if s > 0.0:
                out.append((s, i))
        out.sort(key=lambda p: (-p[0], self.chunks[p[1]].chunk_id))
        return out


def rank(chunks: list, query: str) -> list[tuple[float, int]]:
    return BM25Index(chunks).score(query)
