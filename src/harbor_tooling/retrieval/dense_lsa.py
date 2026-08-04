"""TF-IDF + truncated SVD (LSA) cosine retrieval. numpy only.

THIS IS NOT A NEURAL DENSE RETRIEVER. There is no HF token and no GPU in the
generation environment, so no pretrained embedding model is available. LSA is
the strongest *offline, dependency-free* stand-in: it is a real dense vector
space with latent semantic structure, but its semantics come only from
co-occurrence within this one repository, not from pretraining. It should be
expected to underperform a genuine embedding model. The condition is named
`dense-lsa` everywhere and never claims to be bge/e5/gte or any other model.

sklearn is unavailable, so TruncatedSVD is implemented directly as a
randomized SVD (Halko/Martinsson/Tropp): project onto a small random subspace,
orthonormalize, run power iterations to sharpen the spectrum, then take an
exact SVD of the resulting small matrix. Seeded, so it is reproducible.
"""

from __future__ import annotations

from collections import Counter as _Counter

from . import tokenize

N_COMPONENTS = 192
MIN_DF = 2
MAX_DF_RATIO = 0.60
MAX_FEATURES = 12000
N_POWER_ITER = 4
OVERSAMPLES = 10


def _require_numpy():
    try:
        import numpy as np
        return np
    except ImportError as exc:
        raise SystemExit(
            'condition dense-lsa-rag requires numpy. Use the vendored '
            'interpreter, e.g.\n  mrgbench-vendor/audit/tooling/venv/bin/python '
            'gen_context.py ...'
        ) from exc


def _randomized_svd(np, a, k: int, seed: int):
    rng = np.random.default_rng(seed)
    n_rows, n_cols = a.shape
    k = max(1, min(k, min(n_rows, n_cols) - 1)) if min(n_rows, n_cols) > 1 else 1
    size = min(k + OVERSAMPLES, min(n_rows, n_cols))
    omega = rng.standard_normal((n_cols, size)).astype(np.float32)
    q, _ = np.linalg.qr(a @ omega)
    for _ in range(N_POWER_ITER):
        q, _ = np.linalg.qr(a.T @ q)
        q, _ = np.linalg.qr(a @ q)
    b = q.T @ a
    _, s, vt = np.linalg.svd(b, full_matrices=False)
    return s[:k], vt[:k]


class LSAIndex:
    def __init__(self, chunks: list, seed: int = 0, n_components: int = N_COMPONENTS) -> None:
        np = _require_numpy()
        self._np = np
        self.chunks = chunks
        self.n_docs = len(chunks)

        tfs = [dict(_Counter(tokenize(ch.text))) for ch in chunks]
        df: dict[str, int] = {}
        for tf in tfs:
            for term in tf:
                df[term] = df.get(term, 0) + 1

        max_df = max(MIN_DF, int(self.n_docs * MAX_DF_RATIO))
        kept = [t for t, n in df.items() if MIN_DF <= n <= max_df]
        # Rank by document frequency, ties by term string, so the vocabulary
        # cap is a deterministic function of the corpus alone.
        kept.sort(key=lambda t: (-df[t], t))
        kept = sorted(kept[:MAX_FEATURES])
        self.vocab = {t: i for i, t in enumerate(kept)}
        self.df = df

        n_terms = len(self.vocab)
        if self.n_docs == 0 or n_terms == 0:
            self.doc_vecs = np.zeros((self.n_docs, 1), dtype=np.float32)
            self.components = np.zeros((1, max(n_terms, 1)), dtype=np.float32)
            return

        import math

        self.idf = np.zeros(n_terms, dtype=np.float32)
        for term, col in self.vocab.items():
            self.idf[col] = math.log((1.0 + self.n_docs) / (1.0 + df[term])) + 1.0

        matrix = np.zeros((self.n_docs, n_terms), dtype=np.float32)
        for row, tf in enumerate(tfs):
            for term, count in tf.items():
                col = self.vocab.get(term)
                if col is not None:
                    matrix[row, col] = (1.0 + math.log(count)) * self.idf[col]
        self._l2_normalize(matrix)

        _, vt = _randomized_svd(np, matrix, n_components, seed)
        self.components = vt
        self.doc_vecs = matrix @ vt.T
        self._l2_normalize(self.doc_vecs)

    def _l2_normalize(self, m) -> None:
        np = self._np
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        np.maximum(norms, 1e-8, out=norms)
        m /= norms

    def _embed_query(self, query: str):
        np = self._np
        import math

        vec = np.zeros((1, len(self.vocab)), dtype=np.float32)
        for term, count in _Counter(tokenize(query)).items():
            col = self.vocab.get(term)
            if col is not None:
                vec[0, col] = (1.0 + math.log(count)) * self.idf[col]
        self._l2_normalize(vec)
        out = vec @ self.components.T
        self._l2_normalize(out)
        return out

    def score(self, query: str) -> list[tuple[float, int]]:
        """Return [(cosine, chunk_index)] sorted by score desc, chunk_id asc."""
        if not self.n_docs or not self.vocab:
            return []
        sims = (self.doc_vecs @ self._embed_query(query).T).ravel()
        out = [(float(sims[i]), i) for i in range(self.n_docs) if sims[i] > 0.0]
        out.sort(key=lambda p: (-p[0], self.chunks[p[1]].chunk_id))
        return out


def rank(chunks: list, query: str, seed: int = 0) -> list[tuple[float, int]]:
    return LSAIndex(chunks, seed=seed).score(query)
