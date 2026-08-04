"""Deterministic line-aligned chunking of the surviving repository.

Chunks are ~512 tokens, never split mid-line, and carry the provenance needed
for the inline headers required by the RAG conditions: path, 1-based inclusive
line range, and chunk ordinal within its file.

Determinism: files are visited in path-sorted order; within a file, chunks are
cut greedily front-to-back. No randomness, no dict-ordering dependence.
"""

from __future__ import annotations

from dataclasses import dataclass

TARGET_TOKENS = 512


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    relpath: str
    start_line: int
    end_line: int
    text: str
    tokens: int

    @property
    def provenance(self) -> str:
        return f'{self.relpath}:{self.start_line}-{self.end_line}'


def chunk_file(relpath: str, text: str, counter, target: int = TARGET_TOKENS) -> list[tuple]:
    """Split one file into (start_line, end_line, text, tokens) tuples."""
    lines = text.splitlines()
    if not lines:
        return []
    out: list[tuple] = []
    buf: list[str] = []
    buf_tokens = 0
    start = 1
    for idx, line in enumerate(lines, start=1):
        # +1 approximates the newline's cost under either backend.
        cost = counter.count(line) + 1
        if buf and buf_tokens + cost > target:
            body = '\n'.join(buf)
            out.append((start, idx - 1, body, buf_tokens))
            buf = []
            buf_tokens = 0
            start = idx
        buf.append(line)
        buf_tokens += cost
    if buf:
        out.append((start, len(lines), '\n'.join(buf), buf_tokens))
    return out


def build_corpus(files: list[tuple[str, str]], counter, target: int = TARGET_TOKENS) -> list[Chunk]:
    """files: list of (relpath, text) -- MUST already exclude carved files."""
    chunks: list[Chunk] = []
    cid = 0
    for relpath, text in sorted(files, key=lambda p: p[0]):
        for start, end, body, tok in chunk_file(relpath, text, counter, target):
            if not body.strip():
                continue
            chunks.append(Chunk(cid, relpath, start, end, body, tok))
            cid += 1
    return chunks
