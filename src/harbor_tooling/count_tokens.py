"""Token counting for benchmark context generation.

Two backends:

  tiktoken : cl100k_base, exact for OpenAI-family BPE. Used only if the
             package imports *offline* (no network, no download). tiktoken
             lazily downloads its BPE ranks on first use, so we probe with a
             real encode() call inside a try/except and demote to chars4 if
             anything at all goes wrong.

  chars4   : len(text) / 4, the classic English-prose rule of thumb. On source
             code it runs roughly +-15% against cl100k_base (dense punctuation
             and long identifiers push real token counts up; long runs of
             indentation push them down).

Whichever backend is live is reported by `Counter.method` and MUST be written
into every generated instruction.md. We never silently guess.
"""

from __future__ import annotations

CHARS_PER_TOKEN = 4
CHARS4_ERROR_NOTE = (
    'the chars/4 heuristic carries roughly +-15% error against a real BPE '
    'tokenizer on source code'
)


class Counter:
    """A token counter with a stable, self-describing method name."""

    def __init__(self, method: str, encoding=None) -> None:
        self.method = method
        self._encoding = encoding

    @property
    def exact(self) -> bool:
        return self.method.startswith('tiktoken')

    def count(self, text: str) -> int:
        if self._encoding is not None:
            return len(self._encoding.encode(text, disallowed_special=()))
        return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN

    def truncate(self, text: str, max_tokens: int) -> str:
        """Return the longest prefix of `text` costing <= max_tokens tokens.

        Deterministic. For chars4 this is a straight character slice; for
        tiktoken we slice the token stream and decode back.
        """
        if max_tokens <= 0:
            return ''
        if self._encoding is not None:
            toks = self._encoding.encode(text, disallowed_special=())
            if len(toks) <= max_tokens:
                return text
            return self._encoding.decode(toks[:max_tokens])
        return text[: max_tokens * CHARS_PER_TOKEN]

    def describe(self) -> str:
        if self.exact:
            return (
                'tiktoken cl100k_base (exact BPE count for OpenAI-family '
                'tokenizers; other model families will differ somewhat)'
            )
        return (
            'chars/4 heuristic (tiktoken was not importable offline in the '
            'generation environment, so token counts below are ESTIMATES; '
            + CHARS4_ERROR_NOTE
            + ')'
        )


def _try_tiktoken() -> Counter | None:
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding('cl100k_base')
        # Probe: get_encoding may succeed but the BPE file fetch happens lazily.
        if enc.decode(enc.encode('probe 123', disallowed_special=())) != 'probe 123':
            return None
        return Counter('tiktoken/cl100k_base', enc)
    except Exception:
        return None


def get_counter(preference: str = 'tiktoken') -> Counter:
    """Build a Counter.

    preference='tiktoken' -> use tiktoken if it works offline, else chars4.
    preference='chars4'   -> force the heuristic (useful for reproducibility
                             across machines that differ in tiktoken presence).
    """
    if preference not in ('tiktoken', 'chars4'):
        raise ValueError(f'unknown tokenizer preference: {preference!r}')
    if preference == 'chars4':
        return Counter('chars4')
    got = _try_tiktoken()
    if got is not None:
        return got
    return Counter('chars4')


if __name__ == '__main__':
    import sys

    c = get_counter(sys.argv[1] if len(sys.argv) > 1 else 'tiktoken')
    data = sys.stdin.read()
    print(f'method={c.method} tokens={c.count(data)}')
