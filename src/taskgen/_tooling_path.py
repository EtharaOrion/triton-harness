"""Wire up the two external import roots taskgen borrows from, then shim tree-sitter.

  HARNESS/src                     -> `from parser.py_parser import PyParser`
  harbor-tasks/shared/tooling     -> `import sigextract`, `from retrieval... import`

Both are vendored trees we deliberately do NOT copy: sigextract/bm25/dense_lsa
are the same implementations the shipped harbor dataset was generated with, and
forking them would silently decouple generated tasks from the reference corpus.

Importing this module is the only supported way to reach either root. It calls
`_ts_compat.install()` FIRST, because `parser.py_parser` imports tree_sitter at
module scope and the shim must already be in place by then.
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import _ts_compat

HARNESS_SRC = Path(__file__).resolve().parents[1]
HARNESS = HARNESS_SRC.parent
ROOT = HARNESS.parent.parent
TOOLING = ROOT / 'harbor-tasks' / 'shared' / 'tooling'


def _prepend(path: Path) -> None:
    text = str(path)
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)


def install() -> None:
    """Idempotent. Safe to call from any taskgen module's import."""
    _ts_compat.install()
    if not TOOLING.is_dir():
        raise RuntimeError(
            f'harbor shared tooling not found at {TOOLING}. taskgen reuses '
            'sigextract/count_tokens/manifest/retrieval from there.'
        )
    _prepend(TOOLING)
    _prepend(HARNESS_SRC)


install()
