"""Wire up the two import roots taskgen borrows from, then shim tree-sitter.

  HARNESS/src                  -> `from parser.py_parser import PyParser`
  HARNESS/src/harbor_tooling   -> `import sigextract`, `from retrieval... import`

BOTH LIVE INSIDE THIS WORKTREE. Every taskgen import resolves from the harness
tree; nothing is mapped in from outside it, so the harness is usable on its own.

`harbor_tooling/` is a verbatim vendored copy of `harbor-tasks/shared/tooling`
(its README.md records the upstream commit and the refresh command). Vendoring
has a cost the previous out-of-tree sys.path injection did not: a copy that
drifts from upstream would silently decouple generated tasks from the reference
corpus the shipped harbor dataset was built with. It is therefore refreshed
wholesale, never edited in place.

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
#: Checkout root. Retained for callers that locate external DATA (repos-src,
#: the vendored wabt tarball); it is never used to resolve an import.
ROOT = HARNESS.parent.parent
TOOLING = HARNESS_SRC / 'harbor_tooling'


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
            f'vendored harbor tooling not found at {TOOLING}. taskgen imports '
            'sigextract/count_tokens/manifest/retrieval from there; restore it '
            'with the refresh command in src/harbor_tooling/README.md.'
        )
    _prepend(TOOLING)
    _prepend(HARNESS_SRC)


install()
