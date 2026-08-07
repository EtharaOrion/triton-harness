r"""The SHIPPED Dockerfile's bytes, frozen -- the safety net under the dep-plan seam.

`tests/test_render_gap.py` pins the MEASURE render byte for byte. Nothing pinned
the SHIPPED render: the file all eleven entries build, the one `verify` runs
RED/GREEN and the layer-leak scan against. Its per-language tests assert
STRUCTURE ("the leak gate is present", "no solution mount ships"), and structure
survives an edit that moves bytes -- which is exactly the edit that would
silently republish every task image.

So the digests below are captured from the rendering as it stood BEFORE
`render_dockerfile` grew a `dep_plan` parameter. They are the contract that the
seam is ADDITIVE: threading a resolved plan into the shipped render must not
move a single byte of the plan-free path, on any language, and a `None` plan is
the only thing `emit.py` passes for a parser-backed language ever.

python and go are pinned alongside the four whole-suite languages on purpose.
They have no `render_gap` at all, so a plan can never reach their slot -- which
makes them the control: if their digest moves, the change leaked out of the
`dep_plan is not None` branch and into the shared composition.

Pure: no docker, no network, no LLM, no clock, no host path.
"""

from __future__ import annotations

import hashlib
import inspect

import pytest

from taskgen import langs
from taskgen.langs import base as B

#: sha256 of `get(lang).render_dockerfile(EnvSpec(repo_name=repo))` for every
#: registered language, measured against the tree BEFORE the shipped render took
#: a `dep_plan`. `(lang, repo, sha256, length)` -- the length is carried too so a
#: digest mismatch reports whether the rendering grew, shrank or merely moved.
#: The rust row is the ONE deliberate exception: its strings leak-assert used to
#: grep target/ for the bare crate name and false-positived on every crate whose
#: name is a substring of a std symbol, so the assert now matches by rust
#: mangling structure and the block's bytes moved with it (see
#: `taskgen.langs.rust.leak_symbol_ere`). No other row may ever move.
SHIPPED_DOCKERFILE_GOLDENS: tuple[tuple[str, str, str, int], ...] = (
    ('c', 'c-xs',
     'f6f4dacff417dadbfdcca5ba135ce93d6c9ddb5179fd68a38f81fa98098a8c9d', 2427),
    ('cpp', 'cpp-Rux',
     '5af1d18e4d3f1498ac26609347ff537c0d5268f636695202418559831718408f', 2926),
    ('java', 'java-tamboui',
     'fcf95d7e1891e04e75cac57e91202026f9d678ac678a045d62b90532f9c205ab', 10164),
    ('rust', 'rust-spacewasm',
     'f54fb16d07a3efd52000bc68419cfd1c2dda73cbdd47a54806a89abf530477ef', 7645),
    ('python', 'python-a2a-python',
     '684a019f17ab5794f4a68b9535511a96abed98feccb2733c16cfc3b7774675c5', 4623),
    ('go', 'go-multigres',
     '5d38a2a13eb1d679e458d6a6eb19610218e08ed42360db8bfd620a0d4217a5ca', 4013),
)

#: Every language the golden table must cover. Read off the registry rather than
#: respelled, so a plugin added without a frozen digest fails here instead of
#: shipping an unpinned image.
_GOLDEN_LANGS = {lang for lang, _, _, _ in SHIPPED_DOCKERFILE_GOLDENS}


def _render(lang: str, repo: str) -> str:
    return langs.get(lang).render_dockerfile(B.EnvSpec(repo_name=repo))


@pytest.mark.parametrize(
    ('lang', 'repo', 'digest', 'length'), SHIPPED_DOCKERFILE_GOLDENS,
    ids=[lang for lang, _, _, _ in SHIPPED_DOCKERFILE_GOLDENS],
)
def test_the_shipped_dockerfile_hashes_to_its_frozen_digest(
    lang: str, repo: str, digest: str, length: int,
) -> None:
    """The load-bearing one: the shipped image's bytes have not moved."""
    text = _render(lang, repo)
    assert len(text) == length
    assert hashlib.sha256(text.encode('utf-8')).hexdigest() == digest


@pytest.mark.parametrize(
    ('lang', 'repo'), [(lang, repo) for lang, repo, _, _ in SHIPPED_DOCKERFILE_GOLDENS],
)
def test_the_shipped_dockerfile_is_deterministic(lang: str, repo: str) -> None:
    """Two renderings of one spec are the same string, not merely equal digests."""
    assert _render(lang, repo) == _render(lang, repo)


@pytest.mark.parametrize(
    ('lang', 'repo'), [(lang, repo) for lang, repo, _, _ in SHIPPED_DOCKERFILE_GOLDENS],
)
def test_the_default_render_takes_no_positional_argument_beyond_env(
    lang: str, repo: str,
) -> None:
    """`dep_plan` must be KEYWORD-ONLY, so no caller can pass one by position.

    A positional plan would make `render_dockerfile(env, something)` compile at
    every existing call site and change what ships.
    """
    signature = inspect.signature(type(langs.get(lang)).render_dockerfile)
    positional = [
        name for name, parameter in signature.parameters.items()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert positional == ['self', 'env']


def test_every_registered_language_has_a_frozen_shipped_digest() -> None:
    """A new plugin cannot ship an unpinned image by simply not being listed."""
    assert set(langs.available()) == _GOLDEN_LANGS
