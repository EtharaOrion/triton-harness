"""Build a `TruthInputs` out of taskgen's own artifacts — the git-seam replacement.

The upstream generator derived every input from a git repository: `git diff base..ref`
for the golden diff, `git show <ref>:<file>` for the golden code, a PDF for the
spec. taskgen already holds BOTH sides of that diff in memory, so the whole
discovery layer collapses to pure functions over one `CarveSet`:

    golden_code   carve.originals[rel]      the INTACT oracle
    stub_code     carve.overlay[rel]        the emitted stub
    golden_diff   difflib.unified_diff(stub, oracle) per carved file
    stub_files    carve.carved_relpaths
    fail_to_pass  the linked pytest node ids (tests/allowlist.txt == graded.selectors)
    spec_text     Target.docstring (the prompt's specification), never the body

No git, no PDF, no network, no subprocess. The diff is emitted in the
`--- a/<rel>` / `+++ b/<rel>` shape `manifest.target_files_from_diff` and
`manifest.required_target_files` parse, so the deterministic manifest keeps
working unchanged against a taskgen-built diff.

DIRECTION MATTERS. The diff is stub -> oracle, i.e. what a correct solve ADDS.
That is what the upstream `base..reference` produced, and it is what the TRUTH.md
leak-guard (`truth._golden_added_lines`) scans: reversing it would hand the guard
the deleted stub lines and silently disarm it.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .generators.truth import TruthInputs

__all__ = [
    "CarveLike",
    "SolutionCode",
    "build_truth_inputs",
    "golden_diff",
    "solution_code",
]


class CarveLike(Protocol):
    """The slice of `taskgen.carve.CarveSet` this module reads.

    Stated structurally so `verifier` never imports the carve module at type
    level: the port stays usable from a bare `TruthInputs` caller too. Read-only
    members, because `CarveSet` is a FROZEN dataclass -- a mutable protocol
    attribute would exclude the one type this exists to describe.
    """

    @property
    def carved_relpaths(self) -> tuple[str, ...]: ...

    @property
    def deleted_relpaths(self) -> tuple[str, ...]: ...

    @property
    def overlay(self) -> Mapping[str, str]: ...

    @property
    def originals(self) -> Mapping[str, str]: ...


@dataclass(frozen=True)
class SolutionCode:
    """The two bare code bodies the rubric soundness loop judges against.

    `sound_rubric_loop` anchors every criterion on golden-pass ∧ stub-fail, and
    it judges CODE, not a trajectory — so it needs the concatenated carved
    sources of each side rather than the diff between them.
    """

    golden: str
    stub: str


def _posix(rel: str) -> str:
    return Path(rel).as_posix()


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def golden_diff(carve: CarveLike) -> str:
    """`unified_diff(stub, oracle)` over every carved file, path-sorted.

    A file the carve DELETED outright has no overlay text, so its stub side is
    empty and the whole oracle body reads as added — which is exactly what the
    solver has to write.
    """
    parts: list[str] = []
    for rel in sorted(_posix(r) for r in carve.carved_relpaths):
        oracle = carve.originals.get(rel)
        if oracle is None:
            # Not an oracle-bearing path (nothing to diff against); skipping is
            # safer than inventing an empty golden side, which would read as a
            # deletion of the whole file.
            continue
        stub = carve.overlay.get(rel, "")
        chunk = list(difflib.unified_diff(
            _lines(stub), _lines(oracle),
            fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm="\n",
        ))
        if not chunk:
            continue
        parts.append(f"diff --git a/{rel} b/{rel}\n")
        parts.extend(chunk)
        if parts[-1] and not parts[-1].endswith("\n"):
            parts.append("\n")
    return "".join(parts)


def solution_code(carve: CarveLike) -> SolutionCode:
    """The carved files' oracle text and stub text, each concatenated in path order."""
    golden: list[str] = []
    stub: list[str] = []
    for rel in sorted(_posix(r) for r in carve.carved_relpaths):
        oracle = carve.originals.get(rel)
        if oracle is not None:
            golden.append(f"# ==== {rel} ====\n{oracle}")
        stub.append(f"# ==== {rel} ====\n{carve.overlay.get(rel, '')}")
    return SolutionCode(golden="\n".join(golden), stub="\n".join(stub))


def _derived_fail_to_pass(target: object) -> list[str]:
    """The linked pytest node ids, formatted exactly as tests/allowlist.txt is.

    `taskgen.nodeids.format_nodeid` is the single definition of that format (the
    class segment is dropped for module-level tests); it is imported lazily so
    this package still imports without taskgen on the path.
    """
    tests = getattr(target, "tests", None)
    if not tests:
        return []
    from taskgen.nodeids import format_nodeid

    return sorted({
        format_nodeid(t.relpath, getattr(t, "class_name", "") or "", t.name)
        for t in tests
    })


def _derived_repo_name(target: object) -> str:
    repo = getattr(target, "repo", None)
    return Path(repo).name if repo else ""


def build_truth_inputs(
    carve: CarveLike,
    target: object,
    language: str,
    *,
    repo_name: str = "",
    fail_to_pass: Sequence[str] | None = None,
    spec_text: str | None = None,
) -> TruthInputs:
    """Everything the generators need, from taskgen artifacts alone.

    Only the three positional arguments are load-bearing; the keyword arguments
    let the caller pass the values it already computed authoritatively (the
    graded selectors ARE the emitted allowlist, and `plan.repo_name` is the
    identity that seeds the entry ids) instead of re-deriving them here.
    """
    ids = list(fail_to_pass) if fail_to_pass is not None else _derived_fail_to_pass(target)
    spec = spec_text
    if spec is None:
        # Function scope: the target's docstring IS the specification handed to
        # the solver. File/folder scope has no single target, so fall back to the
        # carve's own docstring view, which is empty there — an empty spec is
        # honest, an invented one is not.
        spec = getattr(target, "docstring", "") or getattr(carve, "docstring", "") or ""
    return TruthInputs(
        repo=repo_name or _derived_repo_name(target),
        language=language,
        spec_text=spec,
        golden_diff=golden_diff(carve),
        fail_to_pass=ids,
        pass_to_pass=[],
        stub_files=sorted(_posix(r) for r in carve.carved_relpaths),
    )
