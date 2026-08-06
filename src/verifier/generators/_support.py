"""Minimal helpers inlined from modules that stayed in the results/eval half.

Only what the ported soundness loop actually calls is here: the golden/stub
criterion validation from the upstream ``rubric_anchor.py``
(``validate_criteria`` + ``CriterionValidation``). The rest of that module —
building anchor code out of a git checkout, caching anchors.json, calling the
judge — is out of scope, so instead of importing ``judge.JudgeResult`` the judge
verdicts are described structurally: any object with ``ok`` and ``by_id()`` fits,
which is also what lets a caller inject its own executor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .rubric import Rubric


class VerdictLike(Protocol):
    criterion_id: str
    passed: bool
    justification: str


class JudgeResultLike(Protocol):
    @property
    def ok(self) -> bool: ...

    def by_id(self) -> dict[str, VerdictLike]: ...


@dataclass
class CriterionValidation:
    criterion_id: str
    kept: bool
    reason: str


def validate_criteria(rubric: Rubric, golden: JudgeResultLike,
                      stub: JudgeResultLike) -> dict[str, CriterionValidation]:
    """Keep an anchorable criterion iff golden passes it AND stub fails it. Process
    (non-anchorable) criteria are always kept (can't be golden-anchored). Requires
    both anchor judgements to be OK; otherwise everything is kept (anchoring skipped)."""
    gv, sv = golden.by_id(), stub.by_id()
    anchors_ok = golden.ok and stub.ok
    out: dict[str, CriterionValidation] = {}
    for c in rubric.criteria:
        if not c.anchorable or not anchors_ok:
            out[c.id] = CriterionValidation(c.id, True,
                                            "process criterion" if not c.anchorable
                                            else "anchors unavailable")
            continue
        g = gv.get(c.id)
        s = sv.get(c.id)
        if g is not None and not g.passed:
            out[c.id] = CriterionValidation(c.id, False, "golden FAILS it (unsound/too-strict)")
        elif s is not None and s.passed:
            out[c.id] = CriterionValidation(c.id, False, "stub PASSES it (non-discriminating)")
        else:
            out[c.id] = CriterionValidation(c.id, True, "validated (golden-pass ∧ stub-fail)")
    return out
